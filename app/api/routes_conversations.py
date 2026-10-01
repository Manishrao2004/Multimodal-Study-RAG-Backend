from __future__ import annotations

import hashlib
import json
from uuid import UUID, uuid4
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_app_state
from app.api.routes_ask import ask
from app.api.routes_vision import ask_about_screenshot
from app.core.generation.memory import prepare_memory
from app.db.conversations import ConversationConflict
from app.models.conversations import (
    ConversationDetail, ConversationQuestion, ConversationSummary, ConversationTitle,
    ConversationImport,
)
from app.models.schemas import AskRequest, AskResponse, AnswerBasis, RetrievalMode
from app.core.security import validate_query_text, SecurityValidationError
from app.state import AppState

router = APIRouter(prefix="/conversations", tags=["conversations"])


def translate_error(exc):
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="Conversation not found.")
    return HTTPException(status_code=409, detail=str(exc))


@router.get("", response_model=list[ConversationSummary])
def list_conversations(state: AppState = Depends(get_app_state)):
    return state.conversations.list()


@router.post("", response_model=ConversationDetail, status_code=201)
def create_conversation(body: ConversationTitle, state: AppState = Depends(get_app_state)):
    return state.conversations.create(body.title)


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: UUID, state: AppState = Depends(get_app_state)):
    try:
        return state.conversations.get(str(conversation_id))
    except KeyError as exc:
        raise translate_error(exc) from exc


@router.post("/import", response_model=ConversationDetail)
def import_conversation(body: ConversationImport, state: AppState = Depends(get_app_state)):
    messages = []
    for message in body.messages:
        entry = {"id": str(uuid4()), "role": message.role, "content": message.content, "created_at": time.time()}
        if message.role == "assistant":
            entry["result"] = AskResponse(
                answer=message.content, evidence=message.evidence, mode=RetrievalMode.full,
                answer_basis=AnswerBasis.sources if message.evidence else AnswerBasis.model,
                provenance_notice="Imported browser history. Original answer quality metrics were not saved.",
            ).model_dump(mode="json")
        messages.append(entry)
    return state.conversations.import_history(str(body.id), body.title, messages)


@router.patch("/{conversation_id}", response_model=ConversationDetail)
def rename_conversation(conversation_id: UUID, body: ConversationTitle, state: AppState = Depends(get_app_state)):
    try:
        return state.conversations.rename(str(conversation_id), body.title)
    except KeyError as exc:
        raise translate_error(exc) from exc


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: UUID, state: AppState = Depends(get_app_state)):
    try:
        state.conversations.delete(str(conversation_id))
    except (KeyError, ConversationConflict) as exc:
        raise translate_error(exc) from exc


async def run_turn(identifier, request_id, revision, fingerprint, query, generate, state):
    store = state.conversations
    try:
        conversation, replay = store.reserve(identifier, request_id, revision, fingerprint)
        if replay:
            return conversation
    except (KeyError, ConversationConflict) as exc:
        raise translate_error(exc) from exc
    try:
        history, summary, covered = await prepare_memory(conversation, state.settings)
        result = await generate(history)
        return store.complete(identifier, request_id, fingerprint, query, result.model_dump(mode="json"), summary, covered)
    except ConversationConflict as exc:
        raise translate_error(exc) from exc
    finally:
        store.release(identifier, request_id)


@router.post("/{conversation_id}/ask", response_model=ConversationDetail)
async def conversation_ask(conversation_id: UUID, body: ConversationQuestion, state: AppState = Depends(get_app_state)):
    parameters = body.model_dump(exclude={"history", "request_id", "revision"}, mode="json")
    fingerprint = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()

    async def generate(history):
        return await ask(AskRequest(**parameters, history=history), state)

    return await run_turn(str(conversation_id), str(body.request_id), body.revision,
                          fingerprint, body.query, generate, state)


@router.post("/{conversation_id}/vision", response_model=ConversationDetail)
async def conversation_vision(
    conversation_id: UUID,
    question: str = Form(...), image: UploadFile = File(...), request_id: UUID = Form(...),
    revision: int = Form(..., ge=0), use_knowledge_base: bool = Form(True),
    save_to_library: bool = Form(False), top_k: int = Form(5, ge=1, le=10),
    state: AppState = Depends(get_app_state),
):
    try:
        question = validate_query_text(question, max_length=4000)
    except SecurityValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raw = await image.read(state.settings.security_max_screenshot_bytes + 1)
    if len(raw) > state.settings.security_max_screenshot_bytes:
        raise HTTPException(status_code=413, detail="Screenshot exceeds the size limit.")
    await image.seek(0)
    fingerprint = hashlib.sha256(raw + json.dumps(
        [question, use_knowledge_base, save_to_library, top_k]
    ).encode()).hexdigest()

    async def generate(history):
        return await ask_about_screenshot(
            question=question, image=image, use_knowledge_base=use_knowledge_base,
            save_to_library=save_to_library, top_k=top_k, state=state,
            history_json=json.dumps([turn.model_dump() for turn in history]),
        )

    return await run_turn(str(conversation_id), str(request_id), revision, fingerprint,
                          question, generate, state)
