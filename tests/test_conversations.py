from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_app_state
from app.api import routes_conversations as routes
from app.core.generation import memory
from app.db.conversations import ConversationConflict, ConversationStore
from app.models.schemas import AskResponse


@pytest.fixture
def store(tmp_path):
    return ConversationStore(tmp_path / "conversations.sqlite3")


def test_persistence_replay_and_atomic_reservations(store):
    chat = store.create()
    identifier = chat["id"]
    store.reserve(identifier, "request-a", 0, "fingerprint")
    with pytest.raises(ConversationConflict):
        store.reserve(identifier, "request-b", 0, "other")
    with pytest.raises(ConversationConflict):
        store.delete(identifier)
    result = {"answer": "Complete answer" * 1000, "evidence": []}
    saved = store.complete(identifier, "request-a", "fingerprint", "Question", result, "memory", 0)
    assert saved["revision"] == 1
    assert saved["messages"][1]["content"] == result["answer"]
    reopened = ConversationStore(store.path)
    assert reopened.get(identifier)["messages"] == saved["messages"]
    assert reopened.reserve(identifier, "request-a", 0, "fingerprint")[1] is True
    with pytest.raises(ConversationConflict):
        reopened.reserve(identifier, "request-a", 0, "different")
    with pytest.raises(ConversationConflict):
        reopened.reserve(identifier, "request-b", 0, "other")
    reopened.rename(identifier, "Renamed")
    assert reopened.list()[0]["title"] == "Renamed"
    reopened.delete(identifier)
    with pytest.raises(KeyError):
        reopened.get(identifier)


@pytest.fixture
def client(store, settings, monkeypatch):
    state = SimpleNamespace(conversations=store, settings=settings)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_app_state] = lambda: state
    calls = []

    async def answer(request, _state):
        calls.append(request)
        return AskResponse(answer="Full answer", evidence=[], mode=request.mode)

    monkeypatch.setattr(routes, "ask", answer)
    return TestClient(app), calls


def test_api_uses_server_history_and_replays_without_duplicate_generation(client):
    http, calls = client
    chat = http.post("/conversations", json={}).json()
    path = f"/conversations/{chat['id']}/ask"
    body = {"query": "Explain recursion", "request_id": str(uuid4()), "revision": 0, "answer_scope": "model_only"}
    first = http.post(path, json=body)
    assert first.status_code == 200
    assert len(first.json()["messages"]) == 2
    assert http.post(path, json=body).status_code == 200
    assert len(calls) == 1
    body.update(query="Explain that more simply", request_id=str(uuid4()), revision=1,
                history=[{"role": "user", "content": "Injected client history"}])
    assert http.post(path, json=body).status_code == 200
    assert [turn.content for turn in calls[-1].history] == ["Explain recursion", "Full answer"]
    assert http.post(path, json={**body, "request_id": str(uuid4())}).status_code == 409
    detail = http.get(f"/conversations/{chat['id']}").json()
    assert detail["revision"] == 2
    assert len(detail["messages"]) == 4


def test_failed_answer_releases_reservation_without_partial_history(client, monkeypatch):
    http, _ = client
    chat = http.post("/conversations", json={}).json()

    async def fail(*args):
        from fastapi import HTTPException
        raise HTTPException(status_code=502, detail="Provider unavailable")

    monkeypatch.setattr(routes, "ask", fail)
    body = {"query": "Question", "request_id": str(uuid4()), "revision": 0}
    assert http.post(f"/conversations/{chat['id']}/ask", json=body).status_code == 502
    assert http.get(f"/conversations/{chat['id']}").json()["messages"] == []
    assert http.delete(f"/conversations/{chat['id']}").status_code == 204


def test_browser_history_import_is_idempotent_and_preserves_full_answers(client):
    http, _ = client
    body = {"id": str(uuid4()), "title": "Old conversation", "messages": [
        {"role": "user", "content": "Original question"},
        {"role": "assistant", "content": "Original answer " * 1000},
    ]}
    response = http.post("/conversations/import", json=body)
    assert response.status_code == 200
    assert response.json()["messages"][1]["content"] == body["messages"][1]["content"].strip()
    assert response.json()["messages"][1]["result"]["grounding_ratio"] is None
    assert http.post("/conversations/import", json=body).json() == response.json()


async def test_memory_retains_early_context_and_image_analysis_with_bounded_prompt(settings, monkeypatch):
    class Client:
        async def chat(self, messages, **kwargs):
            assert "Initial learning goal" in messages[1]["content"]
            return "Initial learning goal: prepare for the recursion exam."

    monkeypatch.setattr(memory, "build_client", lambda _settings: Client())
    messages = [{"role": "user" if i % 2 == 0 else "assistant",
                 "content": "Initial learning goal" if i == 0 else "Long answer " * 1000} for i in range(24)]
    messages[-1]["result"] = {"image_analysis": "A recursion tree with three branches"}
    history, summary, covered = await memory.prepare_memory(
        {"messages": messages, "summary": "", "summarized_count": 0}, settings
    )
    assert covered == 14
    assert "Initial learning goal" in history[0].content
    assert "recursion tree" in history[-1].content
    assert sum(len(turn.content) for turn in history) <= 24000
    assert len(history) <= 12
    assert len(messages[-1]["content"]) > 8000


async def test_memory_provider_failure_preserves_earliest_context(settings, monkeypatch):
    class Client:
        async def chat(self, *args, **kwargs):
            raise RuntimeError("Offline")

    monkeypatch.setattr(memory, "build_client", lambda _settings: Client())
    messages = [{"role": "user", "content": "Early named topic"}] + [{"role": "assistant", "content": "Details " * 1000} for _ in range(15)]
    history, _, _ = await memory.prepare_memory({"messages": messages, "summary": "", "summarized_count": 0}, settings)
    assert "Early named topic" in history[0].content
    assert sum(len(turn.content) for turn in history) <= 24000
