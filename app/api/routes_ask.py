from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_app_state
from app.core.attribution.chunk_attribution import compute_attribution, normalize_to_percentages
from app.core.attribution.citation_validation import (
    citation_validity_rate,
    extract_and_validate_citations,
)
from app.core.attribution.consensus import find_disagreements
from app.core.attribution.token_grounding import compute_token_grounding, grounding_ratio
from app.core.generation.answer import generate_answer
from app.core.generation.llm_client import LLMError
from app.models.schemas import AskRequest, AskResponse, EvidenceItem
from app.state import AppState

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, state: AppState = Depends(get_app_state)):
    if not state.retrieval.is_ready:
        raise HTTPException(
            status_code=409, detail="Knowledge base is empty. Ingest a document first."
        )

    settings = state.settings
    started = time.perf_counter()

    ranked = await state.retrieval.retrieve(
        request.query, top_k=request.top_k, mode=request.mode
    )
    if not ranked:
        raise HTTPException(status_code=404, detail="No relevant evidence found.")

    chunks = [chunk for chunk, _ in ranked]
    scores_by_id = {chunk.chunk_id: score for chunk, score in ranked}

    # Run before generation: the conflict list is fed into the prompt so the
    # model can address both sides in its own words (plan Sec. 5.1).
    disagreements = []
    if request.detect_contradictions:
        try:
            disagreements = await find_disagreements(chunks, settings)
        except Exception:
            disagreements = []  # never fail a query over the optional feature

    try:
        answer = await generate_answer(request.query, chunks, settings, disagreements)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    attribution = compute_attribution(request.query, answer, chunks, settings)
    percentages = normalize_to_percentages(attribution)
    citations = extract_and_validate_citations(answer, chunks)

    context_text = "\n".join(c.text for c in chunks)
    token_scores = compute_token_grounding(answer, context_text)
    ratio = grounding_ratio(token_scores, settings.grounding_threshold)

    evidence = [
        EvidenceItem(
            chunk=chunk,
            rerank_score=scores_by_id[chunk.chunk_id],
            attribution_score=attribution.get(chunk.chunk_id),
            attribution_percent=percentages.get(chunk.chunk_id),
        )
        for chunk in chunks
    ]

    return AskResponse(
        answer=answer,
        evidence=evidence,
        token_grounding=token_scores,
        citations=citations,
        grounding_ratio=ratio,
        citation_validity_rate=citation_validity_rate(citations),
        disagreements=disagreements,
        mode=request.mode,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )
