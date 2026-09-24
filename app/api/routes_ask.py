from __future__ import annotations

import asyncio
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


async def _comprehensive_retrieve(request: AskRequest, state: AppState):
    """Retrieve a compact but source-diverse evidence set for broad questions.

    A regular RAG query rewards the globally best passages, which can make one
    detailed document crowd out the rest of the library. Comprehensive mode
    deliberately asks each source for its strongest passages, then preserves
    the global results as a relevance backstop.
    """
    sources = [document.source_file for document in state.kb.list_documents()]
    per_source = min(3, max(1, request.top_k))
    global_limit = min(20, max(12, request.top_k * 3))
    global_hits = await state.retrieval.retrieve(
        request.query, top_k=global_limit, mode=request.mode
    )
    by_source = await asyncio.gather(
        *(
            state.retrieval.retrieve(
                request.query, top_k=per_source, mode=request.mode, source_file=source
            )
            for source in sources
        )
    )

    selected: list[tuple] = []
    seen: set[str] = set()
    # Give every source with relevant material a chance to contribute before
    # filling the remainder by global relevance.
    for hits in by_source:
        for hit in hits:
            if hit[0].chunk_id not in seen:
                selected.append(hit)
                seen.add(hit[0].chunk_id)
    for hit in global_hits:
        if hit[0].chunk_id not in seen:
            selected.append(hit)
            seen.add(hit[0].chunk_id)

    return selected[:global_limit], sources


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, state: AppState = Depends(get_app_state)):
    if not state.retrieval.is_ready:
        raise HTTPException(
            status_code=409, detail="Knowledge base is empty. Ingest a document first."
        )

    settings = state.settings
    started = time.perf_counter()

    sources_considered: list[str] = []
    if request.comprehensive:
        ranked, sources_considered = await _comprehensive_retrieve(request, state)
    else:
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
        generation_args = (request.query, chunks, settings, disagreements)
        if request.history:
            answer = await generate_answer(*generation_args, history=request.history)
        else:
            answer = await generate_answer(*generation_args)
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
        comprehensive=request.comprehensive,
        sources_considered=sources_considered,
        sources_used=sorted({chunk.source_file for chunk in chunks}),
    )
