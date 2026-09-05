"""Study-productivity endpoints (plan Sec. 5.2): summarise, compare, quiz.

All three retrieve first and generate second, exactly like /ask, so every
output carries the same citation guarantees rather than being free-form LLM
text.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_app_state
from app.core.attribution.chunk_attribution import compute_attribution, normalize_to_percentages
from app.core.attribution.citation_validation import (
    citation_validity_rate,
    extract_and_validate_citations,
)
from app.core.attribution.token_grounding import compute_token_grounding, grounding_ratio
from app.core.generation.llm_client import LLMError
from app.core.generation.study_tools import (
    generate_comparison,
    generate_quiz,
    generate_summary,
)
from app.models.schemas import (
    Chunk,
    CompareRequest,
    CompareResponse,
    EvidenceItem,
    QuizRequest,
    QuizResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from app.state import AppState

router = APIRouter(prefix="/study", tags=["study"])


def _require_index(state: AppState) -> None:
    if not state.retrieval.is_ready:
        raise HTTPException(
            status_code=409, detail="Knowledge base is empty. Ingest a document first."
        )


def _evidence(
    ranked: list[tuple[Chunk, float]], attribution: dict[str, float]
) -> list[EvidenceItem]:
    percentages = normalize_to_percentages(attribution)
    return [
        EvidenceItem(
            chunk=chunk,
            rerank_score=score,
            attribution_score=attribution.get(chunk.chunk_id),
            attribution_percent=percentages.get(chunk.chunk_id),
        )
        for chunk, score in ranked
    ]


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest, state: AppState = Depends(get_app_state)):
    _require_index(state)
    ranked = await state.retrieval.retrieve(
        request.topic, top_k=request.top_k, source_file=request.source_file
    )
    if not ranked:
        raise HTTPException(status_code=404, detail="No relevant material found for that topic.")

    chunks = [c for c, _ in ranked]
    try:
        summary = await generate_summary(request.topic, chunks, state.settings)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    citations = extract_and_validate_citations(summary, chunks)
    token_scores = compute_token_grounding(summary, "\n".join(c.text for c in chunks))
    attribution = compute_attribution(request.topic, summary, chunks, state.settings)

    return SummarizeResponse(
        summary=summary,
        evidence=_evidence(ranked, attribution),
        citations=citations,
        citation_validity_rate=citation_validity_rate(citations),
        grounding_ratio=grounding_ratio(token_scores, state.settings.grounding_threshold),
    )


@router.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest, state: AppState = Depends(get_app_state)):
    _require_index(state)

    known = {d.source_file for d in state.kb.list_documents()}
    for source in (request.source_a, request.source_b):
        if source not in known:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown document '{source}'. Known documents: {sorted(known)}",
            )
    if request.source_a == request.source_b:
        raise HTTPException(status_code=400, detail="Pick two different documents to compare.")

    ranked_a = await state.retrieval.retrieve(
        request.topic, top_k=request.top_k, source_file=request.source_a
    )
    ranked_b = await state.retrieval.retrieve(
        request.topic, top_k=request.top_k, source_file=request.source_b
    )
    if not ranked_a or not ranked_b:
        missing = request.source_a if not ranked_a else request.source_b
        raise HTTPException(
            status_code=404, detail=f"No material about '{request.topic}' found in '{missing}'."
        )

    chunks_a = [c for c, _ in ranked_a]
    chunks_b = [c for c, _ in ranked_b]
    try:
        comparison = await generate_comparison(
            request.topic, chunks_a, chunks_b, request.source_a, request.source_b, state.settings
        )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # One shared numbering across both sources, matching the prompt's context.
    citations = extract_and_validate_citations(comparison, chunks_a + chunks_b)
    attribution = compute_attribution(
        request.topic, comparison, chunks_a + chunks_b, state.settings
    )

    return CompareResponse(
        comparison=comparison,
        evidence_a=_evidence(ranked_a, attribution),
        evidence_b=_evidence(ranked_b, attribution),
        citations=citations,
        citation_validity_rate=citation_validity_rate(citations),
    )


@router.post("/quiz", response_model=QuizResponse)
async def quiz(request: QuizRequest, state: AppState = Depends(get_app_state)):
    _require_index(state)
    ranked = await state.retrieval.retrieve(
        request.topic, top_k=request.top_k, source_file=request.source_file
    )
    if not ranked:
        raise HTTPException(status_code=404, detail="No relevant material found for that topic.")

    chunks = [c for c, _ in ranked]
    try:
        items, dropped = await generate_quiz(
            request.topic, chunks, request.count, request.format, state.settings
        )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    attribution = compute_attribution(
        request.topic, " ".join(i.question + " " + i.answer for i in items), chunks, state.settings
    )
    total = len(items) + dropped
    return QuizResponse(
        items=items,
        evidence=_evidence(ranked, attribution),
        grounded_rate=(len(items) / total) if total else 0.0,
        dropped=dropped,
    )
