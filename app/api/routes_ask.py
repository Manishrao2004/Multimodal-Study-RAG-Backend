from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_app_state
from app.core.attribution.chunk_attribution import compute_attribution
from app.core.attribution.citation_validation import extract_and_validate_citations
from app.core.attribution.consensus import detect_disagreements
from app.core.attribution.token_grounding import compute_token_grounding, grounding_ratio
from app.core.generation.answer import generate_answer
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

    ranked = await state.retrieval.retrieve(request.query, top_k=request.top_k)
    if not ranked:
        raise HTTPException(status_code=404, detail="No relevant evidence found.")

    chunks = [chunk for chunk, _ in ranked]
    scores_by_id = {chunk.chunk_id: score for chunk, score in ranked}

    answer = await generate_answer(request.query, chunks, settings)

    attribution = compute_attribution(request.query, answer, chunks, settings)
    citations = extract_and_validate_citations(answer, chunks)

    context_text = "\n".join(c.text for c in chunks)
    token_scores = compute_token_grounding(answer, context_text)
    ratio = grounding_ratio(token_scores, settings.grounding_threshold)

    disagreements = detect_disagreements(chunks, settings.embedding_model)
    if disagreements:
        notes = "; ".join(
            f"{d.source_a} vs {d.source_b} (similarity {d.similarity:.2f})" for d in disagreements
        )
        answer += f"\n\n⚠ Source disagreement detected: {notes}"

    evidence = [
        EvidenceItem(
            chunk=chunk,
            rerank_score=scores_by_id[chunk.chunk_id],
            attribution_score=attribution.get(chunk.chunk_id),
        )
        for chunk in chunks
    ]

    return AskResponse(
        answer=answer,
        evidence=evidence,
        token_grounding=token_scores,
        citations=citations,
        grounding_ratio=ratio,
    )
