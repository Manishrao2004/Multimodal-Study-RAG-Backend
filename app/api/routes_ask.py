from __future__ import annotations

import asyncio
import re
import time

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_app_state
from app.core.attribution.chunk_attribution import compute_attribution, normalize_to_percentages
from app.core.attribution.citation_validation import (
    citation_validity_rate,
    extract_and_validate_citations,
    normalize_citation_markers,
)
from app.core.attribution.consensus import find_disagreements
from app.core.attribution.token_grounding import compute_token_grounding, grounding_ratio
from app.core.generation.answer import (
    generate_answer,
    generate_general_answer,
    generate_mixed_answer,
)
from app.core.generation.llm_client import LLMError
from app.core.generation.aspect_pipeline import assess_aspects
from app.core.generation.aspect_evidence import apply_map
from app.core.generation.claim_verification import verify_and_repair, source_excerpt_fallback
from app.core.generation.orchestration import (
    evaluate_evidence,
    evaluate_aspect_evidence,
    plan_question,
    preserve_question_scope,
    validate_evidence_decision,
)
from app.core.retrieval.sufficiency import assess_evidence_sufficiency
from app.core.retrieval.context_expansion import expand_local_context, bound_context
from app.core.vision.evidence import inspect_retrieved_figures, requires_visual_evidence
from app.models.schemas import (
    AnswerBasis,
    AnswerScope,
    AskRequest,
    AskResponse,
    ClaimVerification,
    EvidenceItem,
    EvidenceSufficiency,
    QuestionPlan,
    ChunkType,
    EvidenceMethod,
    EvidenceTrace,
)
from app.state import AppState

router = APIRouter(tags=["ask"])


def _requires_source_answer(query: str) -> bool:
    return bool(re.search(
        r"\baccording to (?:the|my|this|that|uploaded|provided|source|notes|document|pdf)\b|\bas stated in\b|\bmy (?:pdf|notes|document|file)\b"
        r"|\b(?:this|that|the|provided|uploaded) (?:figure|diagram|table|chart|document|pdf)\b"
        r"|\bpage\s+\d+\b", query, re.I))

async def _comprehensive_retrieve(request: AskRequest, state: AppState, query: str):
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
        query, top_k=global_limit, mode=request.mode
    )
    by_source = await asyncio.gather(
        *(
            state.retrieval.retrieve(
                query, top_k=per_source, mode=request.mode, source_file=source
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
    method = request.evidence_method or EvidenceMethod(state.settings.evidence_method)
    strict_source = request.answer_scope == AnswerScope.sources_only or (
        method == EvidenceMethod.edurag and request.answer_scope != AnswerScope.model_only)
    if strict_source and not state.retrieval.is_ready:
        raise HTTPException(
            status_code=409, detail="Knowledge base is empty. Ingest a document first."
        )

    settings = state.settings
    started = time.perf_counter()

    sources_considered: list[str] = []
    ranked = []
    try:
        question_plan = (QuestionPlan(standalone_query=request.query) if method == EvidenceMethod.edurag
                         else await plan_question(request.query, request.history, settings))
    except Exception:
        # Planning improves routing and follow-ups but must never make the main
        # answer unavailable when a provider has a transient failure.
        question_plan = QuestionPlan(standalone_query=request.query)
    question_plan = preserve_question_scope(question_plan, request.query)
    resolved_query = question_plan.standalone_query
    if method == EvidenceMethod.aspect and not settings.aspect_decomposition_enabled:
        question_plan = question_plan.model_copy(update={"requested_aspects": [resolved_query]})
    if request.answer_scope != AnswerScope.model_only and state.retrieval.is_ready:
        if request.comprehensive:
            ranked, sources_considered = await _comprehensive_retrieve(
                request, state, resolved_query
            )
        else:
            ranked = await state.retrieval.retrieve(
                resolved_query, top_k=request.top_k, mode=request.mode
            )
    if strict_source and not ranked and method != EvidenceMethod.aspect:
        raise HTTPException(status_code=404, detail="No relevant evidence found.")

    if request.answer_scope == AnswerScope.auto and ranked and method != EvidenceMethod.edurag:
        kb = getattr(state, "kb", None)
        if kb is not None and hasattr(kb, "nearby_chunks"):
            ranked = expand_local_context(ranked, kb)
    if ranked and method != EvidenceMethod.edurag:
        ranked = await inspect_retrieved_figures(resolved_query, ranked, settings)
        if request.answer_scope == AnswerScope.auto or method == EvidenceMethod.aspect:
            ranked = bound_context(ranked)
    visual_unavailable = bool(ranked and requires_visual_evidence(resolved_query) and
                              not any(c.visual_inspection_status == "completed" for c, _ in ranked))
    evidence_decision = None
    evidence_map = []
    evidence_trace = EvidenceTrace()
    check_status = "not_checked"
    check_reason = None
    if request.answer_scope == AnswerScope.model_only:
        sufficiency = EvidenceSufficiency.not_checked
    elif method == EvidenceMethod.aspect:
        ranked, evidence_decision, evidence_map, evidence_trace, check_status = await assess_aspects(
            question_plan, ranked, state, request, evaluate_aspect_evidence, inspect_retrieved_figures)
        visual_unavailable = bool(ranked and requires_visual_evidence(resolved_query) and
                                 not any(c.visual_inspection_status == "completed" for c, _ in ranked))
        sufficiency = evidence_decision.sufficiency if evidence_decision else EvidenceSufficiency.insufficient
        check_reason = evidence_decision.reason if evidence_decision else "Source support could not be assessed."
        if evidence_decision and request.comprehensive and set(sources_considered) - {c.source_file for c, _ in ranked}:
            evidence_decision.missing_aspects.append("Coverage of every requested source document was not established")
            evidence_decision.needs_background = True
            sufficiency = EvidenceSufficiency.partial
        if evidence_decision and visual_unavailable:
            evidence_decision.missing_aspects.append("Requested visual details could not be verified from the actual figure")
            sufficiency = EvidenceSufficiency.partial
        if evidence_decision and (visual_unavailable or (
            request.comprehensive and set(sources_considered) - {c.source_file for c, _ in ranked}
        )):
            for aspect in evidence_map:
                if aspect.status == "supported":
                    aspect.status = "partial"
                    aspect.reason = evidence_decision.missing_aspects[-1]
            evidence_decision = apply_map(evidence_decision, evidence_map)
        if evidence_decision and evidence_decision.missing_aspects:
            check_reason = "Source coverage is incomplete. Missing: " + "; ".join(evidence_decision.missing_aspects)
    elif request.answer_scope == AnswerScope.auto and ranked:
        try:
            evidence_decision = await evaluate_evidence(
                question_plan, [chunk for chunk, _ in ranked], settings
            )
            evidence_decision = validate_evidence_decision(question_plan, [c for c, _ in ranked][:8], evidence_decision)
            check_status = "completed"
            check_reason = evidence_decision.reason
            if request.comprehensive and set(sources_considered) - {c.source_file for c, _ in ranked}:
                evidence_decision.missing_aspects.append("Coverage of every requested source document was not established")
                evidence_decision.needs_background = True
            if visual_unavailable:
                evidence_decision.missing_aspects.append("Requested visual details could not be verified from the actual figure")
                evidence_decision.needs_background = True
                if evidence_decision.sufficiency == EvidenceSufficiency.strong:
                    evidence_decision.sufficiency = EvidenceSufficiency.partial
            sufficiency = evidence_decision.sufficiency
            if evidence_decision.topic_relevant and evidence_decision.needs_background:
                sufficiency = EvidenceSufficiency.partial
            if evidence_decision.missing_aspects:
                check_reason = "Source coverage is incomplete. Missing: " + "; ".join(evidence_decision.missing_aspects)
        except Exception:
            # Keyword overlap and reranker relevance cannot prove answerability.
            # Never upgrade a failed semantic check to a source-grounded answer.
            sufficiency = EvidenceSufficiency.insufficient
            evidence_decision = None
            check_status = "unavailable"
            check_reason = "Source support could not be assessed. Retrieval relevance is not proof of support."
    else:
        sufficiency = assess_evidence_sufficiency(
            resolved_query, ranked, request.mode
        )
    if evidence_decision and evidence_decision.topic_relevant and evidence_decision.topic_items:
        # Enrich generation after assessment without adding requested aspects.
        question_plan = question_plan.model_copy(update={
            "referenced_items": list(dict.fromkeys([
                *question_plan.referenced_items, *evidence_decision.topic_items,
            ]))[:50],
        })
    if strict_source:
        basis = AnswerBasis.sources
    elif request.answer_scope == AnswerScope.model_only:
        basis = AnswerBasis.model
    elif sufficiency == EvidenceSufficiency.insufficient:
        # A document can be useful context yet not fully answer a broader or
        # current request. Keep that provenance visible instead of discarding
        # relevant study material and silently switching to model-only output.
        basis = (
            AnswerBasis.mixed
            if evidence_decision is not None and evidence_decision.topic_relevant
            else AnswerBasis.model
        )
    elif sufficiency == EvidenceSufficiency.partial:
        basis = AnswerBasis.mixed
    else:
        basis = AnswerBasis.sources

    supported_ids = None
    if (request.answer_scope == AnswerScope.auto or method == EvidenceMethod.aspect) and evidence_decision is not None:
        supported_ids = {s.chunk_id for s in evidence_decision.supports}
        for c, _ in ranked:
            if c.type == ChunkType.visual and c.visual_inspection_status != "completed":
                continue
            text = " ".join(c.text.casefold().split())
            if any(" ".join(item.casefold().split()) in text for item in evidence_decision.topic_items):
                supported_ids.add(c.chunk_id)
        if not supported_ids:
            sufficiency = EvidenceSufficiency.insufficient
            if not strict_source:
                basis = AnswerBasis.model
            check_reason = "No traceable source support survived validation."

    # Irrelevant passages must not appear beside a model-only answer: doing so
    # would imply provenance that the answer did not actually use.
    answer_ranked = ranked if basis != AnswerBasis.model else []
    if supported_ids is not None:
        answer_ranked = [(c, score) for c, score in answer_ranked if c.chunk_id in supported_ids]
    chunks = [chunk for chunk, _ in answer_ranked]
    scores_by_id = {chunk.chunk_id: score for chunk, score in ranked}

    # Run before generation: the conflict list is fed into the prompt so the
    # model can address both sides in its own words (plan Sec. 5.1).
    disagreements = []
    if request.detect_contradictions and chunks:
        try:
            disagreements = await find_disagreements(chunks, settings)
        except Exception:
            disagreements = []  # never fail a query over the optional feature

    source_abstained = False
    try:
        if strict_source and method == EvidenceMethod.aspect and (not chunks or check_status != "completed"):
            source_abstained = True
            answer = "I couldn't establish supporting evidence for this question in the retrieved material. Try another source or rephrase the question."
        elif basis == AnswerBasis.model:
            if _requires_source_answer(request.query + " " + resolved_query):
                source_abstained = True
                answer = "I couldn't verify the uploaded material, so I can't establish what that source says or shows about this question."
            else:
                options = {"history": request.history, "plan": question_plan}
                if check_status == "unavailable":
                    options["evidence_limitation"] = check_reason
                answer = await generate_general_answer(resolved_query, settings, **options)
        elif basis == AnswerBasis.mixed:
            answer = await generate_mixed_answer(
                resolved_query,
                chunks,
                settings,
                disagreements,
                history=request.history,
                plan=question_plan,
                covered_aspects=(
                    evidence_decision.covered_aspects if evidence_decision else []
                ),
                missing_aspects=(
                    evidence_decision.missing_aspects if evidence_decision else []
                ),
                **({"evidence_map": evidence_map} if method == EvidenceMethod.aspect else {}),
            )
        else:
            answer = await generate_answer(
                resolved_query,
                chunks,
                settings,
                disagreements,
                history=request.history,
                plan=question_plan,
                **({"evidence_map": evidence_map} if method == EvidenceMethod.aspect else {}),
            )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    answer = normalize_citation_markers(answer)
    claim_verification = ClaimVerification()
    if method == EvidenceMethod.aspect and not source_abstained:
        answer, claim_verification = await verify_and_repair(answer, chunks, settings, basis)
        answer = source_excerpt_fallback(
            answer, evidence_map, chunks, claim_verification, basis,
            topic_items=evidence_decision.topic_items if evidence_decision else None,
        )

    source_released = not (source_abstained or claim_verification.withheld)
    attribution = compute_attribution(request.query, answer, chunks, settings) if chunks and source_released else {}
    percentages = normalize_to_percentages(attribution) if attribution else {}
    citations = extract_and_validate_citations(answer, chunks) if chunks else []

    token_scores = None
    ratio = None
    if chunks and source_released:
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

    notices = {
        AnswerBasis.sources: "Answer grounded in uploaded study material.",
        AnswerBasis.mixed: (
            "Uploaded material was relevant but incomplete; additional model knowledge "
            "is separated and is not represented as document evidence."
        ),
        AnswerBasis.model: (
            "No uploaded evidence with validated support was used. This answer comes "
            "from general model knowledge and has not been verified by web research."
        ),
    }
    provenance_notice = notices[basis]
    if check_status == "unavailable":
        provenance_notice = "Uploaded-source support could not be assessed. This answer is general knowledge; file-specific claims remain unverified."
    elif visual_unavailable:
        provenance_notice += " The requested visual details could not be verified from the actual figure; caption text alone does not establish them."
    if source_abstained:
        provenance_notice = "No source-specific answer was generated because supporting material could not be verified."
    elif claim_verification.fallback_kind == "supported_claims":
        provenance_notice = "Source-supported answer units were retained; other generated source claims were omitted because verification did not establish support. General knowledge remains separate."
    elif claim_verification.fallback_used:
        provenance_notice = "The generated source explanation was withheld; exact retrieved source excerpts are shown instead. These are quotations, not independently verified explanations."
    elif claim_verification.withheld:
        provenance_notice += " The source answer was withheld because claim verification did not establish support."
    if question_plan.verification_required:
        reason = question_plan.verification_reason or "authoritative external evidence is required"
        provenance_notice += f" Verification note: {reason}."
    if basis == AnswerBasis.sources and sufficiency == EvidenceSufficiency.insufficient:
        provenance_notice = (
            "My sources only mode is active, but the retrieved material does not "
            "appear sufficient for a complete answer. No model background was added."
        )

    return AskResponse(
        answer=answer,
        evidence=evidence,
        token_grounding=token_scores,
        citations=citations,
        grounding_ratio=ratio,
        citation_validity_rate=citation_validity_rate(citations) if chunks else None,
        disagreements=disagreements,
        mode=request.mode,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        comprehensive=request.comprehensive,
        sources_considered=sources_considered,
        sources_used=sorted({chunk.source_file for chunk in chunks}) if not (source_abstained or claim_verification.withheld) else [],
        answer_scope=request.answer_scope,
        answer_basis=basis,
        evidence_sufficiency=sufficiency,
        provenance_notice=provenance_notice,
        evidence_check_status=check_status,
        evidence_check_reason=check_reason,
        supported_aspects=evidence_decision.covered_aspects if evidence_decision else [],
        missing_aspects=evidence_decision.missing_aspects if evidence_decision else [],
        evidence_method=method,
        evidence_map=evidence_map,
        evidence_trace=evidence_trace,
        claim_verification=claim_verification,
    )
