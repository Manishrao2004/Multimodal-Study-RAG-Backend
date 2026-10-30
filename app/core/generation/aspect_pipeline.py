"""Assess, retrieve gaps once, and reassess the exact bounded answer context."""
from app.core.generation.aspect_evidence import apply_map, build_evidence_map, retrieve_missing_aspects
from app.core.generation.orchestration import validate_evidence_decision
from app.core.retrieval.context_expansion import bound_context
from app.models.schemas import EvidenceTrace


async def assess_aspects(plan, ranked, state, request, evaluate, inspect):
    trace = EvidenceTrace(
        initial_chunk_ids=[c.chunk_id for c, _ in ranked],
        targeted_retrieval_enabled=state.settings.targeted_retrieval_enabled,
        claim_verification_enabled=state.settings.claim_verification_enabled,
    )
    decision = None
    status = "completed"
    try:
        trace.assessment_calls += 1
        decision = await evaluate(plan, [c for c, _ in ranked], state.settings)
        decision = validate_evidence_decision(plan, [c for c, _ in ranked], decision)
    except Exception:
        status = "unavailable"
    evidence_map = build_evidence_map(plan, [c for c, _ in ranked], decision, unavailable=status == "unavailable")
    trace.initial_map = [a.model_copy(deep=True) for a in evidence_map]
    if status == "completed":
        candidates, trace.retrieval_attempts = await retrieve_missing_aspects(
            plan, evidence_map, ranked, state.retrieval, state.settings, request.mode)
        if any(attempt.added_ids for attempt in trace.retrieval_attempts):
            try:
                candidates = bound_context(await inspect(plan.standalone_query, candidates, state.settings))
                trace.assessment_calls += 1
                updated = await evaluate(plan, [c for c, _ in candidates], state.settings)
                updated = validate_evidence_decision(plan, [c for c, _ in candidates], updated)
                evidence_map = build_evidence_map(plan, [c for c, _ in candidates], updated)
                ranked, decision = candidates, updated
            except Exception:
                # Keep the previously assessed context and map; never attach a
                # new passage as "used" after a failed reassessment.
                for attempt in trace.retrieval_attempts:
                    attempt.status = "unavailable"
                    attempt.added_ids = []
    if decision is not None:
        decision = apply_map(decision, evidence_map)
    trace.final_chunk_ids = [c.chunk_id for c, _ in ranked]
    return ranked, decision, evidence_map, trace, status
