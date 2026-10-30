"""Traceable aspect coverage and one bounded round of gap-directed retrieval.

Quote validation proves provenance, not entailment. Semantic statuses remain
model judgments and are exposed for independent evaluation.
"""
from __future__ import annotations

import asyncio
import time

from app.models.schemas import AspectEvidence, ChunkType, EvidenceQuote, RetrievalAttempt
from app.core.retrieval.context_expansion import bound_context


def normal(text: str) -> str:
    return " ".join(text.casefold().split())


def valid_quote(quote: EvidenceQuote, by_id: dict) -> bool:
    chunk = by_id.get(quote.chunk_id)
    text = " ".join(quote.quote.split())
    return bool(chunk and (chunk.type != ChunkType.visual or chunk.visual_inspection_status == "completed")
                and text and text in " ".join(chunk.text.split()))


def build_evidence_map(plan, chunks, decision, *, unavailable=False) -> list[AspectEvidence]:
    aspects = list(dict.fromkeys(plan.requested_aspects or [plan.standalone_query]))
    by_id = {c.chunk_id: c for c in chunks}
    judgments = {}
    if decision is not None:
        for judgment in decision.aspect_assessments:
            # Duplicate judgments are ambiguous: do not trust the last entry.
            key = normal(judgment.aspect)
            judgments[key] = None if key in judgments else judgment
    result = []
    for index, aspect in enumerate(aspects, 1):
        key = normal(aspect)
        supports, conflicts = [], []
        status, reason = "missing", "Support was not established in the retrieved material."
        if unavailable:
            status, reason = "not_checked", "Evidence assessment was unavailable."
        elif decision is not None:
            judgment = judgments.get(key)
            if judgment is not None:
                supports = [q for q in judgment.supports if valid_quote(q, by_id)]
                conflicts = [q for q in judgment.conflicts if valid_quote(q, by_id)]
                status, reason = judgment.status, judgment.reason
                if status == "missing":
                    supports, conflicts = [], []
                elif not supports:
                    status, reason = "missing", "The supporting quotes could not be validated."
                elif status == "conflicting" and not any(q not in supports for q in conflicts):
                    status, reason = "partial", "Both sides of the reported conflict were not established."
            elif not decision.aspect_assessments:
                supports = [EvidenceQuote(chunk_id=s.chunk_id, quote=s.quote) for s in decision.supports
                            if normal(s.aspect) == key and valid_quote(s, by_id)]
                if supports and decision.topic_relevant:
                    explicitly_missing = any(key == normal(m) for m in decision.missing_aspects)
                    status = "partial" if explicitly_missing else "supported"
                    reason = "Supporting quote provenance validated; semantic coverage is model-assessed."
            if not decision.topic_relevant or decision.sufficiency.value == "insufficient":
                status = "partial" if supports and decision.topic_relevant else "missing"
        result.append(AspectEvidence(aspect_id=f"a{index}", aspect=aspect, status=status,
                                     supports=supports, conflicts=conflicts, reason=reason))
    # Unassigned gaps must not coexist with a falsely all-supported map.
    if decision is not None and result and all(a.status == "supported" for a in result):
        if decision.missing_aspects or decision.needs_background:
            for aspect in result:
                aspect.status = "partial"
                aspect.reason = "; ".join(decision.missing_aspects) or "Complete support was not established."
    if plan.verification_required:
        for aspect in result:
            if aspect.status == "supported":
                aspect.status = "partial"
                aspect.reason = "Current or authoritative external verification has not been established."
    return result


def apply_map(decision, evidence_map):
    """Use only canonical requested aspects in generation coverage metadata."""
    result = decision.model_copy(deep=True)
    from app.core.generation.orchestration import AspectSupport
    result.supports = [AspectSupport(aspect=a.aspect, **quote.model_dump())
                       for a in evidence_map if a.status in {"supported", "partial", "conflicting"}
                       for quote in [*a.supports, *a.conflicts]]
    result.covered_aspects = [a.aspect for a in evidence_map if a.status == "supported"]
    result.missing_aspects = list(dict.fromkeys([
        *[a.aspect for a in evidence_map if a.status != "supported"],
        *decision.missing_aspects,
    ]))
    if any(a.status != "supported" for a in evidence_map):
        from app.models.schemas import EvidenceSufficiency
        result.sufficiency = EvidenceSufficiency.partial if result.topic_relevant else EvidenceSufficiency.insufficient
        result.needs_background = result.topic_relevant
    return result


async def retrieve_missing_aspects(plan, evidence_map, ranked, retrieval, settings, mode):
    gaps = [a for a in evidence_map if a.status in {"partial", "missing"}]
    if not settings.targeted_retrieval_enabled or not gaps or settings.targeted_retrieval_max_queries == 0:
        return ranked, []
    attempts, found = [], []
    known = {c.chunk_id for c, _ in ranked}
    # One round; original question retains subject/constraints. No new HyDE
    # calls: targeted queries use hybrid retrieval when full was requested.
    from app.models.schemas import RetrievalMode
    targeted_mode = RetrievalMode.hybrid if mode == RetrievalMode.full else mode
    for gap in gaps[:settings.targeted_retrieval_max_queries]:
        query = f"{plan.standalone_query}\nFocus on: {gap.aspect}"[:4000]
        started = time.perf_counter()
        attempt = RetrievalAttempt(aspect_ids=[gap.aspect_id], query=query, status="completed")
        try:
            hits = await asyncio.wait_for(retrieval.retrieve(
                query, top_k=settings.targeted_retrieval_top_k, mode=targeted_mode),
                timeout=settings.targeted_retrieval_timeout_seconds)
            attempt.retrieved_ids = [c.chunk_id for c, _ in hits]
            for hit in hits:
                if hit[0].chunk_id not in known:
                    known.add(hit[0].chunk_id)
                    found.append(hit)
        except Exception:
            attempt.status = "unavailable"
        attempt.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        attempts.append(attempt)
    if not found:
        return ranked, attempts
    # Reserve half the bounded context for new candidates; preserve validated
    # initial support first so top-k saturation cannot hide the new evidence.
    supported = {q.chunk_id for a in evidence_map for q in [*a.supports, *a.conflicts]}
    original = sorted(ranked, key=lambda hit: hit[0].chunk_id not in supported)
    combined = bound_context([*original[:4], *found[:4], *original[4:]])
    final_ids = {c.chunk_id for c, _ in combined}
    initial_ids = {c.chunk_id for c, _ in ranked}
    for attempt in attempts:
        attempt.added_ids = list(dict.fromkeys(cid for cid in attempt.retrieved_ids if cid in final_ids and cid not in initial_ids))
    return combined, attempts
