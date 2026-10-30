from types import SimpleNamespace

import pytest

from app.api import routes_ask
from app.core.generation.aspect_evidence import build_evidence_map, retrieve_missing_aspects
from app.core.generation.orchestration import AspectJudgment, AspectSupport, EvidenceDecision
from app.models.schemas import AskRequest, ClaimVerification, EvidenceQuote, QuestionPlan, RetrievalMode
from app.tests.conftest import chunk


def decision(c, aspects=("advantages",), *, missing=("disadvantages",)):
    return EvidenceDecision(sufficiency="partial" if missing else "strong", topic_relevant=True,
        reason="Fixture", missing_aspects=list(missing), supports=[
            AspectSupport(aspect=a, chunk_id=c.chunk_id, quote=c.text) for a in aspects])


def plan():
    return QuestionPlan(standalone_query="Explain advantages and disadvantages of partnerships",
                        requested_aspects=["advantages", "disadvantages"])


def test_map_has_stable_ids_and_missing_aspect():
    c = chunk("Partners share the investment.")
    result = build_evidence_map(plan(), [c], decision(c))
    assert [(a.aspect_id, a.status) for a in result] == [("a1", "supported"), ("a2", "missing")]


def test_forged_quote_cannot_establish_support():
    c = chunk()
    d = decision(c)
    d.supports[0].quote = "An invented advantage."
    assert build_evidence_map(plan(), [c], d)[0].status == "missing"


def test_conflict_needs_two_distinct_validated_positions():
    a, b = chunk("Maximum size is 8.", "a"), chunk("Maximum size is 12.", "b")
    p = QuestionPlan(standalone_query="What is the maximum size?", requested_aspects=["size"])
    d = EvidenceDecision(sufficiency="partial", topic_relevant=True, reason="Conflict",
        aspect_assessments=[AspectJudgment(aspect="size", status="conflicting",
            supports=[EvidenceQuote(chunk_id="a", quote=a.text)],
            conflicts=[EvidenceQuote(chunk_id="b", quote=b.text)])])
    assert build_evidence_map(p, [a, b], d)[0].status == "conflicting"
    d.aspect_assessments[0].conflicts[0].quote = "Invented"
    assert build_evidence_map(p, [a, b], d)[0].status == "partial"


def test_duplicate_assessments_cannot_become_complete_support():
    c = chunk()
    d = decision(c, missing=())
    j = AspectJudgment(aspect="advantages", status="supported", supports=[EvidenceQuote(chunk_id=c.chunk_id, quote=c.text)])
    d.aspect_assessments = [j, j]
    assert build_evidence_map(plan(), [c], d)[0].status == "missing"


@pytest.mark.asyncio
async def test_targeted_round_is_bounded_deduplicated_and_avoids_hyde(settings):
    a, b = chunk("Partners share investment.", "a"), chunk("Partners bear liability.", "b")
    calls = []
    async def retrieve(query, **kwargs):
        calls.append((query, kwargs))
        return [(a, .9), (b, .8), (b, .7)]
    mapped = build_evidence_map(plan(), [a], decision(a))
    ranked, attempts = await retrieve_missing_aspects(plan(), mapped, [(a, .9)],
        SimpleNamespace(retrieve=retrieve), settings, RetrievalMode.full)
    assert len(calls) == 1 and calls[0][1]["mode"] == RetrievalMode.hybrid
    assert "disadvantages" in calls[0][0] and "partnerships" in calls[0][0]
    assert [c.chunk_id for c, _ in ranked] == ["a", "b"]
    assert attempts[0].added_ids == ["b"]


@pytest.mark.asyncio
async def test_new_evidence_gets_a_slot_even_when_initial_context_is_full(settings):
    original = [(chunk(f"Initial context {i}", str(i)), .9) for i in range(8)]
    added = chunk("Partners bear liability.", "new")
    async def retrieve(*args, **kwargs):
        return [(added, .8)]
    mapped = build_evidence_map(plan(), [c for c, _ in original], decision(original[0][0]))
    ranked, _ = await retrieve_missing_aspects(plan(), mapped, original,
        SimpleNamespace(retrieve=retrieve), settings, RetrievalMode.full)
    assert len(ranked) <= 8 and "new" in [c.chunk_id for c, _ in ranked]


def setup(monkeypatch, settings, *, second_failure=False, verification=False):
    a = chunk("Partners share investment.", "a")
    b = chunk("Partners bear liability.", "b")
    calls = []
    async def retrieve(query, **kwargs):
        calls.append(query)
        return [(b, .8)] if "Focus on:" in query else [(a, .9)]
    async def planner(*args):
        return plan()
    async def assess(p, chunks, config):
        if len(chunks) == 2:
            if second_failure:
                raise TimeoutError()
            d = decision(a, missing=())
            d.supports.append(AspectSupport(aspect="disadvantages", chunk_id="b", quote=b.text))
            return d
        return decision(a)
    generated = []
    async def generate(query, chunks, *args, **kwargs):
        generated.extend(c.chunk_id for c in chunks)
        assert kwargs["evidence_map"]
        return "Partners share investment [1]."
    async def verify(answer, chunks, config, basis):
        return answer, ClaimVerification(status="completed")
    monkeypatch.setattr(routes_ask, "plan_question", planner)
    monkeypatch.setattr(routes_ask, "evaluate_aspect_evidence", assess)
    monkeypatch.setattr(routes_ask, "generate_answer", generate)
    monkeypatch.setattr(routes_ask, "generate_mixed_answer", generate)
    async def general(*args, **kwargs):
        return "General explanation."
    monkeypatch.setattr(routes_ask, "generate_general_answer", general)
    monkeypatch.setattr(routes_ask, "compute_attribution", lambda *args: {})
    if not verification:
        monkeypatch.setattr(routes_ask, "verify_and_repair", verify)
    state = SimpleNamespace(settings=settings, retrieval=SimpleNamespace(is_ready=True, retrieve=retrieve))
    return state, calls, generated


@pytest.mark.asyncio
async def test_handler_recovers_gap_and_uses_reassessed_context(settings, monkeypatch):
    state, calls, generated = setup(monkeypatch, settings)
    result = await routes_ask.ask(AskRequest(query=plan().standalone_query, evidence_method="aspect",
                                           detect_contradictions=False), state)
    assert len(calls) == 2 and generated == ["a", "b"]
    assert result.answer_basis.value == "sources"
    assert all(a.status == "supported" for a in result.evidence_map)
    assert result.evidence_trace.initial_map[1].status == "missing"
    assert result.evidence_trace.assessment_calls == 2


@pytest.mark.asyncio
async def test_reassessment_outage_keeps_original_context_and_support(settings, monkeypatch):
    state, _, generated = setup(monkeypatch, settings, second_failure=True)
    result = await routes_ask.ask(AskRequest(query=plan().standalone_query, evidence_method="aspect",
                                           detect_contradictions=False), state)
    assert generated == ["a"] and result.answer_basis.value == "mixed"
    assert result.evidence_trace.final_chunk_ids == ["a"]
    assert result.evidence_trace.retrieval_attempts[0].status == "unavailable"
    assert result.evidence_trace.retrieval_attempts[0].added_ids == []


@pytest.mark.asyncio
async def test_no_targeted_retrieval_ablation(settings, monkeypatch):
    settings.targeted_retrieval_enabled = False
    state, calls, _ = setup(monkeypatch, settings)
    result = await routes_ask.ask(AskRequest(query=plan().standalone_query, evidence_method="aspect",
                                           detect_contradictions=False), state)
    assert len(calls) == 1 and not result.evidence_trace.retrieval_attempts
    assert result.answer_basis.value == "mixed"


@pytest.mark.asyncio
async def test_source_only_assessment_failure_abstains_without_generating(settings, monkeypatch):
    state, _, generated = setup(monkeypatch, settings)
    async def unavailable(*args):
        raise TimeoutError()
    monkeypatch.setattr(routes_ask, "evaluate_aspect_evidence", unavailable)
    result = await routes_ask.ask(AskRequest(query=plan().standalone_query, evidence_method="aspect",
        answer_scope="sources_only", detect_contradictions=False), state)
    assert not generated and not result.sources_used
    assert all(a.status == "not_checked" for a in result.evidence_map)
    assert result.evidence_check_status == "unavailable"


@pytest.mark.asyncio
async def test_whole_question_ablation(settings, monkeypatch):
    settings.aspect_decomposition_enabled = False
    state, _, _ = setup(monkeypatch, settings)
    result = await routes_ask.ask(AskRequest(query=plan().standalone_query, evidence_method="aspect",
                                           detect_contradictions=False), state)
    assert [a.aspect for a in result.evidence_map] == [plan().standalone_query]


@pytest.mark.asyncio
async def test_withheld_answer_does_not_claim_attribution_or_grounding(settings, monkeypatch):
    state, _, _ = setup(monkeypatch, settings, verification=True)
    settings.claim_repair_enabled = False
    # The provider is disabled; the audit must fail closed, not pretend that
    # a successful generation established claim support.
    result = await routes_ask.ask(AskRequest(query=plan().standalone_query, evidence_method="aspect",
                                           detect_contradictions=False), state)
    assert result.claim_verification.withheld
    assert result.evidence and not result.sources_used
    assert result.token_grounding is None and result.grounding_ratio is None
    assert all(e.attribution_score is None for e in result.evidence)


@pytest.mark.asyncio
async def test_edurag_style_baseline_bypasses_new_policy_stages(settings, monkeypatch):
    state, calls, _ = setup(monkeypatch, settings)
    async def forbidden(*args, **kwargs):
        raise AssertionError("New stage reached by strict baseline")
    async def generate(*args, **kwargs):
        assert "evidence_map" not in kwargs
        return "Source fact [1]."
    monkeypatch.setattr(routes_ask, "plan_question", forbidden)
    monkeypatch.setattr(routes_ask, "evaluate_aspect_evidence", forbidden)
    monkeypatch.setattr(routes_ask, "verify_and_repair", forbidden)
    monkeypatch.setattr(routes_ask, "generate_answer", generate)
    result = await routes_ask.ask(AskRequest(query=plan().standalone_query, evidence_method="edurag",
                                           detect_contradictions=False), state)
    assert len(calls) == 1 and result.answer_basis.value == "sources"
    assert not result.evidence_map and result.claim_verification.status == "not_checked"
