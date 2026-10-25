from types import SimpleNamespace

import pytest

from app.api import routes_ask
from app.core.generation import answer
from app.core.generation.orchestration import EvidenceDecision, AspectSupport
from app.models.schemas import AskRequest, QuestionPlan
from app.tests.conftest import chunk


def setup_handler(monkeypatch, settings, *, failure=False, support=True, irrelevant=False):
    c = chunk()
    calls = []
    class Retrieval:
        is_ready = True
        async def retrieve(self, *args, **kwargs):
            return [(c, .999)]
    async def plan(*args):
        return QuestionPlan(standalone_query="Explain a transformer", requested_aspects=["explanation"])
    async def evaluate(*args):
        if failure:
            raise TimeoutError("provider unavailable")
        return EvidenceDecision(sufficiency="insufficient" if irrelevant else "strong",
            topic_relevant=not irrelevant, reason="Test decision", supports=[
                AspectSupport(aspect="explanation", chunk_id="c1", quote=c.text)] if support else [])
    def generate(kind):
        async def run(*args, **kwargs):
            calls.append((kind, kwargs))
            return "A supported fact [1]." if kind != "model" else "General explanation."
        return run
    monkeypatch.setattr(routes_ask, "plan_question", plan)
    monkeypatch.setattr(routes_ask, "evaluate_evidence", evaluate)
    monkeypatch.setattr(routes_ask, "generate_answer", generate("sources"))
    monkeypatch.setattr(routes_ask, "generate_mixed_answer", generate("mixed"))
    monkeypatch.setattr(routes_ask, "generate_general_answer", generate("model"))
    monkeypatch.setattr(routes_ask, "compute_attribution", lambda *a: {"c1": 1})
    return SimpleNamespace(settings=settings, retrieval=Retrieval()), calls


@pytest.mark.asyncio
async def test_failed_check_cannot_be_rescued_by_high_retrieval_score(settings, monkeypatch):
    state, calls = setup_handler(monkeypatch, settings, failure=True)
    response = await routes_ask.ask(AskRequest(query="Explain a transformer", detect_contradictions=False), state)
    assert response.answer_basis.value == "model"
    assert response.evidence_sufficiency.value == "insufficient"
    assert response.evidence_check_status == "unavailable"
    assert response.evidence == [] and response.sources_used == []
    assert "evidence_limitation" in calls[0][1]
    assert "could not be assessed" in response.provenance_notice


@pytest.mark.asyncio
@pytest.mark.parametrize("support,expected", [(True, "sources"), (False, "model")])
async def test_real_handler_requires_traceable_support(settings, monkeypatch, support, expected):
    state, calls = setup_handler(monkeypatch, settings, support=support)
    response = await routes_ask.ask(AskRequest(query="Explain a transformer", detect_contradictions=False), state)
    assert response.answer_basis.value == expected
    assert calls[0][0] == expected
    assert response.evidence_check_status == "completed"
    assert bool(response.supported_aspects) == support


@pytest.mark.asyncio
async def test_irrelevant_passages_are_not_attached_to_general_answers(settings, monkeypatch):
    state, _ = setup_handler(monkeypatch, settings, support=False, irrelevant=True)
    response = await routes_ask.ask(AskRequest(query="Explain a transformer", detect_contradictions=False), state)
    assert response.answer_basis.value == "model"
    assert not response.evidence


@pytest.mark.asyncio
@pytest.mark.parametrize("scope,expected", [("sources_only", "sources"), ("model_only", "model")])
async def test_explicit_policy_overrides_are_preserved(settings, monkeypatch, scope, expected):
    state, calls = setup_handler(monkeypatch, settings, failure=True)
    response = await routes_ask.ask(AskRequest(query="Explain a transformer", answer_scope=scope,
                                             detect_contradictions=False), state)
    assert response.answer_basis.value == expected
    assert calls[0][0] == expected


@pytest.mark.asyncio
async def test_fallback_prompt_forbids_inventing_source_specific_answers(settings, monkeypatch):
    captured = []
    class Client:
        async def chat(self, messages, **kwargs):
            captured.extend(messages)
            return "I cannot establish the document's value."
    monkeypatch.setattr(answer, "build_client", lambda s: Client())
    await answer.generate_general_answer("What fee is in my PDF?", settings, evidence_limitation="Unavailable")
    assert "Never invent document contents" in captured[0]["content"]


@pytest.mark.asyncio
async def test_failed_source_specific_question_declines_without_another_generation_call(settings, monkeypatch):
    state, calls = setup_handler(monkeypatch, settings, failure=True)
    response = await routes_ask.ask(AskRequest(query="What registration fee is in my PDF?",
                                             detect_contradictions=False), state)
    assert "can't establish" in response.answer
    assert not calls and not response.evidence
    assert response.evidence_check_status == "unavailable"


@pytest.mark.asyncio
async def test_specific_figure_question_cannot_be_answered_from_prose_alone(settings, monkeypatch):
    state, calls = setup_handler(monkeypatch, settings)
    async def plan(*args):
        return QuestionPlan(standalone_query="Explain the diagram", requested_aspects=["explanation"])
    monkeypatch.setattr(routes_ask, "plan_question", plan)
    response = await routes_ask.ask(AskRequest(query="Explain the diagram", detect_contradictions=False), state)
    assert response.answer_basis.value == "mixed"
    assert any("visual details" in gap for gap in response.missing_aspects)
    assert "could not be verified" in response.provenance_notice
    assert calls[0][0] == "mixed"


@pytest.mark.asyncio
async def test_unproven_distractor_is_not_passed_to_generation_or_shown_as_used(settings, monkeypatch):
    state, _ = setup_handler(monkeypatch, settings)
    c = chunk()
    distractor = chunk("Unrelated photosynthesis facts.", chunk_id="noise").model_copy(update={"source_file": "other.pdf"})
    async def retrieve(*args, **kwargs):
        return [(c, .95), (distractor, .9)]
    state.retrieval.retrieve = retrieve
    seen = []
    async def generate(query, chunks, *args, **kwargs):
        seen.extend(chunks)
        return "A transformer transfers energy [1]."
    monkeypatch.setattr(routes_ask, "generate_answer", generate)
    response = await routes_ask.ask(AskRequest(query="Explain a transformer", detect_contradictions=False), state)
    assert [c.chunk_id for c in seen] == ["c1"]
    assert response.sources_used == ["notes.pdf"]
    assert [e.chunk.chunk_id for e in response.evidence] == ["c1"]


@pytest.mark.asyncio
async def test_missing_source_material_does_not_trigger_a_guessed_document_answer(settings, monkeypatch):
    state, calls = setup_handler(monkeypatch, settings)
    async def retrieve(*args, **kwargs):
        return []
    state.retrieval.retrieve = retrieve
    response = await routes_ask.ask(AskRequest(query="What is on page 2 of my PDF?", detect_contradictions=False), state)
    assert "can't establish" in response.answer and not calls
    assert "No source-specific answer" in response.provenance_notice
