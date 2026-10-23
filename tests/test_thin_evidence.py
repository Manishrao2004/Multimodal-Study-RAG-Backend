from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_app_state
from app.api import routes_ask
from app.core.generation import answer
from app.core.generation.orchestration import EvidenceDecision
from app.core.retrieval.sufficiency import assess_evidence_sufficiency
from app.models.schemas import EvidenceSufficiency, QuestionPlan, RetrievalMode
from tests.conftest import make_chunk


@pytest.mark.parametrize("text", [
    "Types of partners:\n1. General Partner\n2. Sleeping Partner\n3. Nominal Partner",
    "The types of partners are general, sleeping, and nominal partners.",
])
def test_name_only_material_is_partial_for_explanations(text):
    ranked = [(make_chunk("outline", text), 0.95)]
    assert assess_evidence_sufficiency(
        "Explain the types of partners", ranked, RetrievalMode.full,
    ) == EvidenceSufficiency.partial


def test_a_list_is_sufficient_when_only_names_are_requested():
    ranked = [(make_chunk("outline", "The types of partners are general, sleeping, and nominal partners."), 0.95)]
    assert assess_evidence_sufficiency("List the types of partners", ranked, RetrievalMode.full) == EvidenceSufficiency.strong


def test_a_short_real_definition_is_not_treated_as_a_heading():
    ranked = [(make_chunk("definition", "A sleeping partner contributes capital but does not participate in management."), 0.95)]
    assert assess_evidence_sufficiency("Explain a sleeping partner", ranked, RetrievalMode.full) == EvidenceSufficiency.strong


@pytest.mark.parametrize("scope", ["auto", "sources_only"])
def test_background_is_added_in_auto_but_source_only_remains_strict(settings, monkeypatch, scope):
    chunk = make_chunk("outline", "Types of partners: General Partner, Sleeping Partner, Nominal Partner")
    calls = []

    class Retrieval:
        is_ready = True

        async def retrieve(self, *args, **kwargs):
            return [(chunk, 0.95)]

    async def plan(*args):
        return QuestionPlan(standalone_query="Explain the types of partners")

    async def evaluate(*args):
        # The explicit missing-knowledge signal wins even if the model's
        # sufficiency label is inconsistent with its own assessment.
        return EvidenceDecision(
            sufficiency=EvidenceSufficiency.strong,
            reason="Only topic names are present", topic_relevant=True,
            needs_background=True,
            topic_items=["General Partner", "Sleeping Partner", "Nominal Partner"],
            missing_aspects=["definition of each type"],
        )

    async def mixed(query, chunks, config, *args, **kwargs):
        calls.append(("mixed", kwargs))
        return "The notes list partner types [1].\n\nGeneral knowledge explains their roles."

    async def grounded(*args, **kwargs):
        calls.append(("sources", kwargs))
        return "The notes list partner types [1]."

    monkeypatch.setattr(routes_ask, "plan_question", plan)
    monkeypatch.setattr(routes_ask, "evaluate_evidence", evaluate)
    monkeypatch.setattr(routes_ask, "generate_mixed_answer", mixed)
    monkeypatch.setattr(routes_ask, "generate_answer", grounded)
    monkeypatch.setattr(routes_ask, "compute_attribution", lambda *args: {"outline": 1.0})
    app = FastAPI()
    app.include_router(routes_ask.router)
    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(settings=settings, retrieval=Retrieval())
    response = TestClient(app).post("/ask", json={
        "query": "Explain the types of partners", "answer_scope": scope, "detect_contradictions": False,
    })
    assert response.status_code == 200
    result = response.json()
    if scope == "auto":
        assert result["answer_basis"] == "mixed"
        assert result["evidence_sufficiency"] == "partial"
        assert calls[0][0] == "mixed"
        assert calls[0][1]["plan"].referenced_items == ["General Partner", "Sleeping Partner", "Nominal Partner"]
        assert calls[0][1]["missing_aspects"] == ["definition of each type"]
    else:
        assert result["answer_basis"] == "sources"
        assert calls[0][0] == "sources"


async def test_supplemental_generation_receives_actual_document_topics(settings, monkeypatch):
    prompts = []

    class Client:
        async def chat(self, messages, **kwargs):
            prompts.append(messages)
            return "Partner types [1]." if "CITATION RULES" in messages[0]["content"] else "A concise explanation from general knowledge."

    monkeypatch.setattr(answer, "build_client", lambda _settings: Client())
    result = await answer.generate_mixed_answer(
        "Explain each listed topic",
        [make_chunk("outline", "General Partner\nSleeping Partner\nNominal Partner")],
        settings,
        missing_aspects=["definitions"],
    )
    supplemental = next(messages for messages in prompts if "CITATION RULES" not in messages[0]["content"])
    assert "TOPIC ANCHORS" in supplemental[1]["content"]
    assert "Sleeping Partner" in supplemental[1]["content"]
    assert "Nominal Partner" in supplemental[1]["content"]
    assert "Never add document citation markers" in supplemental[0]["content"]
    assert "## Additional explanation (general knowledge)" in result
