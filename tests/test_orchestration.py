import pytest

from app.core.generation import orchestration
from app.models.schemas import ConversationTurn, EvidenceSufficiency
from tests.conftest import make_chunk


class FakeClient:
    def __init__(self, response: str):
        self.response = response

    async def chat(self, messages, temperature=0.0):
        return self.response


@pytest.mark.asyncio
async def test_langchain_rewrites_follow_up_as_standalone_query(settings, monkeypatch):
    monkeypatch.setattr(
        orchestration,
        "_client",
        lambda _settings: FakeClient(
            '{"standalone_query":"Explain each type of partner",'
            '"referenced_items":["each type of partner"],'
            '"requested_aspects":["explanation"],'
            '"response_format":"item-by-item explanation",'
            '"verification_required":false,"verification_reason":null}'
        ),
    )
    rewritten = await orchestration.rewrite_conversation_query(
        "explain each",
        [ConversationTurn(role="user", content="what types of partners")],
        settings,
    )
    assert rewritten == "Explain each type of partner"


@pytest.mark.asyncio
async def test_langchain_returns_validated_evidence_decision(settings, monkeypatch):
    monkeypatch.setattr(
        orchestration,
        "_client",
        lambda _settings: FakeClient(
            '{"sufficiency":"partial","topic_relevant":true,'
            '"reason":"The passages list types but do not explain them."}'
        ),
    )
    decision = await orchestration.evaluate_evidence(
        orchestration.QuestionPlan(
            standalone_query="Explain each type of partner",
            referenced_items=["general", "sleeping", "nominal", "minor"],
            requested_aspects=["definition for every item"],
        ),
        [make_chunk("c1", "The types are general, sleeping, nominal, and minor partners.")],
        settings,
    )
    assert decision.sufficiency == EvidenceSufficiency.partial


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "history", "standalone"),
    [
        (
            "explain each",
            [ConversationTurn(role="user", content="Compare alpha, beta, and gamma")],
            "Explain alpha, beta, and gamma individually",
        ),
        (
            "give me a two-column comparison",
            [ConversationTurn(role="user", content="What are TCP and UDP?")],
            "Compare TCP and UDP in two columns",
        ),
        ("What changed this month?", [], "What changed this month?"),
    ],
)
async def test_planner_handles_multiple_query_shapes(
    settings, monkeypatch, query, history, standalone
):
    needs_verification = "this month" in query.lower()
    monkeypatch.setattr(
        orchestration,
        "_client",
        lambda _settings: FakeClient(
            orchestration.QuestionPlan(
                standalone_query=standalone,
                requested_aspects=["requested coverage"],
                response_format="requested format",
                verification_required=needs_verification,
                verification_reason=("Time-sensitive request" if needs_verification else None),
            ).model_dump_json()
        ),
    )
    plan = await orchestration.plan_question(query, history, settings)
    assert plan.standalone_query == standalone
    assert plan.verification_required is needs_verification
