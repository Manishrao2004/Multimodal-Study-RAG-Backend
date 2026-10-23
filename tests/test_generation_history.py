import pytest

from app.core.generation import answer
from app.models.schemas import ConversationTurn, QuestionPlan


class FakeClient:
    def __init__(self):
        self.messages = None

    async def chat(self, messages, temperature=0.0):
        self.messages = messages
        return "Student-friendly answer."


@pytest.mark.asyncio
async def test_general_answer_accepts_history_and_plan(settings, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(answer, "build_client", lambda _settings: client)

    result = await answer.generate_general_answer(
        "Explain recursion simply",
        settings,
        history=[ConversationTurn(role="user", content="Tell me about recursion")],
        plan=QuestionPlan(standalone_query="Explain recursion simply"),
    )

    assert result == "Student-friendly answer."
    assert "CONVERSATION CONTEXT" in client.messages[1]["content"]
    assert "RESPONSE PLAN" in client.messages[1]["content"]
