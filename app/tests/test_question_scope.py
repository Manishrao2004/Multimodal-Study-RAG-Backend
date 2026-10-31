from types import SimpleNamespace

import pytest

from app.core.generation import answer, orchestration
from app.models.schemas import AspectEvidence, QuestionPlan
from app.tests.conftest import chunk


def test_planner_cannot_append_its_own_clarification_and_alternative_domains():
    polluted = QuestionPlan(standalone_query="What are the types of partners? (Please specify the context, e.g., business, legal or personal.)",
        requested_aspects=["business partnerships", "personal relationships"])
    clean = orchestration.preserve_question_scope(polluted, "types of partners")
    assert clean.standalone_query == "types of partners" and clean.requested_aspects == ["list"]


def test_genuine_followup_resolution_is_preserved():
    plan = QuestionPlan(standalone_query="Compare TCP and UDP in a table", requested_aspects=["comparison"])
    assert orchestration.preserve_question_scope(plan, "put that in a table") == plan


@pytest.mark.asyncio
async def test_planner_output_is_checked_before_it_can_change_retrieval(settings, monkeypatch):
    polluted = QuestionPlan(standalone_query="Types of partners (Please specify the context)", requested_aspects=["legal", "personal"])
    async def invoke(*args):
        return polluted
    monkeypatch.setattr(orchestration, "_structured_chain", lambda *args, **kwargs: SimpleNamespace(ainvoke=invoke))
    result = await orchestration.plan_question("types of partners", [], settings)
    assert result.standalone_query == "types of partners" and result.requested_aspects == ["list"]


def test_bare_types_request_does_not_invent_a_definition_requirement():
    plan = QuestionPlan(standalone_query="What are the types of partners?", requested_aspects=["definition"], referenced_items=["partners"])
    result = orchestration.preserve_question_scope(plan, "types of partners")
    assert result.requested_aspects == ["list"] and result.referenced_items == []


def test_explicit_request_for_explanations_is_not_changed_to_a_list():
    plan = QuestionPlan(standalone_query="Explain types of partners", requested_aspects=["explanation"])
    assert orchestration.preserve_question_scope(plan, "types of partners with explanations") == plan


def test_bare_types_followup_keeps_the_resolved_topic():
    plan = QuestionPlan(standalone_query="types of network protocols", requested_aspects=["definition"])
    result = orchestration.preserve_question_scope(plan, "types of those")
    assert result.standalone_query == "types of network protocols" and result.requested_aspects == ["list"]


@pytest.mark.asyncio
async def test_mixed_explanation_is_anchored_to_retrieved_topic_and_names(settings, monkeypatch):
    messages = []
    async def chat(sent, **kwargs):
        messages.append(sent)
        return "Fixture"
    monkeypatch.setattr(answer, "build_client", lambda _: SimpleNamespace(chat=chat))
    c = chunk("Types of business partners: Active Partner, Sleeping Partner.")
    plan = QuestionPlan(standalone_query="Explain types of partners", referenced_items=["Active Partner", "Sleeping Partner"])
    await answer.generate_mixed_answer(plan.standalone_query, [c], settings, plan=plan, missing_aspects=["definitions"])
    supplemental = next(sent for sent in messages if "TOPIC ANCHORS" in sent[1]["content"])
    assert "Do not ask the student to choose" in supplemental[0]["content"]
    assert "Active Partner" in supplemental[1]["content"] and "Sleeping Partner" in supplemental[1]["content"]


def test_planner_preserves_explicit_word_count_when_rewrite_omits_it():
    plan = QuestionPlan(standalone_query="Tell me about Fortnite", response_format="concise")
    result = orchestration.preserve_question_scope(plan, "Tell me about Fortnite, around 500 words")
    assert "500 words" in result.response_format


def test_planner_preserves_word_limit_qualifier():
    plan = QuestionPlan(standalone_query="Explain the topic", response_format="brief")
    result = orchestration.preserve_question_scope(plan, "Explain the topic in no more than 100 words")
    assert "no more than 100 words" in result.response_format


@pytest.mark.asyncio
async def test_mixed_general_section_can_explain_missing_aspects(settings, monkeypatch):
    messages = []
    async def chat(sent, **kwargs):
        messages.append(sent)
        return "Fixture"
    monkeypatch.setattr(answer, "build_client", lambda _: SimpleNamespace(chat=chat))
    aspect = AspectEvidence(aspect_id="a1", aspect="gameplay", status="missing")
    await answer.generate_mixed_answer("Tell me about the game in 500 words", [chunk()], settings, evidence_map=[aspect])
    supplemental = next(sent for sent in messages if "TOPIC ANCHORS" in sent[1]["content"])
    assert "allowed to use general model knowledge" in supplemental[0]["content"]
    assert "Explain missing or partial aspects using general knowledge" in supplemental[1]["content"]
    assert "do not fill them from memory" not in supplemental[1]["content"]


@pytest.mark.asyncio
async def test_empty_source_generation_shows_limitation_instead_of_blank_section(settings, monkeypatch):
    async def chat(sent, **kwargs):
        return "" if "RETRIEVED EVIDENCE" in sent[1]["content"] else "General explanation."
    monkeypatch.setattr(answer, "build_client", lambda _: SimpleNamespace(chat=chat))
    result = await answer.generate_mixed_answer("Explain", [chunk()], settings)
    assert "I couldn't generate a source-based explanation" in result
    assert "General explanation." in result
