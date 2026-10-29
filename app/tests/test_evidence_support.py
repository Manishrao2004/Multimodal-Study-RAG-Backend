import pytest

from app.core.generation.orchestration import AspectSupport, EvidenceDecision, validate_evidence_decision
from app.models.schemas import ChunkType, EvidenceSufficiency, QuestionPlan
from app.tests.conftest import chunk


def judge(c, aspect="explanation", **kwargs):
    return EvidenceDecision(sufficiency="strong", topic_relevant=True, reason="Supported",
        supports=[AspectSupport(aspect=aspect, chunk_id=c.chunk_id, quote=c.text)], **kwargs)


def test_real_definition_keeps_source_route():
    c = chunk()
    plan = QuestionPlan(standalone_query="Explain a transformer", requested_aspects=["explanation"])
    result = validate_evidence_decision(plan, [c], judge(c))
    assert result.sufficiency == EvidenceSufficiency.strong
    assert result.covered_aspects == ["explanation"]


def test_quoting_only_a_heading_does_not_borrow_support_from_unquoted_prose():
    c = chunk("Transformer\nA transformer transfers energy through electromagnetic induction.")
    decision = judge(c)
    decision.supports[0].quote = "Transformer"
    result = validate_evidence_decision(QuestionPlan(standalone_query="Explain a transformer",
        requested_aspects=["explanation"]), [c], decision)
    assert result.sufficiency == EvidenceSufficiency.partial


@pytest.mark.parametrize("mutation", ["invented_quote", "wrong_chunk", "wrong_aspect", "changed_negation", "changed_case"])
def test_unsupported_support_never_becomes_strong(mutation):
    c = chunk()
    decision = judge(c)
    support = decision.supports[0]
    if mutation == "wrong_chunk":
        support.chunk_id = "other"
    elif mutation == "wrong_aspect":
        support.aspect = "cost"
    elif mutation == "changed_negation":
        support.quote = c.text.replace("transfers", "does not transfer")
    elif mutation == "changed_case":
        support.quote = c.text.upper()
    else:
        support.quote = "It has zero energy losses."
    result = validate_evidence_decision(QuestionPlan(standalone_query="Explain transformers",
        requested_aspects=["explanation"]), [c], decision)
    assert result.sufficiency != EvidenceSufficiency.strong
    assert not result.supports
    assert "explanation" in result.missing_aspects


def test_whitespace_in_pdf_quotes_is_normalized():
    c = chunk("A transformer\ntransfers energy through electromagnetic induction.")
    decision = judge(c)
    decision.supports[0].quote = c.text.replace("\n", "  ")
    assert validate_evidence_decision(QuestionPlan(standalone_query="Explain transformers",
        requested_aspects=["explanation"]), [c], decision).sufficiency == EvidenceSufficiency.strong


def test_arithmetic_uses_supplied_operands_without_demanding_printed_result():
    c = chunk("Revenue is INR 50000 and expenses are INR 35000.", kind=ChunkType.table)
    result = validate_evidence_decision(QuestionPlan(standalone_query="Calculate profit",
        requested_aspects=["profit calculation"]), [c], judge(c, "profit calculation"))
    assert result.sufficiency == EvidenceSufficiency.strong


def test_comparison_requires_both_items_and_all_requested_aspects():
    c = chunk("TCP provides reliable ordered delivery.")
    plan = QuestionPlan(standalone_query="Compare TCP and UDP", referenced_items=["TCP", "UDP"],
                        requested_aspects=["comparison", "examples"])
    result = validate_evidence_decision(plan, [c], judge(c, "comparison"))
    assert result.sufficiency == EvidenceSufficiency.partial
    assert "examples" in result.missing_aspects
    assert "Support for UDP" in result.missing_aspects


@pytest.mark.parametrize("query,expected", [("Explain these partner types", "partial"), ("List partner types", "strong")])
def test_outline_support_depends_on_requested_depth(query, expected):
    c = chunk("General partner\nSleeping partner\nNominal partner")
    result = validate_evidence_decision(QuestionPlan(standalone_query=query, requested_aspects=["requested content"]),
                                        [c], judge(c, "requested content"))
    assert result.sufficiency.value == expected


def test_uninspected_caption_does_not_establish_visual_support():
    c = chunk("Input A points to output B.", kind=ChunkType.visual)
    plan = QuestionPlan(standalone_query="Explain the diagram", requested_aspects=["explanation"])
    assert validate_evidence_decision(plan, [c], judge(c)).sufficiency == EvidenceSufficiency.partial
    c.visual_inspection_status = "completed"
    assert validate_evidence_decision(plan, [c], judge(c)).sufficiency == EvidenceSufficiency.strong


def test_current_facts_require_verification_even_with_a_matching_quote():
    c = chunk("The registration fee is INR 1000.")
    plan = QuestionPlan(standalone_query="What is the registration fee today?", requested_aspects=["fee"],
                        verification_required=True)
    result = validate_evidence_decision(plan, [c], judge(c, "fee"))
    assert result.sufficiency == EvidenceSufficiency.partial
    assert result.needs_background
    assert validate_evidence_decision(plan, [c], result).missing_aspects == result.missing_aspects


def test_genuine_quotes_do_not_override_an_insufficient_semantic_verdict():
    c = chunk()
    decision = judge(c)
    decision.sufficiency = EvidenceSufficiency.insufficient
    result = validate_evidence_decision(QuestionPlan(standalone_query="Explain transformers",
        requested_aspects=["explanation"]), [c], decision)
    assert result.sufficiency == EvidenceSufficiency.insufficient


def test_invented_topic_names_are_not_injected_into_the_response_plan():
    c = chunk("Types: General Partner, Sleeping Partner")
    decision = judge(c, topic_items=["General Partner", "Alien Partner"])
    result = validate_evidence_decision(QuestionPlan(standalone_query="List types", requested_aspects=["explanation"]), [c], decision)
    assert result.topic_items == ["General Partner"]
