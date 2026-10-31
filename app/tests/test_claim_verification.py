import json
from types import SimpleNamespace

import pytest

from app.core.generation import claim_verification as verifier
from app.models.schemas import AnswerBasis, AspectEvidence, ClaimVerification, EvidenceQuote, VerifiedClaim
from app.tests.conftest import chunk


def mock_client(monkeypatch, responses):
    calls = []
    async def chat(messages, **kwargs):
        calls.append(messages)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return json.dumps(response) if isinstance(response, dict) else response
    monkeypatch.setattr(verifier, "client_for", lambda s: SimpleNamespace(chat=chat))
    return calls


def audit(status="supported", quote="A transformer transfers energy through electromagnetic induction.", marker="c1"):
    return {"judgments": [{"claim_id": marker, "status": status,
        "supports": [{"chunk_id": "c1", "quote": quote}], "reason": "Fixture"}]}


def test_source_units_keep_st_louis_with_the_claim_it_completes():
    answer = "The Rams franchise moved to St. Louis in 1994 [2]. They played their first game there in 1995 [1]."
    units = verifier.source_units(answer)
    assert [unit for _, unit in units] == [
        "The Rams franchise moved to St. Louis in 1994 [2].",
        "They played their first game there in 1995 [1].",
    ]


@pytest.mark.asyncio
async def test_supported_citation_and_verbatim_quote_pass(settings, monkeypatch):
    mock_client(monkeypatch, [audit()])
    answer = "A transformer transfers energy [1]."
    text, result = await verifier.verify_and_repair(answer, [chunk()], settings, AnswerBasis.sources)
    assert text == answer and result.status == "completed"
    assert result.claims[0].status == "supported" and not result.withheld


@pytest.mark.asyncio
async def test_line_citation_still_requires_verbatim_support(settings, monkeypatch):
    mock_client(monkeypatch, [audit()])
    text, result = await verifier.verify_and_repair("A transformer transfers energy 【1†L1-L2】.", [chunk()], settings, AnswerBasis.sources)
    assert text == "A transformer transfers energy [1]."
    assert result.claims[0].status == "supported" and not result.withheld


@pytest.mark.asyncio
@pytest.mark.parametrize("text,quote", [
    ("A transformer transfers energy [2].", "A transformer transfers energy through electromagnetic induction."),
    ("A transformer transfers energy [1].", "A transformer does not transfer energy."),
    ("A transformer transfers energy.", "A transformer transfers energy through electromagnetic induction."),
])
async def test_ids_or_lexical_similarity_cannot_rescue_invalid_claim(settings, monkeypatch, text, quote):
    settings.claim_repair_enabled = False
    mock_client(monkeypatch, [audit(quote=quote)])
    released, result = await verifier.verify_and_repair(text, [chunk()], settings, AnswerBasis.sources)
    assert result.withheld and result.claims[0].status == "unsupported"
    assert text not in released


@pytest.mark.asyncio
async def test_missing_claim_judgment_fails_closed(settings, monkeypatch):
    mock_client(monkeypatch, [{"judgments": []}])
    _, result = await verifier.verify_and_repair("Wrong number [1].", [chunk()], settings, AnswerBasis.sources)
    assert result.withheld and result.status == "unavailable"


@pytest.mark.asyncio
async def test_one_repair_is_rechecked_and_failures_preserved(settings, monkeypatch):
    calls = mock_client(monkeypatch, [audit("contradicted"), "Corrected fact [1].", audit()])
    text, result = await verifier.verify_and_repair("Wrong fact [1].", [chunk()], settings, AnswerBasis.sources)
    assert text == "Corrected fact [1]." and not result.withheld
    assert result.repair_attempted and result.initial_claims[0].status == "contradicted"
    assert len(calls) == result.model_calls == 3


@pytest.mark.asyncio
async def test_failed_repair_never_loops(settings, monkeypatch):
    calls = mock_client(monkeypatch, [audit("unsupported"), "Still wrong [1].", audit("unsupported")])
    _, result = await verifier.verify_and_repair("Wrong [1].", [chunk()], settings, AnswerBasis.sources)
    assert result.withheld and len(calls) == 3


@pytest.mark.asyncio
async def test_fail_open_never_releases_known_unsupported_claim_after_repair_fails(settings, monkeypatch):
    settings.claim_verification_fail_closed = False
    mock_client(monkeypatch, [audit("unsupported"), TimeoutError()])
    text, report = await verifier.verify_and_repair("Wrong [1].", [chunk()], settings, AnswerBasis.sources)
    assert report.withheld and "Wrong [1]" not in text
    assert report.initial_claims[0].status == "unsupported"


@pytest.mark.asyncio
async def test_fail_open_reports_unavailable_when_no_audit_completed(settings, monkeypatch):
    settings.claim_verification_fail_closed = False
    mock_client(monkeypatch, [TimeoutError()])
    text, report = await verifier.verify_and_repair("Draft [1].", [chunk()], settings, AnswerBasis.sources)
    assert text == "Draft [1]." and not report.withheld
    assert report.status == "unavailable" and "unverified" in report.reason


@pytest.mark.asyncio
async def test_mixed_failure_preserves_only_original_delimited_background(settings, monkeypatch):
    mock_client(monkeypatch, [TimeoutError()])
    answer = f"{verifier.SOURCE_HEADING}\n\nWrong source claim [1].\n\n{verifier.BACKGROUND_HEADING}\n\nGeneral explanation [9]."
    text, result = await verifier.verify_and_repair(answer, [chunk()], settings, AnswerBasis.mixed)
    assert result.withheld and "Wrong source claim" not in text
    assert "General explanation" in text and "[9]" not in text


@pytest.mark.asyncio
async def test_malformed_mixed_boundaries_do_not_relabel_source_as_background(settings, monkeypatch):
    calls = mock_client(monkeypatch, [])
    text, result = await verifier.verify_and_repair("Wrong [1].", [chunk()], settings, AnswerBasis.mixed)
    assert result.withheld and "Wrong" not in text and not calls


@pytest.mark.asyncio
async def test_oversize_answer_does_not_silently_audit_a_prefix(settings, monkeypatch):
    calls = mock_client(monkeypatch, [])
    _, result = await verifier.verify_and_repair("x" * 13000, [chunk()], settings, AnswerBasis.sources)
    assert result.withheld and result.status == "unavailable" and not calls and result.model_calls == 0


@pytest.mark.asyncio
async def test_duplicate_mixed_headings_cannot_smuggle_source_claims_into_background(settings, monkeypatch):
    calls = mock_client(monkeypatch, [])
    answer = (f"{verifier.SOURCE_HEADING}\n\nFact [1].\n{verifier.BACKGROUND_HEADING}\n"
              f"Smuggled claim\n{verifier.BACKGROUND_HEADING}\nActual background")
    text, result = await verifier.verify_and_repair(answer, [chunk()], settings, AnswerBasis.mixed)
    assert result.withheld and "Smuggled" not in text and "Actual background" not in text and not calls


@pytest.mark.asyncio
async def test_verification_ablation_and_model_only_do_not_call_judge(settings, monkeypatch):
    calls = mock_client(monkeypatch, [])
    settings.claim_verification_enabled = False
    text, report = await verifier.verify_and_repair("Draft", [chunk()], settings, AnswerBasis.sources)
    assert text == "Draft" and report.status == "not_checked"
    settings.claim_verification_enabled = True
    text, report = await verifier.verify_and_repair("Draft", [], settings, AnswerBasis.model)
    assert text == "Draft" and not calls


@pytest.mark.asyncio
async def test_derived_calculation_can_quote_operands(settings, monkeypatch):
    c = chunk("Revenue is 50000 and expenses are 35000.")
    response = audit(quote=c.text)
    response["judgments"][0]["derived"] = True
    mock_client(monkeypatch, [response])
    claims = await verifier.audit_source("Profit is 15000 [1].", [c], settings)
    assert claims[0].status == "supported" and claims[0].provenance == "derived"


def test_numbered_list_with_shared_citation_is_one_complete_audit_unit():
    answer = "The partner types are:\n\n1. Active Partner\n2. Sleeping Partner\n3. Nominal Partner\n\nThese categories are listed in the source [1]."
    units = verifier.source_units(answer)
    assert units == [("c1", "The partner types are:\n1. Active Partner\n2. Sleeping Partner\n3. Nominal Partner\nThese categories are listed in the source [1].")]
    assert all(text not in {"1.", "2.", "3."} for _, text in units)


def test_list_citation_scope_does_not_extend_to_unrelated_prose():
    answer = "1. TCP\n2. UDP [1]\n\nAll networking is free [2]."
    units = verifier.source_units(answer)
    assert len(units) == 2
    assert "[2]" not in units[0][1] and "free" in units[1][1]


@pytest.mark.asyncio
async def test_cited_list_passes_without_unnecessary_repair(settings, monkeypatch):
    c = chunk("1. Active Partner\n2. Sleeping Partner")
    calls = mock_client(monkeypatch, [audit(quote=c.text)])
    answer = "Types:\n1. Active Partner\n2. Sleeping Partner\n\nThese types are listed in the source [1]."
    text, report = await verifier.verify_and_repair(answer, [c], settings, AnswerBasis.sources)
    assert text == answer and not report.withheld and not report.repair_attempted
    assert len(calls) == 1


def test_failed_audit_can_display_only_validated_literal_source_excerpts():
    c = chunk("1. Active Partner\n2. Sleeping Partner")
    mapped = [AspectEvidence(aspect_id="a1", aspect="types", status="supported", supports=[
        EvidenceQuote(chunk_id=c.chunk_id, quote=c.text),
        EvidenceQuote(chunk_id=c.chunk_id, quote="A fabricated definition"),
    ])]
    report = ClaimVerification(status="unavailable", withheld=True)
    text = verifier.source_excerpt_fallback("Withheld", mapped, [c], report, AnswerBasis.sources)
    assert report.fallback_used
    assert "> 1. Active Partner\n> 2. Sleeping Partner" in text
    assert "fabricated" not in text and "[1]" in text


def test_excerpt_fallback_preserves_separate_background_and_quotes_conflicting_positions():
    a, b = chunk("Maximum is 8.", "a"), chunk("Maximum is 12.", "b")
    mapped = [AspectEvidence(aspect_id="a1", aspect="size", status="conflicting",
        supports=[EvidenceQuote(chunk_id="a", quote=a.text)],
        conflicts=[EvidenceQuote(chunk_id="b", quote=b.text)])]
    answer = verifier.join_answer("Withheld", "General explanation.")
    report = ClaimVerification(status="completed", withheld=True)
    text = verifier.source_excerpt_fallback(answer, mapped, [a,b], report, AnswerBasis.mixed)
    assert "> Maximum is 8." in text and "> Maximum is 12." in text
    assert "[1]" in text and "[2]" in text
    assert text.endswith("General explanation.")


def test_later_repair_failure_preserves_audited_list_without_unsupported_scope_claim():
    c = chunk("1. Active Partner\n2. Sleeping Partner")
    listed = "The types are:\n1. Active Partner\n2. Sleeping Partner\nThese categories are listed in the source [1]."
    mapped = [AspectEvidence(aspect_id="a1", aspect="types", status="supported",
                             supports=[EvidenceQuote(chunk_id=c.chunk_id, quote=c.text)])]
    report = ClaimVerification(status="unavailable", withheld=True, repair_attempted=True, claims=[
        VerifiedClaim(claim_id="c1", text=listed, provenance="source", status="supported",
            citation_markers=[1], supports=[EvidenceQuote(chunk_id=c.chunk_id, quote=c.text)]),
        VerifiedClaim(claim_id="c2", text="Other invented domains cannot be addressed.", provenance="source", status="unsupported"),
    ])
    text = verifier.source_excerpt_fallback("Withheld", mapped, [c], report, AnswerBasis.sources)
    assert text == listed and "invented domains" not in text
    assert report.fallback_kind == "supported_claims" and report.released_claim_ids == ["c1"]


def test_supported_subset_still_requires_valid_quotes_and_citations():
    c = chunk()
    report = ClaimVerification(status="unavailable", withheld=True, claims=[
        VerifiedClaim(claim_id="c1", text="Incorrect [1]", provenance="source", status="supported",
            citation_markers=[1], supports=[EvidenceQuote(chunk_id=c.chunk_id, quote="Invented quote")]),
    ])
    assert verifier.source_excerpt_fallback("Withheld", [], [c], report, AnswerBasis.sources) == "Withheld"
    assert not report.fallback_used


def test_relevant_topic_list_is_available_when_definitions_and_audit_are_missing():
    c = chunk("1. Active Partner -\n2. Sleeping Partner -")
    mapped = [AspectEvidence(aspect_id="a1", aspect="definition", status="missing")]
    report = ClaimVerification(status="unavailable", withheld=True)
    answer = verifier.join_answer("Withheld", "General explanation.")
    text = verifier.source_excerpt_fallback(answer, mapped, [c], report, AnswerBasis.mixed,
                                             topic_items=["Active Partner", "Sleeping Partner"])
    assert "> 1. Active Partner -\n> 2. Sleeping Partner -" in text
    assert text.endswith("General explanation.") and report.fallback_kind == "source_excerpts"


def test_topic_list_fallback_does_not_quote_an_uninspected_visual():
    c = chunk("Active Partner", kind="visual")
    report = ClaimVerification(status="unavailable", withheld=True)
    assert verifier.source_excerpt_fallback("Withheld", [], [c], report, AnswerBasis.sources,
                                            topic_items=["Active Partner"]) == "Withheld"
