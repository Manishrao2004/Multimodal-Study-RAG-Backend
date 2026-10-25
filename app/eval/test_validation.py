"""Tracked evaluation regression tests (the legacy tests directory is ignored)."""
import pytest

from app.eval.validate import CASES, load_cases, check_expectations
from app.eval.validation_metrics import classification, claim_metrics, interval, retrieval_metrics, summarize, wilson_interval, bounded_score


def test_dataset_has_three_answerability_classes_and_valid_contexts():
    data = load_cases(CASES)
    assert {c["expected_basis"] for c in data["cases"]} == {"sources", "mixed", "model"}
    assert len(data["cases"]) >= 12


def test_classification_mistakes_and_unrepresented_classes():
    result = classification(["sources", "mixed", "model"], ["sources", "sources", "model"], ["sources", "mixed", "model"])
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["per_class"]["mixed"]["f1"] == 0
    assert result["per_class"]["sources"]["precision"] == .5
    assert classification([], [], ["sources"])["accuracy"] is None


def test_invalid_classifications_rejected():
    with pytest.raises(ValueError):
        classification(["a"], [], ["a"])
    with pytest.raises(ValueError):
        classification(["b"], ["a"], ["a"])


def test_multirelevant_retrieval_counts_duplicates_once():
    score = retrieval_metrics(["noise", "a", "a", "b"], ["a", "b", "c"], 3)
    assert score["precision"] == pytest.approx(2 / 3)
    assert score["recall"] == pytest.approx(2 / 3)
    assert score["rr"] == .5
    assert score["ap"] == pytest.approx((.5 + 2 / 3) / 3)
    assert 0 < score["ndcg"] < 1
    assert retrieval_metrics([], ["a"])["recall"] == 0
    assert retrieval_metrics(["a"], [])["recall"] is None


def test_claim_id_validity_does_not_imply_entailment():
    result = claim_metrics([
        dict(supported=False, requires_citation=True, cited=True, citation_supported=False),
        dict(supported=True, requires_citation=True, cited=False, citation_supported=False),
        dict(supported=False, requires_citation=False, cited=False, citation_supported=False),
    ])
    assert result["source_claim_support"] == .5
    assert result["claim_citation_completeness"] == 0
    assert result["citation_entailment_precision"] == 0
    assert claim_metrics([])["source_claim_support"] is None
    with pytest.raises(ValueError):
        claim_metrics([dict(supported=True, requires_citation=True, cited=False, citation_supported=True)])


def test_failed_generation_in_end_to_end_denominator():
    good = dict(status="ok", expected_basis="sources", elapsed_ms=10,
                response=dict(answer_basis="sources", evidence_sufficiency="strong", citations=[]))
    result = summarize([good, dict(status="error", expected_basis="mixed")])
    assert result["routing"]["accuracy"] == 1
    assert result["end_to_end_routing_success"] == .5
    assert result["citation_id_validity_micro"] is None
    assert result["failures"] == 1


def test_bootstrap_reproducible_and_empty_is_unknown():
    assert interval([0, 1, 1]) == interval([0, 1, 1])
    assert interval([])["ci95"] is None
    assert interval([1, 1])["ci95"] == [1, 1]
    with pytest.raises(ValueError):
        interval([float("nan")])


def test_binomial_interval_does_not_claim_certainty_from_twelve_passes():
    result = wilson_interval(12, 12)
    assert .75 < result["ci95"][0] < .76
    assert result["ci95"][1] == pytest.approx(1)
    assert wilson_interval(0, 0)["ci95"] is None
    with pytest.raises(ValueError):
        wilson_interval(2, 1)


def test_region_and_verification_checks_fail_on_wrong_or_missing_context():
    assert check_expectations({"expected_currency": "INR"}, {"answer": "Profit is INR 15000."}, {})["currency_label"]
    assert not check_expectations({"expected_currency": "INR"}, {"answer": "Profit is $150."}, {})["currency_label"]
    assert not check_expectations({"expected_verification_required": True}, {"answer": "Fees vary."}, {})["verification_flag"]
    assert check_expectations({"forbidden_patterns": [r"\d+%"]}, {"answer": "Tax is 25%."}, {})["no_forbidden_details"] is False


def test_successful_retry_does_not_hide_initial_failure():
    row = dict(status="ok", expected_basis="sources", elapsed_ms=10,
               previous_attempts=[dict(status="error", error_type="TimeoutError", elapsed_ms=180000)],
               response=dict(answer_basis="sources", evidence_sufficiency="strong", citations=[]))
    result = summarize([row])
    assert result["completion_rate"] == 1
    assert result["handler_attempt_completion_rate"] == .5
    assert result["first_handler_attempt_completion_rate"] == 0
    assert result["first_handler_attempt_routing_success"] == 0
    assert result["retry_attempts"] == 1
    assert result["handler_attempt_latency_ms"]["p95"] == 180000


def test_metric_bounds_tolerate_roundoff_but_reject_invalid_scores():
    assert bounded_score(1.00000001) == 1
    assert bounded_score(-.2, -1) == -.2
    for value in (float("nan"), float("inf"), 1.01, -.1):
        with pytest.raises(ValueError):
            bounded_score(value)
