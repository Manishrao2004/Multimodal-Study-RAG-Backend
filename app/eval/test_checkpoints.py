import copy

import pytest

from app.eval.checkpoints import case_manifest, digest, resume_rows, row_metadata


def fixtures():
    rows = [{"id": "a", "query": "Question A", "reference": "Fact A", "status": "ok", "response": {"answer": "Fact A"}},
            {"id": "b", "query": "Question B", "reference": "Fact B", "status": "error", "error_type": "TimeoutError"}]
    stored = {"judge_model": "judge", "records_sha256": digest(rows), "case_manifest": case_manifest(rows),
              "rows": [{"id": "a", "status": "ok", "scores": {"faithfulness": 1}, **row_metadata(rows[0])},
                       {"id": "b", "status": "not_applicable", **row_metadata(rows[1])}]}
    return rows, stored


def test_recovered_handler_can_be_judged_without_rerunning_unchanged_answers():
    rows, stored = fixtures()
    rows[1].update(status="ok", response={"answer": "Fact B"})
    rows[1].pop("error_type")
    previous, recovered = resume_rows(stored, rows, {"judge_model": "judge"}, ("judge_model",))
    assert list(previous) == ["a"]
    assert recovered == ["b"]


def test_changed_successful_answer_cannot_reuse_its_old_score():
    rows, stored = fixtures()
    rows[0]["response"]["answer"] = "Different answer"
    with pytest.raises(ValueError, match="Completed answer changed"):
        resume_rows(stored, rows, {"judge_model": "judge"}, ("judge_model",))


def test_changed_reference_or_judge_requires_fresh_experiment():
    rows, stored = fixtures()
    changed = copy.deepcopy(rows)
    changed[0]["reference"] = "Different fact"
    with pytest.raises(ValueError, match="reference labels"):
        resume_rows(stored, changed, {"judge_model": "judge"}, ("judge_model",))
    with pytest.raises(ValueError, match="judge configuration"):
        resume_rows(stored, rows, {"judge_model": "different judge"}, ("judge_model",))


def test_duplicate_case_ids_are_invalid():
    with pytest.raises(ValueError, match="Duplicate"):
        case_manifest([{"id": "a"}, {"id": "a"}])


def test_failure_history_update_does_not_reuse_or_invent_a_judgment():
    rows, stored = fixtures()
    rows[1]["previous_attempts"] = [{"status": "error", "error_type": "TimeoutError"}]
    previous, recovered = resume_rows(stored, rows, {"judge_model": "judge"}, ("judge_model",))
    assert list(previous) == ["a"]
    assert recovered == []
