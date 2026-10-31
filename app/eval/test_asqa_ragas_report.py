from app.eval.asqa_ragas_report import summarize
from app.eval.checkpoints import digest
import pytest


def test_separate_excerpts_and_count_missing_and_exclusions():
    inputs = [
        {"id": "1", "sample_id": "1", "method": "aspect", "evaluated_answer_kind": "source_excerpts", "status": "ok"},
        {"id": "2", "sample_id": "2", "method": "aspect", "evaluated_answer_kind": "whole_source_answer", "status": "ok"},
        {"id": "3", "sample_id": "3", "method": "aspect", "evaluated_answer_kind": "whole_source_answer", "status": "excluded", "exclusion_reason": "no_retrieved_context"},
    ]
    output = [{"rows": [{"id": "1", "record_sha256": digest(inputs[0]), "scores": {"faithfulness": 1.0}}, {"id": "2", "record_sha256": digest(inputs[1]), "scores": {}}]}]
    result = summarize(inputs, output)["groups"]
    excerpt = result["aspect:source_excerpts"]
    answer = result["aspect:whole_source_answer"]
    assert excerpt["metrics"]["faithfulness"]["mean"] == 1.0
    assert excerpt["metrics"]["faithfulness"]["ci95"] is None
    assert answer["metrics"]["faithfulness"]["missing"] == 1
    assert answer["excluded"] == 1 and answer["exclusion_reasons"] == {"no_retrieved_context": 1}


def test_rejects_judgment_from_changed_answer():
    row = {"id": "1", "sample_id": "1", "method": "aspect", "evaluated_answer_kind": "whole_source_answer", "status": "ok", "answer": "new"}
    with pytest.raises(ValueError, match="does not match"):
        summarize([row], [{"rows": [{"id": "1", "record_sha256": digest({**row, "answer": "old"})}]}])


def test_pairs_only_same_question_and_excludes_excerpts():
    rows = [
        {"id": "a", "sample_id": "1", "method": "edurag", "evaluated_answer_kind": "whole_source_answer", "status": "ok"},
        {"id": "b", "sample_id": "1", "method": "aspect", "evaluated_answer_kind": "mixed_source_section", "status": "ok"},
        {"id": "c", "sample_id": "2", "method": "edurag", "evaluated_answer_kind": "whole_source_answer", "status": "ok"},
        {"id": "d", "sample_id": "2", "method": "aspect", "evaluated_answer_kind": "source_excerpts", "status": "ok"},
    ]
    judged = [{"id": r["id"], "record_sha256": digest(r), "scores": {"faithfulness": score}} for r, score in zip(rows, [.4, .9, .2, 1.0])]
    paired = summarize(rows, [{"rows": judged}])["paired_aspect_minus_edurag"]["faithfulness"]
    assert paired["n"] == 1 and paired["mean"] == .5 and paired["ci95"] is None


def test_rejects_mixed_judge_models():
    with pytest.raises(ValueError, match="judge configurations"):
        summarize([], [{"judge_model": "one", "rows": []}, {"judge_model": "two", "rows": []}])
