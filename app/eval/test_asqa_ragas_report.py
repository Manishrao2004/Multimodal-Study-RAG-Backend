from app.eval.asqa_ragas_report import summarize


def test_separate_excerpts_and_count_missing_and_exclusions():
    inputs = [
        {"id": "1", "method": "aspect", "evaluated_answer_kind": "source_excerpts", "status": "ok"},
        {"id": "2", "method": "aspect", "evaluated_answer_kind": "whole_source_answer", "status": "ok"},
        {"id": "3", "method": "aspect", "evaluated_answer_kind": "whole_source_answer", "status": "excluded", "exclusion_reason": "no_retrieved_context"},
    ]
    output = [{"rows": [{"id": "1", "scores": {"faithfulness": 1.0}}, {"id": "2", "scores": {}}]}]
    result = summarize(inputs, output)["groups"]
    excerpt = result["aspect:source_excerpts"]
    answer = result["aspect:whole_source_answer"]
    assert excerpt["metrics"]["faithfulness"]["mean"] == 1.0
    assert excerpt["metrics"]["faithfulness"]["ci95"] is None
    assert answer["metrics"]["faithfulness"]["missing"] == 1
    assert answer["excluded"] == 1 and answer["exclusion_reasons"] == {"no_retrieved_context": 1}
