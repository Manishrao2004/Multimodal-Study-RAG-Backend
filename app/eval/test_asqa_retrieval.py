from app.eval.asqa_retrieval import make_cases, report
from app.models.schemas import Chunk


def test_labels_match_shared_passages_and_exclude_placeholder():
    chunks = [Chunk(chunk_id="a", source_file="asqa_a", text="shared", chunk_length=6, type="text"),
              Chunk(chunk_id="b", source_file="asqa_b", text="shared", chunk_length=6, type="text")]
    rows = [{"sample_id": "1", "ambiguous_question": "Who?", "annotations": [],
             "qa_pairs": [{"context": "shared"}]},
            {"sample_id": "2", "ambiguous_question": "What?", "annotations": [],
             "qa_pairs": [{"context": "No context provided"}]}]
    cases, excluded = make_cases(rows, chunks)
    assert cases[0]["relevant_ids"] == ["a", "b"]
    assert len(excluded) == 1


def test_failed_retrieval_is_counted_separately():
    cases = [{"sample_id": "1", "relevant_ids": ["a"]}, {"sample_id": "2", "relevant_ids": ["b"]}]
    records = [{"sample_id": "1", "mode": "hybrid", "status": "ok", "retrieved_ids": ["a"], "elapsed_ms": 1},
               {"sample_id": "2", "mode": "hybrid", "status": "error", "elapsed_ms": 2}]
    metric = report(records, cases)["hybrid"]
    assert metric["failures"] == 1
    assert metric["completion_rate"] == .5
    assert metric["mrr_at_5"]["n"] == 1
