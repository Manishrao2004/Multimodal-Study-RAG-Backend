import json

import pytest

from app.eval.aspect_study import DATASET, ControlledRetrieval, StudyDataset, method_settings, paired_aspect_differences, summarize, write_review
from app.models.schemas import Chunk


def test_development_dataset_is_valid_and_disclaims_independent_review():
    data = StudyDataset.model_validate_json(DATASET.read_text(encoding="utf-8"))
    assert len(data.cases) >= 10 and "not independently reviewed" in data.annotation_status
    assert {s for c in data.cases for s in c.expected_aspects.values()} == {"supported", "partial", "missing", "conflicting"}


def test_removed_evidence_cannot_enter_a_controlled_search():
    data = StudyDataset.model_validate_json(DATASET.read_text(encoding="utf-8"))
    case = next(c for c in data.cases if c.id == "unrecoverable-gap")
    corpus = {p.id: Chunk(chunk_id=p.id, source_file=p.source, text=p.text, chunk_length=len(p.text), type="text") for p in data.passages}
    retrieval = ControlledRetrieval(case, corpus)
    assert "p-risk" not in retrieval.available


def test_failures_stay_in_denominator_and_baseline_missing_metrics_are_null():
    rows = [dict(id="a", group="g", method="aspect", expected_aspects={"x":"supported"}, status="ok",
                 elapsed_ms=10, response={"evidence_map":[dict(aspect="x", status="supported")]}),
            dict(id="b", group="g", method="aspect", expected_aspects={"x":"supported"}, status="error", elapsed_ms=30)]
    report = summarize(rows)["aspect"]
    assert report["completion_rate"] == .5
    assert report["aspect_status_accuracy"] == 1  # explicitly conditional on completed maps
    rows[0]["method"] = "adaptive"
    rows[0]["response"] = {}
    assert summarize(rows)["adaptive"]["aspect_status_accuracy"] is None


def test_cluster_comparison_is_paired_and_deterministic():
    rows = [dict(id="a", group="g", method="aspect", expected_aspects={"x":"supported"}, status="ok",
                 response={"evidence_map":[dict(aspect="x", status="supported")]}),
            dict(id="a", group="g", method="aspect_no_retrieval", expected_aspects={"x":"supported"}, status="error")]
    assert paired_aspect_differences(rows) == paired_aspect_differences(rows)
    assert paired_aspect_differences(rows)[0]["mean_difference"] == 1
    assert paired_aspect_differences(rows)[0]["cluster_bootstrap95"] is None


def test_whole_question_is_not_scored_against_incompatible_aspect_names():
    rows = [dict(id="a", group="g", method="whole_question", expected_aspects={"reliability":"supported"},
        status="ok", elapsed_ms=1, response={"evidence_map":[dict(aspect="Compare TCP and UDP", status="supported")]})]
    assert summarize(rows)["whole_question"]["aspect_status_accuracy"] is None
    assert paired_aspect_differences(rows) == []


def test_invalid_dataset_does_not_allow_initial_gold_outside_available_corpus():
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    payload["cases"][0]["available_context_ids"] = []
    with pytest.raises(ValueError):
        StudyDataset.model_validate(payload)


def test_human_review_export_leaves_review_fields_blank(tmp_path):
    path = tmp_path / "review.csv"
    write_review([dict(id="a", method="aspect", group="g", query="q", reference="r", response={"answer":"answer"})], path)
    import csv
    with path.open(encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    assert row["answer"] == "answer" and row["reviewer"] == row["source_claim_support"] == ""
