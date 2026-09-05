"""Tests for the deterministic explainability layer (EduRAG Tier 3)."""

from __future__ import annotations

from app.core.attribution.chunk_attribution import jaccard_overlap, normalize_to_percentages
from app.core.attribution.citation_validation import (
    citation_validity_rate,
    extract_and_validate_citations,
)
from app.core.attribution.token_grounding import compute_token_grounding, grounding_ratio
from tests.conftest import make_chunk


class TestTokenGrounding:
    def test_exact_matches_score_highest(self):
        scores = compute_token_grounding("photosynthesis", "photosynthesis occurs in chloroplasts")
        assert [s.match_type for s in scores] == ["exact"]
        assert scores[0].score == 1.0

    def test_morphological_variant_falls_back_to_stem(self):
        scores = compute_token_grounding("running", "the process runs continuously")
        assert scores[0].match_type == "stem"
        assert scores[0].score == 0.9

    def test_absent_token_is_ungrounded(self):
        scores = compute_token_grounding("quantum", "photosynthesis occurs in chloroplasts")
        assert scores[0].match_type == "ungrounded"
        assert scores[0].score == 0.2

    def test_stopwords_are_excluded_from_scoring(self):
        scores = compute_token_grounding("the and of chloroplast", "chloroplast")
        assert [s.token for s in scores] == ["chloroplast"]

    def test_grounding_ratio_counts_only_tokens_above_threshold(self):
        context = "mitochondria produce energy"
        scores = compute_token_grounding("mitochondria produce zebras", context)
        # two grounded, one ungrounded at 0.2
        assert grounding_ratio(scores, threshold=0.8) == 2 / 3

    def test_empty_answer_gives_zero_ratio(self):
        assert grounding_ratio([], threshold=0.8) == 0.0


class TestCitationValidation:
    def test_in_range_markers_map_to_their_chunks(self):
        chunks = [make_chunk("a", "x"), make_chunk("b", "y")]
        citations = extract_and_validate_citations("Claim one [1] and two [2].", chunks)
        assert [(c.marker, c.chunk_id, c.valid) for c in citations] == [
            (1, "a", True),
            (2, "b", True),
        ]

    def test_marker_beyond_retrieved_set_is_hallucinated(self):
        chunks = [make_chunk("a", "x")]
        citations = extract_and_validate_citations("A claim [7].", chunks)
        assert citations[0].valid is False
        assert citations[0].chunk_id == ""

    def test_repeated_marker_is_reported_once(self):
        chunks = [make_chunk("a", "x")]
        citations = extract_and_validate_citations("One [1], again [1], still [1].", chunks)
        assert len(citations) == 1

    def test_validity_rate_is_the_fraction_of_real_references(self):
        chunks = [make_chunk("a", "x")]
        citations = extract_and_validate_citations("Good [1], bad [9].", chunks)
        assert citation_validity_rate(citations) == 0.5

    def test_uncited_answer_has_no_citations(self):
        assert extract_and_validate_citations("No markers here.", [make_chunk("a", "x")]) == []


class TestChunkAttribution:
    def test_jaccard_is_one_for_identical_content(self):
        assert jaccard_overlap("mitochondria produce energy", "mitochondria produce energy") == 1.0

    def test_jaccard_is_zero_for_disjoint_content(self):
        assert jaccard_overlap("mitochondria energy", "sonnet poetry") == 0.0

    def test_jaccard_ignores_stopwords(self):
        # differs only by stopwords, so the content sets are identical
        assert jaccard_overlap("the mitochondria", "a mitochondria") == 1.0

    def test_percentages_sum_to_one(self):
        percentages = normalize_to_percentages({"a": 3.0, "b": 1.0})
        assert percentages == {"a": 0.75, "b": 0.25}

    def test_all_zero_attribution_does_not_divide_by_zero(self):
        assert normalize_to_percentages({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}
