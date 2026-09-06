"""Evaluation metrics (plan Sec. 8)."""

from __future__ import annotations

import pytest

from app.eval.benchmark import sample_chunks
from app.eval.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
    rouge_l,
    token_f1,
)
from app.eval.significance import bootstrap_ci, paired_wilcoxon
from app.models.schemas import ChunkType
from tests.conftest import make_chunk


class TestReciprocalRank:
    def test_top_hit_scores_one(self):
        assert reciprocal_rank(["a", "b", "c"], "a") == 1.0

    def test_third_place_scores_one_third(self):
        assert reciprocal_rank(["a", "b", "c"], "c") == 1 / 3

    def test_missing_gold_scores_zero(self):
        assert reciprocal_rank(["a", "b"], "z") == 0.0

    def test_mrr_averages_across_queries(self):
        # ranks 1 and 2 -> (1.0 + 0.5) / 2
        assert mean_reciprocal_rank([["a"], ["x", "b"]], ["a", "b"]) == 0.75

    def test_mrr_of_no_queries_is_zero(self):
        assert mean_reciprocal_rank([], []) == 0.0


class TestRecall:
    def test_recall_at_1_requires_the_top_position(self):
        retrieved = [["a", "b"], ["x", "b"]]
        assert recall_at_k(retrieved, ["a", "b"], k=1) == 0.5

    def test_recall_at_5_is_more_forgiving_than_recall_at_1(self):
        retrieved = [["w", "x", "y", "z", "gold"]]
        assert recall_at_k(retrieved, ["gold"], k=1) == 0.0
        assert recall_at_k(retrieved, ["gold"], k=5) == 1.0


class TestLexicalMetrics:
    def test_token_f1_is_one_for_identical_text(self):
        assert token_f1("the cat sat", "the cat sat") == 1.0

    def test_token_f1_is_zero_with_no_shared_tokens(self):
        assert token_f1("alpha beta", "gamma delta") == 0.0

    def test_token_f1_penalises_padding(self):
        # Same recall, worse precision than the exact match.
        assert token_f1("the cat sat on a very long mat", "the cat sat") < 1.0

    def test_rouge_l_rewards_correct_ordering(self):
        in_order = rouge_l("a b c d", "a b c d")
        shuffled = rouge_l("d c b a", "a b c d")
        assert in_order == 1.0
        assert shuffled < in_order

    def test_metrics_handle_empty_input(self):
        assert token_f1("", "something") == 0.0
        assert rouge_l("", "something") == 0.0


class TestBenchmarkSampling:
    def test_sampling_is_deterministic_for_a_seed(self):
        chunks = [make_chunk(f"c{i}", "x" * 300) for i in range(20)]
        first = [c.chunk_id for c in sample_chunks(chunks, 5, seed=7)]
        second = [c.chunk_id for c in sample_chunks(chunks, 5, seed=7)]
        assert first == second

    def test_short_chunks_are_excluded(self):
        chunks = [make_chunk("short", "tiny"), make_chunk("long", "x" * 300)]
        assert [c.chunk_id for c in sample_chunks(chunks, 5)] == ["long"]

    def test_visual_filter_selects_only_captions(self):
        chunks = [
            make_chunk("t", "x" * 300),
            make_chunk("v", "y" * 300, chunk_type=ChunkType.visual),
        ]
        selected = sample_chunks(chunks, 5, chunk_type=ChunkType.visual)
        assert [c.chunk_id for c in selected] == ["v"]

    def test_requesting_more_than_available_returns_all(self):
        chunks = [make_chunk("a", "x" * 300)]
        assert len(sample_chunks(chunks, 50)) == 1


class TestGenerationScoreReporting:
    """Citation validity and citation coverage measure different failures and
    must not be collapsed: an answer with no markers is a coverage failure,
    while a marker pointing at a non-existent passage is a validity failure.
    Averaging uncited answers into validity would silently redefine the
    base paper's metric and make the numbers incomparable."""

    def _scores(self, **overrides):
        from app.eval.runner import GenerationScores

        defaults = dict(
            queries=2,
            bertscore_f1=0.9,
            bertscore_backend="bert-score",
            rouge_l=0.4,
            token_f1=0.5,
            faithfulness=0.9,
            citation_validity_rate=1.0,
            citation_coverage=0.5,
            token_grounding_ratio=0.7,
        )
        defaults.update(overrides)
        return GenerationScores(**defaults)

    def test_both_citation_metrics_are_reported(self):
        row = self._scores().as_row()
        assert row["Citation Validity Rate"] == 1.0
        assert row["Citation Coverage"] == 0.5

    def test_perfect_validity_can_coexist_with_partial_coverage(self):
        # Exactly the observed case: every marker the model wrote was real,
        # but half the answers carried no marker at all.
        row = self._scores(citation_validity_rate=1.0, citation_coverage=0.44).as_row()
        assert row["Citation Validity Rate"] > row["Citation Coverage"]

    def test_hallucinated_citations_lower_validity_not_coverage(self):
        row = self._scores(citation_validity_rate=0.5, citation_coverage=1.0).as_row()
        assert row["Citation Validity Rate"] == 0.5
        assert row["Citation Coverage"] == 1.0


class TestPairedWilcoxon:
    def test_consistently_higher_arm_is_flagged_significant(self):
        # A clear, consistent improvement across every query.
        values_a = [0.2, 0.3, 0.1, 0.25, 0.15, 0.3, 0.2, 0.1, 0.35, 0.2]
        values_b = [0.9, 0.85, 0.95, 0.8, 0.9, 0.88, 0.92, 0.85, 0.9, 0.87]
        result = paired_wilcoxon("MRR", values_a, values_b)
        assert result.mean_diff > 0
        assert result.p_value is not None
        assert result.p_value < 0.05
        assert result.significant_at_0_05 is True

    def test_identical_arms_are_not_significant(self):
        values = [0.5, 0.6, 0.4, 0.7, 0.3]
        result = paired_wilcoxon("MRR", values, list(values))
        assert result.statistic is None
        assert result.p_value is None
        assert result.significant_at_0_05 is None
        assert "undefined" in result.note

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            paired_wilcoxon("MRR", [0.1, 0.2], [0.1])

    def test_means_are_reported_regardless_of_significance(self):
        result = paired_wilcoxon("MRR", [0.5, 0.5], [0.5, 0.5])
        assert result.mean_a == 0.5
        assert result.mean_b == 0.5


class TestBootstrapCI:
    def test_ci_brackets_the_mean(self):
        values = [0.5, 0.6, 0.55, 0.45, 0.5, 0.6, 0.5]
        mean, lo, hi = bootstrap_ci(values, n_resamples=2000)
        assert lo <= mean <= hi

    def test_empty_input_returns_zeros(self):
        assert bootstrap_ci([]) == (0.0, 0.0, 0.0)

    def test_constant_values_give_a_zero_width_interval(self):
        mean, lo, hi = bootstrap_ci([0.7, 0.7, 0.7], n_resamples=500)
        assert mean == pytest.approx(0.7)
        assert lo == pytest.approx(hi)
        assert lo == pytest.approx(mean)

    def test_same_seed_is_reproducible(self):
        values = [0.1, 0.9, 0.3, 0.7, 0.5]
        first = bootstrap_ci(values, n_resamples=500, seed=1)
        second = bootstrap_ci(values, n_resamples=500, seed=1)
        assert first == second
