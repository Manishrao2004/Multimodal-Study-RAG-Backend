"""Eval runner integration: retrieval scoring and ablation significance
against the real (small, local) index built by the `populated_state` fixture."""

from __future__ import annotations

import pytest

from app.eval.benchmark import BenchmarkItem
from app.eval.runner import ablation_significance, evaluate_retrieval, run_ablation
from app.models.schemas import RetrievalMode


def _benchmark_items() -> list[BenchmarkItem]:
    return [
        BenchmarkItem(
            query="how does reciprocal rank fusion combine ranked lists",
            reference_answer="It sums the reciprocal of each document's rank across retrievers.",
            gold_chunk_id="c1",
            source_file="notes.pdf",
            chunk_type="text",
        ),
        BenchmarkItem(
            query="what does a cross-encoder reranker do",
            reference_answer="It scores the query and document jointly with a transformer.",
            gold_chunk_id="c3",
            source_file="slides.pptx",
            chunk_type="text",
        ),
    ]


@pytest.mark.asyncio
class TestEvaluateRetrieval:
    async def test_carries_per_query_reciprocal_rank_for_pairing(self, populated_state):
        items = _benchmark_items()
        scores = await evaluate_retrieval(items, populated_state.retrieval, RetrievalMode.full)
        assert len(scores.per_query_reciprocal_rank) == len(items)
        assert all(0.0 <= rr <= 1.0 for rr in scores.per_query_reciprocal_rank)

    async def test_mode_and_query_count_are_recorded(self, populated_state):
        items = _benchmark_items()
        scores = await evaluate_retrieval(items, populated_state.retrieval, RetrievalMode.bm25_only)
        assert scores.mode == "bm25_only"
        assert scores.queries == len(items)


@pytest.mark.asyncio
class TestAblationSignificance:
    async def test_returns_one_comparison_per_adjacent_pair_of_arms(self, populated_state):
        ablation = await run_ablation(_benchmark_items(), populated_state.retrieval)
        significance = ablation_significance(ablation)
        assert len(significance) == len(ablation) - 1

    async def test_comparison_labels_name_both_arms(self, populated_state):
        ablation = await run_ablation(_benchmark_items(), populated_state.retrieval)
        significance = ablation_significance(ablation)
        assert "bm25_only" in significance[0].metric
        assert "hybrid" in significance[0].metric


def test_ablation_significance_has_no_pairs_for_a_single_arm():
    from app.eval.runner import RetrievalScores

    single = [RetrievalScores(mode="full", queries=2, mrr=1.0, recall_at_1=1.0, recall_at_5=1.0)]
    assert ablation_significance(single) == []
