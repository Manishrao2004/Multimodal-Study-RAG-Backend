"""Retrieval: fusion maths, index behaviour, and the ablation modes."""

from __future__ import annotations

import pytest

from app.core.retrieval.fusion import reciprocal_rank_fusion
from app.models.schemas import RetrievalMode
from tests.conftest import make_chunk


class TestReciprocalRankFusion:
    def test_document_ranked_well_by_both_retrievers_wins(self):
        bm25 = [("a", 9.0), ("b", 5.0)]
        dense = [("a", 0.9), ("c", 0.7)]
        fused = reciprocal_rank_fusion([bm25, dense], k=60)
        assert fused[0][0] == "a"

    def test_fusion_ignores_raw_scores_and_uses_only_rank(self):
        # 'b' has a huge raw score but a worse rank in both lists than 'a'.
        lists = [[("a", 0.01), ("b", 999.0)], [("a", 0.01), ("b", 999.0)]]
        assert reciprocal_rank_fusion(lists, k=60)[0][0] == "a"

    def test_larger_k_flattens_the_rank_advantage(self):
        lists = [[("a", 1.0), ("b", 1.0)]]
        small = reciprocal_rank_fusion(lists, k=1)
        large = reciprocal_rank_fusion(lists, k=1000)
        gap_small = small[0][1] - small[1][1]
        gap_large = large[0][1] - large[1][1]
        assert gap_large < gap_small

    def test_empty_input_yields_nothing(self):
        assert reciprocal_rank_fusion([], k=60) == []


class TestBM25Index:
    def test_exact_keyword_match_ranks_first(self, populated_state):
        hits = populated_state.retrieval.bm25.search("reciprocal rank fusion", top_k=3)
        assert hits[0][0] == "c1"

    def test_unmatched_query_returns_nothing(self, populated_state):
        assert populated_state.retrieval.bm25.search("xylophone kangaroo", top_k=3) == []

    def test_unbuilt_index_is_not_ready(self, settings, kb):
        from app.core.retrieval.pipeline import RetrievalIndex

        index = RetrievalIndex(kb, settings)
        assert index.is_ready is False
        assert index.bm25.search("anything", top_k=3) == []


class TestFaissIndex:
    def test_semantic_match_without_shared_keywords(self, populated_state):
        # No word overlap with c3's text, but the same concept.
        hits = populated_state.retrieval.faiss.search(
            "joint attention scoring of query and passage together", top_k=3
        )
        assert hits[0][0] == "c3"

    def test_scores_are_cosine_similarities(self, populated_state):
        hits = populated_state.retrieval.faiss.search("bm25 lexical matching", top_k=3)
        assert all(-1.0 <= score <= 1.0 for _, score in hits)


@pytest.mark.asyncio
class TestRetrievalModes:
    async def test_every_ablation_mode_returns_results(self, populated_state):
        for mode in RetrievalMode:
            ranked = await populated_state.retrieval.retrieve(
                "how does reciprocal rank fusion work", top_k=3, mode=mode
            )
            assert ranked, f"{mode.value} returned nothing"

    async def test_dense_finds_what_lexical_matching_cannot(self, populated_state):
        # Paraphrases c3 ("cross-encoder reranker") without reusing any of its
        # content words, so BM25 has nothing to match on. This synonym gap is
        # exactly what adding FAISS to BM25 buys, and what the ablation measures.
        query = "scoring passages alongside the question inside one neural network"
        sparse = await populated_state.retrieval.retrieve(
            query, top_k=3, mode=RetrievalMode.bm25_only
        )
        dense = await populated_state.retrieval.retrieve(
            query, top_k=3, mode=RetrievalMode.dense_only
        )
        assert [c.chunk_id for c, _ in sparse] == []
        assert [c.chunk_id for c, _ in dense][0] == "c3"

    async def test_top_k_is_respected(self, populated_state):
        ranked = await populated_state.retrieval.retrieve("retrieval", top_k=2)
        assert len(ranked) <= 2

    async def test_source_filter_restricts_results(self, populated_state):
        ranked = await populated_state.retrieval.retrieve(
            "reranking and fusion", top_k=5, source_file="slides.pptx"
        )
        assert ranked
        assert {c.source_file for c, _ in ranked} == {"slides.pptx"}

    async def test_empty_index_returns_nothing(self, settings, kb):
        from app.core.retrieval.pipeline import RetrievalIndex

        index = RetrievalIndex(kb, settings)
        index.rebuild()
        assert await index.retrieve("anything") == []


class TestHyDEDisabled:
    @pytest.mark.asyncio
    async def test_disabled_hyde_passes_the_query_through_unchanged(self, settings):
        from app.core.retrieval.hyde import expand_query

        assert await expand_query("what is bm25", settings) == "what is bm25"

    @pytest.mark.asyncio
    async def test_unreachable_hyde_model_falls_back_to_the_raw_query(self, settings):
        # HyDE is an enhancement, not a hard dependency: a dead endpoint must
        # degrade to plain dense retrieval rather than failing the request.
        from app.core.retrieval.hyde import expand_query

        broken = settings.model_copy(
            update={
                "hyde_enabled": True,
                "hyde_provider": "openai_compatible",
                "hyde_base_url": "http://127.0.0.1:1/v1",
            }
        )
        assert await expand_query("what is bm25", broken) == "what is bm25"


class TestKnowledgeBase:
    def test_get_chunks_preserves_requested_order(self, populated_state):
        chunks = populated_state.kb.get_chunks(["c3", "c1"])
        assert [c.chunk_id for c in chunks] == ["c3", "c1"]

    def test_missing_ids_are_skipped_not_errored(self, populated_state):
        chunks = populated_state.kb.get_chunks(["c1", "does-not-exist"])
        assert [c.chunk_id for c in chunks] == ["c1"]

    def test_documents_are_grouped_by_source(self, populated_state):
        docs = {d.source_file: d for d in populated_state.kb.list_documents()}
        assert docs["notes.pdf"].chunk_count == 2
        assert docs["slides.pptx"].chunk_count == 1

    def test_deleting_a_document_removes_only_its_chunks(self, populated_state):
        removed = populated_state.kb.delete_document("notes.pdf")
        assert removed == 2
        assert [c.chunk_id for c in populated_state.kb.all_chunks()] == ["c3"]

    def test_reingesting_the_same_chunk_id_replaces_it(self, populated_state):
        before = populated_state.kb.count()
        populated_state.kb.add_chunks([make_chunk("c1", "replaced text")])
        assert populated_state.kb.count() == before
        assert populated_state.kb.get_chunk("c1").text == "replaced text"
