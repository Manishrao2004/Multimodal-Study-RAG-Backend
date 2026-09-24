"""Hybrid retrieval orchestrator — Algorithm 1 (Stage 1) from the EduRAG paper:
HyDE-expanded FAISS + original-query BM25 -> RRF fusion -> cross-encoder rerank
-> top-k final chunks.

`RetrievalMode` exposes the paper's ablation arms (Table 3) on the live index so
the evaluation harness can measure BM25-only vs. hybrid vs. full system without
rebuilding anything between runs.
"""

from __future__ import annotations

import asyncio

from app.config import Settings
from app.core.retrieval.bm25_index import BM25Index
from app.core.retrieval.faiss_index import FaissIndex
from app.core.retrieval.fusion import reciprocal_rank_fusion
from app.core.retrieval.hyde import expand_query
from app.core.retrieval.reranker import rerank
from app.db.knowledge_base import KnowledgeBase
from app.models.schemas import Chunk, RetrievalMode


class RetrievalIndex:
    """Holds the in-memory BM25 + FAISS indexes for the current knowledge base
    and knows how to rebuild them after ingestion."""

    def __init__(self, kb: KnowledgeBase, settings: Settings):
        self.kb = kb
        self.settings = settings
        self.bm25 = BM25Index()
        self.faiss = FaissIndex(model_name=settings.embedding_model)
        self._lock = asyncio.Lock()

    def rebuild(self) -> None:
        chunks = self.kb.all_chunks()
        self.bm25.build(chunks)
        self.faiss.build(chunks)

    async def rebuild_async(self) -> None:
        """Re-embedding the whole corpus is heavy; run it off the event loop and
        serialise it so two concurrent uploads can't interleave index writes."""
        async with self._lock:
            await asyncio.to_thread(self.rebuild)

    @property
    def is_ready(self) -> bool:
        return self.bm25.is_built and self.faiss.is_built

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: RetrievalMode = RetrievalMode.full,
        source_file: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Returns top-k (chunk, score) sorted best-first.

        The score is the cross-encoder relevance in `full` mode (which chunk
        attribution reuses directly) and the fused/lexical/dense score otherwise.
        """
        if not self.is_ready:
            return []

        settings = self.settings
        limit = top_k or settings.rerank_top_k

        # Only the full pipeline pays for HyDE — the ablation arms must not get
        # its benefit, or the comparison measures nothing.
        dense_query = query
        if mode == RetrievalMode.full:
            dense_query = await expand_query(query, settings)

        candidate_scores: list[tuple[str, float]]
        if mode == RetrievalMode.bm25_only:
            candidate_scores = self.bm25.search(query, top_k=settings.bm25_top_k)
        elif mode == RetrievalMode.dense_only:
            candidate_scores = self.faiss.search(dense_query, top_k=settings.faiss_top_k)
        else:
            bm25_hits = self.bm25.search(query, top_k=settings.bm25_top_k)
            faiss_hits = self.faiss.search(dense_query, top_k=settings.faiss_top_k)
            candidate_scores = reciprocal_rank_fusion([bm25_hits, faiss_hits], k=settings.rrf_k)

        if not candidate_scores:
            return []

        pool = max(settings.bm25_top_k, settings.faiss_top_k)
        candidates = self.kb.get_chunks([cid for cid, _ in candidate_scores[:pool]])
        if source_file:
            candidates = [c for c in candidates if c.source_file == source_file]
        if not candidates:
            return []

        chunk_by_id = {c.chunk_id: c for c in candidates}

        # Without reranking, the retriever's own ordering is the final ordering.
        if mode != RetrievalMode.full:
            ordered = [
                (chunk_by_id[cid], float(score))
                for cid, score in candidate_scores
                if cid in chunk_by_id
            ]
            return ordered[:limit]

        pairs = [(c.chunk_id, c.text) for c in candidates]
        reranked = await asyncio.to_thread(
            rerank, query, pairs, settings.reranker_model, limit
        )
        return [(chunk_by_id[cid], score) for cid, score in reranked if cid in chunk_by_id]
