"""Hybrid retrieval orchestrator — Algorithm 1 (Stage 1) from the EduRAG paper:
HyDE-expanded FAISS + original-query BM25 -> RRF fusion -> cross-encoder rerank
-> top-k final chunks.
"""

from __future__ import annotations

from app.config import Settings
from app.core.retrieval.bm25_index import BM25Index
from app.core.retrieval.faiss_index import FaissIndex
from app.core.retrieval.fusion import reciprocal_rank_fusion
from app.core.retrieval.hyde import expand_query
from app.core.retrieval.reranker import rerank
from app.db.knowledge_base import KnowledgeBase
from app.models.schemas import Chunk


class RetrievalIndex:
    """Holds the in-memory BM25 + FAISS indexes for the current knowledge base
    and knows how to rebuild them after ingestion."""

    def __init__(self, kb: KnowledgeBase, settings: Settings):
        self.kb = kb
        self.settings = settings
        self.bm25 = BM25Index()
        self.faiss = FaissIndex(model_name=settings.embedding_model)

    def rebuild(self) -> None:
        chunks = self.kb.all_chunks()
        self.bm25.build(chunks)
        self.faiss.build(chunks)

    @property
    def is_ready(self) -> bool:
        return self.bm25.is_built and self.faiss.is_built

    async def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        """Returns top-k (chunk, rerank_score) sorted best-first."""
        if not self.is_ready:
            return []

        settings = self.settings
        dense_query = await expand_query(query, settings)

        bm25_hits = self.bm25.search(query, top_k=settings.bm25_top_k)
        faiss_hits = self.faiss.search(dense_query, top_k=settings.faiss_top_k)

        fused = reciprocal_rank_fusion([bm25_hits, faiss_hits], k=settings.rrf_k)
        candidate_ids = [cid for cid, _ in fused[: max(settings.bm25_top_k, settings.faiss_top_k)]]
        candidates = self.kb.get_chunks(candidate_ids)
        if not candidates:
            return []

        pairs = [(c.chunk_id, c.text) for c in candidates]
        reranked = rerank(
            query,
            pairs,
            model_name=settings.reranker_model,
            top_k=top_k or settings.rerank_top_k,
        )

        chunk_by_id = {c.chunk_id: c for c in candidates}
        return [(chunk_by_id[cid], score) for cid, score in reranked if cid in chunk_by_id]
