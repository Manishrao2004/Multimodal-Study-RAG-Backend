"""Cross-encoder reranking — Eq. (5) in the EduRAG paper: full query-document
self-attention for precision reranking of the top RRF candidates."""

from __future__ import annotations

from threading import Lock

import torch
from sentence_transformers import CrossEncoder

_model_cache: dict[str, CrossEncoder] = {}
_lock = Lock()


def get_cross_encoder(model_name: str) -> CrossEncoder:
    with _lock:
        if model_name not in _model_cache:
            # Sigmoid activation so raw logits become the sigma(CrossEncoder(q,d))
            # relevance score from Eq. (5), reused as-is for chunk attribution.
            _model_cache[model_name] = CrossEncoder(
                model_name, activation_fn=torch.nn.Sigmoid()
            )
        return _model_cache[model_name]


def rerank(
    query: str, candidates: list[tuple[str, str]], model_name: str, top_k: int
) -> list[tuple[str, float]]:
    """candidates: list of (chunk_id, chunk_text). Returns top_k (chunk_id, score)
    sorted best-first, score already sigmoid-normalized by the cross-encoder."""
    if not candidates:
        return []
    model = get_cross_encoder(model_name)
    pairs = [(query, text) for _, text in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip([cid for cid, _ in candidates], scores), key=lambda x: x[1], reverse=True)
    return [(cid, float(score)) for cid, score in ranked[:top_k]]


def score_pairs(pairs: list[tuple[str, str]], model_name: str) -> list[float]:
    """Raw sigmoid-normalized scores for arbitrary (query_or_text, text) pairs —
    used by the attribution module to re-score (query+answer, chunk)."""
    if not pairs:
        return []
    model = get_cross_encoder(model_name)
    return [float(s) for s in model.predict(pairs)]
