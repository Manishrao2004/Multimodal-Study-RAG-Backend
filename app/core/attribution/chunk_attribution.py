"""Chunk-level evidence attribution — Eq. (6)-(7) in the EduRAG paper:

Attr(d_i) = alpha * CrossEncoder(q (+) a_clean, d_i) + beta * Jaccard(a_clean, d_i)

No perturbations (SHAP/LIME) — a single cross-encoder forward pass per chunk
plus a Jaccard lexical-overlap computation. O(k) where k = number of retrieved
chunks, not O(2^M).
"""

from __future__ import annotations

from app.config import Settings
from app.core.retrieval.bm25_index import get_stopwords, tokenize
from app.core.retrieval.reranker import score_pairs
from app.models.schemas import Chunk


def _content_token_set(text: str) -> set[str]:
    stopwords = get_stopwords()
    return {t for t in tokenize(text) if t not in stopwords}


def jaccard_overlap(answer: str, chunk_text: str) -> float:
    a_tokens = _content_token_set(answer)
    d_tokens = _content_token_set(chunk_text)
    if not a_tokens or not d_tokens:
        return 0.0
    intersection = len(a_tokens & d_tokens)
    union = len(a_tokens | d_tokens)
    return intersection / union if union else 0.0


def compute_attribution(
    query: str,
    answer_clean: str,
    chunks: list[Chunk],
    settings: Settings,
) -> dict[str, float]:
    """Returns {chunk_id: attribution_score}."""
    if not chunks:
        return {}

    joint_query = f"{query} {answer_clean}".strip()
    pairs = [(joint_query, c.text) for c in chunks]
    cross_scores = score_pairs(pairs, model_name=settings.reranker_model)

    attribution: dict[str, float] = {}
    for chunk, cross_score in zip(chunks, cross_scores):
        overlap = jaccard_overlap(answer_clean, chunk.text)
        attribution[chunk.chunk_id] = (
            settings.attribution_alpha * cross_score + settings.attribution_beta * overlap
        )
    return attribution


def normalize_to_percentages(attribution: dict[str, float]) -> dict[str, float]:
    """Evidence-contribution percentages (e.g. '26.2% to Page 292' in the
    paper's qualitative example)."""
    total = sum(attribution.values())
    if total <= 0:
        return {cid: 0.0 for cid in attribution}
    return {cid: score / total for cid, score in attribution.items()}
