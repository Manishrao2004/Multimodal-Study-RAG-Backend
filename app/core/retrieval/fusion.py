"""Reciprocal Rank Fusion — Eq. (4) in the EduRAG paper.

score_RRF(d) = sum over retrievers r of 1 / (k + rank_r(d))
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]], k: int = 60
) -> list[tuple[str, float]]:
    """Each input is a list of (chunk_id, score) already sorted best-first.
    Returns a fused, sorted list of (chunk_id, fused_score)."""
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(ranked):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
