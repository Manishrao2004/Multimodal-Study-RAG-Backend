"""Citation validation — Eq. (9) in the EduRAG paper. Citation markers like
[1], [2] in the generated answer must reference an index within the retrieved
top-k set; anything else is a hallucinated reference and is flagged invalid."""

from __future__ import annotations

import re

from app.models.schemas import Chunk, Citation

_CITATION_PATTERN = re.compile(r"\[(\d+)]")


def extract_and_validate_citations(answer_text: str, ranked_chunks: list[Chunk]) -> list[Citation]:
    """`ranked_chunks` is the final top-k list in the order presented to the
    LLM, so citation marker N corresponds to ranked_chunks[N - 1]."""
    n = len(ranked_chunks)
    citations: list[Citation] = []
    seen: set[int] = set()
    for match in _CITATION_PATTERN.finditer(answer_text):
        marker = int(match.group(1))
        if marker in seen:
            continue
        seen.add(marker)
        valid = 1 <= marker <= n
        chunk_id = ranked_chunks[marker - 1].chunk_id if valid else ""
        citations.append(Citation(marker=marker, chunk_id=chunk_id, valid=valid))
    return citations


def citation_validity_rate(citations: list[Citation]) -> float:
    if not citations:
        return 0.0
    return sum(1 for c in citations if c.valid) / len(citations)
