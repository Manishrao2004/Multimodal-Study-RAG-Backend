"""Source consensus / cross-document contradiction detection.

Base mechanism (EduRAG Sec. III-C4): pairwise cosine similarity between
retrieved chunk embeddings flags source disagreement.

Extended per the project's novel-contribution plan (Sec. 5.1): when two
top-ranked chunks *from different source documents* score below the
similarity threshold, surface both sides explicitly instead of silently
picking one ("Your textbook says X, your lecture slides say Y") — the
generation layer can use `flagged_pairs` to prompt the LLM to acknowledge
both perspectives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.retrieval.faiss_index import get_embedding_model
from app.models.schemas import Chunk

DEFAULT_DISAGREEMENT_THRESHOLD = 0.55


@dataclass
class ConsensusPair:
    chunk_id_a: str
    chunk_id_b: str
    source_a: str
    source_b: str
    similarity: float


def detect_disagreements(
    chunks: list[Chunk],
    embedding_model_name: str,
    threshold: float = DEFAULT_DISAGREEMENT_THRESHOLD,
) -> list[ConsensusPair]:
    """Only compares chunks from *different* source_file — same-document
    chunks are expected to be similar/complementary, not conflicting."""
    if len(chunks) < 2:
        return []

    model = get_embedding_model(embedding_model_name)
    embeddings = model.encode(
        [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    flagged: list[ConsensusPair] = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            if chunks[i].source_file == chunks[j].source_file:
                continue
            similarity = float(np.dot(embeddings[i], embeddings[j]))
            if similarity < threshold:
                flagged.append(
                    ConsensusPair(
                        chunk_id_a=chunks[i].chunk_id,
                        chunk_id_b=chunks[j].chunk_id,
                        source_a=chunks[i].source_file,
                        source_b=chunks[j].source_file,
                        similarity=similarity,
                    )
                )
    return flagged
