"""Source consensus / cross-document contradiction detection.

Base mechanism (EduRAG Sec. III-C4): pairwise cosine similarity between
retrieved chunk embeddings flags source disagreement.

Extended per the project's novel-contribution plan (Sec. 5.1). Two changes over
the paper's version:

1. **Banded similarity, not low similarity.** A *low* cosine score means the two
   passages are about different things, not that they disagree — so the paper's
   bare threshold flags mostly unrelated noise. Measured on
   bge-small-en-v1.5, unrelated pairs score 0.47-0.52 while genuine
   contradictions score 0.82-0.89: contradictions are among the *most* similar
   pairs, not the least. Candidates are therefore taken from a band.

2. **LLM verification, because similarity cannot decide this.** The same
   measurement puts paraphrases at 0.91-1.00, overlapping the contradiction
   range — no threshold on cosine similarity can separate "these agree" from
   "these conflict", since both are statements about the same subject in
   similar language. The band is only a cheap prefilter that discards unrelated
   pairs and verbatim duplicates; an entailment-style check over the few
   surviving pairs makes the actual judgement and produces the "your textbook
   says X, your slides say Y" explanation the plan calls for.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

import numpy as np

from app.config import Settings
from app.core.generation.llm_client import LLMClient, LLMConfig
from app.core.retrieval.faiss_index import get_embedding_model
from app.models.schemas import Chunk, Disagreement

MAX_CLAIM_CHARS = 400

_VERIFY_PROMPT = (
    "You are checking two passages from a student's study material for factual "
    "conflict. They come from different documents.\n\n"
    "Answer with a single JSON object and nothing else:\n"
    '{"conflict": true|false, "explanation": "<one sentence>"}\n\n'
    "Set conflict to true ONLY if the passages make incompatible claims about "
    "the same thing (different values, definitions, counts, orderings, or "
    "contradictory statements). Set it to false if they merely cover different "
    "topics, or if one is more detailed than the other while agreeing."
)


@dataclass
class ConsensusPair:
    chunk_id_a: str
    chunk_id_b: str
    source_a: str
    source_b: str
    similarity: float


def _excerpt(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= MAX_CLAIM_CHARS:
        return collapsed
    return collapsed[:MAX_CLAIM_CHARS].rsplit(" ", 1)[0] + "…"


def detect_disagreements(
    chunks: list[Chunk],
    embedding_model_name: str,
    related_min: float,
    agreement_max: float,
    embeddings: np.ndarray | None = None,
) -> list[ConsensusPair]:
    """Candidate conflicts: cross-document chunk pairs whose similarity falls in
    the "same topic, different content" band. Sorted most-similar-first, since
    the closer two passages are while still differing, the more likely the
    difference is a real contradiction rather than a topic shift.

    `embeddings` may be passed in to reuse vectors already computed elsewhere.
    """
    if len(chunks) < 2:
        return []

    if embeddings is None:
        model = get_embedding_model(embedding_model_name)
        embeddings = model.encode(
            [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False
        )
    embeddings = np.asarray(embeddings, dtype="float32")

    flagged: list[ConsensusPair] = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            if chunks[i].source_file == chunks[j].source_file:
                continue  # same document: expected to be complementary, not conflicting
            similarity = float(np.dot(embeddings[i], embeddings[j]))
            if related_min <= similarity <= agreement_max:
                flagged.append(
                    ConsensusPair(
                        chunk_id_a=chunks[i].chunk_id,
                        chunk_id_b=chunks[j].chunk_id,
                        source_a=chunks[i].source_file,
                        source_b=chunks[j].source_file,
                        similarity=similarity,
                    )
                )
    flagged.sort(key=lambda p: p.similarity, reverse=True)
    return flagged


async def _verify_pair(
    chunk_a: Chunk, chunk_b: Chunk, settings: Settings
) -> tuple[bool, str | None]:
    client = LLMClient(
        LLMConfig(
            provider=settings.llm_provider,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
    )
    messages = [
        {"role": "system", "content": _VERIFY_PROMPT},
        {
            "role": "user",
            "content": (
                f"Passage A (from {chunk_a.source_file}):\n{_excerpt(chunk_a.text)}\n\n"
                f"Passage B (from {chunk_b.source_file}):\n{_excerpt(chunk_b.text)}"
            ),
        },
    ]
    try:
        raw = await client.chat(messages, temperature=0.0)
    except Exception:
        return False, None

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return False, None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False, None
    return bool(payload.get("conflict")), (payload.get("explanation") or None)


async def find_disagreements(
    chunks: list[Chunk],
    settings: Settings,
    verify: bool = True,
    embeddings: np.ndarray | None = None,
) -> list[Disagreement]:
    """End-to-end contradiction detection: band prefilter, then (optionally) LLM
    verification of the strongest candidates. Only verified conflicts are
    returned when verification runs, so the UI never shows a false alarm from
    the cheap signal alone."""
    candidates = detect_disagreements(
        chunks,
        settings.embedding_model,
        related_min=settings.consensus_related_min,
        agreement_max=settings.consensus_agreement_max,
        embeddings=embeddings,
    )
    if not candidates:
        return []

    by_id = {c.chunk_id: c for c in chunks}
    budget = candidates[: settings.consensus_max_pairs_verified]

    if not verify or settings.llm_provider == "none":
        return [
            Disagreement(
                chunk_id_a=p.chunk_id_a,
                chunk_id_b=p.chunk_id_b,
                source_a=p.source_a,
                source_b=p.source_b,
                similarity=p.similarity,
                claim_a=_excerpt(by_id[p.chunk_id_a].text),
                claim_b=_excerpt(by_id[p.chunk_id_b].text),
                verified=False,
            )
            for p in budget
        ]

    verdicts = await asyncio.gather(
        *(_verify_pair(by_id[p.chunk_id_a], by_id[p.chunk_id_b], settings) for p in budget)
    )

    return [
        Disagreement(
            chunk_id_a=pair.chunk_id_a,
            chunk_id_b=pair.chunk_id_b,
            source_a=pair.source_a,
            source_b=pair.source_b,
            similarity=pair.similarity,
            claim_a=_excerpt(by_id[pair.chunk_id_a].text),
            claim_b=_excerpt(by_id[pair.chunk_id_b].text),
            verified=True,
            explanation=explanation,
        )
        for pair, (is_conflict, explanation) in zip(budget, verdicts)
        if is_conflict
    ]
