"""Grounded generation with citation-constrained prompting — Tier 3 /
Algorithm 1 Stage 2 of the EduRAG paper. Answers must be built strictly from
the top-k retrieved chunks and cite them inline as [1], [2], ... in the order
the chunks are listed."""

from __future__ import annotations

from app.config import Settings
from app.core.generation.llm_client import LLMClient, LLMConfig, strip_think_tags
from app.models.schemas import Chunk, ChunkType, Disagreement

__all__ = ["build_client", "format_context", "generate_answer", "strip_think_tags"]

_SYSTEM_PROMPT = (
    "You are a study assistant. Answer the student's question using ONLY the "
    "numbered context passages below.\n\n"
    "CITATION RULES (these are mandatory):\n"
    "- Every sentence that states a fact must end with a citation marker such "
    "as [1], or [2][3] when it draws on more than one passage.\n"
    "- Use only the numbers of the passages given to you. Never cite a number "
    "that is not in the list.\n"
    "- An answer containing no citation markers at all is invalid. Even a "
    "one-sentence answer must carry one.\n"
    "- Cite the passage the claim actually came from, not the nearest number.\n\n"
    "If the passages disagree with each other, say so explicitly and give both "
    "sides rather than silently picking one. If the answer is not contained in "
    "the passages, say you don't have enough information instead of guessing."
)

# Passage kinds are labelled so the model can qualify its wording — a caption is
# a description of a figure, not prose from the book, and a transcript line is
# speech. Without this the model tends to assert caption content as textbook fact.
_TYPE_LABELS = {
    ChunkType.visual: "figure caption",
    ChunkType.audio: "lecture transcript",
    ChunkType.table: "table",
}


def build_client(settings: Settings) -> LLMClient:
    return LLMClient(
        LLMConfig(
            provider=settings.llm_provider,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
    )


def format_context(chunks: list[Chunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        location = chunk.locator()
        label = _TYPE_LABELS.get(chunk.type)
        if label:
            location += f", {label}"
        parts.append(f"[{i}] ({location}) {chunk.text}")
    return "\n\n".join(parts)


def _disagreement_note(disagreements: list[Disagreement], chunks: list[Chunk]) -> str:
    """Tells the model which specific passages conflict so it can present both
    sides in its own prose, instead of the conflict being bolted onto the answer
    as a footer after generation."""
    if not disagreements:
        return ""
    index_by_id = {c.chunk_id: i for i, c in enumerate(chunks, start=1)}
    lines = []
    for d in disagreements:
        a = index_by_id.get(d.chunk_id_a)
        b = index_by_id.get(d.chunk_id_b)
        if a is None or b is None:
            continue
        detail = f" ({d.explanation})" if d.explanation else ""
        lines.append(f"- Passage [{a}] ({d.source_a}) conflicts with [{b}] ({d.source_b}){detail}")
    if not lines:
        return ""
    return (
        "\n\nThese passages have been detected as conflicting:\n"
        + "\n".join(lines)
        + "\n\nPresent both sides explicitly, naming the source documents, and tell "
        "the student the two disagree rather than choosing one silently."
    )


async def generate_answer(
    query: str,
    chunks: list[Chunk],
    settings: Settings,
    disagreements: list[Disagreement] | None = None,
) -> str:
    context = format_context(chunks)
    user_content = f"Context passages:\n\n{context}\n\nQuestion: {query}"
    user_content += _disagreement_note(disagreements or [], chunks)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return await build_client(settings).chat(messages, temperature=0.1)
