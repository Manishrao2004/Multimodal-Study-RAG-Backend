"""Grounded generation with citation-constrained prompting — Tier 3 /
Algorithm 1 Stage 2 of the EduRAG paper. Answers must be built strictly from
the top-k retrieved chunks and cite them inline as [1], [2], ... in the order
the chunks are listed."""

from __future__ import annotations

import re

from app.config import Settings
from app.core.generation.llm_client import LLMClient, LLMConfig
from app.models.schemas import Chunk

_THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_SYSTEM_PROMPT = (
    "You are a study assistant. Answer the student's question using ONLY the "
    "numbered context passages below. Every factual claim must be followed by "
    "an inline citation marker like [1] or [2] referencing the passage it came "
    "from. If the passages disagree with each other, say so explicitly rather "
    "than silently picking one. If the answer is not contained in the "
    "passages, say you don't have enough information instead of guessing."
)


def format_context(chunks: list[Chunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        location = f"{chunk.source_file}"
        if chunk.page_number is not None:
            location += f", p.{chunk.page_number}"
        parts.append(f"[{i}] ({location}) {chunk.text}")
    return "\n\n".join(parts)


def strip_think_tags(text: str) -> str:
    """Removes <think>...</think> reasoning blocks some local models (e.g.
    DeepSeek-R1) emit before the final answer."""
    return _THINK_TAG_PATTERN.sub("", text).strip()


async def generate_answer(query: str, chunks: list[Chunk], settings: Settings) -> str:
    context = format_context(chunks)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context passages:\n\n{context}\n\nQuestion: {query}",
        },
    ]
    client = LLMClient(
        LLMConfig(
            provider=settings.llm_provider,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
    )
    raw_answer = await client.chat(messages, temperature=0.1)
    return strip_think_tags(raw_answer)
