"""HyDE query expansion: generate a hypothetical answer document for the
*dense* branch only, while BM25 keeps the original query — matches Eq. (1)-(3)
in the EduRAG paper. Uses whatever LLM provider is configured for HyDE
(typically a small/fast local model, independent of the main answer model)."""

from __future__ import annotations

from app.config import Settings
from app.core.generation.llm_client import LLMClient, LLMConfig

_HYDE_SYSTEM_PROMPT = (
    "You are helping expand a student's question into a short hypothetical "
    "textbook passage that would answer it. Write 2-4 sentences of plausible "
    "factual prose as if it were an excerpt from a course textbook. Do not "
    "mention that this is hypothetical."
)


async def expand_query(query: str, settings: Settings) -> str:
    if not settings.hyde_enabled or settings.hyde_provider == "none":
        return query

    client = LLMClient(
        LLMConfig(
            provider=settings.hyde_provider,
            model=settings.hyde_model,
            base_url=settings.hyde_base_url,
            api_key=settings.hyde_api_key,
        )
    )
    messages = [
        {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    try:
        hypothetical_doc = await client.chat(messages, temperature=0.1)
    except Exception:
        # HyDE is an enhancement, not a hard dependency — fall back to the
        # raw query if the HyDE model is unavailable.
        return query
    return f"{query}\n{hypothetical_doc.strip()}"
