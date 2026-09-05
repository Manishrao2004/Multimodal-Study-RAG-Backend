"""Vision-language captioning for diagrams/figures -> searchable 'visual chunks'.

Phase-C feature (see project plan). Stubbed out behind the same provider
abstraction as the text LLM so it's not tied to any one VLM: set VLM_PROVIDER
to "ollama" (e.g. a llava/llava-phi3 tag) or "openai_compatible" (any vision
chat endpoint) once image extraction from Docling is wired up here.
"""

from __future__ import annotations

import base64

from app.config import Settings
from app.core.generation.llm_client import LLMClient, LLMConfig

MAX_CAPTION_CHARS = 800


async def caption_image(image_bytes: bytes, settings: Settings) -> str | None:
    if settings.vlm_provider == "none":
        return None

    client = LLMClient(
        LLMConfig(
            provider=settings.vlm_provider,
            model=settings.vlm_model,
            base_url=settings.vlm_base_url,
            api_key=settings.vlm_api_key,
        )
    )
    b64 = base64.b64encode(image_bytes).decode()
    messages = [
        {
            "role": "user",
            "content": "Describe this educational diagram/figure in one to two "
            "dense, factual sentences suitable for search indexing. Mention "
            "labelled components and relationships shown.",
            "images": [b64],
        }
    ]
    caption = await client.chat(messages, temperature=0.1)
    return caption.strip()[:MAX_CAPTION_CHARS]
