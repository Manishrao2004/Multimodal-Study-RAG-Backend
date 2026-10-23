"""Vision-language captioning for diagrams/figures -> searchable 'visual chunks'.

EduRAG Tier-1 (Sec. III-A): rather than building a separate visual embedding
index, each surviving figure is described by a VLM and the caption is indexed
as ordinary text. A diagram then becomes retrievable by the exact same
BM25 + FAISS + rerank path as prose, which is why the paper reports near-perfect
visual MRR for so little extra machinery.

Provider-agnostic: set VLM_PROVIDER to "ollama" (llava / llava-phi3) or
"openai_compatible" (any vision chat endpoint — Groq's Qwen-VL, GPT-4o, etc.).
"""

from __future__ import annotations

import asyncio
import base64
import logging

from app.config import Settings
from app.core.generation.llm_client import LLMClient, LLMConfig, build_vision_message
from app.core.ingestion.figures import ExtractedFigure

MAX_CAPTION_CHARS = 800
logger = logging.getLogger(__name__)

_CAPTION_PROMPT = (
    "Describe this educational diagram or figure in one to two dense, factual "
    "sentences suitable for search indexing. Name the labelled components and "
    "the relationships shown between them. Do not start with 'This image shows' "
    "and include the final outputs or outcomes. For charts, preserve axis labels, "
    "units, legend names, and clearly readable values; qualify approximate values "
    "and never invent unreadable text or numbers. "
    "— state the content directly. If the image carries no educational content "
    "(a logo, a decorative rule, a blank area), reply with exactly: SKIP"
)


def _client(settings: Settings) -> LLMClient:
    return LLMClient(
        LLMConfig(
            provider=settings.vlm_provider,
            model=settings.vlm_model,
            base_url=settings.vlm_base_url,
            api_key=settings.vlm_api_key,
        )
    )


async def caption_image(image_bytes: bytes, settings: Settings) -> str | None:
    """Returns a caption, or None if captioning is disabled or the VLM judged
    the image to carry no educational content."""
    if settings.vlm_provider == "none":
        return None

    b64 = base64.b64encode(image_bytes).decode()
    message = build_vision_message(_CAPTION_PROMPT, b64, settings.vlm_provider)
    caption = await _client(settings).chat([message], temperature=0.1)

    caption = caption.strip()
    if not caption or caption.upper().startswith("SKIP"):
        return None
    return caption[:MAX_CAPTION_CHARS]


async def caption_figures(
    figures: list[ExtractedFigure], settings: Settings, concurrency: int = 3,
    warnings: list[str] | None = None,
) -> dict[int, str]:
    """Captions figures concurrently, returning {figure.index: caption}.

    A single failing image must not fail a whole document upload, so per-figure
    failures are counted separately from intentionally skipped decorative images.
    Optional warnings can be surfaced in the upload response without exposing
    provider responses or credentials.
    """
    if settings.vlm_provider == "none" or not figures:
        return {}
    if concurrency < 1:
        raise ValueError("Caption concurrency must be at least 1.")

    semaphore = asyncio.Semaphore(concurrency)
    captions: dict[int, str] = {}
    failed: list[int] = []
    skipped: list[int] = []

    async def run(figure: ExtractedFigure) -> None:
        async with semaphore:
            try:
                caption = await caption_image(figure.png_bytes, settings)
            except Exception as exc:
                failed.append(figure.index)
                logger.warning("Figure %s captioning failed (%s)", figure.index, type(exc).__name__)
                return
            if caption:
                captions[figure.index] = caption
            else:
                skipped.append(figure.index)

    await asyncio.gather(*(run(f) for f in figures))
    if warnings is not None:
        if failed:
            warnings.append(
                f"{len(failed)} figure(s) could not be captioned. Text was preserved; "
                "check the vision provider configuration or availability and retry the upload."
            )
        if skipped:
            warnings.append(f"{len(skipped)} figure(s) produced no educational caption and were skipped.")
    return captions
