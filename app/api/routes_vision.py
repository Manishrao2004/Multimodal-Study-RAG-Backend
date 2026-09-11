"""Interactive screenshot questions.

Pasted images are processed in memory and discarded by default. When the user
explicitly chooses to save one, its normalized PNG and question-aware visual
chunk are added to the same knowledge base used by document figures.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_app_state
from app.core.attribution.chunk_attribution import compute_attribution, normalize_to_percentages
from app.core.attribution.citation_validation import (
    citation_validity_rate,
    extract_and_validate_citations,
)
from app.core.attribution.token_grounding import compute_token_grounding, grounding_ratio
from app.core.generation.answer import generate_answer
from app.core.generation.llm_client import LLMError
from app.core.security import SecurityValidationError, validate_query_text, validate_upload
from app.core.vision.screenshot import (
    SCREENSHOT_EXTENSIONS,
    analyze_screenshot,
    normalize_screenshot,
)
from app.models.schemas import (
    Chunk,
    ChunkType,
    EvidenceItem,
    RetrievalMode,
    ScreenshotAskResponse,
)
from app.state import AppState

router = APIRouter(prefix="/vision", tags=["vision"])


@router.post("/ask", response_model=ScreenshotAskResponse)
async def ask_about_screenshot(
    question: str = Form(...),
    image: UploadFile = File(...),
    use_knowledge_base: bool = Form(True),
    save_to_library: bool = Form(False),
    top_k: int = Form(5, ge=1, le=10),
    state: AppState = Depends(get_app_state),
):
    settings = state.settings
    if settings.vlm_provider == "none":
        raise HTTPException(
            status_code=503,
            detail="Screenshot questions are disabled. Configure VLM_PROVIDER first.",
        )

    try:
        clean_question = validate_query_text(
            question,
            max_length=settings.security_max_query_length,
            field_name="question",
        )
        raw_bytes = await image.read()
        validated = validate_upload(
            image.filename or "",
            raw_bytes,
            supported_extensions=SCREENSHOT_EXTENSIONS,
            max_size_bytes=settings.security_max_screenshot_bytes,
        )
        screenshot = normalize_screenshot(
            raw_bytes,
            max_pixels=settings.security_max_screenshot_pixels,
            max_dimension=settings.screenshot_max_dimension_px,
        )
    except SecurityValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        image_analysis = await analyze_screenshot(screenshot, clean_question, settings)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    token = uuid.uuid4().hex[:12]
    saved_name = f"screenshot_{token}.png"
    screenshot_chunk = Chunk(
        chunk_id=f"screenshot_v_{token}",
        source_file=saved_name if save_to_library else validated.safe_filename,
        section_path="Pasted screenshot",
        text=image_analysis,
        chunk_length=len(image_analysis),
        type=ChunkType.visual,
        image_ref=saved_name if save_to_library else None,
    )

    ranked_kb: list[tuple[Chunk, float]] = []
    if use_knowledge_base and state.retrieval.is_ready:
        retrieval_query = f"{clean_question}\n{image_analysis}"
        ranked_kb = await state.retrieval.retrieve(
            retrieval_query,
            top_k=top_k,
            mode=RetrievalMode.full,
        )

    ranked: list[tuple[Chunk, float]] = [(screenshot_chunk, 1.0), *ranked_kb]
    chunks = [chunk for chunk, _ in ranked]
    try:
        answer = await generate_answer(clean_question, chunks, settings)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    attribution = compute_attribution(clean_question, answer, chunks, settings)
    percentages = normalize_to_percentages(attribution)
    citations = extract_and_validate_citations(answer, chunks)
    token_scores = compute_token_grounding(answer, "\n".join(c.text for c in chunks))

    if save_to_library:
        settings.image_dir.mkdir(parents=True, exist_ok=True)
        (settings.image_dir / saved_name).write_bytes(screenshot.png_bytes)
        state.kb.add_chunks([screenshot_chunk])
        await state.retrieval.rebuild_async()

    evidence = [
        EvidenceItem(
            chunk=chunk,
            rerank_score=score,
            attribution_score=attribution.get(chunk.chunk_id),
            attribution_percent=percentages.get(chunk.chunk_id),
        )
        for chunk, score in ranked
    ]

    return ScreenshotAskResponse(
        answer=answer,
        image_analysis=image_analysis,
        image_width=screenshot.width,
        image_height=screenshot.height,
        evidence=evidence,
        citations=citations,
        grounding_ratio=grounding_ratio(token_scores, settings.grounding_threshold),
        citation_validity_rate=citation_validity_rate(citations),
        used_knowledge_base=bool(ranked_kb),
        saved=save_to_library,
        saved_chunk_id=screenshot_chunk.chunk_id if save_to_library else None,
    )
