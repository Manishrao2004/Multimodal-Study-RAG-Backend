from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.api.deps import get_app_state
from app.config import Settings
from app.core.generation.llm_client import LLMError
from app.core.ingestion.pipeline import AUDIO_SUFFIXES, DOCUMENT_SUFFIXES, ingest_file
from app.core.security import SecurityValidationError, validate_upload
from app.models.schemas import ChunkType, IngestResponse
from app.state import AppState

router = APIRouter(prefix="/ingest", tags=["ingest"])

_DOCUMENT_EXTENSIONS = frozenset(s.lstrip(".") for s in DOCUMENT_SUFFIXES)
_AUDIO_EXTENSIONS = frozenset(s.lstrip(".") for s in AUDIO_SUFFIXES)
_ALL_EXTENSIONS = _DOCUMENT_EXTENSIONS | _AUDIO_EXTENSIONS


@router.post("", response_model=IngestResponse)
async def ingest_document(file: UploadFile, state: AppState = Depends(get_app_state)):
    settings: Settings = state.settings
    raw_bytes = await file.read()

    # Uploaded content is untrusted: validate extension, magic bytes and size
    # *before* anything touches disk or a parser. Audio and documents get
    # different size ceilings since a lecture recording is legitimately much
    # larger than a PDF.
    is_audio_candidate = Path(file.filename or "").suffix.lower().lstrip(".") in _AUDIO_EXTENSIONS
    max_size = settings.security_max_audio_bytes if is_audio_candidate else settings.security_max_document_bytes
    try:
        validated = validate_upload(
            file.filename or "",
            raw_bytes,
            supported_extensions=_ALL_EXTENSIONS,
            max_size_bytes=max_size,
        )
    except SecurityValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    dest_path = settings.upload_dir / validated.safe_filename
    dest_path.write_bytes(raw_bytes)

    try:
        result = await ingest_file(dest_path, settings)
    except LLMError as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not parse document: {exc}") from exc

    if not result.chunks:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="No extractable content found in document.")

    state.kb.add_chunks(result.chunks)
    await state.retrieval.rebuild_async()

    chunks = result.chunks
    return IngestResponse(
        source_file=validated.safe_filename,
        chunks_added=len(chunks),
        text_chunks=sum(1 for c in chunks if c.type == ChunkType.text),
        table_chunks=sum(1 for c in chunks if c.type == ChunkType.table),
        visual_chunks=sum(1 for c in chunks if c.type == ChunkType.visual),
        audio_chunks=sum(1 for c in chunks if c.type == ChunkType.audio),
        figures_detected=result.figures_detected,
        figures_captioned=result.figures_captioned,
        warnings=result.warnings,
    )
