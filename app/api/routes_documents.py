"""Knowledge-base inspection and management.

The demo needs to show what has been ingested and let a document be removed
without wiping the whole database — deleting a source drops its chunks and
rebuilds the indexes, since BM25 and FAISS are both built from the KB in one
pass and hold no per-document deletion path of their own.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from concurrent.futures import TimeoutError as RenderTimeout
from app.core.pdf_preview import renderer, render_page
from app.core.pdf_preview import cached_page_count
from app.core.document_text import full_document_text
from app.core.office_preview import ensure_pdf, OfficePreviewError
from subprocess import TimeoutExpired
import asyncio

from app.api.deps import get_app_state
from app.models.schemas import DocumentSummary, KnowledgeBaseStats
from app.state import AppState
from app.core.source_files import source_path, source_response
from app.models.schemas import Chunk

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentSummary])
async def list_documents(state: AppState = Depends(get_app_state)):
    documents = state.kb.list_documents()
    for document in documents:
        path = source_path(state.settings.upload_dir, document.source_file)
        if path is not None and path.suffix.lower() == ".pdf":
            try:
                stat = path.stat()
                document.pages = await asyncio.to_thread(cached_page_count, str(path), stat.st_mtime_ns, stat.st_size)
            except Exception:
                pass  # Preserve indexed page information when the original cannot be read.
    return documents


@router.get("/stats", response_model=KnowledgeBaseStats)
async def stats(state: AppState = Depends(get_app_state)):
    return KnowledgeBaseStats(
        documents=len(state.kb.list_documents()),
        chunks=state.kb.count(),
        by_type=state.kb.count_by_type(),
        index_ready=state.retrieval.is_ready,
    )


@router.delete("/{source_file}")
async def delete_document(source_file: str, state: AppState = Depends(get_app_state)):
    removed = state.kb.delete_document(source_file)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No document named '{source_file}'.")
    await state.retrieval.rebuild_async()
    return {"source_file": source_file, "chunks_removed": removed}


@router.get("/{source_file}/file")
async def open_source(source_file: str, state: AppState = Depends(get_app_state)):
    if not state.kb.chunks_for_source(source_file):
        raise HTTPException(status_code=404, detail="Source not found in the library.")
    return source_response(state.settings.upload_dir, source_file)


@router.get("/{source_file}/preview")
async def preview_source(source_file: str, state: AppState = Depends(get_app_state)):
    chunks: list[Chunk] = state.kb.chunks_for_source(source_file)
    if not chunks:
        raise HTTPException(status_code=404, detail="Source not found in the library.")
    chunks.sort(key=lambda c: (c.page_number or 0, c.timestamp_start or 0, c.chunk_id))
    path = source_path(state.settings.upload_dir, source_file)
    full_sections = []
    warning = None
    if path is not None:
        try:
            full_sections = await asyncio.to_thread(full_document_text, path)
        except Exception:
            warning = "Full document text could not be read. Indexed passages are still available."
    return {"source_file": source_file,
            "original_available": path is not None,
            "full_sections": full_sections, "preview_warning": warning,
            "chunks": chunks}


@router.get("/{source_file}/pages/{page_number}")
def preview_pdf_page(source_file: str, page_number: int, state: AppState = Depends(get_app_state)):
    if not state.kb.chunks_for_source(source_file):
        raise HTTPException(status_code=404, detail="Source not found in the library.")
    path = source_path(state.settings.upload_dir, source_file)
    if path is None:
        raise HTTPException(status_code=404, detail="Original source is no longer available.")
    if path.suffix.lower() not in {".pdf", ".docx", ".pptx"}:
        raise HTTPException(status_code=415, detail="Page images are available for PDF, Word and PowerPoint sources.")
    try:
        path = ensure_pdf(path, state.settings.upload_dir.parent / "previews")
        image, count = renderer().submit(render_page, str(path), page_number).result(timeout=30)
    except OfficePreviewError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except IndexError as exc:
        raise HTTPException(status_code=404, detail="Page not found.") from exc
    except (RenderTimeout, TimeoutExpired) as exc:
        raise HTTPException(status_code=504, detail="Page preview timed out. Try again.") from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail="This document could not be displayed. Open the extracted text instead.") from exc
    return Response(image, media_type="image/png", headers={"X-Page-Count": str(count), "Cache-Control": "private, max-age=60"})
