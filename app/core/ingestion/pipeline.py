"""Document ingestion: Docling parsing -> page/section-aware chunks.

Mirrors EduRAG's Tier-1 (structural parsing with Docling, chunk_id preserving
page + section metadata, text/table/visual chunk types), extended with audio
chunks per the project plan Sec. 5.3.

The unifying idea, taken straight from the paper: every modality is converted
to text at ingestion time — a diagram becomes its VLM caption, a lecture
becomes its transcript — so retrieval, reranking, attribution and grounding all
operate on exactly one kind of object. There is no separate visual or audio
index anywhere downstream.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

from app.config import Settings
from app.core.ingestion.figures import extract_figures
from app.core.ingestion.transcription import (
    AUDIO_SUFFIXES,
    group_segments,
    transcribe,
)
from app.core.ingestion.vlm_caption import caption_figures
from app.models.schemas import Chunk, ChunkType

DOCUMENT_SUFFIXES = {".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".txt"}
SUPPORTED_SUFFIXES = DOCUMENT_SUFFIXES | AUDIO_SUFFIXES


@dataclass
class IngestResult:
    chunks: list[Chunk]
    figures_detected: int = 0
    figures_captioned: int = 0
    warnings: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _get_converter() -> DocumentConverter:
    """Picture images are off by default in Docling and are what the figure
    pipeline needs, so the PDF backend is configured explicitly. The converter
    loads layout models on first use, hence the cache."""
    pdf_options = PdfPipelineOptions()
    pdf_options.generate_picture_images = True
    pdf_options.images_scale = 2.0
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
    )


def _make_chunk_id(source_file: str, index: int, kind: str = "c") -> str:
    digest = hashlib.sha1(f"{source_file}:{kind}:{index}".encode()).hexdigest()[:10]
    return f"{Path(source_file).stem}_{kind}{index:04d}_{digest}"


def parse_document(file_path: Path) -> list[Chunk]:
    """Parse a single document into text/table chunks (no figures, no audio).
    Kept as a standalone function because the evaluation harness and tests
    ingest corpora without needing VLM calls."""
    chunks, _doc = _parse_document_with_doc(file_path)
    return chunks


def _parse_document_with_doc(file_path: Path):
    result = _get_converter().convert(str(file_path))
    doc = result.document

    chunker = HierarchicalChunker()
    chunks: list[Chunk] = []
    for i, chunk in enumerate(chunker.chunk(doc)):
        text = chunk.text.strip()
        if not text:
            continue

        meta = chunk.meta
        page_number = None
        is_table = False

        doc_items = getattr(meta, "doc_items", None) or []
        for item in doc_items:
            prov = getattr(item, "prov", None) or []
            if prov:
                page_number = getattr(prov[0], "page_no", None)
            label = str(getattr(item, "label", "")).lower()
            if "table" in label:
                is_table = True

        headings = getattr(meta, "headings", None)
        section_path = " > ".join(headings) if headings else None

        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(file_path.name, i),
                source_file=file_path.name,
                page_number=page_number,
                section_path=section_path,
                text=text,
                chunk_length=len(text),
                type=ChunkType.table if is_table else ChunkType.text,
                image_ref=None,
            )
        )
    return chunks, doc


async def ingest_file(file_path: Path, settings: Settings) -> IngestResult:
    """Full multimodal ingestion for one uploaded file."""
    if file_path.suffix.lower() in AUDIO_SUFFIXES:
        return await _ingest_audio(file_path, settings)
    return await _ingest_document(file_path, settings)


async def _ingest_document(file_path: Path, settings: Settings) -> IngestResult:
    # Docling parsing is CPU-bound and synchronous; keep the event loop free so
    # concurrent uploads and in-flight /ask requests aren't blocked by it.
    chunks, doc = await asyncio.to_thread(_parse_document_with_doc, file_path)
    result = IngestResult(chunks=chunks)

    try:
        figures = await asyncio.to_thread(extract_figures, doc, settings, file_path.stem)
    except Exception as exc:  # figure extraction must never fail an upload
        result.warnings.append(f"Figure extraction failed: {exc}")
        return result

    result.figures_detected = len(figures)
    if not figures:
        return result

    if settings.vlm_provider == "none":
        result.warnings.append(
            f"{len(figures)} figure(s) detected but VLM_PROVIDER is 'none', so no "
            "visual chunks were created. Set VLM_PROVIDER to caption them."
        )
        return result

    captions = await caption_figures(figures, settings)
    result.figures_captioned = len(captions)
    if len(captions) < len(figures):
        result.warnings.append(
            f"{len(figures) - len(captions)} figure(s) were skipped or failed captioning."
        )

    for figure in figures:
        caption = captions.get(figure.index)
        if not caption:
            continue
        result.chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(file_path.name, figure.index, kind="v"),
                source_file=file_path.name,
                page_number=figure.page_number,
                section_path="Figure",
                text=caption,
                chunk_length=len(caption),
                type=ChunkType.visual,
                image_ref=str(figure.image_path.name) if figure.image_path else None,
            )
        )
    return result


async def _ingest_audio(file_path: Path, settings: Settings) -> IngestResult:
    segments = await transcribe(file_path, settings)
    if not segments:
        return IngestResult(chunks=[], warnings=["Transcription produced no text."])

    grouped = group_segments(segments, settings.asr_segment_target_chars)
    chunks = [
        Chunk(
            chunk_id=_make_chunk_id(file_path.name, i, kind="a"),
            source_file=file_path.name,
            page_number=None,
            section_path="Lecture transcript",
            text=segment.text,
            chunk_length=len(segment.text),
            type=ChunkType.audio,
            timestamp_start=segment.start,
            timestamp_end=segment.end,
        )
        for i, segment in enumerate(grouped)
    ]
    return IngestResult(chunks=chunks)
