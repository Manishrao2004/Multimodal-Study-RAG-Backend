"""Constrained access to indexed source files for library and citation viewers."""
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse

MEDIA_TYPES = {
    ".pdf": "application/pdf", ".mp3": "audio/mpeg", ".mpga": "audio/mpeg",
    ".wav": "audio/wav", ".m4a": "audio/mp4", ".flac": "audio/flac",
    ".ogg": "audio/ogg", ".webm": "video/webm", ".mp4": "video/mp4",
    ".txt": "text/plain", ".md": "text/plain", ".html": "text/html", ".htm": "text/html",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def source_path(upload_dir: Path, name: str) -> Path | None:
    root = upload_dir.resolve()
    candidate = (root / name).resolve()
    return candidate if root in candidate.parents and candidate.is_file() else None


def source_response(upload_dir: Path, name: str) -> FileResponse:
    path = source_path(upload_dir, name)
    if path is None:
        raise HTTPException(status_code=404, detail="Original source is no longer available.")
    suffix = path.suffix.lower()
    if suffix not in MEDIA_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported source format.")
    # HTML and Office files are downloads; never execute uploaded HTML in our origin.
    disposition = "attachment" if suffix in {".html", ".htm", ".docx", ".pptx"} else "inline"
    return FileResponse(path, media_type=MEDIA_TYPES[suffix], filename=path.name,
                        content_disposition_type=disposition,
                        headers={"X-Content-Type-Options": "nosniff"})
