"""Security-boundary helpers: upload validation, query sanitization, and
defenses against retrieved-content prompt injection.

None of this claims to make the system secure against a determined attacker —
there is no authentication and no multi-tenant isolation, and none is in
scope for a local-first, single-user study assistant. What belongs here is
narrower: uploaded files and retrieved document text are both untrusted input
(the student didn't necessarily author every PDF they upload, and a chunk's
text is whatever was in that PDF), so they get the same treatment any
untrusted input would — validate shape/size before touching it, never let it
be mistaken for an instruction, and never leak internals when something goes
wrong.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_SAFE_FILENAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


class SecurityValidationError(ValueError):
    """A user-facing validation failure. The message is safe to return as-is
    to a client — never wrap a raw exception or stack trace in this."""


def sanitize_filename(filename: str) -> str:
    """Reduce an untrusted filename to a safe basename: no directory
    components (blocks path traversal) and no characters outside a small
    allowlist (blocks null bytes, shell metacharacters, and anything a
    filesystem might interpret unexpectedly)."""
    from pathlib import Path

    name = Path(filename or "").name
    cleaned = "".join(ch if ch in _SAFE_FILENAME_CHARS else "_" for ch in name)
    cleaned = cleaned.strip("._")
    return cleaned or "upload"


# Magic-byte signatures for formats that have one. DOCX/PPTX are both ZIP
# containers and share the ZIP local-file-header signature. HTML/MD/TXT have
# no reliable binary signature — left out of this map deliberately, and
# `_looks_like_declared_type` treats "no known signature" as "cannot check,
# don't false-reject" rather than guessing at one.
_DOCUMENT_MAGIC_BYTES: dict[str, tuple[bytes, ...]] = {
    "pdf": (b"%PDF-",),
    "docx": (b"PK\x03\x04",),
    "pptx": (b"PK\x03\x04",),
    "png": (b"\x89PNG\r\n\x1a\n",),
    "jpg": (b"\xff\xd8\xff",),
    "jpeg": (b"\xff\xd8\xff",),
}

# WAV/RIFF, MP3 (ID3 tag or a raw MPEG frame sync), OGG, FLAC, and M4A/MP4
# (ftyp box at offset 4). A cheap, dependency-free sanity check that the bytes
# at least resemble the claimed format before they reach a transcription API.
_AUDIO_MAGIC_BYTES: dict[str, tuple[bytes, ...]] = {
    "wav": (b"RIFF",),
    "mp3": (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"),
    "ogg": (b"OggS",),
    "flac": (b"fLaC",),
    "webm": (b"\x1a\x45\xdf\xa3",),
}
_M4A_LIKE_EXTENSIONS = frozenset({"m4a", "mp4", "mpga"})


def _looks_like_declared_type(data: bytes, extension: str) -> bool:
    if extension in _M4A_LIKE_EXTENSIONS:
        return data[4:8] == b"ftyp"
    if extension == "webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    signatures = _DOCUMENT_MAGIC_BYTES.get(extension) or _AUDIO_MAGIC_BYTES.get(extension)
    if signatures is None:
        return True
    return any(data.startswith(sig) for sig in signatures)


@dataclass(frozen=True)
class ValidatedUpload:
    safe_filename: str
    extension: str
    size_bytes: int


def validate_upload(
    filename: str,
    data: bytes,
    *,
    supported_extensions: frozenset[str],
    max_size_bytes: int,
) -> ValidatedUpload:
    """Validate an in-memory upload's shape before anything parses it.
    Raises `SecurityValidationError` with a message safe to show the client."""
    if not filename or not filename.strip():
        raise SecurityValidationError("A filename is required.")

    safe_name = sanitize_filename(filename)
    extension = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""

    if extension not in supported_extensions:
        supported = ", ".join(sorted(supported_extensions))
        raise SecurityValidationError(
            f"Unsupported file type '.{extension}'. Supported: {supported}."
        )

    if not data:
        raise SecurityValidationError("The uploaded file is empty.")

    if len(data) > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        raise SecurityValidationError(f"The uploaded file exceeds the {max_mb:.0f}MB limit.")

    if not _looks_like_declared_type(data, extension):
        raise SecurityValidationError(
            f"The uploaded file does not look like a valid .{extension} file."
        )

    return ValidatedUpload(safe_filename=safe_name, extension=extension, size_bytes=len(data))


# --- Query text validation -------------------------------------------------

_ALLOWED_CONTROL_CHARS = frozenset({"\t", "\n", "\r"})


def contains_disallowed_control_chars(text: str) -> bool:
    """A legitimate question never needs raw control characters; allowing
    them through risks corrupting logs or terminals that later display the
    query verbatim."""
    return any(
        unicodedata.category(ch) == "Cc" and ch not in _ALLOWED_CONTROL_CHARS for ch in text
    )


def validate_query_text(text: str, *, max_length: int, field_name: str = "query") -> str:
    """Validate free-text user input (a question or topic). Returns the
    stripped text, or raises `SecurityValidationError`."""
    stripped = text.strip()
    if not stripped:
        raise SecurityValidationError(f"{field_name} must not be empty.")
    if len(stripped) > max_length:
        raise SecurityValidationError(
            f"{field_name} exceeds the {max_length}-character limit."
        )
    if contains_disallowed_control_chars(stripped):
        raise SecurityValidationError(f"{field_name} contains unsupported control characters.")
    return stripped


# --- Retrieved-content prompt injection defenses ----------------------------
#
# Two independent layers, matching how the rest of this app treats retrieved
# text as *data*: the prompt itself is evidence-is-data framed (see
# app/core/generation/answer.py), and this module additionally (1) breaks any
# literal occurrence of the prompt's own section markers inside chunk text so
# a malicious document can't spoof a section boundary, and (2) scans evidence
# for phrasing shaped like an injection attempt, purely for logging — it never
# blocks or alters a request. Neither guarantees a model can't be influenced
# by adversarial text in its context; both raise the bar without pretending
# to eliminate the risk.

RESERVED_PROMPT_MARKERS: tuple[str, ...] = (
    "=== SYSTEM INSTRUCTIONS ===",
    "=== RETRIEVED EVIDENCE",
    "=== STUDENT QUESTION ===",
    "=== ANSWER ===",
)


def neutralize_prompt_markers(text: str) -> str:
    """Break any literal occurrence of a reserved section marker inside
    untrusted text, while leaving the rest human-readable."""
    neutralized = text
    for marker in RESERVED_PROMPT_MARKERS:
        if marker in neutralized:
            neutralized = neutralized.replace(marker, marker.replace("===", "[evidence text]"))
    return neutralized


# Multi-word phrase shapes with essentially no legitimate use in study
# material, rather than single keywords — "ignore" or "system" alone appear
# constantly in ordinary text ("ignore the sign", "the immune system") and
# must not trigger on their own.
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ignore_instructions", re.compile(r"ignore\s+(all\s+)?(previous|prior|the\s+above)\s+instructions", re.I)),
    ("reveal_system_prompt", re.compile(r"reveal\s+(your|the)\s+(system\s+)?(prompt|instructions)", re.I)),
    ("developer_mode", re.compile(r"\b(developer|debug|admin)\s+mode\b", re.I)),
    ("disregard_rules", re.compile(r"disregard\s+(the\s+)?(above|previous|prior|these)\s+(rules|instructions)", re.I)),
    ("act_as_persona", re.compile(r"you\s+are\s+now\s+(a|an|in)\b", re.I)),
    ("exfiltrate_secrets", re.compile(r"(reveal|print|output|leak)\s+(the\s+)?(api\s*key|env(ironment)?\s*var)", re.I)),
    ("external_fetch", re.compile(r"\b(fetch|curl|download|POST)\s+https?://", re.I)),
)


def detect_injection_signals(text: str) -> tuple[str, ...]:
    """Names of injection-shaped phrase patterns found in `text`, or an empty
    tuple (the common case). Detection only — never used to block a chunk."""
    return tuple(name for name, pattern in _INJECTION_PATTERNS if pattern.search(text))
