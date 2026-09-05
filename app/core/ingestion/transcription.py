"""Lecture-audio transcription -> timestamped text chunks (plan Sec. 5.3).

This is deliberately a one-off preprocessing step, not a streaming system: an
uploaded recording is transcribed once, split into chunks at segment
boundaries, and fed through the *same* index as prose. Audio therefore needs no
new retrieval logic — it becomes another chunk type, exactly as EduRAG treats
table and visual chunks.

Two providers:
  - "openai_compatible" -> any /audio/transcriptions endpoint (Groq's Whisper,
                           OpenAI's). Nothing runs locally.
  - "faster_whisper"    -> local transcription, no network and no API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

from app.config import Settings
from app.core.generation.llm_client import LLMError

AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4", ".mpga"}


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


def format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


async def transcribe(file_path: Path, settings: Settings) -> list[TranscriptSegment]:
    provider = settings.asr_provider
    if provider == "none":
        raise LLMError(
            "Audio ingestion is disabled. Set ASR_PROVIDER to 'openai_compatible' "
            "or 'faster_whisper' to enable it."
        )
    if provider == "openai_compatible":
        return await _transcribe_openai_compatible(file_path, settings)
    if provider == "faster_whisper":
        return _transcribe_faster_whisper(file_path, settings)
    raise ValueError(f"Unknown ASR provider: {provider!r}")


async def _transcribe_openai_compatible(
    file_path: Path, settings: Settings
) -> list[TranscriptSegment]:
    url = f"{settings.asr_base_url.rstrip('/')}/audio/transcriptions"
    headers = {}
    if settings.asr_api_key:
        headers["Authorization"] = f"Bearer {settings.asr_api_key}"

    # verbose_json is what carries per-segment timestamps; without it we get a
    # single undifferentiated blob and lose the ability to cite a moment.
    data = {"model": settings.asr_model, "response_format": "verbose_json"}
    files = {"file": (file_path.name, file_path.read_bytes())}

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(url, headers=headers, data=data, files=files)
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPStatusError as exc:
        raise LLMError(
            f"Transcription failed ({exc.response.status_code}) for model "
            f"'{settings.asr_model}': {exc.response.text[:400]}"
        ) from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"Could not reach ASR endpoint at {settings.asr_base_url}: {exc}") from exc

    segments = payload.get("segments")
    if not segments:
        text = (payload.get("text") or "").strip()
        return [TranscriptSegment(start=0.0, end=0.0, text=text)] if text else []

    return [
        TranscriptSegment(
            start=float(s.get("start", 0.0)),
            end=float(s.get("end", 0.0)),
            text=(s.get("text") or "").strip(),
        )
        for s in segments
        if (s.get("text") or "").strip()
    ]


def _transcribe_faster_whisper(file_path: Path, settings: Settings) -> list[TranscriptSegment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise LLMError(
            "ASR_PROVIDER='faster_whisper' requires the optional dependency: "
            "install it with `uv add faster-whisper`."
        ) from exc

    model = WhisperModel(settings.asr_model, device="auto", compute_type="int8")
    segments, _info = model.transcribe(str(file_path))
    return [
        TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
        for s in segments
        if s.text.strip()
    ]


def group_segments(
    segments: list[TranscriptSegment], target_chars: int
) -> list[TranscriptSegment]:
    """Merges Whisper's short segments into chunk-sized passages, preserving the
    start time of the first segment and the end time of the last so each chunk
    still points at a real moment in the recording."""
    grouped: list[TranscriptSegment] = []
    buffer: list[TranscriptSegment] = []
    length = 0

    for segment in segments:
        buffer.append(segment)
        length += len(segment.text) + 1
        if length >= target_chars:
            grouped.append(
                TranscriptSegment(
                    start=buffer[0].start,
                    end=buffer[-1].end,
                    text=" ".join(s.text for s in buffer).strip(),
                )
            )
            buffer, length = [], 0

    if buffer:
        grouped.append(
            TranscriptSegment(
                start=buffer[0].start,
                end=buffer[-1].end,
                text=" ".join(s.text for s in buffer).strip(),
            )
        )
    return grouped
