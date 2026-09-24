"""LLM-as-teacher benchmark construction — plan Sec. 8 / EduRAG Sec. IV-A.

For each sampled chunk, an LLM acting as a teacher writes a question that the
chunk answers, plus a reference answer. The chunk's id is the ground truth, so
retrieval can be scored without any human labelling.

Two properties matter for the benchmark to be honest:

  * The question must be answerable from the chunk *alone*, or retrieval is
    being asked to find something the chunk doesn't contain.
  * The question must not quote the chunk verbatim, or BM25 wins trivially and
    the ablation shows nothing. The prompt forbids copying distinctive phrases.

Visual queries are generated from `visual` chunks specifically, giving the
separate diagram benchmark the plan asks for.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import Settings
from app.core.generation.answer import build_client
from app.models.schemas import Chunk, ChunkType

MIN_CHUNK_CHARS = 200

_TEACHER_PROMPT = (
    "You are a teacher writing exam questions from a course passage.\n\n"
    "Write ONE question that this passage fully answers, and a reference answer "
    "of 1-3 sentences drawn only from the passage.\n\n"
    "Rules:\n"
    "- The question must be answerable from this passage alone.\n"
    "- Do NOT copy distinctive phrases from the passage into the question; ask "
    "it the way a student would, in their own words.\n"
    "- Do not refer to 'the passage', 'the text', or 'the figure' — ask about "
    "the subject matter directly.\n\n"
    'Respond with only a JSON object: {"question": "...", "answer": "..."}'
)


@dataclass
class BenchmarkItem:
    query: str
    reference_answer: str
    gold_chunk_id: str
    source_file: str
    chunk_type: str


def _parse(raw: str) -> tuple[str, str] | None:
    match = re.search(r"\{.*}", raw, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    question = str(payload.get("question") or "").strip()
    answer = str(payload.get("answer") or "").strip()
    if not question or not answer:
        return None
    return question, answer


def sample_chunks(
    chunks: list[Chunk],
    count: int,
    chunk_type: ChunkType | None = None,
    seed: int = 42,
) -> list[Chunk]:
    """Deterministic sampling so a benchmark can be regenerated identically."""
    pool = [c for c in chunks if len(c.text) >= MIN_CHUNK_CHARS]
    if chunk_type is not None:
        pool = [c for c in pool if c.type == chunk_type]
    if not pool:
        return []
    rng = random.Random(seed)
    return rng.sample(pool, min(count, len(pool)))


async def generate_benchmark(
    chunks: list[Chunk],
    settings: Settings,
    count: int = 97,
    chunk_type: ChunkType | None = None,
    seed: int = 42,
    concurrency: int = 4,
) -> list[BenchmarkItem]:
    """Default count of 97 matches the base paper's benchmark size, so the
    resulting table is directly comparable in structure."""
    selected = sample_chunks(chunks, count, chunk_type=chunk_type, seed=seed)
    if not selected:
        return []

    client = build_client(settings)
    semaphore = asyncio.Semaphore(concurrency)
    items: list[BenchmarkItem] = []

    async def build(chunk: Chunk) -> None:
        async with semaphore:
            messages = [
                {"role": "system", "content": _TEACHER_PROMPT},
                {"role": "user", "content": f"Passage:\n{chunk.text}"},
            ]
            try:
                raw = await client.chat(messages, temperature=0.3)
            except Exception:
                return
            parsed = _parse(raw)
            if parsed is None:
                return
            question, answer = parsed
            items.append(
                BenchmarkItem(
                    query=question,
                    reference_answer=answer,
                    gold_chunk_id=chunk.chunk_id,
                    source_file=chunk.source_file,
                    chunk_type=chunk.type.value,
                )
            )

    await asyncio.gather(*(build(c) for c in selected))
    return items


def save_benchmark(items: list[BenchmarkItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(i) for i in items], indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_benchmark(path: Path) -> list[BenchmarkItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkItem(**entry) for entry in payload]
