"""Offline retrieval study over the populated ASQA dev subset.

This is an ASQA-derived retrieval probe, not the official ASQA answer score.
Its relevance labels are the dataset's annotated/QA contexts that are present
in the selected corpus. Labels are incomplete: other passages can also answer.
The input database and cached Arrow file are read only; the index is isolated.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import platform
import sqlite3
import tempfile
import time

import pyarrow as pa

from app.config import get_settings
from app.core.retrieval.pipeline import RetrievalIndex
from app.db.knowledge_base import KnowledgeBase
from app.eval.validation_metrics import interval, retrieval_metrics
from app.models.schemas import Chunk, RetrievalMode


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def load_arrow(path: Path, limit: int):
    with pa.memory_map(str(path), "r") as data:
        reader = pa.ipc.open_stream(data)
        return reader.read_all().slice(0, limit).to_pylist()


def load_corpus(db_path: Path):
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        raw = db.execute("SELECT * FROM chunks WHERE source_file LIKE 'asqa_%' ORDER BY chunk_id").fetchall()
    return [Chunk.model_validate(dict(row)) for row in raw
            if row["text"].strip().casefold() != "no context provided"]


def make_cases(rows, chunks):
    by_text = defaultdict(list)
    for chunk in chunks:
        by_text[text_hash(chunk.text)].append(chunk.chunk_id)
    cases = []
    excluded = []
    for row in rows:
        texts = [item.get("content", "") for ann in row.get("annotations", []) for item in ann.get("knowledge", [])]
        texts += [item.get("context", "") for item in row.get("qa_pairs", [])]
        labels = sorted({cid for text in texts if text and text.strip().casefold() != "no context provided"
                         for cid in by_text.get(text_hash(text), [])})
        query = row.get("ambiguous_question", "").strip()
        if not query or not labels:
            excluded.append({"sample_id": str(row.get("sample_id")), "reason": "no question or matched annotated context in corpus"})
            continue
        cases.append({"sample_id": str(row["sample_id"]), "query": query, "relevant_ids": labels})
    return cases, excluded


def corpus_hash(chunks):
    payload = [(c.chunk_id, c.source_file, c.text) for c in chunks]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def report(records, cases):
    output = {}
    for mode in dict.fromkeys(r["mode"] for r in records):
        rows = [r for r in records if r["mode"] == mode]
        ok = [r for r in rows if r["status"] == "ok"]
        gold = {c["sample_id"]: c["relevant_ids"] for c in cases}
        scores = [retrieval_metrics(r["retrieved_ids"], gold[r["sample_id"]], 5) for r in ok]
        output[mode] = {
            "attempted": len(rows), "completed": len(ok), "failures": len(rows) - len(ok),
            "completion_rate": len(ok) / len(rows) if rows else None,
            "mrr_at_5": interval([s["rr"] for s in scores]),
            "recall_at_5": interval([s["recall"] for s in scores]),
            "hit_rate_at_5": interval([float(s["rr"] > 0) for s in scores]),
            "ndcg_at_5": interval([s["ndcg"] for s in scores]),
            "latency_ms_all_attempts": interval([r["elapsed_ms"] for r in rows]),
        }
    return output


async def execute(cases, chunks, settings, modes, out):
    records = []
    # Rebuild only a disposable SQLite DB, leaving the user's live KB intact.
    with tempfile.TemporaryDirectory(prefix="asqa-index-") as folder:
        kb = KnowledgeBase(Path(folder) / "kb.sqlite3")
        kb.add_chunks(chunks)
        index = RetrievalIndex(kb, settings)
        await asyncio.to_thread(index.rebuild)
        for case in cases:
            for mode in modes:
                started = time.perf_counter()
                row = {"sample_id": case["sample_id"], "mode": mode, "status": "error"}
                try:
                    hits = await index.retrieve(case["query"], top_k=5, mode=RetrievalMode(mode))
                    row.update(status="ok", retrieved_ids=[c.chunk_id for c, _ in hits])
                except Exception as exc:
                    row["error_type"] = type(exc).__name__
                row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
                records.append(row)
                (out / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            if len(records) % 15 == 0:
                print(f"Completed {len(records)} retrieval attempts", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow", type=Path, required=True, help="Cached ASQA dev Arrow file")
    parser.add_argument("--db", type=Path, required=True, help="Populated KB, opened read-only")
    parser.add_argument("--out", type=Path, required=True, help="New empty output directory")
    parser.add_argument("--limit", type=int, default=100, help="First N ASQA dev samples originally imported")
    parser.add_argument("--modes", nargs="+", default=["bm25_only", "hybrid", "full"],
                        choices=[m.value for m in RetrievalMode])
    args = parser.parse_args()
    if args.limit < 1 or len(set(args.modes)) != len(args.modes):
        parser.error("Limit must be positive and modes unique")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use a new empty output directory")
    rows = load_arrow(args.arrow, args.limit)
    chunks = load_corpus(args.db)
    cases, excluded = make_cases(rows, chunks)
    if not chunks or not cases:
        parser.error("No ASQA corpus or matched labeled cases")
    args.out.mkdir(parents=True, exist_ok=True)
    settings = get_settings().model_copy(update={"hyde_enabled": False, "hyde_provider": "none"})
    code = [Path(__file__), *sorted(Path("app/core/retrieval").glob("*.py")), Path("app/eval/validation_metrics.py")]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": "din0s/asqa dev cached Arrow",
        "arrow_sha256": hashlib.sha256(args.arrow.read_bytes()).hexdigest(),
        "corpus_sha256": corpus_hash(chunks), "corpus_chunks": len(chunks), "sample_limit": args.limit,
        "selected_cases": len(cases), "excluded_cases": excluded, "modes": args.modes,
        "model": settings.embedding_model, "reranker": settings.reranker_model, "hyde_enabled": False,
        "python": platform.python_version(), "code_sha256": {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
        "design": "ASQA dev question contexts matched by exact text to an isolated ASQA-only corpus; incomplete relevance labels; full means hybrid plus reranker WITHOUT HyDE; not official ASQA scoring"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out / "cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Indexing {len(chunks)} chunks for {len(cases)} labeled questions; {len(excluded)} excluded.", flush=True)
    records = asyncio.run(execute(cases, chunks, settings, args.modes, args.out))
    (args.out / "summary.json").write_text(json.dumps({**manifest, "metrics": report(records, cases)}, indent=2), encoding="utf-8")
    raise SystemExit(0 if all(r["status"] == "ok" for r in records) else 1)


if __name__ == "__main__":
    main()
