"""End-to-end ASQA development probe through an already-running local API.

This uses the live knowledge base/index and configured providers. No uploads,
conversation writes, or database mutations are made. ASQA answer-string recall
is diagnostic only and is not the official ASQA metric or an accuracy score.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import httpx

from app.core.generation.build_identity import pipeline_source_sha256
from app.eval.asqa_answers import short_answer_recall, summarize, write_review
from app.eval.asqa_retrieval import load_arrow, load_corpus, make_cases


def kb_fingerprint(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute("SELECT chunk_id, source_file, text FROM chunks ORDER BY chunk_id").fetchall()
    ids = [r[0] for r in rows]
    return {
        "count": len(rows),
        "ids_sha256": hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest(),
        "contents_sha256": hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest(),
    }


async def read_config(client, base):
    response = await client.get(f"{base}/config", timeout=10)
    response.raise_for_status()
    data = response.json()
    # Endpoint locations and keys do not belong in run artifacts.
    data.get("llm", {}).pop("base_url", None)
    return data


def validate_server(config, kb, expected_sha):
    if config.get("loaded_pipeline_sha256") != expected_sha:
        raise ValueError("The running backend does not match the local pipeline sources; restart it before testing.")
    retrieval = config.get("retrieval", {})
    if retrieval.get("indexed_chunks") != kb["count"] or retrieval.get("indexed_ids_sha256") != kb["ids_sha256"]:
        raise ValueError("The running BM25 index does not match the database IDs; restart/rebuild before testing.")


async def execute(cases, methods, out, db_path, base, manifest, previous):
    records = []
    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        for case in cases:
            for method in methods:
                key = (case["sample_id"], method)
                old = previous.get(key)
                if old and old["status"] == "ok":
                    records.append(old)
                    print(f"{case['sample_id']}/{method}: reused", flush=True)
                    continue
                config = await read_config(client, base)
                current_kb = kb_fingerprint(db_path)
                validate_server(config, current_kb, manifest["loaded_pipeline_sha256"])
                if config != manifest["server_config"] or current_kb != manifest["knowledge_base"]:
                    raise ValueError("Server configuration or knowledge base changed; start a separate run.")
                row = {"sample_id": case["sample_id"], "query": case["query"], "method": method, "status": "error"}
                if old:
                    row["previous_attempts"] = old.get("previous_attempts", []) + [{k: old[k] for k in ("status", "error_type", "http_status", "elapsed_ms", "llm_calls") if k in old}]
                started = time.perf_counter()
                try:
                    response = await client.post(f"{base}/ask", json={"query": case["query"], "answer_scope": "auto",
                        "evidence_method": method, "detect_contradictions": False, "mode": "full", "top_k": 5})
                    row["http_status"] = response.status_code
                    response.raise_for_status()
                    result = response.json()
                    row.update(status="ok", response=result,
                        short_answer_presence=short_answer_recall(result.get("answer", ""), case["qa_pairs"]),
                        llm_calls=result.get("model_calls", []))
                except Exception as exc:
                    row["error_type"] = type(exc).__name__
                    row["llm_calls"] = []  # server-side attempts unavailable for failed HTTP responses
                row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
                records.append(row)
                previous[key] = row
                (out / "records.json").write_text(json.dumps(list(previous.values()), indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"{case['sample_id']}/{method}: {row['status']} ({row['elapsed_ms']/1000:.1f}s)", flush=True)
    return records


async def prepare(args):
    raw = load_arrow(args.arrow, 100)
    corpus = load_corpus(args.db)
    cases, excluded = make_cases(raw, corpus)
    if not cases:
        raise ValueError("No ASQA questions match the populated corpus")
    by_id = {str(r["sample_id"]): r for r in raw}
    cases = [{**c, "qa_pairs": by_id[c["sample_id"]]["qa_pairs"]} for c in cases[:args.count]]
    kb = kb_fingerprint(args.db)
    async with httpx.AsyncClient(trust_env=False) as client:
        config = await read_config(client, args.base_url.rstrip("/"))
    identity = pipeline_source_sha256()
    validate_server(config, kb, identity)
    manifest = {
        "design": "Live /ask through running application, first eligible ASQA dev samples from imported first 100; full current knowledge base; primary and fallback models recorded per response. Not an official ASQA score.",
        "arrow_sha256": hashlib.sha256(args.arrow.read_bytes()).hexdigest(),
        "loaded_pipeline_sha256": identity, "knowledge_base": kb,
        "server_config": config, "methods": args.methods, "count": args.count,
        "excluded_no_matched_context": len(excluded), "case_ids": [c["sample_id"] for c in cases],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return cases, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--methods", nargs="+", choices=["edurag", "adaptive", "aspect"], default=["edurag", "aspect"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not re_is_loopback(args.base_url):
        parser.error("Only localhost/127.0.0.1 HTTP backends are supported")
    if args.count < 1 or len(args.methods) != len(set(args.methods)):
        parser.error("Count must be positive and methods unique")
    cases, manifest = asyncio.run(prepare(args))
    previous = {}
    if args.resume:
        if not (args.out / "manifest.json").exists() or not (args.out / "records.json").exists():
            parser.error("Resume requires manifest and records")
        old_manifest = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))
        old_manifest.pop("started_utc", None)
        if old_manifest != manifest:
            parser.error("Pipeline, data or configuration changed; use a new output directory")
        valid = {(c["sample_id"], m) for c in cases for m in args.methods}
        for row in json.loads((args.out / "records.json").read_text(encoding="utf-8")):
            key = (row["sample_id"], row["method"])
            if key not in valid or key in previous:
                parser.error("Invalid or duplicate saved records")
            previous[key] = row
    else:
        if args.out.exists() and any(args.out.iterdir()):
            parser.error("Use a new empty output directory")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
        (args.out / "manifest.json").write_text(json.dumps({**manifest, "started_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    records = asyncio.run(execute(cases, args.methods, args.out, args.db, args.base_url.rstrip("/"), manifest, previous))
    final_kb = kb_fingerprint(args.db)
    if final_kb != manifest["knowledge_base"]:
        raise ValueError("Knowledge base changed during the run; results cannot be combined.")
    (args.out / "summary.json").write_text(json.dumps({"manifest": manifest, "metrics": summarize(records)}, indent=2), encoding="utf-8")
    write_review(records, args.out)
    raise SystemExit(0 if all(r["status"] == "ok" for r in records) else 1)


def re_is_loopback(url):
    from urllib.parse import urlsplit
    parts = urlsplit(url)
    return parts.scheme == "http" and parts.hostname in {"localhost", "127.0.0.1"} and not parts.username and not parts.password and not parts.path.strip("/") and not parts.query and not parts.fragment


if __name__ == "__main__":
    main()
