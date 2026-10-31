"""Small controlled answer comparison on original ASQA dev questions.

Original annotated contexts from the populated corpus are provided initially;
gap searches use local BM25 over that corpus. Exact substring recall of ASQA
short answers is a diagnostic, NOT official ASQA, entailment or citation score.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace

from app.api.routes_ask import ask
from app.config import get_settings
from app.core.generation.llm_client import capture_llm_calls
from app.core.retrieval.bm25_index import BM25Index
from app.eval.asqa_retrieval import load_arrow, load_corpus, make_cases, corpus_hash
from app.eval.validation_metrics import interval
from app.models.schemas import AskRequest


def normalize(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def short_answer_recall(answer, pairs):
    """Each disambiguated QA pair counts once if any provided alias appears."""
    haystack = f" {normalize(answer)} "
    eligible = [p for p in pairs if any(normalize(a) for a in p.get("short_answers", []))]
    if not eligible:
        return {"matched": 0, "eligible": 0, "recall": None}
    matched = sum(any(f" {normalize(alias)} " in haystack for alias in p["short_answers"] if normalize(alias)) for p in eligible)
    return {"matched": matched, "eligible": len(eligible), "recall": matched / len(eligible)}


class ControlledCorpus:
    def __init__(self, case, corpus):
        self.by_id = {c.chunk_id: c for c in corpus}
        self.initial = [self.by_id[cid] for cid in case["relevant_ids"][:5]]
        self.index = BM25Index()
        self.index.build(corpus)
        self.is_ready = bool(corpus)
        self.calls = 0

    async def retrieve(self, query, top_k=5, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return [(c, 1.0) for c in self.initial[:top_k]]
        return [(self.by_id[cid], float(score)) for cid, score in self.index.search(query, top_k)]


def config_digest(dataset_path, chunks, settings, methods, count):
    code = [Path("app/api/routes_ask.py"), Path("app/config.py"), Path("app/models/schemas.py"),
            Path(__file__), *sorted(Path("app/core/generation").glob("*.py")),
            *sorted(Path("app/core/retrieval").glob("*.py")), *sorted(Path("app/core/attribution").glob("*.py"))]
    return {"arrow_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(), "corpus_sha256": corpus_hash(chunks),
        "code_sha256": {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
        "models": {"provider": settings.llm_provider, "model": settings.llm_model,
            "endpoint_sha256": hashlib.sha256(settings.llm_base_url.encode()).hexdigest(),
            "fallback_provider": settings.llm_fallback_provider if settings.llm_fallback_base_url else None,
            "fallback_model": settings.llm_fallback_model if settings.llm_fallback_base_url else None,
            "fallback_endpoint_sha256": hashlib.sha256(settings.llm_fallback_base_url.encode()).hexdigest() if settings.llm_fallback_base_url else None,
            "reranker": settings.reranker_model},
        "settings": settings.model_dump(include={"evidence_check_timeout_seconds", "evidence_method",
            "targeted_retrieval_enabled", "targeted_retrieval_max_queries", "targeted_retrieval_top_k",
            "targeted_retrieval_timeout_seconds", "claim_verification_enabled", "claim_verification_fail_closed",
            "claim_verification_max_chars", "claim_verification_max_claims", "claim_repair_enabled",
            "aspect_decomposition_enabled"}),
        "methods": methods, "count": count,
        "design": "First eligible ASQA dev questions from imported first 100; fixed annotated contexts up to five, then optional local BM25 gap search. Short-answer string recall is a surface diagnostic, not factual correctness or official ASQA score."}


def summarize(records):
    output = {}
    for method in dict.fromkeys(r["method"] for r in records):
        rows = [r for r in records if r["method"] == method]
        completed = [r for r in rows if r["status"] == "ok"]
        scored = [r["short_answer_presence"]["recall"] for r in completed if r["short_answer_presence"]["recall"] is not None]
        output[method] = {"attempted": len(rows), "completed": len(completed), "failures": len(rows) - len(completed),
            "completion_rate": len(completed) / len(rows) if rows else None,
            "short_answer_surface_recall_completed": interval(scored),
            "latency_ms_completed": interval([r["elapsed_ms"] for r in completed]),
            "latency_ms_all_attempts": interval([r["elapsed_ms"] for r in rows]),
            "audit_unavailable": sum(r["response"].get("claim_verification", {}).get("status") == "unavailable" for r in completed),
            "withheld": sum(r["response"].get("claim_verification", {}).get("withheld", False) for r in completed),
            "fallback_provider_calls": sum(c.get("status") == "fallback_used" for r in rows for c in r.get("llm_calls", [])),
            "note": "Conditional surface matching; does not check negation, factual support, citations, or whether all reference facts are present in supplied context."}
    return output


async def execute(cases, corpus, settings, methods, out, previous):
    records = []
    for case in cases:
        for method in methods:
            key = (case["sample_id"], method)
            old = previous.get(key)
            if old and old["status"] == "ok":
                records.append(old)
                print(f"{case['sample_id']}/{method}: reused", flush=True)
                continue
            row = {"sample_id": case["sample_id"], "query": case["query"], "method": method, "status": "error"}
            if old:
                row["previous_attempts"] = old.get("previous_attempts", []) + [{k: old[k] for k in ("status", "error_type", "elapsed_ms", "llm_calls") if k in old}]
            state = SimpleNamespace(settings=settings, retrieval=ControlledCorpus(case, corpus))
            started = time.perf_counter()
            with capture_llm_calls() as calls:
                try:
                    result = await asyncio.wait_for(ask(AskRequest(query=case["query"], answer_scope="auto",
                        evidence_method=method, detect_contradictions=False), state), timeout=300)
                    row.update(status="ok", response=result.model_dump(mode="json"),
                        short_answer_presence=short_answer_recall(result.answer, case["qa_pairs"]))
                except Exception as exc:
                    row["error_type"] = type(exc).__name__
                row["llm_calls"] = result.model_calls if row["status"] == "ok" else calls
            row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
            records.append(row)
            previous[key] = row
            (out / "records.json").write_text(json.dumps(list(previous.values()), indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{case['sample_id']}/{method}: {row['status']} ({row['elapsed_ms'] / 1000:.1f}s)", flush=True)
    return records


def write_review(records, out):
    with (out / "human-review.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["sample_id", "method", "question", "answer", "evidence_json", "reviewer", "factual_correctness", "source_claim_support", "citation_correctness", "coverage", "notes"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for r in records:
            writer.writerow({"sample_id": r["sample_id"], "method": r["method"], "question": r["query"],
                "answer": r.get("response", {}).get("answer", ""),
                "evidence_json": json.dumps(r.get("response", {}).get("evidence", []), ensure_ascii=False)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--methods", nargs="+", choices=["edurag", "adaptive", "aspect"], default=["edurag", "aspect"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable-fallback", action="store_true", help="Use only the configured primary model")
    parser.add_argument("--fail-closed", action="store_true", help="Withhold source claims if the audit fails")
    args = parser.parse_args()
    if args.count < 1 or len(set(args.methods)) != len(args.methods):
        parser.error("count must be positive and methods unique")
    corpus = load_corpus(args.db)
    raw = load_arrow(args.arrow, 100)
    cases, _ = make_cases(raw, corpus)
    by_id = {str(r["sample_id"]): r for r in raw}
    cases = [{**case, "qa_pairs": by_id[case["sample_id"]]["qa_pairs"]} for case in cases[:args.count]]
    if not cases:
        parser.error("No matching ASQA cases")
    updates = {"vlm_provider": "none"}
    if args.disable_fallback:
        updates["llm_fallback_base_url"] = ""
    if args.fail_closed:
        updates["claim_verification_fail_closed"] = True
    settings = get_settings().model_copy(update=updates)
    if settings.llm_provider == "none":
        parser.error("A configured answer model is required")
    config = config_digest(args.arrow, corpus, settings, args.methods, args.count)
    previous = {}
    if args.resume:
        if not (args.out / "manifest.json").exists() or not (args.out / "records.json").exists():
            parser.error("Resume requires manifest.json and records.json")
        old_manifest = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))
        if old_manifest != config:
            parser.error("Code, data, model, settings, methods or count changed; use a new directory")
        old_rows = json.loads((args.out / "records.json").read_text(encoding="utf-8"))
        valid = {(c["sample_id"], m) for c in cases for m in args.methods}
        for row in old_rows:
            key = (row["sample_id"], row["method"])
            if key in previous or key not in valid:
                parser.error("Invalid or duplicate saved records")
            previous[key] = row
    else:
        if args.out.exists() and any(args.out.iterdir()):
            parser.error("Use a new empty output directory")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "manifest.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (args.out / "cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
        (args.out / "corpus.json").write_text(json.dumps([c.model_dump(mode="json") for c in corpus], ensure_ascii=False), encoding="utf-8")
    records = asyncio.run(execute(cases, corpus, settings, args.methods, args.out, previous))
    (args.out / "summary.json").write_text(json.dumps({"manifest": config, "metrics": summarize(records)}, indent=2), encoding="utf-8")
    write_review(records, args.out)
    raise SystemExit(0 if all(r["status"] == "ok" for r in records) else 1)


if __name__ == "__main__":
    main()
