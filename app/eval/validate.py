"""Controlled adaptive regression through the production ask handler.

uv run python -m app.eval.validate --live --out artifacts/validation
uv run --group eval python -m app.eval.ragas_eval artifacts/validation/records.json

Never ingests into or changes the student's knowledge base. Fixed passages
isolate routing/generation; this is NOT a live retrieval benchmark.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from app.eval.validation_metrics import summarize

CASES = Path(__file__).with_name("validation_cases.json")


def load_cases(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = [row["id"] for row in data["cases"]]
    corpus_ids = [row["id"] for row in data["corpus"]]
    if not ids or len(ids) != len(set(ids)) or len(corpus_ids) != len(set(corpus_ids)):
        raise ValueError("Case and corpus IDs must be unique and cases nonempty")
    for case in data["cases"]:
        if set(case["contexts"]) - set(corpus_ids):
            raise ValueError(f"Missing context in {case['id']}")
        if case["expected_basis"] not in {"sources", "mixed", "model"}:
            raise ValueError("Invalid expected basis")
    return data


class FixedRetrieval:
    """Controlled context delivery, no claims about retriever performance."""
    is_ready = True

    def __init__(self, chunks):
        self.chunks = chunks

    async def retrieve(self, query, top_k=5, mode=None, **kwargs):
        return [(chunk, .9 - i * .05) for i, chunk in enumerate(self.chunks[:top_k])]


def check_expectations(case: dict, response: dict, trace: dict) -> dict:
    checks = {}
    answer = response["answer"]
    if "expected_verification_required" in case:
        checks["verification_flag"] = trace.get("question_plan", {}).get("verification_required") == case["expected_verification_required"]
    if case.get("expected_currency"):
        patterns = {"INR": r"₹|\bINR\b|\brupees?\b", "USD": r"\$|\bUSD\b|\bUS dollars?\b"}
        target = case["expected_currency"]
        other = "USD" if target == "INR" else "INR"
        checks["currency_label"] = bool(re.search(patterns[target], answer, re.I)) and not re.search(patterns[other], answer, re.I)
    if case.get("forbidden_patterns"):
        checks["no_forbidden_details"] = not any(re.search(pattern, answer) for pattern in case["forbidden_patterns"])
    return checks


async def run_live(data: dict, out: Path, resume: bool = False) -> list[dict]:
    from app.api import routes_ask
    from app.config import get_settings
    from app.core.generation.runtime_context import student_context
    from app.db.knowledge_base import KnowledgeBase
    from app.models.schemas import AskRequest, Chunk, ChunkType
    from app.state import AppState

    settings = get_settings()
    out.mkdir(parents=True, exist_ok=True)
    tracked_sources = [Path(__file__), Path(__file__).parents[1] / "api/routes_ask.py",
                       Path(__file__).parents[1] / "core/generation/orchestration.py",
                       Path(__file__).parents[1] / "core/generation/answer.py"]
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked_sources},
                "context": {"timezone": "Asia/Kolkata", "country": "India"},
                "contradictions_enabled": False, "fixed_contexts": True}
    previous = {}
    if resume and (out / "records.json").exists():
        old_data = json.loads((out / "cases.json").read_text(encoding="utf-8"))
        old_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if old_data != data:
            raise ValueError("Cannot resume a changed dataset")
        # Harness changes may add bookkeeping, but successful answers must stay
        # tied to the exact same production prompts and handler.
        for path, digest in manifest["source_sha256"].items():
            if path != Path(__file__).as_posix() and old_manifest["source_sha256"].get(path) != digest:
                raise ValueError("Cannot resume answers from different production code")
        if (out / "summary.json").exists():
            models = json.loads((out / "summary.json").read_text(encoding="utf-8"))["models"]
            if models != {"answer": settings.llm_model, "provider": settings.llm_provider, "embedding": settings.embedding_model}:
                raise ValueError("Cannot resume a changed model configuration")
        previous = {r["id"]: r for r in json.loads((out / "records.json").read_text(encoding="utf-8"))}
        if set(previous) - {c["id"] for c in data["cases"]}:
            raise ValueError("Saved records contain unknown cases")
        manifest["previous_manifest"] = old_manifest
    (out / "cases.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    corpus = {row["id"]: Chunk(chunk_id=row["id"], source_file=row["source"],
                               page_number=1, text=row["text"], chunk_length=len(row["text"]),
                               type=ChunkType.text) for row in data["corpus"]}
    records = []
    original_plan = routes_ask.plan_question
    original_assessment = routes_ask.evaluate_evidence
    trace = {}

    async def capture_plan(*args, **kwargs):
        plan = await original_plan(*args, **kwargs)
        trace["question_plan"] = plan.model_dump(mode="json")
        return plan

    async def capture_assessment(*args, **kwargs):
        decision = await original_assessment(*args, **kwargs)
        trace["evidence_decision"] = decision.model_dump(mode="json")
        return decision

    # This isolated CLI runs sequentially. Wrappers observe production functions
    # without changing their results and are restored even after a failed run.
    routes_ask.plan_question = capture_plan
    routes_ask.evaluate_evidence = capture_assessment
    with tempfile.TemporaryDirectory(prefix="rag-validation-") as folder:
        isolated = settings.model_copy(update={"kb_db_path": Path(folder) / "kb.sqlite3"})
        kb = KnowledgeBase(isolated.kb_db_path)
        kb.add_chunks(list(corpus.values()))
        context_token = student_context.set(("Asia/Kolkata", "India"))
        try:
            for case in data["cases"]:
                old = previous.get(case["id"])
                # Freeze every completed response, including wrong routes. Only
                # failed handler attempts may be retried, avoiding score selection.
                if old and old["status"] == "ok":
                    records.append(old)
                    print(f"{case['id']}: reusing completed response", flush=True)
                    continue
                trace.clear()
                chunks = [corpus[key] for key in case["contexts"]]
                state = AppState(isolated, kb, FixedRetrieval(chunks))
                request = AskRequest(query=case["query"], answer_scope=case.get("scope", "auto"),
                                     history=case.get("history", []),
                                     detect_contradictions=False)
                record = {**case, "retrieved_contexts": [c.text for c in chunks],
                          "status": "error"}
                if old:
                    record["previous_attempts"] = old.get("previous_attempts", []) + [
                        {key: old[key] for key in ("status", "error_type", "elapsed_ms") if key in old}]
                started = time.perf_counter()
                try:
                    result = await asyncio.wait_for(routes_ask.ask(request, state), timeout=180)
                    record.update(status="ok", response=result.model_dump(mode="json"))
                except Exception as exc:
                    # Keep provider error text and credentials out of stored reports.
                    record["error_type"] = type(exc).__name__
                record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
                record["trace"] = dict(trace)
                if record["status"] == "ok":
                    record["checks"] = check_expectations(case, record["response"], trace)
                records.append(record)
                (out / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
                actual = record.get("response", {}).get("answer_basis", record.get("error_type"))
                print(f"{case['id']}: expected={case['expected_basis']} actual={actual}", flush=True)
        finally:
            student_context.reset(context_token)
            routes_ask.plan_question = original_plan
            routes_ask.evaluate_evidence = original_assessment
    return records


def write_report(records: list[dict], data: dict, out: Path, cases_path: Path) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    from app.config import get_settings
    settings = get_settings()
    summary = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
               "git_commit": commit, "python": platform.python_version(),
               "dataset_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
               "annotation_status": data["annotation_status"], "seed": 42,
               "design": "controlled contexts through production ask handler; no retrieval accuracy claim",
               "models": {"answer": settings.llm_model, "provider": settings.llm_provider,
                          "embedding": settings.embedding_model},
               "metrics": summarize(records)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = ["# Adaptive validation results", "", f"Date UTC: {summary['created_utc']}",
            "", summary["design"], "", f"Annotations: {data['annotation_status']}", "",
            "| Case | Expected | Actual | Status |", "| --- | --- | --- | --- |"]
    rows.extend(f"| {r['id']} | {r['expected_basis']} | {r.get('response', {}).get('answer_basis', 'n/a')} | {r['status']} |" for r in records)
    rows += ["", "## Metrics", "", "```json", json.dumps(summary["metrics"], indent=2), "```", "",
             "Bootstrap intervals describe this small developer fixture set only. Independent human review and a held out student dataset are required before generalization claims."]
    (out / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--out", type=Path, default=Path("artifacts/validation"))
    parser.add_argument("--live", action="store_true", help="Call configured LLMs on controlled source passages")
    parser.add_argument("--resume", action="store_true", help="Retry failed handler calls only; freeze all completed answers")
    parser.add_argument("--records", type=Path, help="Recompute summaries from captured records without LLM calls")
    args = parser.parse_args()
    data = load_cases(args.cases)
    if bool(args.live) == bool(args.records):
        parser.error("Choose exactly one of --live or --records")
    if args.resume and not args.live:
        parser.error("--resume requires --live")
    records = asyncio.run(run_live(data, args.out, args.resume)) if args.live else json.loads(args.records.read_text(encoding="utf-8"))
    if [r["id"] for r in records] != [c["id"] for c in data["cases"]]:
        parser.error("Records must include every dataset case in order")
    summary = write_report(records, data, args.out, args.cases)
    metrics = summary["metrics"]
    # A failed call or wrong route must fail automation, even when successful-only means look good.
    raise SystemExit(0 if metrics["end_to_end_routing_success"] == 1.0 and metrics["constraint_failures"] == 0 else 1)


if __name__ == "__main__":
    main()
