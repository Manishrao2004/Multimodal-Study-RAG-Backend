"""Isolated, controlled comparison of evidence-aware answering methods.

Default --check validates a dataset without model calls. --live invokes the
configured answer provider. Initial contexts are fixed; targeted searches use
local BM25 over each case's allowed corpus. This is not an end-to-end retrieval
benchmark or independent claim-support measurement.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
from pathlib import Path
import platform
from datetime import datetime, timezone
import tempfile
import time
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.api.routes_ask import ask
from app.config import get_settings
from app.core.retrieval.bm25_index import BM25Index
from app.eval.validation_metrics import interval
from app.models.schemas import AskRequest, Chunk, ChunkType

DATASET = Path(__file__).with_name("aspect_cases.json")
METHODS = ("edurag", "adaptive", "aspect", "aspect_no_retrieval", "aspect_no_verification", "whole_question")


class Passage(BaseModel):
    id: str
    source: str
    text: str = Field(min_length=1, max_length=4000)


class StudyCase(BaseModel):
    id: str
    group: str
    query: str
    initial_context_ids: list[str]
    available_context_ids: list[str]
    expected_aspects: dict[str, Literal["supported", "partial", "missing", "conflicting"]]
    scope: Literal["auto", "sources_only"] = "auto"
    reference: str


class StudyDataset(BaseModel):
    version: str
    annotation_status: str
    passages: list[Passage]
    cases: list[StudyCase]

    @model_validator(mode="after")
    def consistent(self):
        ids = {p.id for p in self.passages}
        if len(ids) != len(self.passages) or len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("Passage and case IDs must be unique")
        for c in self.cases:
            if not set(c.initial_context_ids) <= set(c.available_context_ids) <= ids:
                raise ValueError(f"Invalid or unavailable context in {c.id}")
            if not c.expected_aspects:
                raise ValueError("Every case needs aspect labels")
        return self


class ControlledRetrieval:
    """Fixed initial retrieval; additional searches cannot access removed evidence."""
    def __init__(self, case, corpus):
        self.initial = [corpus[cid] for cid in case.initial_context_ids]
        self.available = {cid: corpus[cid] for cid in case.available_context_ids}
        self.index = BM25Index()
        self.index.build(list(self.available.values()))
        self.calls = 0
        self.is_ready = bool(self.available)

    async def retrieve(self, query, top_k=5, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return [(c, .9) for c in self.initial[:top_k]]
        return [(self.available[cid], score) for cid, score in self.index.search(query, top_k)]


def method_settings(settings, method):
    return settings.model_copy(update={
        "evidence_method": method if method in {"edurag", "adaptive"} else "aspect",
        "targeted_retrieval_enabled": method != "aspect_no_retrieval",
        "claim_verification_enabled": method != "aspect_no_verification",
        "aspect_decomposition_enabled": method != "whole_question",
        "vlm_provider": "none",  # text-only protocol; separate study for vision
    })


def summarize(records):
    output = {}
    for method in dict.fromkeys(r["method"] for r in records):
        rows = [r for r in records if r["method"] == method]
        ok = [r for r in rows if r["status"] == "ok"]
        scored, correct = 0, 0
        for row in ok:
            predictions = {" ".join(a["aspect"].casefold().split()): a["status"]
                           for a in row["response"].get("evidence_map", [])}
            # Baselines that do not produce maps have no aspect-status score.
            if predictions:
                for name, expected in row["expected_aspects"].items():
                    scored += 1
                    correct += predictions.get(" ".join(name.casefold().split())) == expected
        output[method] = {
            "attempted": len(rows), "completed": len(ok), "failures": len(rows) - len(ok),
            "completion_rate": len(ok) / len(rows) if rows else None,
            "aspect_status_accuracy": correct / scored if scored else None,
            "aspect_status_denominator": scored,
            "latency_ms_completed": interval([r["elapsed_ms"] for r in ok]),
            "latency_ms_all_attempts": interval([r["elapsed_ms"] for r in rows]),
            "source_withheld": sum(r["response"].get("claim_verification", {}).get("withheld", False) for r in ok),
            "note": "Automatic aspect labels only; source-claim correctness and usefulness require independent review.",
        }
    return output


def paired_aspect_differences(records):
    """Pairs by case ID; failed/absent maps score zero for end-to-end recovery.

    Bootstrap uncertainty is grouped by source/course group rather than treating
    related cases as independent. This is a development statistic, not a claim
    of significant superiority.
    """
    import random
    results = []
    for other in ("aspect_no_retrieval", "aspect_no_verification", "whole_question"):
        left = {r["id"]: r for r in records if r["method"] == "aspect"}
        right = {r["id"]: r for r in records if r["method"] == other}
        pairs = []
        for cid in sorted(left.keys() & right.keys()):
            def score(row):
                predictions = {a["aspect"].casefold(): a["status"] for a in row.get("response", {}).get("evidence_map", [])}
                expected = row["expected_aspects"]
                return sum(predictions.get(a.casefold()) == status for a, status in expected.items()) / len(expected)
            pairs.append((left[cid]["group"], score(left[cid]) - score(right[cid])))
        if not pairs:
            continue
        groups = sorted({g for g, _ in pairs})
        rng = random.Random(42)
        estimates = []
        for _ in range(2000):
            sampled = [d for g in rng.choices(groups, k=len(groups)) for group, d in pairs if group == g]
            estimates.append(sum(sampled) / len(sampled))
        estimates.sort()
        results.append({"comparison": f"aspect minus {other}", "paired_cases": len(pairs),
                        "groups": len(groups), "mean_difference": sum(d for _, d in pairs) / len(pairs),
                        "cluster_bootstrap95": [estimates[49], estimates[1949]],
                        "note": "Small developer fixtures can give degenerate intervals; not independent evidence of generalization."})
    return results


def write_review(records, path):
    fields = ["case_id", "method", "group", "question", "reference", "answer", "evidence_json",
              "reviewer", "source_claim_support", "citation_correctness", "aspect_coverage", "usefulness", "notes"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({"case_id": row["id"], "method": row["method"], "group": row["group"],
                "question": row["query"], "reference": row["reference"],
                "answer": row.get("response", {}).get("answer", ""),
                "evidence_json": json.dumps(row.get("response", {}).get("evidence", []), ensure_ascii=False)})


def manifest(dataset_path, dataset, settings, methods):
    root = Path(__file__).resolve().parents[2]
    code_paths = sorted(set([
        *sorted((root / "app/core/generation").glob("*.py")),
        *sorted((root / "app/core/retrieval").glob("*.py")),
        *sorted((root / "app/core/attribution").glob("*.py")),
        root / "app/api/routes_ask.py", root / "app/config.py",
        root / "app/models/schemas.py", Path(__file__),
    ]))
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "dataset_version": dataset.version, "annotation_status": dataset.annotation_status,
        "python": platform.python_version(), "methods": methods,
        "models": {"answer": settings.llm_model, "provider": settings.llm_provider,
                   "answer_endpoint_sha256": hashlib.sha256(settings.llm_base_url.encode()).hexdigest(),
                   "reranker": settings.reranker_model, "embedding": settings.embedding_model},
        "settings": {method: method_settings(settings, method).model_dump(include={
            "evidence_method", "targeted_retrieval_enabled", "targeted_retrieval_max_queries",
            "targeted_retrieval_top_k", "targeted_retrieval_timeout_seconds", "evidence_check_timeout_seconds",
            "claim_verification_enabled", "claim_repair_enabled", "claim_verification_max_claims",
            "claim_verification_max_chars", "aspect_decomposition_enabled"}) for method in methods},
        "code_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
        "design": "Controlled initial contexts; targeted local BM25; no end-to-end retrieval or learning-effect claim",
    }


async def run_live(dataset, methods, settings, out):
    corpus = {p.id: Chunk(chunk_id=p.id, source_file=p.source, text=p.text,
                         chunk_length=len(p.text), type=ChunkType.text) for p in dataset.passages}
    records = []
    with tempfile.TemporaryDirectory(prefix="aspect-study-") as folder:
        for method in methods:
            config = method_settings(settings, method).model_copy(update={"kb_db_path": Path(folder) / "unused.sqlite3"})
            for case in dataset.cases:
                state = SimpleNamespace(settings=config, retrieval=ControlledRetrieval(case, corpus))
                row = {**case.model_dump(), "method": method, "status": "error"}
                started = time.perf_counter()
                try:
                    response = await asyncio.wait_for(ask(AskRequest(query=case.query, answer_scope=case.scope,
                        evidence_method=config.evidence_method, detect_contradictions=False), state), timeout=300)
                    row.update(status="ok", response=response.model_dump(mode="json"))
                except Exception as exc:
                    row["error_type"] = type(exc).__name__
                row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
                records.append(row)
                (out / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"{method}/{case.id}: {row['status']}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--check", action="store_true")
    choice.add_argument("--live", action="store_true")
    choice.add_argument("--records", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--out", type=Path, default=Path("artifacts/aspect-study"))
    args = parser.parse_args()
    if len(args.methods) != len(set(args.methods)):
        parser.error("Methods must be unique")
    dataset = StudyDataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
    if args.check:
        print(f"Valid: {len(dataset.cases)} cases, {len(dataset.passages)} passages. {dataset.annotation_status}")
        return
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use an empty output directory to preserve previous runs")
    args.out.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    if args.live and settings.llm_provider == "none":
        parser.error("Configure a local or hosted answer provider for --live")
    if args.live:
        run_manifest = manifest(args.dataset, dataset, settings, args.methods)
    else:
        manifest_path = args.records.with_name("manifest.json")
        if not manifest_path.exists():
            parser.error("Replay requires the original manifest.json beside records.json")
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if run_manifest["dataset_sha256"] != hashlib.sha256(args.dataset.read_bytes()).hexdigest():
            parser.error("Dataset hash differs from the original run")
    (args.out / "manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    (args.out / "dataset.json").write_text(args.dataset.read_text(encoding="utf-8"), encoding="utf-8")
    records = asyncio.run(run_live(dataset, args.methods, settings, args.out)) if args.live else json.loads(args.records.read_text(encoding="utf-8"))
    expected = {(case.id, method) for case in dataset.cases for method in args.methods}
    observed = [(row["id"], row["method"]) for row in records]
    if set(observed) != expected or len(observed) != len(expected):
        parser.error("Records must contain exactly every selected case/method pair")
    for row in records:
        case = next(c for c in dataset.cases if c.id == row["id"])
        if any(row.get(k) != v for k, v in case.model_dump().items()):
            parser.error("Saved records do not match this dataset version")
    report = {**run_manifest,
        "operation": "live" if args.live else "replay; recorded outputs, not current model behavior",
        "metrics": summarize(records), "paired_aspect_status": paired_aspect_differences(records)}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_review(records, args.out / "human-review.csv")
    raise SystemExit(0 if all(r["status"] == "ok" for r in records) else 1)


if __name__ == "__main__":
    main()
