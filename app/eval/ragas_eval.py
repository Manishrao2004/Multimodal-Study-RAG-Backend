"""Optional real Ragas scoring of captured responses. No heuristic substitutes.

EVAL_MODEL / EVAL_BASE_URL / EVAL_API_KEY select a separate judge. By default
use project configuration, and disclose if the judge equals the answer model.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import hashlib
import json
import os
import logging
from pathlib import Path

from app.eval.validation_metrics import interval, bounded_score
from app.eval.checkpoints import case_manifest, resume_rows, row_metadata
from app.eval.provider_errors import error_metadata


async def evaluate(records: list[dict], output: Path, resume: bool = False) -> dict:
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    # Provider bodies can contain account identifiers. Store exception types
    # ourselves instead of letting dependency loggers print response bodies.
    logging.getLogger("instructor").setLevel(logging.CRITICAL)
    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness, FactualCorrectness
    from app.config import get_settings

    settings = get_settings()
    model = os.getenv("EVAL_MODEL", settings.llm_model)
    client = AsyncOpenAI(api_key=os.getenv("EVAL_API_KEY", settings.llm_api_key) or "local",
                         base_url=os.getenv("EVAL_BASE_URL", settings.llm_base_url),
                         timeout=90, max_retries=int(os.getenv("EVAL_PROVIDER_RETRIES", "2")))
    max_tokens = int(os.getenv("EVAL_MAX_TOKENS", "4096"))
    if not 256 <= max_tokens <= 32768:
        raise ValueError("EVAL_MAX_TOKENS must be between 256 and 32768")
    # Reasoning models can exhaust the library's 1024-token default before
    # producing the structured claim/entailment output.
    llm = llm_factory(model, client=client, max_tokens=max_tokens, temperature=0.0)
    embeddings = HuggingFaceEmbeddings(model=settings.embedding_model)
    scorers = {"faithfulness": Faithfulness(llm=llm),
               "context_precision": ContextPrecision(llm=llm),
               "context_recall": ContextRecall(llm=llm),
               "factual_correctness": FactualCorrectness(llm=llm),
               "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=3)}
    payload = {"ragas_version": importlib.metadata.version("ragas"), "judge_model": model,
               "records_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
               "same_model_as_generator": model == settings.llm_model,
               "embedding_model": settings.embedding_model, "answer_relevancy_strictness": 3,
               "judge_max_tokens": max_tokens, "temperature": 0.0,
               "case_manifest": case_manifest(records),
               "scope": "source-grounded answers only; mixed/background answers require separate claim annotation",
               "rows": []}
    previous = {}
    if resume and output.exists():
        stored = json.loads(output.read_text(encoding="utf-8"))
        previous, recovered = resume_rows(stored, records, payload,
                                          ("judge_model", "ragas_version", "embedding_model", "judge_max_tokens", "temperature"))
        payload["recovered_handler_cases"] = recovered
        payload["previous_records_sha256"] = stored.get("records_sha256")
    try:
        for row in records:
            if row.get("status") != "ok" or row["response"]["answer_basis"] != "sources":
                payload["rows"].append({"id": row["id"], "status": "not_applicable", **row_metadata(row)})
                continue
            result = previous.get(row["id"], {"id": row["id"], "status": "ok", "scores": {}, "errors": {}})
            result["status"] = "ok"
            result.update(row_metadata(row))
            payload["rows"].append(result)
            for name, scorer in scorers.items():
                if name in result["scores"]:
                    stale_detail = result.get("error_details", {}).pop(name, None)
                    if stale_detail:
                        result.setdefault("previous_errors", []).append({"metric": name, "details": stale_detail})
                    continue
                if name in result["errors"]:
                    result.setdefault("previous_errors", []).append({
                        "metric": name, "error_type": result["errors"].pop(name),
                        "details": result.get("error_details", {}).pop(name, None)})
                kwargs = {"user_input": row["query"], "retrieved_contexts": row["retrieved_contexts"]}
                kwargs["response" if name == "faithfulness" else "reference"] = (
                    row["response"]["answer"] if name == "faithfulness" else row["reference"])
                if name == "factual_correctness":
                    kwargs = {"response": row["response"]["answer"], "reference": row["reference"]}
                elif name == "answer_relevancy":
                    kwargs = {"user_input": row["query"], "response": row["response"]["answer"]}
                try:
                    score = await asyncio.wait_for(scorer.ascore(**kwargs), timeout=180)
                    value = bounded_score(score.value, -1. if name == "answer_relevancy" else 0.)
                    result["scores"][name] = value
                except Exception as exc:
                    result["errors"][name] = type(exc).__name__
                    result.setdefault("error_details", {})[name] = error_metadata(exc)
                    result["status"] = "error"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            expected = {name: row[f"expected_{name}"] for name in scorers if f"expected_{name}" in row}
            if expected:
                result["control_passed"] = all(name in result["scores"] and abs(result["scores"][name] - value) <= .05 for name, value in expected.items())
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Ragas {row['id']}: {result['scores']} errors={result['errors']}", flush=True)
    finally:
        await client.close()
    payload["summary"] = {name: interval([r["scores"][name] for r in payload["rows"] if name in r.get("scores", {})]) for name in scorers}
    payload["failed_rows"] = sum(r["status"] == "error" for r in payload["rows"])
    payload["eligible_rows"] = sum(r["status"] != "not_applicable" for r in payload["rows"])
    payload["failed_controls"] = sum(not r["control_passed"] for r in payload["rows"] if "control_passed" in r)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/validation/ragas.json"))
    parser.add_argument("--resume", action="store_true", help="Retry only missing or failed metrics for identical inputs and judge")
    args = parser.parse_args()
    result = asyncio.run(evaluate(json.loads(args.records.read_text(encoding="utf-8")), args.out, args.resume))
    raise SystemExit(1 if result["failed_rows"] or result["failed_controls"] or not result["eligible_rows"] else 0)


if __name__ == "__main__":
    main()
