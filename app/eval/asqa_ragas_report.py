"""Summarize ASQA RAGAS batches without mixing methods or answer kinds."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from app.eval.validation_metrics import interval
from app.eval.checkpoints import digest

METRICS = ("faithfulness", "context_precision", "context_recall", "factual_correctness", "answer_relevancy")


def metric_summary(measured, metric):
    values = [r["scores"][metric] for r in measured if metric in r.get("scores", {})]
    result = interval(values)
    if len(values) < 2:
        result["ci95"] = None
    result["missing"] = len(measured) - len(values)
    return result


def summarize(inputs, outputs):
    cases = {row["id"]: row for row in inputs}
    if len(cases) != len(inputs):
        raise ValueError("Duplicate input IDs")
    pairs = [(row["sample_id"], row["method"]) for row in inputs]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate sample/method pairs")
    judge_fields = ("ragas_version", "judge_model", "embedding_model", "judge_max_tokens", "temperature", "same_model_as_generator")
    if outputs:
        judge = tuple(outputs[0].get(field) for field in judge_fields)
        if any(tuple(batch.get(field) for field in judge_fields) != judge for batch in outputs[1:]):
            raise ValueError("Cannot combine different judge configurations")
    judged = {}
    for batch in outputs:
        for row in batch["rows"]:
            if row["id"] not in cases or row["id"] in judged:
                raise ValueError("Unknown or duplicate judged ID")
            if row.get("record_sha256") != digest(cases[row["id"]]):
                raise ValueError("Judgment does not match the input answer and reference")
            judged[row["id"]] = row
    groups = defaultdict(list)
    for case in inputs:
        groups[(case["method"], case["evaluated_answer_kind"])].append(case)
    report = {}
    for (method, kind), rows in groups.items():
        eligible = [r for r in rows if r["status"] == "ok"]
        measured = [judged.get(r["id"], {}) for r in eligible]
        report[f"{method}:{kind}"] = {
            "total": len(rows), "eligible": len(eligible), "excluded": len(rows) - len(eligible),
            "exclusion_reasons": dict(Counter(r.get("exclusion_reason") or "unknown" for r in rows if r["status"] != "ok")),
            "metrics": {m: metric_summary(measured, m) for m in METRICS},
        }
    paired = {}
    for metric in METRICS:
        by_sample = defaultdict(dict)
        for case in inputs:
            if case["status"] != "ok" or case["evaluated_answer_kind"] == "source_excerpts":
                continue
            if case["method"] not in {"edurag", "aspect"}:
                continue
            score = judged.get(case["id"], {}).get("scores", {}).get(metric)
            if score is not None:
                by_sample[case["sample_id"]][case["method"]] = score
        differences = [v["aspect"] - v["edurag"] for v in by_sample.values()
                       if "aspect" in v and "edurag" in v]
        paired[metric] = interval(differences)
        if len(differences) < 2:
            paired[metric]["ci95"] = None
    return {"groups": report, "paired_aspect_minus_edurag": paired,
            "note": "Judge scores are not proof of factuality; source excerpts are scored separately. Excluded and missing metrics are not included in means. Paired differences use only eligible generated source sections for both methods; mixed answers contribute only their source section."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("outputs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.inputs.read_text(encoding="utf-8")),
                       [json.loads(p.read_text(encoding="utf-8")) for p in args.outputs])
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
