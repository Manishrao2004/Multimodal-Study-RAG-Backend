"""Auditable metrics for adaptive answers. Missing observations are never perfect scores."""
from __future__ import annotations

import math
import random
from collections import Counter


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def bounded_score(value: float, minimum: float = 0., maximum: float = 1.) -> float:
    value = float(value)
    if not math.isfinite(value) or value < minimum - 1e-6 or value > maximum + 1e-6:
        raise ValueError("Metric is outside its finite mathematical range")
    return max(minimum, min(maximum, value))


def wilson_interval(successes: int, attempted: int) -> dict:
    """95% Wilson binomial interval, including all-pass and all-fail samples."""
    if attempted < 0 or not 0 <= successes <= attempted:
        raise ValueError("Invalid binomial counts")
    if attempted == 0:
        return {"n": 0, "proportion": None, "ci95": None}
    z = 1.959963984540054
    p = successes / attempted
    denominator = 1 + z * z / attempted
    center = (p + z * z / (2 * attempted)) / denominator
    margin = z * math.sqrt(p * (1 - p) / attempted + z * z / (4 * attempted ** 2)) / denominator
    return {"n": attempted, "proportion": p, "ci95": [max(0., center - margin), min(1., center + margin)]}


def classification(expected: list[str], predicted: list[str], labels: list[str]) -> dict:
    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted labels must have equal lengths")
    if any(label not in labels for label in expected + predicted):
        raise ValueError("Unknown classification label")
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for a, b in zip(expected, predicted):
        matrix[a][b] += 1
    scores = {}
    for label in labels:
        tp = matrix[label][label]
        actual = sum(matrix[label].values())
        guessed = sum(matrix[a][label] for a in labels)
        scores[label] = {
            "support": actual, "precision": ratio(tp, guessed),
            "recall": ratio(tp, actual), "f1": ratio(2 * tp, actual + guessed),
        }
    represented = [v["f1"] or 0.0 for v in scores.values() if v["support"]]
    return {"n": len(expected), "accuracy": ratio(sum(a == b for a, b in zip(expected, predicted)), len(expected)),
            "macro_f1_present_classes": sum(represented) / len(represented) if represented else None,
            "per_class": scores, "confusion_matrix": matrix}


def retrieval_metrics(retrieved: list[str], relevant: list[str], k: int = 5) -> dict:
    if k < 1:
        raise ValueError("k must be positive")
    # Duplicate IDs cannot increase precision, recall or gain.
    ranked = list(dict.fromkeys(retrieved))[:k]
    gold = set(relevant)
    if not gold:
        return {"precision": None, "recall": None, "rr": None, "ndcg": None, "ap": None}
    hits = [int(item in gold) for item in ranked]
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    ap = sum(sum(hits[:i + 1]) / (i + 1) for i, hit in enumerate(hits) if hit)
    return {"precision": sum(hits) / len(ranked) if ranked else 0.0,
            "recall": sum(hits) / len(gold),
            "rr": next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0),
            "ndcg": dcg / ideal, "ap": ap / min(len(gold), k)}


def interval(values: list[float], seed: int = 42, resamples: int = 2000) -> dict:
    if resamples < 1 or any(not math.isfinite(v) for v in values):
        raise ValueError("Finite values and positive resamples required")
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(resamples))
    return {"n": len(values), "mean": sum(values) / len(values),
            "ci95": [means[int(.025 * (resamples - 1))], means[int(.975 * (resamples - 1))]]}


def claim_metrics(claims: list[dict]) -> dict:
    """Human or judge annotations; cited IDs existing does not establish support.

    Each claim has supported, requires_citation, cited, citation_supported.
    General background must be explicitly marked requires_citation=False.
    """
    for claim in claims:
        if any(type(claim.get(key)) is not bool for key in
               ("supported", "requires_citation", "cited", "citation_supported")):
            raise ValueError("Claims require four boolean annotations")
        if claim["citation_supported"] and not claim["cited"]:
            raise ValueError("An uncited claim cannot have a supported citation")
    required = [c for c in claims if c["requires_citation"]]
    cited = [c for c in required if c["cited"]]
    return {"claims": len(claims), "source_claims": len(required),
            "source_claim_support": ratio(sum(c["supported"] for c in required), len(required)),
            "unsupported_source_claim_rate": ratio(sum(not c["supported"] for c in required), len(required)),
            "claim_citation_completeness": ratio(sum(c["cited"] and c["citation_supported"] for c in required), len(required)),
            "citation_entailment_precision": ratio(sum(c["citation_supported"] for c in cited), len(cited))}


def summarize(records: list[dict]) -> dict:
    completed = [r for r in records if r.get("status") == "ok"]
    labels = ["sources", "mixed", "model"]
    routes = classification([r["expected_basis"] for r in completed],
                            [r["response"]["answer_basis"] for r in completed], labels)
    sufficiency_rows = [r for r in completed if r.get("expected_sufficiency")]
    sufficiency = classification([r["expected_sufficiency"] for r in sufficiency_rows],
                                [r["response"]["evidence_sufficiency"] for r in sufficiency_rows],
                                ["strong", "partial", "insufficient", "not_checked"])
    citations = [c for r in completed for c in (r["response"].get("citations") or [])]
    latency = sorted(r["elapsed_ms"] for r in completed)
    strata = Counter(r["expected_basis"] for r in records)
    checks = [passed for r in completed for passed in r.get("checks", {}).values()]
    retries = sum(len(r.get("previous_attempts", [])) for r in records)
    first_completed = [r for r in completed if not r.get("previous_attempts")]
    all_latency = sorted([r["elapsed_ms"] for r in records if "elapsed_ms" in r] +
                         [a["elapsed_ms"] for r in records for a in r.get("previous_attempts", []) if "elapsed_ms" in a])
    return {"attempted": len(records), "completed": len(completed), "failures": len(records) - len(completed),
            "completion_rate": ratio(len(completed), len(records)),
            "handler_attempts": len(records) + retries, "retry_attempts": retries,
            "handler_attempt_completion_rate": ratio(len(completed), len(records) + retries),
            "first_handler_attempt_completion_rate": ratio(len(first_completed), len(records)),
            "first_handler_attempt_routing_success": ratio(sum(r["expected_basis"] == r["response"]["answer_basis"] for r in first_completed), len(records)),
            "first_handler_routing_wilson95": wilson_interval(sum(r["expected_basis"] == r["response"]["answer_basis"] for r in first_completed), len(records)),
            "end_to_end_routing_success": ratio(sum(r["expected_basis"] == r["response"]["answer_basis"] for r in completed), len(records)),
            "routing": routes, "sufficiency": sufficiency, "expected_route_counts": dict(strata),
            "routing_accuracy_interval": interval([float(r["expected_basis"] == r["response"]["answer_basis"]) for r in completed]),
            "end_to_end_routing_wilson95": wilson_interval(sum(r["expected_basis"] == r["response"]["answer_basis"] for r in completed), len(records)),
            "citation_id_validity_micro": ratio(sum(c["valid"] for c in citations), len(citations)),
            "citation_count": len(citations),
            "constraint_checks": len(checks), "constraint_failures": sum(not passed for passed in checks),
            "constraint_pass_rate": ratio(sum(checks), len(checks)),
            "latency_ms": {"n": len(latency), "p50": latency[math.ceil(.5 * len(latency)) - 1] if latency else None,
                           "p95": latency[math.ceil(.95 * len(latency)) - 1] if latency else None},
            "handler_attempt_latency_ms": {"n": len(all_latency),
                                           "p50": all_latency[math.ceil(.5 * len(all_latency)) - 1] if all_latency else None,
                                           "p95": all_latency[math.ceil(.95 * len(all_latency)) - 1] if all_latency else None},
            "claim_support": "not measured until annotations or Ragas judges run"}
