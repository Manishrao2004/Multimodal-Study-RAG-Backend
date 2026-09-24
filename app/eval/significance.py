"""Paired statistical comparison between two retrieval/generation
configurations evaluated on the same benchmark queries — turns the ablation
table (plan Sec. 8 / EduRAG Table 3) from three point estimates into a claim
that the differences are actually distinguishable from noise, which matters
once the benchmark is small (a handful of dozens of queries, not thousands).

Wilcoxon signed-rank, not a paired t-test: per-query metric values (recall,
reciprocal rank, grounding ratio) are bounded in [0, 1] and rarely
approximately normal, and benchmark sizes here are small enough that a
normality assumption is not safe. Ties (queries scoring identically under
both configurations) reduce the test's power and are reported explicitly
rather than hidden in a single p-value.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class PairedTestResult:
    metric: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    test: str
    statistic: float | None
    p_value: float | None
    significant_at_0_05: bool | None
    note: str


def paired_wilcoxon(metric: str, values_a: list[float], values_b: list[float]) -> PairedTestResult:
    """Compares `values_b` against `values_a` (b - a is the reported
    direction) on the same queries, in the same order."""
    n = len(values_a)
    if n != len(values_b):
        raise ValueError(f"Paired samples must be the same length: {n} vs {len(values_b)}.")

    mean_a = sum(values_a) / n if n else 0.0
    mean_b = sum(values_b) / n if n else 0.0
    diffs = [b - a for a, b in zip(values_a, values_b)]
    nonzero = [d for d in diffs if d != 0]

    if not nonzero:
        return PairedTestResult(
            metric=metric,
            n=n,
            mean_a=mean_a,
            mean_b=mean_b,
            mean_diff=mean_b - mean_a,
            test="wilcoxon_signed_rank",
            statistic=None,
            p_value=None,
            significant_at_0_05=None,
            note="All paired differences are zero — Wilcoxon is undefined; no measurable difference.",
        )

    from scipy.stats import wilcoxon

    try:
        statistic, p_value = wilcoxon(values_a, values_b)
    except ValueError as exc:
        return PairedTestResult(
            metric=metric,
            n=n,
            mean_a=mean_a,
            mean_b=mean_b,
            mean_diff=mean_b - mean_a,
            test="wilcoxon_signed_rank",
            statistic=None,
            p_value=None,
            significant_at_0_05=None,
            note=f"Wilcoxon test could not be computed: {exc}",
        )

    return PairedTestResult(
        metric=metric,
        n=n,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=mean_b - mean_a,
        test="wilcoxon_signed_rank",
        statistic=float(statistic),
        p_value=float(p_value),
        significant_at_0_05=bool(p_value < 0.05),
        note=(
            f"n={n}; {len(nonzero)} of {n} pairs differ. Small-n results are a "
            "directional signal, not a strong significance claim."
        ),
    )


def bootstrap_ci(
    values: list[float], n_resamples: int = 10_000, seed: int = 42
) -> tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high): a 95% percentile bootstrap CI, which
    makes no distributional assumption — appropriate for the same bounded,
    non-normal per-query metrics `paired_wilcoxon` is used for."""
    if not values:
        return 0.0, 0.0, 0.0

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    return sum(values) / n, lo, hi
