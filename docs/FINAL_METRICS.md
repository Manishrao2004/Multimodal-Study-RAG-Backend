# Saved evaluation results: Aspect and Adaptive

This document records the final **saved pilot results**, not a completed
publication evaluation of the present configuration. The Aspect and Adaptive
results come from different experiments, datasets, code revisions and model
configurations. They do not constitute a controlled comparison with each
other. The EduRAG-style baseline in the Aspect experiment is a separate method,
not Adaptive or a verified reproduction of the original paper.

## Aspect: ASQA answer pilot

On 31 October 2026, the live `/ask` pilot evaluated ten ASQA development
questions using `aspect` and `edurag`: 20 successful requests in total. The
SQL knowledge base contained 987 chunks. The answer model was
`openai/gpt-oss-20b`, which also served as the RAGAS judge; the judge was not
independent. The first ASQA long-answer annotation was used as the reference.
This was a convenience sample, not the official ASQA answer evaluation.

The manifest recorded HyDE, targeted retrieval, claim verification and repair
as enabled, with claim-verification fail-closed disabled. It also recorded a
configured fallback. The primary/fallback model calls are recorded per
response; the presence of fallback configuration alone does not establish
that it was used. These settings differ from the current local `.env` snapshot
described in [configuration](CONFIGURATION.md).

| RAGAS metric | EduRAG-style baseline: 8 source answers | Aspect: 6 mixed source sections | Aspect: 1 source-only answer | Aspect: 2 excerpt fallbacks |
| --- | ---: | ---: | ---: | ---: |
| Faithfulness | 0.792 | 0.917 | 1.000 | 0.909 |
| Context precision | 0.854 | 1.000 | 1.000 | 0.500 |
| Context recall | 0.542 | 0.528 | 0.667 | 0.500 |
| Factual correctness | 0.413 | 0.372 | 0.400 | 0.200 |
| Answer relevancy | 0.910 | 0.824 | 0.991 | 0.719 |

All five metrics were completed for the 17 eligible outputs, giving 85
scores after retries. Two baseline responses lacked retrieved context and one
Aspect response was non-source and excluded. The groups contain different
outputs, so comparing their averages is not a paired comparison. Excerpts are
not generated explanations; mixed responses were scored only on their source
sections, not their general-knowledge sections.

For the seven questions where both methods produced eligible generated source
content, the saved report gives these paired mean differences:

| Metric | Aspect minus EduRAG-style baseline | Paired questions |
| --- | ---: | ---: |
| Faithfulness | +0.167 | 7 |
| Context precision | +0.167 | 7 |
| Context recall | +0.071 | 7 |
| Factual correctness | -0.049 | 7 |
| Answer relevancy | -0.063 | 7 |

These differences are preliminary, conditional on eligibility, and based on
a very small sample with an automatic judge. They do not establish a reliable
improvement or superiority over the base paper. Mean observed request times
were approximately 21.0 seconds for Aspect and 8.8 seconds for the baseline;
these depend on the provider and load.

This pilot predates the `St.` sentence-splitting fix and later code changes.
The three-case post-fix retest was qualitative and has no new RAGAS score. A
retrieved passage and ASQA reference disagreed on a date in one case, showing
that source agreement is not necessarily real-world correctness. See the local
`artifacts/asqa-claim-retest-20261031/RETEST_NOTES.md`.

Raw outputs, per-case scores, confidence intervals, exclusions and settings:
`artifacts/asqa-expanded-20261031/records.json`, `ragas-summary.json`,
`ragas-*.json` and `manifest.json`. They are local, Git-ignored artifacts;
archive them separately. See [ASQA validation](ASQA_VALIDATION.md).

## Adaptive: controlled developer-fixture pilot

On 25 October 2026, the prior Adaptive pipeline was tested through the then
production answer handler with **fixed contexts**, not live retrieval. The 16
developer-authored cases have not been independently reviewed. The answer
model was `openai/gpt-oss-120b`; the RAGAS judge was `openai/gpt-oss-20b`.
This is a different generator model, test set, and code revision from the
Aspect pilot above. A different model does not by itself make the judge
independent in the human-review sense.

| Measure | Saved result | Denominator / interpretation |
| --- | ---: | --- |
| Completed unique cases after retries | 16/16 | 16 developer cases |
| Routing accuracy after retries | 1.000 | 16 completed cases: 7 sources, 4 mixed, 5 model |
| Routing macro F1 after retries | 1.000 | Three represented classes, 16 cases |
| First-attempt completion and routing success | 15/16 (0.938) | 16 initial handler attempts |
| Handler-attempt completion | 16/19 (0.842) | All handler attempts, including three retries |
| Citation ID validity | 1.000 | 12 citation IDs; valid numbering is not claim support |
| Successful-request latency p50 / p95 | 16.85 s / 46.94 s | 16 successful requests |
| All-attempt latency p50 / p95 | 22.46 s / 180.01 s | 19 attempts, including failures |

The post-retry routing success has a Wilson 95% interval of approximately
0.806–1.000. Perfect results on a small development fixture set do not imply
perfect real-world behavior. First-attempt reliability and all-attempt latency
must remain visible alongside the post-retry results.

RAGAS was applicable to seven source answers from these 16 cases; mixed and
model-only cases were not scored as source answers. All five metrics were
completed for those seven answers:

| RAGAS metric | Adaptive mean | Eligible answers |
| --- | ---: | ---: |
| Faithfulness | 0.980 | 7 |
| Context precision | 1.000 | 7 |
| Context recall | 0.929 | 7 |
| Factual correctness | 0.620 | 7 |
| Answer relevancy | 0.739 | 7 |

The fixed passages and developer-authored references make this a development
diagnostic, not an independent retrieval or answer-quality benchmark. Some
references describe expected behavior, and the follow-up question's RAGAS
input omitted conversation history. Judge errors and reference limitations are
discussed in [metrics and validation](METRICS_VALIDATION.md). These scores are
not comparable to the ASQA table above.

Saved inputs and details: `artifacts/validation-final/summary.json`,
`report.md`, `records.json`, `ragas.json`, `manifest.json` and
`human_review.csv`. These artifacts are local and ignored by Git. The summary
captures base commit `8f97173251b9e27734a8c0044fd86647aa0fb402`; the manifest
records source hashes. Do not assume the commit alone captures all working-tree
changes made during the historical run.

## Shared ASQA retrieval probe

The earlier ASQA-derived retrieval probe measured retrieval configurations,
not Aspect-versus-Adaptive answer quality. It covered 99 questions and 437
annotated-context passages, with 297 completed attempts and one excluded case.

| Retrieval configuration | MRR@5 | Recall@5 | Hit rate@5 | NDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.939 | 0.708 | 0.970 | 0.801 |
| Hybrid | 0.959 | 0.782 | 0.990 | 0.862 |
| Hybrid + reranker, without HyDE | 0.985 | 0.805 | 0.990 | 0.894 |

The answer-related corpus and incomplete relevance labels make this easier
than independent-corpus retrieval. Details and limitations are in
[ASQA validation](ASQA_VALIDATION.md).

## What remains unmeasured

There is no saved same-question, same-model, same-corpus paired RAGAS comparison
of Adaptive versus Aspect. The current configuration, multimodal and study
tool paths, and student learning outcomes also lack a complete held-out,
independently reviewed evaluation. Use the saved numbers as pilot findings and
record a new controlled run before making current-version or publication claims.
