# ASQA validation notes

## Retrieval probe (2026-10-30)

The experiment in `artifacts/asqa-retrieval-20261030` uses the first 100 cached
`din0s/asqa` development questions. It matches annotation and question-answer
contexts by exact text against the populated ASQA passages, then builds a
temporary ASQA-only index. One question had no matched annotated context and
was excluded; 99 questions and 437 passages remained. Each question was run
against all three methods. All 297 retrieval attempts completed.

| Configuration | MRR@5 | Recall@5 | Hit rate@5 | NDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.939 | 0.708 | 0.970 | 0.801 |
| Hybrid | 0.959 | 0.782 | 0.990 | 0.862 |
| Hybrid + reranker, without HyDE | 0.985 | 0.805 | 0.990 | 0.894 |

The JSON summary contains question-level bootstrap intervals and latency.
These are preliminary retrieval measurements, not official ASQA answer scores.
The corpus consists of annotated answer-related passages, and relevance labels
are incomplete; this makes it easier than retrieval from an independently
constructed collection and may mark other useful passages as irrelevant. The
sample is the first 100 rather than a random sample, one case was excluded, and
the full application configuration can include HyDE. Do not compare these
values directly with the base paper or claim that the differences are
statistically significant without a suitable paired analysis.

## Answer experiments

### 2026-10-31 remaining-question pilot

To avoid regenerating the three previously attempted questions, the live run
in `artifacts/asqa-remaining-20261031` covered eligible ASQA questions 4–8
only, using the current application code, the full 987-chunk SQL knowledge
base and `openai/gpt-oss-20b` for both methods. All ten requests returned HTTP
200. A temporary local proxy switched between the four provided Groq keys on
HTTP 429; it did not change the model. The earlier answers use a different
pipeline version and model configuration, so do not pool them with this run.

Nine of the ten outputs were eligible for the source-section RAGAS diagnostic;
one baseline output had no returned evidence and was excluded. All 45 metrics
for eligible outputs eventually completed, including a retry of one failed
faithfulness judgment. Raw responses, excluded cases, judge checkpoints and
`ragas-summary.json` are saved locally in that directory (artifacts are ignored
by Git). The summary separates methods and generated source sections from
fallback excerpts and reports counts, missing values and exclusions. Two
aspect outputs were excerpt fallbacks, not verified generated explanations.
The 20B judge was also the generation model, so it is not an independent judge.
The five-question convenience sample is far too small for publication claims,
and the excerpt results must not be presented as generated-answer faithfulness.
Mixed-answer background text was not evaluated by source-section RAGAS.

`python -m app.eval.asqa_live_http` evaluates the actual local `/ask` API against
the current complete knowledge base. It checks the running pipeline and index
identity, logs configured models and successful response model calls, stores
each result as it completes, and supports resuming an unchanged run. Run
`--help` for arguments. A failed HTTP request cannot provide server-side model
telemetry. To avoid changing models midway, configure a single provider and no
fallback before starting; a fallback-enabled run records fallback use but is
not a controlled single-model comparison.

`python -m app.eval.asqa_answers` instead supplies annotated contexts for a
controlled, in-process probe. It is a different experiment from live retrieval.
Both runners generate a human-review CSV. Their short-answer string matching is
a diagnostic only: it cannot determine factual correctness, grounding, citation
quality or negation, and it is not the official ASQA score. An interrupted or
failed run must not be presented as a completed evaluation. The attempt in
`artifacts/asqa-answers-20261030` could not reach the hosted model from this
environment and is incomplete.

The live experiment under `artifacts/asqa-live-20261030` was interrupted after
six attempts on three questions. It predates the line-citation normalization
fix and must not be combined with post-fix results. Two aspect responses
withheld their generated source explanations and one request returned HTTP 502.
The RAGAS pilot uses the first ASQA long-answer annotation as its reference and
the evidence returned by the API. Its `--source-sections` option evaluates the
delivered source section of a mixed answer separately; a fallback excerpt is
explicitly marked as such. Scores for excerpts do not demonstrate that the
withheld generated explanation was faithful. The judge hit a token-per-minute
limit, and the missing metric was subsequently retried. This tiny pilot is not
a representative estimate or a publishable comparison.

Before publication, use a held-out representative set, a defined retrieval
corpus and relevance judgments, the official ASQA evaluation where appropriate,
paired comparisons with fixed models and settings, and independent human review
of factuality, coverage and citations. Record failures, exclusions, fallback,
token usage and latency; report uncertainty and the limitations above.

## Import safety

`scripts/populate_asqa.py` now defaults to an isolated evaluation database under
`data/eval/asqa`, and skips placeholder QA contexts. Use `--db` explicitly to
choose a different database. Existing data is not modified by this change.
