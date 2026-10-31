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

### Expanded ten-question pilot (2026-10-31)

`artifacts/asqa-expanded-20261031` contains a separate run for the next ten
eligible ASQA dev questions (offset 8). It uses the full live 987-chunk SQL
knowledge base, a fixed `openai/gpt-oss-20b` answer model, both `edurag` and
`aspect` methods, and the same 20B model as RAGAS judge through the temporary
Groq key-switching proxy. All 20 HTTP requests completed; each method answered
ten questions. Two aspect responses withheld their generated source explanation.
The code fingerprint is recorded in the manifest. Subsequent non-behavioral
documentation changes are not part of that recorded fingerprint.

Post-run inspection found that the claim splitter broke `St. Louis` at `St.`
and could release a sentence fragment. The splitter was subsequently corrected
and tested against that example. The numbers below predate that fix and must
not be presented as measurements of the updated code.

Seventeen of the 20 responses were eligible for the source-section diagnostic:
eight baseline answers, seven generated aspect source sections, and two aspect
excerpt fallbacks. Two baseline responses had no returned evidence and one
aspect response had a non-source basis. All 85 eligible RAGAS scores completed
after retries; raw batch results retain prior judge errors. The judge was not
independent, the reference is the first ASQA long-answer annotation, and the
mixed-answer background was not evaluated. Passage citations and source
sections must be reviewed separately by a human.

| RAGAS metric | EduRAG baseline: 8 source answers | Aspect: 6 mixed source sections | Aspect: 1 source-only answer | Aspect: 2 excerpt fallbacks |
| --- | ---: | ---: | ---: | ---: |
| Faithfulness | 0.792 | 0.917 | 1.000 | 0.909 |
| Context precision | 0.854 | 1.000 | 1.000 | 0.500 |
| Context recall | 0.542 | 0.528 | 0.667 | 0.500 |
| Factual correctness | 0.413 | 0.372 | 0.400 | 0.200 |
| Answer relevancy | 0.910 | 0.824 | 0.991 | 0.719 |

These column means use different sets of outputs and are not paired results.
The two excerpt fallbacks are not generated explanations; the mixed-answer
scores cover the source section only. These model-judged scores are not proof
of factuality or citation correctness.

The report contains seven paired cases where both methods had eligible generated
source content. Mean aspect-minus-baseline differences for these seven cases
were +0.167 faithfulness, +0.167 context precision, +0.071 context recall,
-0.049 factual correctness and -0.063 answer relevancy. This is a tiny,
convenience sample and is not proof of a statistically significant improvement
or a comparison with the base paper. In particular, excerpt fallbacks and
unscored outputs are excluded from these paired differences and are reported
separately. The JSON summary records confidence intervals and denominators.

Mean observed request time was approximately 8.8 seconds for the baseline and
21.0 seconds for aspect in this run; these values are provider- and load-specific.
Short-answer surface recall in `summary.json` is a lexical diagnostic, not
official ASQA accuracy. Raw records, RAGAS inputs, batch checkpoints, and
`ragas-summary.json` are local artifacts and are not included in Git.

To repeat an experiment, run these from the backend repository with the
evaluation dependencies installed. Set `$arrow` to the cached ASQA dev Arrow
file and `$database` to the populated SQLite database. Use a new output
directory and start the backend with the intended fixed model first.

```powershell
uv sync --group dev --group eval
$arrow = 'PATH_TO_ASQA_DEV_ARROW'
$database = 'data/kb/knowledge_base.sqlite3'
uv run python -m app.eval.asqa_live_http --arrow $arrow --db $database --out artifacts/my-asqa-run --count 8 --methods edurag aspect
uv run python -m app.eval.asqa_ragas_records artifacts/my-asqa-run/records.json --arrow $arrow --out artifacts/my-asqa-run/ragas-input.json --source-sections
uv run --group eval python -m app.eval.ragas_eval artifacts/my-asqa-run/ragas-input.json --out artifacts/my-asqa-run/ragas.json
uv run python -m app.eval.asqa_ragas_report artifacts/my-asqa-run/ragas-input.json artifacts/my-asqa-run/ragas.json --out artifacts/my-asqa-run/ragas-summary.json
```

To retry failed model requests or judgments, repeat the respective command
with `--resume`; the runner rejects changed data, models or settings. A failed
HTTP response does not expose server-side model attempts, so its model calls
cannot be inferred. Never change `.env` keys during a run without documenting
the provider configuration and starting a new run where required.

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

The reporting command `python -m app.eval.asqa_ragas_report --help` aggregates
RAGAS batches by method and answer kind, checks each judgment against the exact
input record, and reports missing scores and exclusions. Paired differences use
only the same question with eligible generated source content from both
methods; excerpt fallbacks and missing scores are excluded. For mixed answers
only the source section is scored, so these differences do not assess general
background. A single paired case has no confidence interval. Store raw inputs
and judge outputs alongside the report for audit and human review.

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
