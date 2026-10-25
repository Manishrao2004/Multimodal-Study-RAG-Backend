# Study RAG Metrics and Validation

This document defines how to evaluate the study assistant for a final year major project. It covers retrieval, adaptive answering, factual support, citations, student usefulness, document viewing, reliability, and reproducibility. Use it with the machine readable run artifacts to explain what was tested and what the results establish.

The central research question is whether adaptive answering selects an appropriate evidence policy and gives a useful explanation while accurately identifying which claims come from uploaded material. A high similarity score alone cannot answer that question. Each metric below has a specific denominator and a separate interpretation.

## 1 Evaluation architecture

There are four complementary evaluation layers.

1. Deterministic regression tests verify the metric formulas, invalid inputs, missing values, duplicate retrieval IDs, and failure denominators. They establish that the scoring implementation behaves correctly.
2. Controlled adaptive tests supply fixed passages to the production `ask` handler. The configured live models perform planning, evidence assessment, generation, citation processing, and attribution. This isolates adaptive behavior from retrieval quality. The student knowledge base is not modified.
3. Retrieval evaluation uses the existing indexed corpus and benchmark. BM25, hybrid retrieval, and the full retrieval pipeline are compared on identical queries. The existing teacher generated benchmark contains one gold chunk per query.
4. Ragas and claim audits score saved live responses. Ragas is an optional dependency group, separate from the normal application. Judge errors remain visible and never become invented scores. Human review is needed to calibrate model judgments and establish trustworthy reference labels.

The controlled corpus contains five passages with synthetic filenames representing business notes, a worked example, an outline, network notes, and irrelevant gardening text. These filenames label passage origins; this suite does not parse Office files or test native document rendering. Format and viewer integration need their own tests.

## 2 Adaptive answering and answerability

The expected evidence states are strong, partial, and insufficient. Strong means the passages support all requested factual aspects. Partial means relevant source material exists but useful parts of the requested answer require additional knowledge. Insufficient means the passages cannot support a useful answer. The model only policy reports not checked because retrieval is intentionally bypassed.

For automatic answering, strong evidence normally selects sources, partial evidence selects mixed, and irrelevant or missing evidence selects model. Relevant but insufficient material may still select mixed to preserve useful provenance. Source only and model only are explicit policy overrides. Evaluate evidence sufficiency and the selected answer basis separately; they are related but are not interchangeable labels.

Simple arithmetic from supplied quantities, paraphrasing, and changing supported facts into a table or steps should not create a missing knowledge judgment. An outline that names partner types supports listing the names, but it does not support explaining their definitions. These are explicit regression cases.

| Metric | Definition | Interpretation |
| --- | --- | --- |
| Routing accuracy | Correct answer basis predictions divided by completed cases | Conditional on requests completing |
| End to end routing success | Correct completed routes divided by all attempted cases | Failed calls count against the system |
| Class precision | TP divided by TP plus FP | How often a predicted route is right |
| Class recall | TP divided by TP plus FN | How often an expected route is recovered |
| Class F1 | 2 TP divided by 2 TP plus FP plus FN | Balance of precision and recall |
| Macro F1 | Mean class F1 across classes represented in reference labels | Prevents a common class hiding poor rare class performance |
| Confusion matrix | Count for every expected and predicted class pair | Reveals systematic source versus mixed mistakes |
| Sufficiency accuracy | Correct evidence states divided by completed labeled cases | Measures assessment separately from policy overrides |

Always report class support. A missing class cannot establish that the system handles that class. Undefined precision is null when a class is never predicted; F1 can still be zero when the class has reference examples. The summary names its macro convention explicitly.

## 3 Retrieval quality

Let R be the ordered retrieved IDs and G the set of relevant reference IDs. Remove duplicate retrieved IDs before scoring. K is a positive retrieval depth. No gold relevance labels means the score is unavailable rather than zero.

| Metric | Formula | Notes |
| --- | --- | --- |
| Reciprocal rank | 1 divided by rank of first relevant result, or 0 | Only the first relevant result contributes |
| MRR at K | Mean reciprocal rank over queries, truncated at K | State retrieval depth |
| Precision at K | Relevant returned IDs divided by returned IDs in the first K | Short lists use their actual returned length in the new helper |
| Recall at K | Relevant returned IDs divided by all reference relevant IDs | Supports multiple relevant passages |
| AP at K | Sum of precision at each relevant rank divided by min of gold count and K | State this normalization when comparing tools |
| nDCG at K | DCG divided by ideal DCG with binary relevance | DCG sums relevance divided by log base 2 of rank plus 1 |

The legacy benchmark has a single gold chunk. Its Recall at 1 and Recall at 5 are consequently equivalent to hit rate for that gold chunk. They do not establish recall over every passage that could support an answer. The new multi relevant helper provides precision, recall, reciprocal rank, AP, and nDCG; running it requires independently annotated relevant ID sets.

Use the existing CLI for indexed retrieval ablation. Default arms are BM25 only, hybrid, and full. Dense only is available as a retrieval mode but is not included by default. Full combines multiple changes, so gains from hybrid to full cannot isolate the separate effects of HyDE and reranking. Add separate arms to make that causal attribution.

The teacher benchmark is useful for development but is generated from passages already known to contain answers. It underrepresents unanswerable, partial, conversational, adversarial, and multi source questions. Matching the query count of a paper does not make scores comparable across different documents, labels, models, or retrieval depths.

## 4 Response correctness and relevance

Token F1 computes multiset token overlap precision and recall against a reference answer, then their harmonic mean. ROUGE L uses the longest common token subsequence and its precision and recall harmonic mean. Both are sensitive to wording and can penalize a correct paraphrase. A wrong answer can still share many words with the reference.

BERTScore F1 uses contextual token alignment when the bert score package is available. The existing fallback is sentence embedding cosine similarity, which must be identified by its backend name and reported as a different metric. The legacy function catches import errors; other model loading or runtime errors may still fail the evaluation.

Response relevance asks whether the answer addresses the question, including all requested items. Ragas AnswerRelevancy generates three questions from each answer and compares their embeddings with the original question, including a noncommittal response check. The adapter uses the configured local embedding model. This signal is sensitive to embedding choice and is not a direct checklist of requested items. The custom instruction following rubric supplies a separate judgment of requested scope and format.

The implemented Ragas adapter runs faithfulness, context precision, context recall, factual correctness F1, and answer relevancy. Factual correctness decomposes response and reference claims and estimates agreement through entailment judgments. It depends on reference completeness: a true extra detail missing from the reference can lower the score. It does not measure real world truth independently of the supplied reference. These five metrics are a selected suite rather than every metric in the Ragas catalog.

Reference answers should enumerate essential facts, accepted variants, units, numerical tolerances, and prohibited unsupported assertions. For a profit calculation, check the actual arithmetic and currency, not just lexical similarity. An expert should adjudicate valid answers that differ from the written reference.

## 5 Claim support and hallucination

Relevancy is a cosine based metric, so the adapter accepts its mathematical range from minus one to one. The other selected Ragas metrics are constrained to zero through one. Numerical excursions within one millionth of a boundary are clamped as rounding error; nonfinite values and larger excursions are recorded as errors. An on topic wrong number can score highly for relevance while failing faithfulness, as the numeric controls demonstrate.

Ragas Faithfulness decomposes an answer into claims and judges whether the retrieved context supports those claims. Its conceptual ratio is supported answer claims divided by all answer claims. This is an LLM judged estimate, with uncertainty from claim extraction and entailment decisions. It does not establish that the uploaded source itself is factually correct.

The legacy project faithfulness function computes the maximum embedding cosine similarity between the answer and any single retrieved chunk. Reports now identify it as an answer context cosine proxy. It measures semantic proximity and can miss a changed number, an incorrect negation, or a fabricated detail about the same topic. Do not present it as claim entailment or Ragas Faithfulness.

Mixed answers intentionally include explicitly labeled general background. Scoring the entire mixed answer as if every sentence must come from a source would penalize the intended behavior. The real Ragas adapter therefore scores source grounded answers only and records the other cases as not applicable. A separate claim audit checks provenance across all answer modes.

| Claim metric | Numerator | Denominator |
| --- | --- | --- |
| Source claim support | Source attributed claims supported by passages | All source attributed claims |
| Unsupported source claim rate | Source attributed claims lacking source support | All source attributed claims |
| Claim citation completeness | Source claims with an entailing citation | All source claims requiring citation |
| Citation entailment precision | Cited source claims entailed by their cited passages | All cited source claims |

The annotation schema requires four boolean labels for each atomic claim: supported, requires citation, cited, and citation supported. Each claim also carries its text and a reason. Explicitly separated general background is excluded from the source claim denominator. This does not establish that the background is correct; reference checking or expert review must assess that separately.

No claims requiring source citations means the relevant source metrics are null. It does not mean perfect citation performance. A judge that extracts no claims is marked as an error requiring manual review. A supported citation must be both present and supported. The audit rejects contradictory labels.

## 6 Citation validity and the linked reader

Audit version 2 additionally requires the numeric answer section ID and an exact quote from that section. The application identifies source and general knowledge sections from its own explicit output headings. A claim whose citation requirement conflicts with that section is rejected and reasked. A missing mixed answer boundary is an error. These checks establish provenance consistency, not factual correctness of a judge verdict.

The application deduplicates repeated marker numbers within each answer. In the new micro validity score, the denominator is therefore distinct answer and marker pairs, rather than all repeated textual occurrences. The passage IDs can be valid while their claims are unsupported.

Citation ID validity checks whether a citation points to a real retrieved chunk. Micro validity pools valid distinct answer-marker pairs and divides by all distinct answer-marker pairs. The existing generation evaluator instead averages per answer validity over cited answers. These are different weighting conventions and should be labeled.

An existing ID can point to an irrelevant passage. Citation entailment checks the actual meaning of the cited passage. Citation completeness checks whether source dependent claims have citations. All three are needed.

The legacy Citation Coverage field is retained for compatibility and now includes its definition: fraction of successful answers containing any citation marker. An answer with ten factual claims and one citation can score one on that field. It is not claim citation completeness.

The reader should be evaluated independently with PDF, PPTX, DOCX, Markdown, HTML, and plain text examples as supported by the application. Required checks include correct source identity, correct page or slide locator, first and last page navigation, source unavailable behavior, no automatic download, compact rendering, full document route, zoom behavior, keyboard navigation, and narrow viewport layout. Native Office conversion should preserve tables, diagrams, and formatting. Text extraction is a fallback rather than proof of visual fidelity.

Suggested viewer metrics are source selection accuracy, locator accuracy, successful page rendering rate, time to first page, and navigation completion rate. Record sample counts per format and conversion backend. These metrics are not produced by the controlled answer suite.

## 7 Ragas context precision and recall

Ragas context precision with a reference judges whether retrieved passages are useful for answering the question and reference, with a rank sensitive average precision calculation. This differs from simple ID precision and requires judge calls. Context recall estimates how much reference answer information the retrieved context supports. It differs from ID recall because it judges information rather than matching chunk identifiers.

The optional adapter uses the collections API in the locked Ragas version and saves per case metric values, failures, eligible counts, and the judge model. No embedding or string overlap score is silently substituted when a Ragas call fails. Failed measurements are absent from the metric mean and remain counted as failures. Read each metric n before interpreting the mean.

Configure an independent judge model using EVAL_MODEL, EVAL_BASE_URL, and EVAL_API_KEY environment variables. Defaults use the application answer model and explicitly disclose that model reuse. Independent model choice reduces one source of correlated error but still requires expert calibration.

The claim audit uses JSON object output and an explicit EVAL_MAX_TOKENS budget (8192 by default); the Ragas adapter defaults to 4096. The claim audit rejects a response whose finish reason reports token exhaustion. These settings are saved and checked on resume. Changing them requires a fresh output file. Earlier audits that relied on provider defaults are retained separately from the budgeted diagnostic runs, which were stopped to respect the free tier constraint.

## 8 Student usefulness and regional context

The custom audit uses five anchored criteria. Each is scored zero for poor, one for partial, and two for satisfactory; reported normalized means divide by two.

| Criterion | Satisfactory behavior | Failure example |
| --- | --- | --- |
| Clarity | Direct explanation understandable to the intended student | Dense or vague explanation that obscures the answer |
| Jargon explained | Necessary unfamiliar terms receive a short explanation | Unexplained technical vocabulary blocks understanding |
| Instruction following | Requested format, scope, items, and brevity are followed | Essay when a short table or two sentences was requested |
| Regional appropriateness | India and INR for relevant examples unless explicitly overridden | Defaulting to USD for an Indian business example |
| Verification caution | Current or authoritative facts are qualified when unverified | Invented current registration fees or exchange rates |

A criterion that needs no special action can be satisfactory, such as regional appropriateness on a networking question. This avoids rewarding needless country mentions or disclaimers. Record the scoring reasons and review disagreements.

The request clock provides the current date and timezone, not evidence for current facts. The default approximate location is India; it does not establish an exact city. Explicit country or currency instructions take priority. Add separate cases for date rollover, timezone differences, user overrides, current fees, and stable concepts. This is necessary before claiming broad temporal or regional validation.

## 9 Reliability latency and cost

Dataset version 2 adds a current fee request, a default INR example, an explicit USD override, and a conversational comparison follow up. It rejects numeric tax percentages in the conceptual company comparison fixture. Currency checks inspect labels rather than exchange rate correctness. The verification check inspects the planner flag rather than whether authoritative facts were obtained. Date rollover and live external verification remain outside this suite.

Completion rate is successful requests divided by all attempted requests. A failed provider call is a system failure even if the other answers are excellent. Store exception types without credentials or provider response bodies.

Latency in the controlled suite measures wall clock time around the production handler. It includes generation and attribution but excludes process startup and index building. Report p50 and p95 with the nearest rank convention and the sample count. Provider rate limiting, retries, machine load, cold model loading, and network conditions affect the result. Twelve observations do not support a production service latency claim.

The current adapter does not measure provider tokens, monetary cost, concurrent throughput, or frontend rendering time. Add request usage telemetry and the actual model tariff at evaluation time before reporting cost. Do not infer token counts from character counts. Separate ingestion time, conversion time, retrieval time, time to first token, complete answer latency, and viewer latency when profiling the entire product.

## 10 Dataset construction and human validation

Adaptive validation also accepts --live --resume for failed handler attempts. It preserves every completed response, including incorrect answers, and rejects changes to the dataset, production code, or captured model configuration. The summary reports unique case completion after retries, total handler attempts, and first attempt completion and routing success separately. A timed out first attempt remains visible in retry history and first attempt denominators. Internal provider HTTP retries are a separate unmeasured count.

The summary separates latency of successful requests from latency across all handler attempts, including timeouts and retry history. A fast successful answer mean must not conceal a long failed request. A resumed command can succeed after all cases complete while its first attempt availability and total attempt completion rate remain below one.

The shipped controlled fixtures are developer authored regression cases awaiting independent human review. They are designed to expose specific mistakes and cannot serve as an independent test set after the prompts are tuned on them.

For a defensible project study, collect a separate held out set with at least 90 questions as an initial project target, including approximately 30 fully supported, 30 partially supported, and 30 unsupported requests. This is a practical study plan, not a universal minimum or a guarantee of statistical power. Include multiple subjects, document formats, paraphrases, calculations, tables, diagrams, cross document questions, follow ups, misspellings, and irrelevant context. Keep audio outside this text focused study.

Use two reviewers with subject knowledge. Each independently labels reference sufficiency, expected answer policy, relevant chunk IDs, reference facts, and claim citation requirements. Reviewers should inspect the actual source pages. Resolve disagreements with written reasons. Calculate agreement before adjudication using accuracy and Cohen kappa for nominal labels; for ordered rubric scores, consider weighted kappa with the weighting declared. Preserve original annotations rather than only the final consensus.

Split development and final evaluation by source document or course so near duplicate passages cannot leak across sets. Freeze test questions, corpus checksums, and references before final evaluation. Do not tune thresholds or rewrite expected labels after seeing final results. If a label is wrong, version the dataset and disclose the correction. Judge calibration should use a separately reviewed subset containing subtle numeric errors, negations, irrelevant valid citations, and well labeled background.

## 11 Uncertainty ablation and acceptance

For end to end routing success, the summary also reports a Wilson 95 percent binomial confidence interval. Unlike resampling an all pass sample, this interval retains uncertainty: 12 successes in 12 trials has a lower confidence bound of about 0.76. The calculation assumes independent Bernoulli trials and does not resolve dataset bias or dependence between related questions.

The new summaries include a seeded percentile bootstrap interval for a mean using 2000 resamples. Empty input returns null. Identical small samples can produce a degenerate interval, which does not prove certainty about future questions. The current interval treats cases as independent; for questions sharing a document or student, use a cluster bootstrap in a larger study.

The existing significance helper implements a paired two sided Wilcoxon test on per query reciprocal ranks. Preserve pairing across ablation arms and report the paired mean difference as well as p values. Small sample size, ties, dependent queries, and assumptions about paired differences limit interpretation. Correct for multiple planned comparisons, for example with Holm adjustment. A statistically detectable improvement can still be too small to matter educationally.

Suggested project acceptance targets, to be agreed before held out evaluation, are routing accuracy at least 0.90, macro F1 at least 0.85, source claim support at least 0.95, claim citation completeness at least 0.90, and zero unresolved critical fabricated source claims. These are proposed engineering targets, not Ragas standards or measured achievements. Report confidence intervals and failures even when means exceed targets. Do not combine unrelated metrics into a single unexplained quality percentage.

Mechanical tests should pass before accepting a scoring change. Live model runs should have complete per case traces and disclosed missing judgments. Final project claims additionally need independent labels and reviewed judge accuracy. Good scores on a small development fixture set cannot establish perfect behavior.

## 12 Reproduction and result artifacts

Run these commands from the backend repository using the checked in lockfile. Live commands use configured providers and incur their normal usage.

```powershell
uv sync --group dev --group eval
uv run python -m pytest app/eval/test_validation.py app/eval/test_judge_eval.py -q
uv run python -m app.eval.validate --live --out artifacts/validation
uv run --group eval python -m app.eval.ragas_eval artifacts/validation/records.json
uv run python -m app.eval.judge_eval artifacts/validation/records.json
uv run python -m app.eval.validate --records artifacts/validation/records.json --out artifacts/validation
uv run python -m app.eval.cli run --skip-generation --report artifacts/retrieval.md --json artifacts/retrieval.json
```

The controlled run exits unsuccessfully if a call fails or its route differs from the expected label. This makes routing regressions visible to automation. Replaying saved records recomputes summaries without generating answers again. It requires every dataset case in order. The optional Ragas and claim audit commands exit unsuccessfully on missing eligible results or judge errors.

Both judge commands accept --resume to retain completed judgments and retry failed or missing measurements for the same records and judge configuration. Input hashes prevent mixing results from different answer runs. The Ragas adapter checkpoints each metric. The claim audit reasks at most twice when JSON or annotation consistency is invalid, records repair counts, and never coerces a contradictory verdict into a success. Earlier failed audit rows remain in the saved first attempt artifact for the recorded experiment. Run the two judge commands sequentially when they share a provider quota.

Records include every answer, reference, context list, expected route, actual basis, evidence state, citation mapping, elapsed time, and error status. Summary metadata includes UTC time, code commit, Python version, dataset SHA256, model identifiers, design, seed, and annotation status. The code commit identifies the base revision; working tree edits should also be recorded when reporting an uncommitted experiment. Never include secret API keys in artifacts.

The checked in evaluation regression tests live under app eval because the existing repository ignores the legacy tests directory. A GitHub Actions workflow runs the deterministic metric contract tests without model credentials. This workflow definition is supplied locally; its presence does not imply a remote CI run occurred.

## 13 Evidence limits and remaining study work

Judge checkpoints use per answer hashes and an immutable case manifest. If a failed handler call later produces its first answer, resume can score that new answer while reusing judgments for unchanged completed answers. Changing a successful answer, reference label, question, or judge configuration requires a fresh output. This prevents a score from being silently attached to a different response.

Controlled route tests bypass retrieval with fixed passages and disable contradiction detection. They exercise the actual answer handler but do not exercise HTTP middleware, browser context headers, upload parsing, vision captioning, document conversion, or the frontend. Each needs a targeted integration study. Two successful diagram examples from development are demonstrations rather than a statistically meaningful vision benchmark.

The existing indexed benchmark evaluates retrieval and strict source generation, while the controlled suite evaluates adaptive routing and generation. Keep their result tables separate. An independent student study is needed to claim improved learning, quiz performance, or usability. Measure learning with a planned pretest and posttest design and appropriate comparison conditions rather than inferring it from answer quality scores.

## 14 Development findings and references

### Development findings about judge reliability

The first custom claim audit incorrectly classified several claims from the explicitly labeled general knowledge section as source claims. Its pooled source support result is therefore unsuitable for a final quality claim. Version 2 adds section IDs, verbatim answer quotes, and consistency checks. Comparing the two protocols would measure a scoring change as well as any answer change.

In the saved twelve case experiment, the independent 20B Ragas judge assigned factual correctness zero to a profit answer that correctly stated 50000 minus 35000 equals 15000, while assigning faithfulness one. The discrepancy is retained in the raw results. Its cause has not been established. This demonstrates why a judge score requires calibration and independent adjudication; neither the high nor the low number should be accepted automatically. The simple positive and wrong number faithfulness controls passed, but they do not validate every metric or every claim type.

The expanded experiment also contains a conversational follow-up, "Put that comparison in a small table." The production handler receives conversation history, but the current Ragas adapter supplies only that latest query as user_input. Its relevance score therefore does not evaluate the complete conversational request. References in this development set include expected-behavior descriptions as well as factual answers. Reference-based scores need independent review for reference adequacy. Do not interpret the follow-up relevance or factual correctness scores as a validated measure of conversational quality. A future conversational study should freeze independently reviewed standalone questions and factual references before collecting new scores.

The conceptual company comparison also volunteered unrequested tax details during development. Generation instructions now keep ordinary comparisons conceptual and require authoritative verification for requested current legal or financial details even when the planner fails to flag them. The expanded regression suite checks this behavior. These are observed development findings rather than guarantees about all future questions.

Ragas metric catalog: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/

Ragas Faithfulness definition and collections API: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/

Ragas Context Precision variants and ID based distinction: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_precision/

Ragas Context Recall variants: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_recall/

The implementation and lockfile are the authority for the exact package versions used in a saved run. Documentation pages can change, so retain the run version and metric names alongside the exported scores.
