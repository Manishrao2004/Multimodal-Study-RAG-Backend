# New pipeline validation handoff — 2026-10-30

The user asked for end-to-end validation on the populated ASQA passages, real
RAGAS scoring, and fixes for problems encountered. Work was stopped at their
request to preserve progress. Do not present this as complete or publication-ready.
The generated artifacts mentioned below are local and ignored by Git.

## What was completed

- `pytest app/tests app/eval`: **133 passed**, after the citation and fallback
  telemetry fixes. Frontend TypeScript and targeted ESLint checks passed on the
  earlier provenance-label change.
- ASQA-derived offline retrieval: 99 matched dev questions and 437 SQL passages;
  297/297 retrieval attempts completed. Results and methodological caveats are
  in `docs/ASQA_VALIDATION.md` and `artifacts/asqa-retrieval-20261030`.
- A pre-fix live run against the actual SQL-backed API completed five of six
  attempted requests across three questions; one aspect request returned 502.
  Two aspect answers withheld their generated source explanations. The planned
  eight-question run did not finish. Raw results and manifest are in
  `artifacts/asqa-live-20261030`.
- Real RAGAS pilot scoring was completed using the configured hosted
  `openai/gpt-oss-20b` judge. The baseline pilot has one eligible answer; the
  aspect pilot has one eligible *fallback excerpt*, not a generated source
  explanation. One metric hit a token-per-minute quota and succeeded on resume.
  Raw pilot input and outputs are in the live artifact directory. These tiny,
  pre-fix results cannot validate current code or establish comparative quality.
- A further post-citation-fix run produced one model-only answer and exposed a
  configured fallback HTTP 404; it was stopped after a telemetry fix changed the
  source fingerprint. Its partial records are in
  `artifacts/asqa-aspect-postfix-20261030` and should not be mixed with other runs.
- The final run with both fixes and fallback disabled was stopped at the user's
  request before its first response was saved. `artifacts/asqa-current-20261030`
  has only cases and manifest; it is not a completed run or a resumable checkpoint.
  The temporary backend used for that run was stopped.

## Changes and known limitations

- Fail-open verification no longer releases claims already found unsupported;
  unavailable checks are labelled unverified in the frontend.
- Citation normalization recognizes unambiguous `【1†L1-L2】`-style markers;
  claim support still requires valid cited passages and verbatim quotes.
- Model telemetry records fallback and now marks failed fallback as failed.
  The configured OpenRouter fallback `google/gemma-2-9b-it:free` returned HTTP
  404 in a live attempt; do not assume it is available. Primary Groq quota also
  returned HTTP 429. No configuration was changed permanently.
- The importer defaults to a separate evaluation database, and evaluation
  baselines no longer receive artificial aspect-score zeroes.
- New `app.eval.asqa_live_http` checkpoints live requests and checks code/index
  identity. `app.eval.asqa_ragas_records` converts the saved response evidence
  and the first ASQA long-answer annotation into RAGAS inputs. Its optional
  source-section mode labels excerpt fallbacks separately. Existing RAGAS
  evaluates source answers only; mixed, withheld, failed and excluded cases
  must be counted and reviewed separately, not silently discarded.
- The first 100 ASQA dev samples and their annotated passages are a convenient,
  biased subset. The oracle-derived corpus and incomplete relevance judgments
  make the retrieval probe easier than independent-corpus evaluation. RAGAS
  judges are fallible; human review and a held-out study remain necessary.

## Resume work

1. Confirm the backend is responsive and its `/config` pipeline fingerprint
   matches current source and its index matches the intended SQL database.
   The normal server at port 8000 was intermittently unresponsive. A temporary
   backend can be run on another port without disturbing it; use a fixed model
   and disable fallback for a controlled comparison, documenting all settings.
2. Start a **new** output directory with `python -m app.eval.asqa_live_http`;
   `--help` lists required Arrow and SQL paths. The cached Arrow path is
   `C:\Users\gandl\.cache\huggingface\datasets\din0s___asqa\default\0.0.0\084060f16b46f3165318f760b2339208b19a0bde\asqa-dev.arrow`;
   the current database is `data/kb/knowledge_base.sqlite3`. Once a run has
   saved records, `--resume` requires identical settings, source and data.
3. Convert completed records with `python -m app.eval.asqa_ragas_records` and
   score using `python -m app.eval.ragas_eval`; use `--resume` to retry judge
   failures, keep original and failed results, and avoid concurrent quota-heavy
   generation and judging. Check every metric's eligible count and failures.
4. Validate source-claim support, citation correctness, aspect coverage,
   abstention and mixed answers separately with independent human review, plus
   representative held-out cases and multimodal tests. Publish no claims based
   on the current pilots alone.
