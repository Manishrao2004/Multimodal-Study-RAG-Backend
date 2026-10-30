# Aspect-level evidence-aware answering

## Implementation and research boundary

The `aspect` method extends the existing adaptive answer policy. It is an
implementation to evaluate, not evidence that the system outperforms EduRAG.
The ten shipped examples are developer-authored fixtures, not independently
reviewed labels or a held-out publication dataset. No live provider experiment
was run as part of implementing this feature.

The existing ingestion, BM25/FAISS, HyDE, cross-encoder, deterministic token
grounding, and chunk attribution remain the foundation. The contribution to
investigate is whether explicit aspect coverage, gap-directed retrieval, and
source-claim checks improve useful coverage without adding unsupported source
claims.

## Answer pipeline

1. Resolve the question and explicitly requested aspects with the existing
   planner. Stable IDs (`a1`, `a2`, ...) follow request order. A failed planner
   falls back to the whole question. No extra comparison criteria are invented.
2. Retrieve and bound context to eight chunks / 16,000 characters, with at most
   4,000 characters per chunk. The semantic assessment and generation use this
   same bounded text. Visual support requires a completed figure inspection.
3. Assess each requested aspect as supported, partial, missing, or conflicting.
   Supporting/conflicting quotes must exist verbatim after whitespace
   normalization in their named chunks. An exact quote establishes provenance,
   not entailment. Missing/duplicate aspect judgments invalidate a live audit.
4. Search for at most two partial/missing aspects in one round, preserving the
   full question and focusing on one gap per query. Full retrieval uses hybrid
   retrieval for these extra queries, avoiding additional HyDE calls. Each
   query has a timeout and retrieves at most three candidates. Other retrieval
   modes preserve their selected mode.
5. Deduplicate new passages and reserve slots in the bounded context for them.
   Retain initially supported passages first. Inspect figures and reassess the
   combined context. If this reassessment fails, restore the original assessed
   context and mark the attempt unavailable. Never silently attach unassessed
   passages as used evidence.
6. Generate from the aspect map. My sources answers supported portions and
   states gaps; Auto may add a separately labelled general explanation. Present
   both source positions for conflicting aspects. External/time-sensitive
   verification remains a separate unmet requirement; no web search is added.
7. Audit deterministic source sentence/line units against their cited passages
   in one model call. A unit can contain multiple claims, all of which must be
   supported. Tables retain rows; headings are included for classification.
   Numbered and bulleted lists with shared citations are audited as complete
   blocks, including introductions and explicit closing list attribution.
   Numbering is never treated as an independent claim. Citation scope does not
   extend to unrelated following prose.
   The audit cannot omit IDs. Validate citation bounds and supporting quotes,
   including quoted operands for derived arithmetic. General-knowledge text is
   not treated as verified source content.
8. Optionally repair the source section once, then audit it again. Retain the
   initial failed judgments in the response. If support remains unresolved or
   the audit fails/exceeds its budget, withhold the entire source section. A
   well-delimited mixed answer may retain its original general-knowledge
   section; never relabel failed source text as background. Duplicate section
   boundaries fail closed.

When a source explanation is withheld, the application can still show up to
four exact, provenance-validated excerpts from the aspect map. Every line is
rendered as a quotation with its source marker. No generated definition is
substituted, no failed draft is released, and no independent fact verification
is claimed. `fallback_used` distinguishes this display from a verified
generated explanation; the original `withheld` state remains true. A safe
exception class name in `failure_type` records audit failures without provider
messages or credentials. Existing saved responses are not rewritten when the
verifier changes.

If some source units already passed the audit before a later repair failed,
retain those units and omit unsupported units (`fallback_kind=supported_claims`,
with `released_claim_ids`). Revalidate quote provenance and citation bounds
before releasing the subset. For conflicting aspects prefer quoted excerpts
of both positions rather than releasing a potentially one-sided subset. If no
audited units can be retained, use `fallback_kind=source_excerpts`. Neither
fallback counts as a fully successful audit; `withheld` still records that the
full source draft was not released.

Question planning may resolve conversational references but must not append its
own clarification question or invent alternative-domain requests. Recognizable
clarification insertions are rejected before retrieval. Retrieved course topics
and validated item names anchor mixed-answer generation, so a domain already
established by the notes is not turned into an unnecessary clarification.

The source checker is model-assisted and fallible. A `supported` status is a
model judgment with validated quote provenance, not a proof of correctness.
Nonfactual classifications and arithmetic judgments also need independent
evaluation. The checker uses the configured answer model and may share its
errors. Conservative withholding can reduce usefulness or over-abstain.

## API and configuration

`AskRequest.evidence_method` optionally overrides the server default:

| Method | Policy |
| --- | --- |
| `edurag` | Strict source-grounded generation; no planner, aspect map, gap retrieval, or claim audit |
| `adaptive` | Existing Auto policy with whole-answer evidence assessment |
| `aspect` | Aspect assessment, one gap retrieval round, and source-claim audit |

Explicit General knowledge bypasses source processing. `edurag` is an
EduRAG-style baseline, not a bit-for-bit reproduction: this project retains
its modified BM25, storage, provider client, and optional conflict detector.
Set `detect_contradictions=false` in controlled comparisons, as the runner does.

The default is `EVIDENCE_METHOD=aspect`. `.env.example` lists the flags. The
most useful ablation settings are `TARGETED_RETRIEVAL_ENABLED=false`,
`CLAIM_VERIFICATION_ENABLED=false`, and `ASPECT_DECOMPOSITION_ENABLED=false`.
Changing them changes inference behavior; record them with every experiment.
Setting `EVIDENCE_METHOD=adaptive` restores the previous answer policy. The old
`app.eval.validate` runner explicitly retains that adaptive baseline.

Responses add `evidence_map`, `evidence_trace`, `claim_verification`, and
`evidence_method` while retaining existing fields. Conversation storage saves
the complete response using its existing JSON representation. Old conversations
deserialize with empty metadata defaults. The frontend provides collapsed
question coverage and source-claim check disclosures and source inspection.

Trace fields distinguish initial/final context IDs, initial status map, extra
retrieval attempts, admitted new IDs, assessment attempts, audit/repair call
attempts, and withholding. A failed reassessment clears admitted IDs but retains
raw retrieved IDs. Extra retrieval scores are not cross-encoder probabilities.
Withheld source answers do not report source-attribution/grounding scores or
claim that source files were used in the released answer. Retrieved evidence
remains inspectable.

## Budgets

At default settings there is at most one initial evidence assessment and one
reassessment, two extra retrieval queries (20 seconds each), and three claim
processing calls (audit, optional repair, re-audit). Assessment and audit calls
use `EVIDENCE_CHECK_TIMEOUT_SECONDS` with no HTTP retries. The aspect assessment
has a 6,000-token output cap. Claim calls have a 6,000-token output cap, audit
input is limited to 12,000 source-answer characters and 40 units, and excess
content is withheld rather than partially checked. These are budgets, not
measured latency or token-usage statistics. Planning, original retrieval/HyDE,
generation, figure inspection and optional conflict detection have their own
costs. The new runner is serial and puts a 300-second cap on each whole handler
attempt. Provider token usage is not currently recorded.

## Isolated comparison runner

Validate the dataset without any model calls:

```powershell
.venv/Scripts/python.exe -m app.eval.aspect_study --check
```

The runner accepts a versioned JSON dataset with passages and cases. A case
contains `id`, `group` (document/course grouping), `query`,
`initial_context_ids`, `available_context_ids`, `expected_aspects`, `reference`,
and optional `scope`. Initial evidence must be a subset of available evidence.
For missing-evidence cases remove the relevant passages from the available
set as well. Check that answers cannot be recovered from duplicate passages.
Use separate document groups and human-reviewed labels for the final dataset.

The `--live` command invokes whichever local or hosted answer provider is
configured; it is never run by `--check`. Example, after explicitly choosing
the provider and reviewing the dataset:

```powershell
.venv/Scripts/python.exe -m app.eval.aspect_study --live `
  --out artifacts/aspect-study-run-1
```

The default six configurations are `edurag`, `adaptive`, `aspect`,
`aspect_no_retrieval`, `aspect_no_verification`, and `whole_question`. Select
fewer with `--methods aspect adaptive`. Each method receives the same fixed
initial passages; extra retrieval uses local BM25 over the case's available
corpus. This isolates policy/gap-retrieval behavior, not the real FAISS/HyDE
retrieval pipeline or PDF ingestion. Run the existing retrieval benchmark
separately for those claims. Vision and automatic conflict detection are
disabled in this text-only controlled runner.

Outputs require an empty directory and include `manifest.json`, a frozen
`dataset.json`, incremental `records.json`, `summary.json`, and
`human-review.csv`. Manifests capture model identifiers, a hash of the provider
endpoint, effective policy settings, code hashes, dataset hash, and creation
time without API keys. Crashes retain their manifest and completed rows, but
this first version does not resume incomplete live runs. Every attempted row
remains a success or failure; failed calls are not silently dropped.

Replay a complete run without model calls, retaining its original manifest:

```powershell
.venv/Scripts/python.exe -m app.eval.aspect_study `
  --dataset artifacts/aspect-study-run-1/dataset.json `
  --records artifacts/aspect-study-run-1/records.json `
  --out artifacts/aspect-study-replay
```

Automatic summaries report completion/failure rates, latency over completed
and all attempts, source withholding counts, and exact-aspect-name status
accuracy with an explicit denominator. Baselines with no aspect map get null
aspect accuracy, not perfect scores. Planner wording differences may reduce
exact-name alignment; independent aspect alignment is needed for final claims.
Paired ablation differences match case IDs and use a seeded cluster bootstrap
over document/course groups. Missing/failed predictions count as zero in these
end-to-end paired aspect scores. Small development sets can give degenerate
intervals. A self-audit's acceptance is not independent claim-support accuracy.

The human-review CSV deliberately leaves reviewer and quality fields empty.
Use independent review of source claims, citations, coverage, and usefulness;
keep reviewer disagreements and adjudication. A held-out educational dataset,
independent claim evaluation, a broader end-to-end retrieval study, and any
student learning-effect study remain research work, not generated results.

## Verification performed for this implementation

Network-blocked regression tests cover map provenance, conflicts, bounded gap
retrieval, saturated context admission, failed reassessment rollback, explicit
source policy, no-targeted-retrieval/whole-question ablations, incomplete and
malformed claim audits, citation bounds, invented quotes, one repair/recheck,
mixed-section isolation, and audit budgets. Evaluation tests cover dataset
integrity, excluded evidence, missing metrics, paired comparisons, and blank
human-review exports. Frontend TypeScript, targeted ESLint, and production
build pass. Browser inspection was unavailable in the environment; no live
visual review or live model-quality measurement is claimed.
