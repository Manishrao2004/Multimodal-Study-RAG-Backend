# Project status and research evidence

Verity is a local multimodal study assistant. The current text-question path
uses aspect-aware evidence assessment and source-claim checking when source
evidence is used. Study tools and screenshot questions have separate paths and
have not been evaluated as if they used the same claim audit. See the root
README for setup, modes, storage and limitations.

## Engineering verification

- Backend regression suite: 439 passed, two optional Office-rendering tests
  skipped. The optional tests require a local Office or LibreOffice renderer.
- Frontend build, lint and TypeScript checks passed after the study-metric label
  corrections. Generated artifacts and personal data remain outside Git.
- These checks verify code behavior using fixtures; they do not prove factual
  quality across real documents or establish student learning outcomes.

## Measured research pilots

- ASQA-derived retrieval: 99 questions, 437 annotated-context corpus chunks,
  297 completed retrieval attempts. Full retrieval here means hybrid and
  reranking without HyDE. The corpus and relevance labels make this easier than
  open retrieval; it is not the official ASQA answer evaluation.
- Expanded answer pilot: ten questions, two methods, 20 successful HTTP
  responses. Seventeen outputs were eligible for a source-section RAGAS
  diagnostic; all 85 eligible scores completed after retries. Two of the 17
  were excerpt fallbacks, not generated explanations; three outputs were
  excluded. Seven questions produced eligible generated source sections for
  both methods and therefore contribute to paired metric differences.
- The configured 20B answer model also served as judge. The selected questions,
  curated corpus, small paired sample and automatic judge limit interpretation.
  An earlier five-question pilot used a different code fingerprint and is kept
  separate. Detailed values, limitations and local artifact paths are in
  [ASQA validation](ASQA_VALIDATION.md).

## Remaining work before a paper claim

Define a held-out, representative benchmark and a corpus constructed without
answer leakage; use the official ASQA metrics when appropriate. Evaluate both
methods on the same questions with frozen configuration and report failures,
latency, token usage, exclusions and uncertainty. Obtain independent review of
factual correctness, support for each cited claim, coverage, abstentions and
mixed background. Evaluate summaries, comparisons, quizzes, flashcards,
screenshots and figures independently on realistic examples. A human-reviewed
study is needed for any claim of improved learning. Local ignored artifacts
must be archived with the experiment before sharing results.
