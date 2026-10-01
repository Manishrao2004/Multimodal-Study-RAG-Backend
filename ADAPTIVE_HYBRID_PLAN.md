# Adaptive Hybrid Study Assistant Plan

## Product goal

Turn the current source-only RAG application into a provenance-aware tutor that
can use uploaded course material, general model knowledge, and eventually
verified web research without presenting one knowledge source as another.

## Answer policy

The student chooses one of three knowledge modes:

1. **Auto** — search uploaded material first, assess its sufficiency, and route
   to a document-grounded, mixed, or general-knowledge answer.
2. **My sources only** — use uploaded evidence exclusively and abstain when it
   is insufficient.
3. **General knowledge** — answer from the configured language model without
   implying that uploaded documents or the web were consulted.

Each response records its `answer_basis`, evidence-sufficiency decision, source
list, and a user-facing provenance notice.

## Delivery phases

### Phase 1 — adaptive local hybrid (implemented)

- Use LangChain prompt templates, async runnables, and Pydantic output parsers
  for validated query rewriting and evidence-routing decisions while retaining
  the existing provider-neutral model client.
- Separate student-facing answer scope from retrieval-ablation mode.
- Rewrite conversational follow-ups into standalone retrieval queries using the
  configured model, so retrieval carries the actual topic across turns.
- Judge evidence completeness against the standalone request with a constrained
  model decision; use lexical coverage and calibrated reranker scores only as a
  failure fallback.
- Route Auto requests to source-only, mixed, or model-only generation.
- Force mixed answers to separate document claims from general explanation.
- Suppress irrelevant evidence beside model-only answers.
- Display provenance and use `N/A` instead of misleading zero integrity scores.

### Phase 2 — robust sufficiency evaluation

- Build a labelled dataset of answerable, partially answerable, and unrelated
  student questions from the project corpus.
- Calibrate thresholds on that dataset rather than treating heuristic values as
  universal.
- Add claim decomposition to detect which requested subtopics remain missing.
- Evaluate routing accuracy, false-source rate, abstention quality, latency, and
  cost.

### Phase 3 — verified research mode

- Add an explicit Research mode backed by a configurable search provider.
- Trigger it for current, legal, medical, financial, or explicitly web-oriented
  questions; never claim browsing when no search occurred.
- Store URL, title, publisher, publication date, access date, and extracted
  passage for every web citation.
- Rank authoritative and primary sources ahead of secondary summaries.
- Keep document and web citation namespaces visually distinct.

### Phase 4 — durable conversation memory

- Add SQLite `chat_sessions`, `chat_messages`, and `message_sources` tables.
- Persist answer scope, basis, retrieved chunk IDs, web citations, integrity
  metrics, and model configuration with every assistant message.
- Add create/list/open/rename/delete APIs and migrate browser-only sessions.
- Summarize older turns for context while preserving full transcripts for the UI.

### Phase 5 — adaptive tutoring

- Classify requests such as explanation, worked problem, hint, quiz, comparison,
  misconception diagnosis, and study planning.
- Track concepts, attempts, confidence, and recurring misconceptions.
- Recommend prerequisite material and spaced-repetition reviews.
- Evaluate actual learning gains with delayed retention tests, not answer quality
  alone.

## Required evaluation

- Routing accuracy across the three evidence-sufficiency classes.
- Retrieval recall and source/section coverage.
- Citation correctness, completeness, and faithfulness.
- Rate of unsupported claims incorrectly presented as document-grounded.
- Quality of mixed-answer provenance boundaries.
- End-to-end latency and model/search cost per route.
- Student task completion, quiz improvement, and delayed retention.

## Safety rules

- Conversation history is context, never factual evidence.
- Model knowledge never receives document citations.
- Web claims require captured web evidence and access dates.
- Current or high-stakes questions must not be answered as verified facts from
  model memory alone.
- Retrieved documents and web pages remain untrusted prompt input.
