# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI backend implementing the four-tier architecture of the base paper
**EduRAG: A Multimodal Explainable RAG System With Deterministic Token Grounding**
(IEEE Access 14, 2026), extended with contradiction detection, a study-tools
layer, audio ingestion and an evaluation harness. It is a final-year major
project, so *comparability with the paper matters*: metric definitions and
pipeline stages are chosen to line up with its Table 2 / Table 3, and changing
their semantics silently breaks the report those tables feed.

## Commands

```bash
uv sync                                    # install
uv run uvicorn app.main:app --reload       # dev server -> /docs
uv run pytest                              # full suite (~3 min, no network)
uv run pytest tests/test_retrieval.py -q   # one file
uv run pytest -k "consensus" -q            # one pattern
uv run pytest tests/test_api.py::TestAsk::test_top_k_is_honoured   # one test
```

Evaluation (makes real LLM calls):

```bash
uv run python -m app.eval.cli ingest ./corpus        # offline index, text+tables only
uv run python -m app.eval.cli build-benchmark --count 97
uv run python -m app.eval.cli build-benchmark --count 15 --visual
uv run python -m app.eval.cli run --report data/eval/report.md
```

There is no linter or formatter configured. Match surrounding style.

## Architecture

### The load-bearing invariant

**Every modality becomes text at ingestion time.** A diagram becomes its VLM
caption; a lecture becomes its Whisper transcript. Both are stored as ordinary
`Chunk` rows and indexed by the same BM25 + FAISS pass as prose. There is no
separate visual or audio index anywhere downstream, and adding one would
undo the main simplification inherited from the paper. New modalities belong in
`app/core/ingestion/` producing `Chunk`s, not in the retrieval layer.

### Request flow

`/ask` → `RetrievalIndex.retrieve` (HyDE → BM25 + FAISS → RRF → cross-encoder)
→ `find_disagreements` → `generate_answer` → attribution + token grounding +
citation validation → `AskResponse`.

Contradiction detection runs **before** generation on purpose: detected
conflicts are injected into the prompt so the model presents both sides in its
own prose. Do not move it after generation and append a footer — that was the
original design and it is worse.

### Provider abstraction

Nothing calls a vendor SDK. All chat traffic goes through `LLMClient.chat`
(`app/core/generation/llm_client.py`), which supports `ollama`,
`openai_compatible` and `none`. LLM / HyDE / VLM / ASR each choose their
provider independently via env vars, so a hosted answer model can sit alongside
local embeddings. Embeddings and the reranker always run locally on CPU.

- Vision messages differ per provider — build them with `build_vision_message`,
  never inline. Ollama takes a parallel `images` list; OpenAI-compatible takes a
  content array with a `data:` URL.
- `strip_think_tags` is applied centrally inside `LLMClient.chat`, so reasoning
  models are handled everywhere at once. Don't re-strip at call sites.
- `_post` retries 429 and 5xx honouring `Retry-After`. Groq's free tier is
  **8k tokens/minute**, which a benchmark run will hit constantly; without the
  retry, evaluation silently drops queries and averages over the survivors.

### Explainability (the project's centrepiece)

Three independent deterministic signals, all in `app/core/attribution/`:

| signal | what it answers |
| --- | --- |
| chunk attribution | which evidence drove the answer (0.7·cross-encoder + 0.3·Jaccard) |
| token grounding | is each answer token traceable (exact → stem → substring → ungrounded) |
| citation validation | does each `[n]` marker point at a really-retrieved chunk |

No SHAP/LIME, no perturbations — that is the paper's core claim and the reason
the whole thing is O(k) rather than O(2^M).

## Things that will bite you

**Two tokenizers, deliberately.** `tokenize` keeps stopwords (token grounding
and ROUGE-L/token-F1 need the full stream); `tokenize_for_retrieval` strips them
(BM25 must not match chunks on "the"). Both live in `bm25_index.py`. Using the
wrong one silently degrades retrieval or skews grounding — this was a real bug.

**Citation Validity ≠ Citation Coverage.** Validity is "of the markers written,
how many are real" (the paper's definition, currently 1.0). Coverage is "how
many answers cited anything at all". Folding uncited answers into validity as
zeroes conflates a prompt-compliance problem with a hallucination problem and
makes the number incomparable to the paper. `evaluate_generation` keeps them
apart; keep it that way.

**The consensus band is embedding-model-specific.** Measured on
`bge-small-en-v1.5`: unrelated 0.47–0.52, real contradictions 0.82–0.89,
paraphrases 0.91–1.00. Contradictions and paraphrases *overlap*, so cosine
similarity alone cannot decide conflict — the band only discards unrelated pairs
and duplicates, and an LLM entailment check makes the actual call. Re-measure
and retune `CONSENSUS_*` if `EMBEDDING_MODEL` changes.

**Ablation arms must stay handicapped.** Only `RetrievalMode.full` gets HyDE
expansion and cross-encoder reranking. If the other modes acquire either, the
ablation measures nothing. See `RetrievalIndex.retrieve`.

**Index rebuilds are whole-corpus.** BM25 and FAISS are built from the KB in one
pass with no per-document deletion path, so ingest and delete both call
`rebuild_async()`. Fine at this scale (the paper uses ~5k chunks); revisit only
if the corpus grows a lot.

**Figures need `generate_picture_images=True`.** Docling omits picture bitmaps
by default, and without it `extract_figures` silently yields nothing. The
converter is `lru_cache`d because it loads layout models on first use.

**Audio chunks have no page number.** Use `Chunk.locator()` rather than
formatting `source_file`/`page_number` yourself; it emits `file @ 2:05` for
transcripts and `file, p.42` otherwise.

## Tests

`tests/conftest.py` sets every provider to `"none"` and points storage at a
`tmp_path`, so the suite runs offline against real BM25/FAISS/cross-encoder
models (both are cached locally by HuggingFace). Tests that need generation
monkeypatch `generate_answer` rather than mocking HTTP.

`asyncio_mode = "auto"` is set in `pyproject.toml`, so `async def` tests need no
decorator.

## Scope decisions (do not "fix" these)

Deliberately out of scope per the project plan: a dedicated visual embedding
model (caption-to-text achieves near-perfect visual MRR for far less risk),
PostgreSQL + pgvector + Redis + Docker (no accuracy benefit at this scale),
real-time streaming audio, and LLM-as-judge/RAGAS faithfulness scoring (the
paper avoids it for latency and comparability).
