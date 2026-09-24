# AI-Powered Multimodal Study Assistant — Backend

An explainable, multimodal RAG backend for study material, built on the
four-tier architecture of **EduRAG: A Multimodal Explainable RAG System With
Deterministic Token Grounding** (Amal, Niranjan & Thushara, *IEEE Access* 14,
2026) and extended per the project plan.

Ask questions over your own notes, textbooks, slides, diagrams and lecture
recordings, and get back an answer where **every claim is traceable** — inline
citations validated against the retrieved set, a per-token grounding heatmap,
per-chunk attribution percentages, and explicit flagging when two of your
sources disagree.

---

## What's implemented

| Tier | Capability | Status |
| --- | --- | --- |
| 1 | Docling parsing → page/section-aware text & table chunks | ✅ |
| 1 | Figure extraction → geometry filter → pHash dedup → VLM captions → visual chunks | ✅ |
| 1 | Lecture audio → Whisper transcript → timestamped chunks | ✅ |
| 1 | Pasted screenshot/image → question-aware VLM analysis → optional visual chunk | ✅ |
| 2 | BM25 (sparse) + FAISS (dense) dual index | ✅ |
| 2 | HyDE query expansion, RRF fusion, cross-encoder reranking | ✅ |
| 3 | Citation-constrained grounded generation | ✅ |
| 3 | Chunk attribution (cross-encoder + Jaccard), token grounding cascade, citation validation | ✅ |
| 3 | Cross-document contradiction detection *(extended beyond the paper)* | ✅ |
| 4 | Summaries, document comparison, quiz/MCQ/flashcard generation | ✅ |
| 4 | FastAPI application layer | ✅ |
| — | Evaluation harness: LLM-as-teacher benchmark, 3-way ablation, metrics, paired significance testing | ✅ |
| — | Security hardening: upload validation, query sanitization, injection defenses | ✅ |

Every modality is converted to text at ingestion time — a diagram becomes its
caption, a lecture becomes its transcript — so retrieval, reranking, attribution
and grounding all operate on one kind of object. There is no separate visual or
audio index anywhere downstream.

---

## Setup

```bash
uv sync
cp .env.example .env      # then fill in your provider + key
```

Embeddings and the reranker always run **locally on CPU**. The LLM, HyDE, VLM
and ASR each pick their own provider, so you can mix hosted and local freely.

**Hosted (no GPU needed)** — any OpenAI-compatible gateway:

```ini
LLM_PROVIDER=openai_compatible
LLM_MODEL=openai/gpt-oss-120b
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your-key
```

**Fully local** — the base paper's setup, via Ollama:

```ini
LLM_PROVIDER=ollama
LLM_MODEL=gemma2:9b
LLM_BASE_URL=http://localhost:11434
VLM_PROVIDER=ollama
VLM_MODEL=llava-phi3
ASR_PROVIDER=faster_whisper
```

Run it:

```bash
uv run uvicorn app.main:app --reload
```

Interactive API docs at <http://localhost:8000/docs>; `GET /config` shows which
models are actually wired up (secrets reported only as present/absent).

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/ingest` | Upload a document or audio file; parses, captions figures, indexes |
| `POST` | `/ask` | Grounded answer + citations + token grounding + attribution |
| `POST` | `/vision/ask` | Ask about a pasted image; optionally retrieve notes and save it |
| `GET` | `/evidence/{chunk_id}` | Fetch a cited chunk |
| `GET` | `/evidence/{chunk_id}/image` | The source figure behind a visual chunk |
| `POST` | `/study/summarize` | Citation-grounded topic summary |
| `POST` | `/study/compare` | How two sources treat the same topic |
| `POST` | `/study/quiz` | MCQs, short-answer or flashcards, each traceable to a chunk |
| `GET` | `/documents` | What's ingested |
| `GET` | `/documents/stats` | Chunk counts by type, index readiness |
| `DELETE` | `/documents/{source_file}` | Remove a document and reindex |
| `GET` | `/health`, `/config` | Liveness, active model configuration |

`POST /ask` accepts a `mode` for the ablation arms: `bm25_only`, `dense_only`,
`hybrid`, `full` (default).

### Screenshot questions

`POST /vision/ask` accepts multipart form data with an image (`png`, `jpg`,
`jpeg`, or `webp`) and a text `question`. Pasted screenshots are normalized and
processed in memory by default. Set `use_knowledge_base=true` to augment the
answer with relevant uploaded study material, or `save_to_library=true` to keep
the normalized image and its VLM analysis as a searchable visual chunk.

A frontend can send clipboard images from a paste event directly to this
endpoint; no separate document-ingestion step is required.

---

## Evaluation

Reproduces the base paper's methodology on your own corpus.

```bash
# 1. Index a corpus offline (text + tables; no LLM calls)
uv run python -m app.eval.cli ingest ./study_material

# 2. Generate the LLM-as-teacher benchmark
uv run python -m app.eval.cli build-benchmark --count 97
uv run python -m app.eval.cli build-benchmark --count 15 --visual

# 3. Score: 3-way ablation + generation & grounding metrics
uv run python -m app.eval.cli run --report data/eval/report.md
```

Reported: **MRR**, **Recall@1/@5** per ablation arm; **BERTScore F1**,
**ROUGE-L**, **Token F1**, **Faithfulness**, **Citation Validity Rate**,
**Citation Coverage**, **Token Grounding Ratio** for the full system. Output is
Markdown, ready to paste beside the paper's Table 2 / Table 3.

> **BERTScore:** `bert-score` is a dependency, so the real metric is used. If it
> is ever missing, the harness falls back to embedding cosine similarity — a
> coarser, sentence-level measure that is **not** comparable to the paper's
> numbers — and labels the column with which backend produced it.

### Sample run

Corpus: the EduRAG paper itself (155 chunks), 9 benchmark queries,
`openai/gpt-oss-120b`. A small run — treat absolute values as illustrative.

**Ablation (Table 3 equivalent)**

| mode | MRR | Recall@1 | Recall@5 |
| --- | --- | --- | --- |
| bm25_only | 0.7778 | 0.6667 | 0.8889 |
| hybrid | 0.8167 | 0.7778 | 0.8889 |
| full | 0.9444 | 0.8889 | 1.0000 |

The same monotonic gain the paper reports: adding dense retrieval to BM25 helps,
and adding HyDE + cross-encoder reranking on top helps again.

**Generation & grounding (Table 2 equivalent)**

| metric | this run | EduRAG paper |
| --- | --- | --- |
| BERTScore F1 | 0.8816 | 0.8952 |
| Faithfulness | 0.9244 | 0.8924 |
| Citation Validity Rate | 1.0000 | 0.94 |
| Token Grounding Ratio | 0.7284 | 0.855 |
| ROUGE-L | 0.3855 | — |
| Citation Coverage | 0.7778 | *(not reported)* |

**Why two citation metrics.** They catch different failures, and collapsing them
hides which one is happening:

- *Validity* — of the markers the model wrote, how many point at a real
  retrieved chunk. **1.0 here: no hallucinated references at all.**
- *Coverage* — how many answers carried any marker. Initially **0.44**: the
  model was answering correctly from real evidence but silently omitting
  citations on over half the answers. Making the citation rules explicit and
  mandatory in the system prompt raised this to **0.78** with no change to
  retrieval.

Averaging uncited answers into validity as zeroes would have shown a single
misleading "0.44 citation validity" and pointed at a hallucination problem that
did not exist.

---

## Notes on the extensions

**Contradiction detection.** The paper flags source disagreement with a bare
cosine-similarity threshold. Measured on `bge-small-en-v1.5`:

| pair type | similarity |
| --- | --- |
| unrelated topics | 0.47 – 0.52 |
| genuine contradictions | 0.82 – 0.89 |
| paraphrases / agreement | 0.91 – 1.00 |

A *low* score means "unrelated", not "conflicting" — and contradictions overlap
paraphrases, so **no threshold on cosine similarity can separate agreement from
conflict**. The band here is only a cheap prefilter that discards unrelated
pairs and duplicates; an entailment-style LLM check over the few survivors makes
the actual judgement and writes the "your textbook says X, your slides say Y"
explanation. Retune the band if you change `EMBEDDING_MODEL` — the ranges are
model-specific.

**Grounded quiz generation.** A generated question whose citation doesn't
resolve to a real retrieved chunk is *dropped*, not shown; `grounded_rate`
reports the fraction kept. This extends the paper's citation-validation idea to
a new output format rather than trusting the model to behave.

**Rate limits.** Free API tiers enforce tight per-minute token budgets (Groq's
is 8k TPM). The client retries on 429 honouring `Retry-After`; without it an
evaluation run silently loses a third of its queries and reports metrics over
whatever survived.

**Significance testing.** The ablation table (BM25 → hybrid → full) is three
point estimates on a benchmark of tens of queries — not enough to eyeball
whether a gain is real. `app/eval/significance.py` runs a paired Wilcoxon
signed-rank test (not a t-test: per-query MRR is bounded in [0, 1] and not
normally distributed) between each consecutive ablation arm on the *same*
queries, and reports it alongside the table so a reported improvement is a
tested claim, not just a bigger number.

---

## Security

Uploaded files and retrieved chunk text are both untrusted input — a student's
own material, but not necessarily benign or well-formed. There is no
authentication and no multi-tenant isolation (out of scope for a local-first,
single-user tool); what's here is narrower:

- **Upload validation** (`app/core/security.py`): magic-byte checks against
  the declared extension (PDF/DOCX/PPTX/audio), a size ceiling per category
  (documents vs. the larger lecture-audio limit), and filename sanitization
  that strips path components *and* any character outside a safe allowlist —
  not just directory traversal.
- **Query sanitization**: every free-text field (`/ask` query, `/study` topic)
  is length-capped and rejects raw control characters via a pydantic
  validator, so a malformed request never reaches retrieval or gets logged
  verbatim with control bytes intact.
- **Evidence-is-data prompt framing**: retrieved chunk text is explicitly
  labelled as data to reason about, never as instructions, and delimited from
  the system prompt and the student's question with literal section markers.
  `neutralize_prompt_markers` breaks any literal occurrence of those markers
  *inside* a chunk's own text, closing the "marker spoofing" gap that framing
  alone doesn't — a malicious document can't fake a new section boundary.
- **Injection-signal logging** (log-only, never blocking): evidence is scanned
  for multi-word phrasing shaped like a prompt-injection attempt before it
  enters a prompt. Tuned against single-keyword false positives — "ignore" or
  "system" alone are ordinary academic vocabulary ("ignore the sign", "the
  immune system") and must not trigger on their own.
- **Safe error responses**: a catch-all exception handler logs the real
  exception server-side and always returns a generic `{"detail": "Internal
  server error."}` — never a stack trace or internal path.

None of this claims prompt injection is solved — no prompt-engineering or
detection layer can guarantee a model is never influenced by adversarial text
in its context window. It raises the bar and keeps failures from leaking
internals, without pretending to eliminate the underlying risk.

---

## Tests

```bash
uv run pytest
```

193 tests; every hosted provider is disabled in the test fixtures, so the suite
exercises retrieval, attribution and grounding against a local index. A clean
machine may download the configured embedding/reranker models and NLTK tokenizer
data on its first run; subsequent runs use the local caches.

---

## Deliberately out of scope

Per the project plan: a dedicated visual embedding model (caption-to-text
achieves near-perfect visual MRR for far less risk), PostgreSQL + pgvector +
Redis + Docker (no accuracy benefit at this corpus scale), real-time streaming
audio, and LLM-as-judge faithfulness scoring (the paper avoids it for latency
and comparability).
