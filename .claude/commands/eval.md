---
description: Run the retrieval ablation and generation evaluation, and report the results table
---

Run the project's evaluation harness and report the numbers.

Arguments (optional): `$ARGUMENTS` — e.g. `--skip-generation` for retrieval metrics
only, or a benchmark size such as `97`.

## Steps

1. Check the knowledge base is populated:

   `uv run python -c "from app.config import get_settings; from app.db.knowledge_base import KnowledgeBase; kb=KnowledgeBase(get_settings().kb_db_path); print(kb.count(), 'chunks', len(kb.list_documents()), 'docs')"`

   If it is empty, stop and ask which corpus directory to ingest — do not invent one.

2. If `data/eval/benchmark.json` does not exist, build it:
   `uv run python -m app.eval.cli build-benchmark --count 97`

   The generator drops items it cannot parse, so the file usually holds fewer
   items than requested. That is expected; report the real count.

3. Run the evaluation:
   `uv run python -m app.eval.cli run --report data/eval/report.md --json data/eval/results.json`

   This makes one LLM call per query and will take several minutes — free-tier
   token limits (Groq: 8k TPM) trigger backoff between calls. Run it in the
   background and wait rather than shortening the benchmark to make it finish.

## Reporting

Show the ablation table and the generation table, then say explicitly:

- Whether `Generation failures` is 0. Anything above 0 means queries were
  dropped and the averages cover only the survivors — investigate before
  quoting the numbers.
- Which `BERTScore backend` produced the column. `embedding-cosine` is the
  fallback and is **not** comparable to the base paper's BERTScore F1.
- Citation **Validity** and **Coverage** separately — they measure different
  failures (see CLAUDE.md). Low coverage with high validity is a prompt
  compliance problem, not hallucination.

Do not round numbers up, present a partial run as complete, or compare a
fallback-backend BERTScore against the paper.
