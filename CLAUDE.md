# Developer guide

Verity is a local, single-user FastAPI study assistant inspired by EduRAG. The
`edurag` method is an evaluation baseline, not an exact reproduction of the
paper. Start with [README.md](README.md), [configuration](docs/CONFIGURATION.md),
[aspect evidence](docs/ASPECT_EVIDENCE.md), and [saved metrics](docs/FINAL_METRICS.md).

## Commands

```powershell
uv sync --group dev --group eval
uv run uvicorn app.main:app --reload
uv run pytest
uv run python -m app.eval.aspect_study --check
```

Live evaluation commands call configured models and can incur costs or quota
limits. Do not substitute incomplete runs for completed results. See
[ASQA validation](docs/ASQA_VALIDATION.md) and
[metrics and validation](docs/METRICS_VALIDATION.md) before running experiments.

## Architecture

Ingestion extracts text, optional figure captions and optional audio transcripts
into chunks. SQLite stores the knowledge base and conversation metadata;
retrieval uses BM25 and local embeddings with optional HyDE and reranking.
Question-aware figure inspection and screenshot questions also have vision paths;
not every image is converted into a durable caption before a question is asked.

`/ask` combines retrieval, planning, evidence assessment, answer generation,
attribution and citation processing. The default internal evidence method is
`aspect`; `adaptive` and `edurag` are selectable baselines. Auto, My sources and
General knowledge are frontend answer scopes, not separate LLMs. Targeted gap
retrieval, claim audit, repair and fail-closed behavior are configurable, so
check effective settings and response traces before describing a run. Study
tools and screenshot questions use separate paths and do not run the full chat
claim audit.

The provider client supports Ollama and OpenAI-compatible chat endpoints.
LLM, HyDE, vision and speech settings are independent. Comma-separated keys
can rotate on rate limits where supported; limits may be shared. Never log
credentials. A valid citation ID, quotation, lexical grounding, or model-assisted
claim judgment does not by itself prove factual correctness.

Uploaded documents and retrieved passages are untrusted inputs. The local app
does not provide multi-user authentication or isolation. Keep source history,
user uploads, databases, `.env` and generated artifacts out of commits.

## Research and tests

The saved Aspect and Adaptive metrics use different samples and cannot be
compared directly. Historical results do not validate the current settings or
code revision. Preserve experiment manifests, per-case results, exclusions,
provider changes and reviewer annotations. Citation validity and coverage are
different measurements; source-supported answers and general-knowledge
sections must not be silently pooled.

Regression tests live in `tests`, `app/tests`, and `app/eval`; optional document
rendering tests require Office or LibreOffice. Tests exercise implementation
behavior, not general factual reliability. Models may need a local cache.
