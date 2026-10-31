# Verity — multimodal study assistant backend

Verity answers questions using a student's uploaded material. It supports text, PDFs, Office documents, images and lecture audio, and offers summaries, source comparisons, quizzes and flashcards. The frontend is the sibling `Multimodal-Study-RAG-Frontend` repository.

## How it works

1. Ingestion extracts text; configured vision and speech models can caption figures and transcribe audio. Chunks and source metadata are saved in SQLite.
2. Retrieval combines BM25 and local embeddings, optionally expands the query with HyDE, and reranks passages with a local cross-encoder.
3. For text questions, the default `aspect` method assesses requested aspects, optionally searches for gaps, and audits generated source claims. It can withhold claims or show excerpts if verification fails. These checks are model-assisted and can be wrong or unavailable.
4. Auto decides between source-based, mixed and general-knowledge answers. My sources restricts answers to uploaded material. General knowledge bypasses document retrieval and source-claim checks. `aspect` is an internal evidence method, not a frontend mode or separate model.

`edurag` and `adaptive` remain available as evaluation baselines; they are not claims of an exact reproduction of the base paper. Study tools use retrieval and their own generation and citation checks, but do not yet run the full aspect/claim-audit path. Screenshot questions use the separate vision path. Valid citation numbers and token overlap do not prove factual support.

## Setup

Requirements: Python 3.12+, `uv`, Node.js 22.13+ for the frontend, and local disk space for embedding and reranking models. Copy `.env.example` to `.env` and configure an answer provider and key. Enable and configure VLM or ASR providers only if needed. Never commit `.env` or API keys.

```powershell
uv sync --group dev --group eval
uv run uvicorn app.main:app --reload
```

The API runs at `http://127.0.0.1:8000`; `/docs` lists routes and `/config` shows configured models and evidence settings without verifying connectivity. The frontend setup is in its README. Models may need to be downloaded on first use. Hosted calls can incur usage and rate limits.

## Data and history

Uploaded material, extracted images, the knowledge database and conversations are stored locally under `data/` by default. Conversation history is stored by the backend. Study-tool sessions are stored in the browser, with up to 12 results; reopening practice starts that practice again. Screenshots are saved to the library only when explicitly requested. Back up local data separately; generated databases, user files, credentials and evaluation artifacts are not included in Git. This is a local single-user application, not a multi-user service with authentication or isolation.

## Tests and evaluation

```powershell
uv run pytest
uv run python -m app.eval.aspect_study --check
```

The ASQA retrieval and answer runners use the user's populated SQL corpus and cached ASQA dev data. The RAGAS runner requires the optional `eval` group and a configured judge model. See [ASQA validation](docs/ASQA_VALIDATION.md) for saved results, exclusions, rate limits and interpretation, and [metrics and validation](docs/METRICS_VALIDATION.md) for definitions. See [aspect evidence](docs/ASPECT_EVIDENCE.md) and [evidence safety](docs/EVIDENCE_SAFETY.md) for implementation details.

The existing RAGAS results are a small pilot, not a completed publication evaluation. The retrieval probe uses answer-related ASQA contexts and incomplete relevance labels; it is not the official ASQA answer score. Evaluation runs must record dataset and code versions, provider/model, failures, exclusions and review status. A paired held-out study and independent human review of claim support, citations, coverage and abstentions remain necessary for a paper.
