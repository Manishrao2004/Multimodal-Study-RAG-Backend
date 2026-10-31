# Configuration and operations

Copy `.env.example` to `.env` in the backend working directory. Restart the
backend after editing it: settings are cached when the process starts. Keep
`.env` and API keys private. `/config` describes effective settings but does
not test provider access or prove a particular response was verified.

## Current local setup

The local `.env` inspected on 31 October 2026 uses Groq's OpenAI-compatible
endpoint, `openai/gpt-oss-20b` for answers, and `qwen/qwen3.8-27b` for the
fallback and vision. Audio transcription uses `whisper-large-v3-turbo`.
Embeddings and reranking run locally. These are a snapshot, not repository
defaults; check the actual environment and `/config` after changes. Groq lists
Qwen 3.8 as a vision-capable model: https://console.groq.com/docs/vision.

The snapshot selects `EVIDENCE_METHOD=aspect`, but disables HyDE, targeted
retrieval, claim verification and claim repair. It also sets fail-closed to
false; that setting matters when claim verification is enabled and an audit
fails. Therefore do not describe current responses as claim-verified or as
having used the full gap-retrieval and audit pipeline. Aspect assessment and
citation checks are not independent proof of factual support.

The saved ASQA and RAGAS pilots used earlier code and settings. Their numbers
are historical results, not validation of this exact configuration. Before a
new comparison, record the effective configuration, code revision, corpus,
model, fallback use and exclusions. See [ASQA validation](ASQA_VALIDATION.md).

## Provider keys and retries

For OpenAI-compatible chat, `LLM_API_KEY` may contain comma-separated keys for
the same provider. Calls start on successive keys, and a 429 rotates through
the pool before waiting and retrying. Keys on one account or organization may
share quota; rotation cannot create capacity. Retries also apply to transient
server errors. If the primary still returns 429 after its retry budget and the
fallback URL is configured, the fallback model is called; other errors do not
trigger that fallback. VLM, ASR and HyDE have separate settings; do not assume
their requests use the answer model's key pool. Never put credentials in logs,
presentations or reports.

## Choosing evidence settings

The settings are independent of the frontend's Auto, My sources and General
knowledge modes. Auto can choose a mixed or general answer; My sources confines
the answer to uploaded material; General knowledge bypasses document retrieval.

- `TARGETED_RETRIEVAL_ENABLED=true` adds a bounded search for missing aspects.
- `CLAIM_VERIFICATION_ENABLED=true` audits generated source claims using a
  model. `CLAIM_REPAIR_ENABLED=true` permits a repair attempt when appropriate.
- `CLAIM_VERIFICATION_FAIL_CLOSED=true` withholds unresolved source text when
  verification fails. Setting it to false allows an explicitly unverified
  response instead. Neither setting makes general knowledge verified.
- `HYDE_ENABLED=true` may add an LLM query-expansion call when applicable.

Enabling these stages can improve the checks performed but adds model calls and
latency and does not guarantee accuracy. Verify the actual response status and
evidence; model-assisted assessment can be wrong.

## Slow answers and troubleshooting

An answer can involve planning, retrieval, generation and, if enabled, targeted
retrieval, image inspection and claim verification. A provider may rate-limit
any of these calls. The client retries 429 responses and can switch to the
fallback only after the primary retry budget is exhausted, so the fallback log
does not mean the previous attempts were instantaneous. Local embeddings and
reranking also add startup or CPU time. Provider-wide quotas can affect all
keys. Check backend logs and response telemetry, `/config`, provider quota, and
model availability; do not paste API keys into reports.

Vision and audio require compatible configured models and provider access.
The presence of a model name alone does not demonstrate successful image or
audio processing. Test each path with a small, non-sensitive sample.
