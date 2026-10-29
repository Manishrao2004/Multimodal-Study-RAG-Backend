# Evidence support and conservative answering

This change applies to ordinary text, tables, calculations, comparisons and extracted figures. Auto mode should use sources only when the requested information has traceable support. Topic relevance alone is insufficient.

## Text and tables

The existing evidence-check call now returns an aspect, chunk identifier and exact supporting quotation for each requested aspect. The server rejects unknown chunk IDs, invented or altered quotations, and unrequested aspect names. Whitespace differences from PDF extraction are allowed; factual wording and case must match. Requested items must appear in the accepted quotations. Headings alone cannot establish an explanation or comparison. Arithmetic may use quoted operands without requiring the final calculation to be printed in the source.

The model still judges semantic sufficiency. A valid quotation does not independently prove entailment, completeness or real-world truth. These checks constrain the model's decision; they do not certify perfect answers.

## Context recovery

Before the check, the server can add up to two nearby chunks for each of the first two retrieved passages. The SQLite query stays within the same document and neighboring pages. This uses no extra retrieval, HyDE or generation call. Expanded passages have a zero reranker score because they were not reranked. Auto mode bounds the checked and generated context to eight chunks, 16000 text characters overall, and 4000 characters per chunk. Omitted material cannot supply quotations for the check. Comprehensive mode records a coverage gap when some requested documents are absent.

## Failed and incomplete checks

A failed evidence check is marked unavailable. Keyword overlap and retrieval score cannot promote it to strong support. A general educational question may receive explicitly unverified general knowledge without source evidence attached. A document-specific question without usable support receives a limitation instead of guessed document contents. An incomplete successful check uses mixed mode, with missing aspects sent to both generation prompts. Missing source-specific details must remain unknown.

Source-only mode remains restricted to uploaded material; model-only mode remains independent of retrieval. Current or authoritative facts flagged by the planner are not externally verified merely because an uploaded passage contains a matching number.

## Figure inspection

Caption text remains useful for retrieval. When the question concerns a figure, or a figure is the top retrieval result, the server can inspect its saved image against the question. The resulting observations are labeled as model interpretations and kept in the response only; the stored caption is not overwritten. Figure identity, source location and image reference are preserved for the evidence viewer.

An unreadable image, invalid response, missing image, disabled provider or failed request cannot establish full visual support. An uninspected caption alone cannot pass the strong-support gate. This implementation inspects extracted figure images; it does not automatically inspect a PDF page whose diagram was never extracted. Such a question retains a visual-support gap.

## Free tier limits

- Text uses the existing planner and evidence-check calls. Both have no automatic provider retries, a 25-second default request timeout, and a 2048-token output cap.
- Figure inspection defaults to at most one figure per question, with no provider retries, a 25-second deadline, and a 1536-token output cap.
- Successful observations are cached in memory for up to 64 image-content, question and model combinations. Changing the image, question, endpoint or model invalidates reuse. Cache entries disappear on restart; simultaneous uncached requests are not coalesced.
- Set `VISUAL_QUERY_ENABLED=false` or `VISUAL_QUERY_MAX_FIGURES=0` to prevent additional query-time vision calls. `EVIDENCE_CHECK_TIMEOUT_SECONDS` and `VISUAL_QUERY_TIMEOUT_SECONDS` are configurable.
- Contradiction detection, HyDE, answer generation and ingestion captioning retain their separate existing budgets. A mixed answer still uses separate source and general-knowledge generation calls.

## Validation

Run the tracked regression tests locally:

```powershell
uv run pytest app/tests app/eval -q
```

The new safety tests block HTTP transport and use controlled provider responses. They cover text support, fabricated quotes, missing comparison items, arithmetic, outline depth, current-fact verification, routing failures, explicit mode overrides, neighboring-source boundaries, context limits, image payload construction, cache invalidation, unreadable images, provider failures and figure-call limits.

The earlier live Ragas scores and PDF/Word metrics report describe the saved experiment before this change. They have not been regenerated or reused as evidence of improved live accuracy for this implementation. No live model calls were made to test this change. A future live comparison should use fresh artifacts, frozen independent references and an explicitly agreed API budget.
