# Evaluation artifacts

The shareable report is `../output/pdf/Study_RAG_Metrics_Validation.pdf`.
Its editable text is `Study_RAG_Metrics_Validation.md`. Both are generated from
the protocol in `../docs/METRICS_VALIDATION.md` and the saved measurements.

Generated runs and reports are ignored by Git and remain on disk. Archive the
run directories with the report when submitting the project; source fixtures,
evaluation code, and document builders remain available in the repository.

## Experiments

- `validation-baseline`: original twelve controlled cases before the routing fixes.
- `validation-arithmetic-fix`: intermediate routing experiment.
- `validation`: twelve case retest, original claim audit, and initial real Ragas measurements.
- `validation-final`: expanded sixteen case experiment with policy checks, retry history, and final audits.
- `judge-controls`: initial positive and wrong-number faithfulness controls.
- `judge-controls-final`: controls with the final judge settings and factual correctness expectations.
- `retrieval.json` and `retrieval.md`: indexed retrieval ablation, per query ranks, corpus and benchmark checksums, and paired comparisons.
- `evaluation-tests.xml`: local regression test results.

Developer fixtures are not independently reviewed ground truth. The review CSV
leaves reviewer fields blank for independent annotation. Keep the original
records when reviewing or correcting labels. Successful responses are frozen
during resume, including incorrect responses. Provider failures and schema
repairs remain recorded separately from answer quality.

Word rendering intermediates are excluded from version control and are not
deliverables. A Word process may still hold the initial DOCX open; it contains
an earlier report draft. The PDF and Markdown named above are the final report.
