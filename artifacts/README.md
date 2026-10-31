# Evaluation artifacts

The older `../output/pdf/Study_RAG_Metrics_Validation.pdf` and
`Study_RAG_Metrics_Validation.md` describe the historical Adaptive experiment;
they do not include the later Aspect ASQA pilot or current settings. They must
not be presented as a current combined report. For consolidated results use
[saved metrics](../docs/FINAL_METRICS.md) and [ASQA notes](../docs/ASQA_VALIDATION.md).

Generated runs and reports are ignored by Git and remain on disk. Archive the
run directories with the report when submitting the project; source fixtures,
evaluation code, and document builders remain available in the repository.

## Experiments

- `validation-baseline`: original twelve controlled cases before the routing fixes.
- `validation-arithmetic-fix`: intermediate routing experiment.
- `validation`: twelve case retest, original claim audit, and initial real Ragas measurements.
- `validation-final`: expanded sixteen case experiment with policy checks, retry history, and final audits.
- `asqa-expanded-20261031`: ten-question Aspect versus EduRAG-style pilot,
  including source-section RAGAS inputs, results, exclusions and manifest.
- `asqa-retrieval-20261030`: ASQA-derived 99-question retrieval probe.
- `asqa-claim-retest-20261031`: three qualitative post-fix examples, without
  new RAGAS scores.
- `judge-controls`: initial positive and wrong-number faithfulness controls.
- `judge-controls-final`: controls with the final judge settings and factual correctness expectations.
- `retrieval.json` and `retrieval.md`: indexed retrieval ablation, per query ranks, corpus and benchmark checksums, and paired comparisons.
- `evaluation-tests.xml`: local regression test results.

Developer fixtures are not independently reviewed ground truth. The review CSV
leaves reviewer fields blank for independent annotation. Keep the original
records when reviewing or correcting labels. Successful responses are frozen
during resume, including incorrect responses. Provider failures and schema
repairs remain recorded separately from answer quality.

The historical PDF and its editable Markdown remain records of their
experiment, but are not a final report for later code or configurations.
