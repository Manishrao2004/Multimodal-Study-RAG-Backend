"""Build the report with the authoritative Codex artifact Python runtime."""
from pathlib import Path
import json
import re
import argparse
import csv
import xml.etree.ElementTree as ET

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/validation-final")
args = parser.parse_args()
RUN = args.run_dir.resolve()
doc = Document()
sec = doc.sections[0]
sec.top_margin = sec.bottom_margin = Inches(.7)
sec.left_margin = sec.right_margin = Inches(.75)
sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
for name in ("Normal", "Title", "Subtitle", "Heading 1", "Heading 2"):
    style = doc.styles[name]
    style.font.name = "Calibri"
    style.font.color.rgb = RGBColor(0, 0, 0)
doc.styles["Normal"].font.size = Pt(10.5)
doc.styles["Normal"].paragraph_format.space_after = Pt(6)
doc.styles["Normal"].paragraph_format.line_spacing = 1.08
doc.styles["Title"].font.size = Pt(25)
doc.styles["Heading 1"].font.size = Pt(15)
doc.styles["Heading 1"].paragraph_format.space_before = Pt(14)
doc.styles["Heading 1"].paragraph_format.space_after = Pt(7)
collect_results = False
result_markdown = []


def display(value):
    if value is None:
        return "Not measured"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return " to ".join(display(item) for item in value)
    return str(value)


def paragraph(text, style=None):
    if collect_results:
        prefix = "## " if style == "Heading 1" else "### " if style == "Heading 2" else ""
        result_markdown.extend([prefix + text, ""])
    return doc.add_paragraph(text.replace("`", ""), style)


def table(lines):
    if collect_results:
        result_markdown.extend(lines + [""])
    rows = [[v.strip().replace("`", "") for v in line.strip().strip("|").split("|")] for line in lines]
    rows = [r for r in rows if not all(re.fullmatch(r"[-: ]+", cell) for cell in r)]
    t = doc.add_table(rows=0, cols=len(rows[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    widths = [1.5, 2.45, 2.82] if len(rows[0]) == 3 else [6.77 / len(rows[0])] * len(rows[0])
    for col, width in zip(t.columns, widths):
        col.width = Inches(width)
    props = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement("w:" + edge)
        for attr, value in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
            e.set(qn("w:" + attr), value)
        borders.append(e)
    props.append(borders)
    for i, values in enumerate(rows):
        cells = t.add_row().cells
        if i == 0:
            repeat = OxmlElement("w:tblHeader")
            t.rows[i]._tr.get_or_add_trPr().append(repeat)
        for cell, value, width in zip(cells, values, widths):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell.text = value
            pr = cell._tc.get_or_add_tcPr()
            margins = OxmlElement("w:tcMar")
            for edge in ("top", "left", "bottom", "right"):
                e = OxmlElement("w:" + edge)
                e.set(qn("w:w"), "95")
                e.set(qn("w:type"), "dxa")
                margins.append(e)
            pr.append(margins)
            shade = OxmlElement("w:shd")
            shade.set(qn("w:fill"), "E7E7E7" if i == 0 else ("F7F7F7" if i % 2 == 0 else "FFFFFF"))
            pr.append(shade)
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(3)
                p.paragraph_format.line_spacing = 1.05
                for run in p.runs:
                    run.font.size = Pt(9)
                    run.bold = i == 0
    paragraph("")


lines = (ROOT / "docs/METRICS_VALIDATION.md").read_text(encoding="utf-8").splitlines()
i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith("| "):
        batch = []
        while i < len(lines) and lines[i].startswith("| "):
            batch.append(lines[i]); i += 1
        table(batch)
        continue
    if line.startswith("# "):
        paragraph(line[2:], "Title")
        paragraph("Evaluation methods and project evidence", "Subtitle")
    elif line.startswith("## "):
        paragraph(line[3:], "Heading 1")
    elif line.startswith("```"):
        pass
    elif line:
        paragraph(line)
    i += 1

collect_results = True
paragraph("15 Recorded experiment results", "Heading 1")
summary_path = RUN / "summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    m = summary["metrics"]
    paragraph(f"Summary calculated UTC {summary['created_utc']}. Dataset SHA256 {summary['dataset_sha256']}. "
              f"Answer model {summary['models']['answer']}. These are development fixture results.")
    table(["| Metric | Value | Interpretation |", "| --- | --- | --- |",
           f"| Attempts and completed | {m['attempted']} and {m['completed']} | Failures {m['failures']} |",
           f"| Routing accuracy | {m['routing']['accuracy']} | Conditional on completion |",
           f"| End to end routing success | {m['end_to_end_routing_success']} | All attempted cases |",
           f"| Sufficiency accuracy | {m['sufficiency']['accuracy']} | Evidence state labels |",
           f"| Macro F1 | {m['routing']['macro_f1_present_classes']} | Represented route classes |",
           f"| Citation ID validity | {m['citation_id_validity_micro']} | {m['citation_count']} markers not entailment |",
           f"| Latency p50 and p95 ms | {m['latency_ms']['p50']} and {m['latency_ms']['p95']} | Small controlled sample |"])
    paragraph(f"Wilson 95 percent interval for end to end routing success: {display(m.get('end_to_end_routing_wilson95', {}).get('ci95'))}. "
              "A perfect fixture run does not establish perfect performance on new student questions.")
    paragraph(f"Handler attempts including retries: {m.get('handler_attempts', m['attempted'])}; "
              f"retry attempts: {m.get('retry_attempts', 0)}; first attempt completion: {display(m.get('first_handler_attempt_completion_rate'))}; "
              f"first attempt routing success: {display(m.get('first_handler_attempt_routing_success'))}. "
              f"First attempt Wilson interval: {display(m.get('first_handler_routing_wilson95', {}).get('ci95'))}. "
              "Completed responses are frozen, including incorrect ones; resume retries failed calls only. Internal HTTP retries are not counted separately.")
    attempt_latency = m.get("handler_attempt_latency_ms", {})
    paragraph(f"Across {attempt_latency.get('n', 0)} handler attempts including failures, p50 latency was {display(attempt_latency.get('p50'))} ms "
              f"and p95 was {display(attempt_latency.get('p95'))} ms. "
              "The earlier latency table is conditional on successful requests; failed and retried calls must be considered for service reliability.")
    records = json.loads((RUN / "records.json").read_text(encoding="utf-8"))
    table(["| Case | Expected route | Actual route |", "| --- | --- | --- |"] +
          [f"| {r['id'].replace('_', ' ')} | {r['expected_basis']} | {r.get('response', {}).get('answer_basis', r['status'])} |" for r in records])
    paragraph(f"Additional policy checks: {m.get('constraint_checks', 0)}; failures: {m.get('constraint_failures', 0)}. "
              "These inspect currency labels, the verification flag, and the absence of unrequested tax percentages, rather than certifying legal facts.")
    with (RUN / "human_review.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_id", "query", "developer_expected_basis", "actual_basis", "reviewer_id", "reviewed_expected_basis", "reference_adequate", "factual_errors", "citation_errors", "notes"])
        for row in records:
            writer.writerow([row["id"], row["query"], row["expected_basis"], row.get("response", {}).get("answer_basis", "error"), "", "", "", "", "", ""])
else:
    paragraph("No completed controlled experiment summary is available for this report build.")
baseline = ROOT / "artifacts/validation-baseline/summary.json"
if baseline.exists():
    previous = json.loads(baseline.read_text(encoding="utf-8"))
    paragraph(f"The saved baseline routed {previous['metrics']['routing']['accuracy']:.2%} of completed cases correctly. "
              "Development found unnecessary mixed routing for simple arithmetic and a small comparison table. "
              "The planner was adding unrequested comparison criteria. Prompts now distinguish presentation from missing facts "
              "and retain only explicitly requested aspects. The original twelve case retest passed all routes and evidence states. "
              "The final expanded set adds four cases; its results are reported separately. This is development progress rather than a held out improvement estimate.")
retrieval = ROOT / "artifacts/retrieval.json"
if retrieval.exists():
    payload = json.loads(retrieval.read_text(encoding="utf-8"))
    paragraph("Indexed retrieval experiment", "Heading 2")
    query_count = payload["ablation"][0]["queries"] if payload["ablation"] else 0
    paragraph(f"The saved indexed experiment has {query_count} teacher generated queries and {payload.get('corpus_chunks', 'unrecorded')} chunks. "
              "Single gold chunk hit rates and the small development query set limit generalization. Per query ranks and source identifiers are retained in the JSON artifact.")
    table(["| Arm | MRR | Recall at 1 and 5 |", "| --- | --- | --- |"] +
          [f"| {row['mode']} | {row['MRR']} | {row['Recall@1']} and {row['Recall@5']} |" for row in payload["ablation"]])
    if payload.get("significance"):
        table(["| Comparison | Paired mean difference | Unadjusted p value |", "| --- | --- | --- |"] +
              [f"| {s['metric']} | {s['mean_diff']:.4f} | {s['p_value']} |" for s in payload["significance"]])
    paragraph("The observed gains were not statistically significant at 0.05 in this small paired experiment. "
              "Differences across repeat runs can arise from live HyDE generation. No causal claim about an individual component follows.")
claim_filename = "claims.json"
for filename, heading in (("ragas.json", "Ragas scores"), (claim_filename, "Claim audit scores")):
    path = RUN / filename
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        paragraph(heading, "Heading 2")
        paragraph(f"Judge {payload['judge_model']}. Same model as generator {payload['same_model_as_generator']}.")
        if filename == "ragas.json":
            paragraph(f"Ragas version {payload['ragas_version']}. Eligible answers {payload['eligible_rows']}; failed rows {payload['failed_rows']}.")
            table(["| Metric | Mean and count | Bootstrap interval |", "| --- | --- | --- |"] +
                  [f"| {name.replace('_', ' ')} | {display(value['mean'])} from {value['n']} | {display(value['ci95'])} |" for name, value in payload["summary"].items()])
            paragraph("These means exclude failed metric judgments. The per metric sample counts and error rows must accompany every reported mean; incomplete judging is not a quality pass.")
        else:
            paragraph("This table uses the saved 20B audit with fourteen completed cases. "
                      "Two judgments remain incomplete. Later budgeted and 120B diagnostic runs were stopped to respect "
                      "the free tier API constraint and are excluded from these means. No more provider calls are required to build this report.")
            paragraph(f"Audit response format {payload.get('response_format', 'provider default')}; "
                      f"maximum completion tokens {payload.get('judge_max_tokens', 'provider default')}. "
                      "The updated audit client uses explicit output budgets for future runs; those settings were not used in this saved run.")
            paragraph(f"Completed audits {payload['completed']}; errors {payload['failures']}. These are model judgments awaiting human calibration.")
            paragraph("Only the successfully audited cases contribute to the claim and rubric means. Missing audits are not random evidence of success; inspect the case IDs before generalizing a perfect partial mean to mixed or model-only answers.")
            table(["| Metric | Value | Scope |", "| --- | --- | --- |"] +
                  [f"| {name.replace('_', ' ')} | {display(value)} | Pooled audited claims |" for name, value in payload["claim_micro"].items()])
            table(["| Rubric | Normalized mean | Sample count |", "| --- | --- | --- |"] +
                  [f"| {name.replace('_', ' ')} | {display(value['mean'])} | {value['n']} |" for name, value in payload["rubrics"].items()])
controls = ROOT / "artifacts/judge-controls-final/ragas.json"
if not controls.exists():
    controls = ROOT / "artifacts/judge-controls/ragas.json"
if controls.exists():
    payload = json.loads(controls.read_text(encoding="utf-8"))
    paragraph("Judge calibration controls", "Heading 2")
    paragraph("Two synthetic controls test a supported profit statement and an otherwise identical wrong number. "
              "They are judge checks and are excluded from application quality means.")
    table(["| Control | Faithfulness | Expected |", "| --- | --- | --- |"] +
          [f"| {r['id'].replace('_', ' ')} | {r.get('scores', {}).get('faithfulness')} | {'0.0' if 'wrong' in r['id'] else '1.0'} |" for r in payload["rows"]])
    paragraph(f"Failed calibration controls: {payload.get('failed_controls', 'not recorded in the initial run')}. "
              "The final controls also check factual correctness against identical and wrong-number reference comparisons.")
paragraph("Validation status", "Heading 2")
test_report = ROOT / "artifacts/evaluation-tests.xml"
if test_report.exists():
    root = ET.parse(test_report).getroot()
    suites = list(root) if root.tag == "testsuites" else [root]
    total = sum(int(s.attrib.get("tests", 0)) for s in suites)
    failed = sum(int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0)) for s in suites)
    paragraph(f"Recorded local evaluation regression tests: {total}; failures and errors: {failed}. "
              "The XML artifact retains per test results. This is a local run, not a remote CI result.")
paragraph("This evidence supports a working evaluation implementation and fixes for the observed development failures. "
          "Independent review, a held out dataset, broader format and temporal tests, and judge calibration remain necessary "
          "for a final research claim. The current study does not certify perfect answers or improved student learning.")
out = ROOT / "artifacts/document-qa/Study_RAG_Metrics_Validation.docx"
out.parent.mkdir(parents=True, exist_ok=True)
doc.save(out)
source_text = (ROOT / "docs/METRICS_VALIDATION.md").read_text(encoding="utf-8")
if summary_path.exists():
    abstract = (
        "## Recorded result summary\n\n"
        f"The local regression suite records {total if test_report.exists() else 'unrecorded'} tests. "
        f"The expanded live experiment completed {m['completed']} of {m['attempted']} unique cases; "
        f"incomplete cases: {m['failures']}. Completed-case routing accuracy is {m['routing']['accuracy']:.2%}; "
        f"first handler attempt routing success is {m.get('first_handler_attempt_routing_success', 0):.2%}. "
        "The report retains timeouts and retry history rather than treating them as correct answers.\n\n"
        "The original twelve-case development retest passed every route and evidence-state label after fixes for unnecessary "
        "arithmetic and comparison expansion. The indexed full pipeline achieved MRR 0.9444 on nine teacher generated questions; "
        "paired comparisons did not establish statistical significance. These development results require a separate held out study.\n\n"
        "Real Ragas and claim-level audits are included with per metric sample counts, failed judgments, "
        "judge identities, and calibration controls. Earlier provider quota failures are retained in the experiment history. "
        "Independent source and claim review remains necessary before a final research accuracy claim.\n\n"
    )
    source_text = source_text.replace("## 1 Evaluation architecture", abstract + "## 1 Evaluation architecture", 1)
(ROOT / "artifacts/Study_RAG_Metrics_Validation.md").write_text(source_text + "\n" + "\n".join(result_markdown), encoding="utf-8")
print(out)

# Export the same complete text as the PDF, including its result summary.
collect_results = False
for element in list(doc._element.body):
    if element.tag != qn("w:sectPr"):
        doc._element.body.remove(element)
word_lines = (ROOT / "artifacts/Study_RAG_Metrics_Validation.md").read_text(encoding="utf-8").splitlines()
i = 0
while i < len(word_lines):
    line = word_lines[i]
    if line.startswith("| "):
        batch = []
        while i < len(word_lines) and word_lines[i].startswith("| "):
            batch.append(word_lines[i])
            i += 1
        table(batch)
        continue
    if line.startswith("# "):
        paragraph(line[2:], "Title")
    elif line.startswith("## "):
        paragraph(line[3:], "Heading 1")
    elif line.startswith("### "):
        paragraph(line[4:], "Heading 2")
    elif line and not line.startswith("```"):
        paragraph(line)
    i += 1
word_out = ROOT / "output/word/Study_RAG_Metrics_Validation.docx"
word_out.parent.mkdir(parents=True, exist_ok=True)
doc.save(word_out)
print(word_out)
