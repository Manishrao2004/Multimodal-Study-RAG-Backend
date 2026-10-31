"""Convert saved live ASQA responses to the existing RAGAS record format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.eval.asqa_retrieval import load_arrow


def convert(records, samples, source_sections=False):
    by_id = {str(sample["sample_id"]): sample for sample in samples}
    converted = []
    for row in records:
        sample = by_id[row["sample_id"]]
        references = [a.get("long_answer", "").strip() for a in sample.get("annotations", []) if a.get("long_answer", "").strip()]
        result = dict(row.get("response", {}))
        original_basis = result.get("answer_basis")
        section_kind = "whole_source_answer"
        if source_sections and original_basis == "sources" and result.get("claim_verification", {}).get("fallback_used"):
            section_kind = "source_excerpts"
        if source_sections and original_basis == "mixed":
            from app.core.generation.claim_verification import split_answer
            try:
                source, _ = split_answer(result.get("answer", ""), original_basis)
            except ValueError:
                source = ""
            result["answer"] = source
            result["answer_basis"] = "sources"
            section_kind = "source_excerpts" if result.get("claim_verification", {}).get("fallback_used") else "mixed_source_section"
        reason = None
        if row.get("status") != "ok":
            reason = "pipeline_request_failed"
        elif not references:
            reason = "no_reference_answer"
        elif result.get("claim_verification", {}).get("withheld") and not (source_sections and section_kind == "source_excerpts"):
            reason = "source_answer_withheld"
        elif result.get("claim_verification", {}).get("fallback_used") and not (source_sections and section_kind == "source_excerpts"):
            reason = "source_answer_replaced_by_excerpts"
        elif result.get("answer_basis") != "sources":
            reason = "non_source_answer"
        contexts = [item["chunk"]["text"] for item in result.get("evidence", []) if item.get("chunk", {}).get("text")]
        if not reason and not contexts:
            reason = "no_retrieved_context"
        if not reason and not result.get("answer", "").strip():
            reason = "empty_source_section"
        converted.append({"id": f"{row['sample_id']}:{row['method']}", "sample_id": row["sample_id"],
            "method": row["method"], "query": row["query"], "reference": references[0] if references else "",
            "original_answer_basis": original_basis, "evaluated_answer_kind": section_kind,
            "reference_selection": "first ASQA long-answer annotation", "retrieved_contexts": contexts,
            "status": "excluded" if reason else "ok", "exclusion_reason": reason,
            "response": result})
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--arrow", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-sections", action="store_true", help="Score the delivered source section of mixed answers, including separately labelled fallback excerpts")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output file")
    rows = convert(json.loads(args.records.read_text(encoding="utf-8")), load_arrow(args.arrow, 100), args.source_sections)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Converted {len(rows)} records; {sum(r['status'] == 'ok' for r in rows)} eligible for source-only RAGAS.")


if __name__ == "__main__":
    main()
