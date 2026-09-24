"""Evaluation CLI.

    # 1. Ingest a corpus (offline, no server needed)
    python -m app.eval.cli ingest ./my_study_material

    # 2. Build the LLM-as-teacher benchmark from the ingested chunks
    python -m app.eval.cli build-benchmark --count 97
    python -m app.eval.cli build-benchmark --count 15 --visual --out data/eval/visual.json

    # 3. Score it: 3-way ablation + end-to-end generation metrics
    python -m app.eval.cli run --report data/eval/report.md

Results print as Markdown tables so they can be pasted straight into the
report next to the base paper's Table 2 / Table 3.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.config import get_settings
from app.core.ingestion.pipeline import DOCUMENT_SUFFIXES, parse_document
from app.db.knowledge_base import KnowledgeBase
from app.eval.benchmark import generate_benchmark, load_benchmark, save_benchmark
from app.eval.metrics import bertscore_available
from app.eval.runner import (
    ablation_significance,
    evaluate_generation,
    evaluate_retrieval,
    run_ablation,
)
from app.models.schemas import ChunkType, RetrievalMode
from app.state import build_app_state

DEFAULT_BENCHMARK = Path("data/eval/benchmark.json")
DEFAULT_VISUAL_BENCHMARK = Path("data/eval/visual_benchmark.json")


def _markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "_(no results)_"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def cmd_ingest(args: argparse.Namespace) -> None:
    settings = get_settings()
    kb = KnowledgeBase(settings.kb_db_path)
    root = Path(args.path)

    files = [p for p in root.rglob("*") if p.suffix.lower() in DOCUMENT_SUFFIXES] if root.is_dir() else [root]
    if not files:
        print(f"No supported documents found under {root}")
        return

    total = 0
    for path in files:
        try:
            chunks = parse_document(path)
        except Exception as exc:
            print(f"  !! {path.name}: {exc}")
            continue
        kb.add_chunks(chunks)
        total += len(chunks)
        print(f"  ok {path.name}: {len(chunks)} chunks")

    print(f"\nIngested {total} chunks from {len(files)} file(s). KB now holds {kb.count()}.")
    print("Note: this path indexes text and tables only — figures and audio go through /ingest.")


async def _build_benchmark(args: argparse.Namespace) -> None:
    settings = get_settings()
    kb = KnowledgeBase(settings.kb_db_path)
    chunks = kb.all_chunks()
    if not chunks:
        print("Knowledge base is empty — run `ingest` first.")
        return

    chunk_type = ChunkType.visual if args.visual else None
    out = Path(args.out) if args.out else (DEFAULT_VISUAL_BENCHMARK if args.visual else DEFAULT_BENCHMARK)

    if chunk_type is ChunkType.visual and not any(c.type == ChunkType.visual for c in chunks):
        print("No visual chunks in the knowledge base — ingest documents with VLM_PROVIDER set.")
        return

    print(f"Generating {args.count} benchmark items with {settings.llm_model}…")
    items = await generate_benchmark(
        chunks, settings, count=args.count, chunk_type=chunk_type, seed=args.seed
    )
    if not items:
        print("No benchmark items were produced — check the LLM provider settings.")
        return

    save_benchmark(items, out)
    print(f"Wrote {len(items)} items to {out}")


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    state = build_app_state()
    if not state.retrieval.is_ready:
        print("Knowledge base is empty — run `ingest` first.")
        return

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        print(f"Benchmark not found at {benchmark_path} — run `build-benchmark` first.")
        return
    items = load_benchmark(benchmark_path)
    print(f"Loaded {len(items)} benchmark queries from {benchmark_path}\n")

    sections: list[str] = ["# Evaluation Report\n"]
    sections.append(
        f"Corpus: **{state.kb.count()} chunks** across "
        f"**{len(state.kb.list_documents())} document(s)**  \n"
        f"Answer model: `{settings.llm_model}` ({settings.llm_provider})  \n"
        f"Embeddings: `{settings.embedding_model}`  ·  Reranker: `{settings.reranker_model}`\n"
    )

    print("Running retrieval ablation…")
    ablation = await run_ablation(items, state.retrieval)
    sections.append("## Retrieval ablation\n")
    sections.append(_markdown_table([s.as_row() for s in ablation]))
    for score in ablation:
        print(f"  {score.mode:<12} MRR={score.mrr:.4f}  R@1={score.recall_at_1:.4f}  R@5={score.recall_at_5:.4f}")

    significance = ablation_significance(ablation)
    if significance:
        sections.append("\n### Significance (paired Wilcoxon on per-query MRR)\n")
        sections.append(
            _markdown_table(
                [
                    {
                        "comparison": s.metric,
                        "n": s.n,
                        "mean diff": round(s.mean_diff, 4),
                        "p-value": round(s.p_value, 4) if s.p_value is not None else "n/a",
                        "significant (p<0.05)": s.significant_at_0_05,
                        "note": s.note,
                    }
                    for s in significance
                ]
            )
        )
        for s in significance:
            p_display = f"{s.p_value:.4f}" if s.p_value is not None else "n/a"
            print(f"  {s.metric}: diff={s.mean_diff:+.4f} p={p_display} sig={s.significant_at_0_05}")

    visual_path = Path(args.visual_benchmark)
    if visual_path.exists():
        print("\nRunning visual-retrieval benchmark…")
        visual_items = load_benchmark(visual_path)
        visual = await evaluate_retrieval(visual_items, state.retrieval, RetrievalMode.full)
        sections.append("\n## Visual retrieval\n")
        sections.append(_markdown_table([visual.as_row()]))
        print(f"  visual       MRR={visual.mrr:.4f}  R@5={visual.recall_at_5:.4f}")

    if not args.skip_generation:
        print("\nRunning generation evaluation (this calls the LLM once per query)…")
        generation = await evaluate_generation(items, state.retrieval, settings)
        sections.append("\n## Generation & grounding (full system)\n")
        sections.append(_markdown_table([generation.as_row()]))
        if generation.bertscore_backend == "embedding-cosine":
            sections.append(
                "\n> **Note:** `bert-score` is not installed, so the BERTScore column "
                "holds embedding cosine similarity instead — a coarser, sentence-level "
                "measure that is *not* comparable to the base paper's BERTScore F1. "
                "Install it with `uv add bert-score` for the real metric.\n"
            )
        for key, value in generation.as_row().items():
            print(f"  {key}: {value}")

    report = "\n".join(sections) + "\n"
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"\nReport written to {report_path}")
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ablation": [s.as_row() for s in ablation]}
        if not args.skip_generation:
            payload["generation"] = generation.as_row()
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Raw results written to {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="app.eval.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index a file or directory of study material")
    p_ingest.add_argument("path")

    p_build = sub.add_parser("build-benchmark", help="Generate LLM-as-teacher Q&A pairs")
    p_build.add_argument("--count", type=int, default=97)
    p_build.add_argument("--visual", action="store_true", help="Build from visual chunks only")
    p_build.add_argument("--seed", type=int, default=42)
    p_build.add_argument("--out")

    p_run = sub.add_parser("run", help="Score the benchmark: ablation + generation metrics")
    p_run.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    p_run.add_argument("--visual-benchmark", default=str(DEFAULT_VISUAL_BENCHMARK))
    p_run.add_argument("--report", help="Write a Markdown report here")
    p_run.add_argument("--json", help="Write raw results here")
    p_run.add_argument("--skip-generation", action="store_true", help="Retrieval metrics only")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "build-benchmark":
        asyncio.run(_build_benchmark(args))
    elif args.command == "run":
        if not bertscore_available():
            print("(bert-score not installed — generation will fall back to embedding cosine)\n")
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
