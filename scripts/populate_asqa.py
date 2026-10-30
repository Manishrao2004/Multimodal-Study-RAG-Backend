"""Populate SQLite Knowledge Base directly using Hugging Face's `datasets` package."""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package not found.", file=sys.stderr)
    print("Please install it with: pip install datasets", file=sys.stderr)
    sys.exit(1)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = BACKEND_ROOT / "data" / "kb" / "knowledge_base.sqlite3"


def ensure_db_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id     TEXT PRIMARY KEY,
        source_file  TEXT NOT NULL,
        page_number  INTEGER,
        section_path TEXT,
        text         TEXT NOT NULL,
        chunk_length INTEGER NOT NULL,
        type         TEXT NOT NULL,
        image_ref    TEXT,
        timestamp_start REAL,
        timestamp_end REAL
    );
    CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_file);
    CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(type);
    """)


def main():
    parser = argparse.ArgumentParser(description="Populate KB with ASQA using Hugging Face datasets.")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev"], help="Dataset split to use")
    parser.add_argument("--limit", type=int, default=100, help="Number of questions to ingest (default: 100)")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Path to knowledge_base.sqlite3")
    args = parser.parse_args()

    print(f"Loading ASQA ('din0s/asqa', split='{args.split}') from Hugging Face...")
    dataset = load_dataset("din0s/asqa", split=args.split)

    total_samples = len(dataset)
    limit = min(args.limit, total_samples)
    print(f"Dataset loaded. Processing {limit} of {total_samples} samples...")

    conn = sqlite3.connect(args.db)
    ensure_db_schema(conn)

    chunks_to_insert = []
    seen_texts = set()

    for idx in range(limit):
        item = dataset[idx]
        q_id = item.get("sample_id") or f"asqa_{idx}"

        # Extract passages from annotations knowledge & qa_pairs context
        passages = []
        for ann in item.get("annotations", []):
            for kn in ann.get("knowledge", []):
                content = (kn.get("content") or "").strip()
                title = kn.get("wikipage") or "Wikipedia"
                if content and content not in seen_texts:
                    seen_texts.add(content)
                    passages.append((title, content))

        for qa in item.get("qa_pairs", []):
            ctx = (qa.get("context") or "").strip()
            title = qa.get("wikipage") or "Wikipedia"
            if ctx and ctx not in seen_texts:
                seen_texts.add(ctx)
                passages.append((title, ctx))

        for p_idx, (title, text) in enumerate(passages, 1):
            chunk_id = f"{q_id}_p{p_idx}"
            safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(title)).strip("_").lower() or "doc"
            source_file = f"asqa_{safe_title}.txt"
            chunks_to_insert.append((
                chunk_id,
                source_file,
                1,              # page_number
                str(title),     # section_path
                text,
                len(text),
                "text",
                None,           # image_ref
                None,           # timestamp_start
                None,           # timestamp_end
            ))

    print(f"Inserting {len(chunks_to_insert)} extracted reference chunks into {args.db}...")
    cur = conn.cursor()
    cur.executemany("""
        INSERT OR REPLACE INTO chunks
        (chunk_id, source_file, page_number, section_path, text, chunk_length, type, image_ref, timestamp_start, timestamp_end)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, chunks_to_insert)
    conn.commit()

    cur.execute("SELECT count(*) FROM chunks")
    total_db_chunks = cur.fetchone()[0]
    conn.close()

    print(f"Done! Successfully populated DB. Total chunks in Knowledge Base: {total_db_chunks}")


if __name__ == "__main__":
    main()
