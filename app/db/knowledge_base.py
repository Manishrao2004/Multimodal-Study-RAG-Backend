"""SQLite-backed knowledge base. Plays the same role as EduRAG's single CSV
(chunk_id, source_file, page_number, section_path, text, chunk_length, type,
image_ref) but with indexed lookups, since we're serving live queries instead
of running one-off offline scripts.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.models.schemas import Chunk, ChunkType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    source_file  TEXT NOT NULL,
    page_number  INTEGER,
    section_path TEXT,
    text         TEXT NOT NULL,
    chunk_length INTEGER NOT NULL,
    type         TEXT NOT NULL,
    image_ref    TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_file);
"""


class KnowledgeBase:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_chunks(self, chunks: list[Chunk]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO chunks
                   (chunk_id, source_file, page_number, section_path, text,
                    chunk_length, type, image_ref)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        c.chunk_id,
                        c.source_file,
                        c.page_number,
                        c.section_path,
                        c.text,
                        c.chunk_length,
                        c.type.value,
                        c.image_ref,
                    )
                    for c in chunks
                ],
            )

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
        by_id = {row["chunk_id"]: self._row_to_chunk(row) for row in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def all_chunks(self) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM chunks").fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def count(self) -> int:
        with self._connect() as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return n

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            source_file=row["source_file"],
            page_number=row["page_number"],
            section_path=row["section_path"],
            text=row["text"],
            chunk_length=row["chunk_length"],
            type=ChunkType(row["type"]),
            image_ref=row["image_ref"],
        )
