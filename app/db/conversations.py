"""Durable local conversations with atomic turn reservations and replay protection."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class ConversationConflict(Exception):
    pass


class ConversationStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at REAL NOT NULL,
                updated_at REAL NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
                messages TEXT NOT NULL DEFAULT '[]', summary TEXT NOT NULL DEFAULT '',
                summarized_count INTEGER NOT NULL DEFAULT 0,
                pending_id TEXT, pending_at REAL)""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def decode(row):
        result = dict(row)
        result["messages"] = json.loads(result["messages"])
        result.pop("pending_id", None)
        result.pop("pending_at", None)
        return result

    def create(self, title="New conversation"):
        identifier, now = str(uuid.uuid4()), time.time()
        with self.connect() as db:
            db.execute("INSERT INTO conversations (id,title,created_at,updated_at) VALUES (?,?,?,?)",
                       (identifier, title, now, now))
        return self.get(identifier)

    def get(self, identifier):
        with self.connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError(identifier)
        return self.decode(row)

    def import_history(self, identifier, title, messages):
        now = time.time()
        with self.connect() as db:
            db.execute("""INSERT OR IGNORE INTO conversations
                (id,title,created_at,updated_at,messages,revision) VALUES (?,?,?,?,?,?)""",
                       (identifier, title, now, now, json.dumps(messages), len(messages) // 2))
        return self.get(identifier)

    def list(self):
        with self.connect() as db:
            rows = db.execute("""SELECT id,title,created_at,updated_at,revision
                FROM conversations ORDER BY updated_at DESC LIMIT 200""").fetchall()
        return [dict(row) for row in rows]

    def rename(self, identifier, title):
        with self.connect() as db:
            if not db.execute("UPDATE conversations SET title=? WHERE id=?", (title, identifier)).rowcount:
                raise KeyError(identifier)
        return self.get(identifier)

    def delete(self, identifier):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT pending_at FROM conversations WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            if row[0] and time.time() - row[0] < 900:
                raise ConversationConflict("Wait for the current answer before deleting this conversation.")
            db.execute("DELETE FROM conversations WHERE id=?", (identifier,))

    def reserve(self, identifier, request_id, revision, fingerprint):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM conversations WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            conversation = self.decode(row)
            for message in conversation["messages"]:
                if message.get("request_id") == request_id and message["role"] == "assistant":
                    if message.get("fingerprint") != fingerprint:
                        raise ConversationConflict("This request ID was already used for a different question.")
                    return conversation, True
            if row["revision"] != revision:
                raise ConversationConflict("Conversation changed. Reload it before sending again.")
            if row["pending_id"] and time.time() - row["pending_at"] < 900:
                raise ConversationConflict("An answer is already being generated in this conversation.")
            db.execute("UPDATE conversations SET pending_id=?,pending_at=? WHERE id=?",
                       (request_id, time.time(), identifier))
        return conversation, False

    def release(self, identifier, request_id):
        with self.connect() as db:
            db.execute("UPDATE conversations SET pending_id=NULL,pending_at=NULL WHERE id=? AND pending_id=?",
                       (identifier, request_id))

    def complete(self, identifier, request_id, fingerprint, query, result, summary, summarized_count):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM conversations WHERE id=?", (identifier,)).fetchone()
            if row is None or row["pending_id"] != request_id:
                raise ConversationConflict("This answer no longer owns the conversation reservation.")
            messages = json.loads(row["messages"])
            now = time.time()
            messages.extend([
                {"id": str(uuid.uuid4()), "role": "user", "content": query, "created_at": now},
                {"id": str(uuid.uuid4()), "role": "assistant", "content": result["answer"],
                 "created_at": now, "result": result, "request_id": request_id, "fingerprint": fingerprint},
            ])
            title = query[:80] if row["revision"] == 0 and row["title"] == "New conversation" else row["title"]
            db.execute("""UPDATE conversations SET messages=?,title=?,updated_at=?,revision=revision+1,
                summary=?,summarized_count=?,pending_id=NULL,pending_at=NULL WHERE id=?""",
                       (json.dumps(messages), title, now, summary, summarized_count, identifier))
        return self.get(identifier)
