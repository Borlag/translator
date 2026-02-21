from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def normalize_text(text: str) -> str:
    # Keep newlines but collapse whitespace inside lines.
    lines = [" ".join(line.split()) for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def sha256_hex(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


@dataclass
class TMHit:
    source_hash: str
    target_text: str
    meta: dict[str, Any]


class TMStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_db()

    def close(self) -> None:
        self.conn.close()

    def _init_db(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tm_exact (
              source_hash TEXT PRIMARY KEY,
              source_norm TEXT NOT NULL,
              target_text TEXT NOT NULL,
              meta_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS progress (
              segment_id TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              source_hash TEXT,
              error TEXT,
              updated_at TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def get_exact(self, source_hash: str) -> Optional[TMHit]:
        cur = self.conn.execute(
            "SELECT source_hash, target_text, meta_json FROM tm_exact WHERE source_hash = ?",
            (source_hash,),
        )
        row = cur.fetchone()
        if not row:
            return None
        meta = json.loads(row[2]) if row[2] else {}
        return TMHit(source_hash=row[0], target_text=row[1], meta=meta)

    def put_exact(self, source_hash: str, source_norm: str, target_text: str, meta: dict[str, Any]) -> None:
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        meta_json = json.dumps(meta, ensure_ascii=False)
        self.conn.execute(
            """
            INSERT INTO tm_exact(source_hash, source_norm, target_text, meta_json, created_at, updated_at)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(source_hash) DO UPDATE SET
              source_norm=excluded.source_norm,
              target_text=excluded.target_text,
              meta_json=excluded.meta_json,
              updated_at=excluded.updated_at
            """,
            (source_hash, source_norm, target_text, meta_json, now, now),
        )
        self.conn.commit()

    def set_progress(self, segment_id: str, status: str, source_hash: str | None = None, error: str | None = None) -> None:
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self.conn.execute(
            """
            INSERT INTO progress(segment_id, status, source_hash, error, updated_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(segment_id) DO UPDATE SET
              status=excluded.status,
              source_hash=excluded.source_hash,
              error=excluded.error,
              updated_at=excluded.updated_at
            """,
            (segment_id, status, source_hash, error, now),
        )
        self.conn.commit()

    def get_progress(self, segment_id: str) -> Optional[dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT segment_id, status, source_hash, error, updated_at FROM progress WHERE segment_id=?",
            (segment_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "segment_id": row[0],
            "status": row[1],
            "source_hash": row[2],
            "error": row[3],
            "updated_at": row[4],
        }
