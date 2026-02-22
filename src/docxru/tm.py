from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
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


@dataclass
class FuzzyTMHit:
    source_hash: str
    source_norm: str
    target_text: str
    similarity: float
    meta: dict[str, Any]


class TMStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._fts_enabled = False
        self._init_db()

    def close(self) -> None:
        self.conn.close()

    @property
    def fts_enabled(self) -> bool:
        return bool(self._fts_enabled)

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
        self._fts_enabled = self._init_fts()
        self.conn.commit()

    def _init_fts(self) -> bool:
        try:
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS tm_exact_fts
                USING fts5(source_hash UNINDEXED, source_norm);
                """
            )
            self.conn.execute(
                """
                INSERT INTO tm_exact_fts(source_hash, source_norm)
                SELECT source_hash, source_norm FROM tm_exact
                WHERE source_hash NOT IN (SELECT source_hash FROM tm_exact_fts);
                """
            )
            return True
        except sqlite3.OperationalError:
            return False

    def _sync_fts_entry(self, source_hash: str, source_norm: str) -> None:
        if not self._fts_enabled:
            return
        self.conn.execute("DELETE FROM tm_exact_fts WHERE source_hash = ?", (source_hash,))
        self.conn.execute(
            "INSERT INTO tm_exact_fts(source_hash, source_norm) VALUES(?, ?)",
            (source_hash, source_norm),
        )

    @staticmethod
    def _build_fts_query(text: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9]{2,}", (text or "").lower())
        unique_tokens = list(dict.fromkeys(tokens))
        if not unique_tokens:
            return ""
        return " OR ".join(f'"{token}"' for token in unique_tokens[:24])

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

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

    def get_fuzzy(
        self,
        source_norm: str,
        *,
        top_k: int = 3,
        min_similarity: float = 0.75,
    ) -> list[FuzzyTMHit]:
        if not self._fts_enabled:
            return []

        query = self._build_fts_query(source_norm)
        if not query:
            return []

        top_limit = max(1, int(top_k))
        candidate_limit = max(top_limit * 8, 24)
        try:
            cur = self.conn.execute(
                """
                SELECT e.source_hash, e.source_norm, e.target_text, e.meta_json
                FROM tm_exact_fts f
                JOIN tm_exact e ON e.source_hash = f.source_hash
                WHERE tm_exact_fts MATCH ?
                LIMIT ?
                """,
                (query, candidate_limit),
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            return []

        hits: list[FuzzyTMHit] = []
        for row in rows:
            try:
                candidate_source = str(row[1])
                similarity = float(SequenceMatcher(None, source_norm, candidate_source).ratio())
                if similarity < float(min_similarity):
                    continue
                meta = json.loads(row[3]) if row[3] else {}
                hits.append(
                    FuzzyTMHit(
                        source_hash=str(row[0]),
                        source_norm=candidate_source,
                        target_text=str(row[2]),
                        similarity=similarity,
                        meta=meta,
                    )
                )
            except Exception:
                continue

        hits.sort(key=lambda hit: hit.similarity, reverse=True)
        return hits[:top_limit]

    def put_exact(self, source_hash: str, source_norm: str, target_text: str, meta: dict[str, Any]) -> None:
        now = self._utc_now_iso()
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
        self._sync_fts_entry(source_hash, source_norm)
        self.conn.commit()

    def set_progress(self, segment_id: str, status: str, source_hash: str | None = None, error: str | None = None) -> None:
        now = self._utc_now_iso()
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
