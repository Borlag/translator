from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sqlite3
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


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
    def __init__(
        self,
        path: str | Path,
        *,
        fuzzy_token_regex: str = r"[A-Za-zА-Яа-яЁё0-9]{2,}",
        fuzzy_rank_mode: str = "hybrid",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fts_enabled = False
        token_pattern = (fuzzy_token_regex or "").strip() or r"[A-Za-zА-Яа-яЁё0-9]{2,}"
        try:
            self._fuzzy_token_re = re.compile(token_pattern, flags=re.UNICODE)
        except re.error:
            self._fuzzy_token_re = re.compile(r"[A-Za-zА-Яа-яЁё0-9]{2,}", flags=re.UNICODE)
        rank_mode = (fuzzy_rank_mode or "").strip().lower()
        self._fuzzy_rank_mode = rank_mode if rank_mode in {"sequence", "hybrid"} else "hybrid"
        self.conn = self._connect_with_recovery()

    def close(self) -> None:
        self.conn.close()

    @property
    def fts_enabled(self) -> bool:
        return bool(self._fts_enabled)

    def _connect_sqlite(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.DatabaseError:
            conn.close()
            raise
        return conn

    def _quarantine_corrupt_sqlite(self) -> None:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        for suffix in ("", "-wal", "-shm"):
            src = Path(f"{self.path}{suffix}")
            if not src.exists():
                continue
            dst = Path(f"{src}.corrupt-{stamp}")
            try:
                src.replace(dst)
            except OSError:
                # Best effort: if quarantine move fails, next connect may still work.
                continue

    def _connect_with_recovery(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect_sqlite()
            self.conn = conn
            self._init_db()
            return conn
        except sqlite3.DatabaseError:
            if conn is not None:
                conn.close()
            self._quarantine_corrupt_sqlite()
            conn = self._connect_sqlite()
            self.conn = conn
            self._init_db()
            return conn

    @staticmethod
    def _is_corruption_error(exc: sqlite3.DatabaseError) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "database disk image is malformed",
                "file is not a database",
                "not a database",
                "malformed",
            )
        )

    def _recover_runtime_corruption(self) -> None:
        with suppress(sqlite3.Error):
            self.conn.close()
        self._quarantine_corrupt_sqlite()
        self.conn = self._connect_sqlite()
        self._init_db()

    def _run_with_recovery(self, operation: Callable[[], T]) -> T:
        try:
            return operation()
        except sqlite3.DatabaseError as exc:
            if not self._is_corruption_error(exc):
                raise
            self._recover_runtime_corruption()
            return operation()

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
                USING fts5(source_hash UNINDEXED, source_norm, tokenize='unicode61');
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

    def _tokenize_fuzzy(self, text: str) -> list[str]:
        tokens = [m.group(0).lower() for m in self._fuzzy_token_re.finditer(text or "")]
        return [token for token in tokens if token]

    def _build_fts_query(self, text: str) -> str:
        tokens = self._tokenize_fuzzy(text)
        unique_tokens = list(dict.fromkeys(tokens))
        if not unique_tokens:
            return ""
        return " OR ".join(f'"{token}"' for token in unique_tokens[:24])

    def _similarity_score(self, source_norm: str, candidate_source: str) -> float:
        sequence_score = float(SequenceMatcher(None, source_norm, candidate_source).ratio())
        if self._fuzzy_rank_mode == "sequence":
            return sequence_score
        source_tokens = set(self._tokenize_fuzzy(source_norm))
        candidate_tokens = set(self._tokenize_fuzzy(candidate_source))
        if not source_tokens and not candidate_tokens:
            token_score = 0.0
        else:
            union = source_tokens | candidate_tokens
            token_score = float(len(source_tokens & candidate_tokens) / max(1, len(union)))
        # Hybrid ranking favors lexical overlap while keeping character-level sensitivity.
        return (0.65 * token_score) + (0.35 * sequence_score)

    @staticmethod
    def _utc_now_iso() -> str:
        return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    def get_exact(self, source_hash: str) -> TMHit | None:
        def _operation() -> TMHit | None:
            cur = self.conn.execute(
                "SELECT source_hash, target_text, meta_json FROM tm_exact WHERE source_hash = ?",
                (source_hash,),
            )
            row = cur.fetchone()
            if not row:
                return None
            meta = json.loads(row[2]) if row[2] else {}
            return TMHit(source_hash=row[0], target_text=row[1], meta=meta)

        return self._run_with_recovery(_operation)

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
        def _operation() -> list[FuzzyTMHit]:
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
                    similarity = self._similarity_score(source_norm, candidate_source)
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

        return self._run_with_recovery(_operation)

    def put_exact(self, source_hash: str, source_norm: str, target_text: str, meta: dict[str, Any]) -> None:
        def _operation() -> None:
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

        self._run_with_recovery(_operation)

    def set_progress(self, segment_id: str, status: str, source_hash: str | None = None, error: str | None = None) -> None:
        def _operation() -> None:
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

        self._run_with_recovery(_operation)

    def get_progress(self, segment_id: str) -> dict[str, Any] | None:
        def _operation() -> dict[str, Any] | None:
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

        return self._run_with_recovery(_operation)

    def get_progress_bulk(self, segment_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not segment_ids:
            return {}

        def _operation() -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, Any]] = {}
            # Keep below common SQLite host parameter limits.
            chunk_size = 900
            for i in range(0, len(segment_ids), chunk_size):
                chunk = [seg_id for seg_id in segment_ids[i : i + chunk_size] if seg_id]
                if not chunk:
                    continue
                placeholders = ",".join("?" for _ in chunk)
                cur = self.conn.execute(
                    f"SELECT segment_id, status, source_hash, error, updated_at FROM progress WHERE segment_id IN ({placeholders})",
                    tuple(chunk),
                )
                for row in cur.fetchall():
                    out[str(row[0])] = {
                        "segment_id": row[0],
                        "status": row[1],
                        "source_hash": row[2],
                        "error": row[3],
                        "updated_at": row[4],
                    }
            return out

        return self._run_with_recovery(_operation)
