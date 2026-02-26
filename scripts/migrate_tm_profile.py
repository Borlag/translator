from __future__ import annotations

import argparse
import hashlib
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from docxru.config import load_config
from docxru.pipeline import _TM_RULESET_VERSION, _build_tm_profile_key


def _read_optional_text(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _profile_key_from_config(config_path: Path) -> str:
    cfg = load_config(config_path)
    custom_system_prompt = _read_optional_text(cfg.llm.system_prompt_path)
    glossary_text = _read_optional_text(cfg.llm.glossary_path)
    return _build_tm_profile_key(
        cfg,
        custom_system_prompt=custom_system_prompt,
        glossary_text=glossary_text,
    )


def _source_hash(*, profile_key: str, source_norm: str) -> str:
    payload = f"{_TM_RULESET_VERSION}\n{profile_key}\n{source_norm}"
    h = hashlib.sha256()
    h.update(payload.encode("utf-8"))
    return h.hexdigest()


def _backup_sqlite(db_path: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = db_path.with_name(f"{db_path.name}.bak-{stamp}")
    shutil.copy2(db_path, backup_path)
    for suffix in ("-wal", "-shm"):
        src = Path(f"{db_path}{suffix}")
        if src.exists():
            shutil.copy2(src, Path(f"{backup_path}{suffix}"))
    return backup_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate TM/progress hashes from one TM profile to another so --resume can continue "
            "after config changes (for example sequential -> grouped turbo)."
        )
    )
    parser.add_argument("--tm-db", default="translation_cache.sqlite", help="Path to TM sqlite DB.")
    parser.add_argument("--old-config", required=True, help="Old config path (profile currently used by progress cache).")
    parser.add_argument("--new-config", required=True, help="New config path (target profile).")
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without writing DB.")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating sqlite backup before write (not recommended).",
    )
    args = parser.parse_args()

    db_path = Path(args.tm_db).resolve()
    old_config = Path(args.old_config).resolve()
    new_config = Path(args.new_config).resolve()

    if not db_path.exists():
        raise FileNotFoundError(f"TM DB not found: {db_path}")
    if not old_config.exists():
        raise FileNotFoundError(f"Old config not found: {old_config}")
    if not new_config.exists():
        raise FileNotFoundError(f"New config not found: {new_config}")

    old_profile_key = _profile_key_from_config(old_config)
    new_profile_key = _profile_key_from_config(new_config)

    if old_profile_key == new_profile_key:
        print("Old/new TM profile keys are identical. Nothing to migrate.")
        return 0

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT source_hash, source_norm, target_text, meta_json FROM tm_exact")
        rows = cur.fetchall()

        migrate_rows: list[tuple[str, str, str, str]] = []
        hash_map: dict[str, str] = {}
        for source_hash, source_norm, target_text, meta_json in rows:
            source_hash_s = str(source_hash)
            source_norm_s = str(source_norm)
            old_hash = _source_hash(profile_key=old_profile_key, source_norm=source_norm_s)
            if source_hash_s != old_hash:
                continue
            new_hash = _source_hash(profile_key=new_profile_key, source_norm=source_norm_s)
            if new_hash == source_hash_s:
                continue
            migrate_rows.append((new_hash, source_norm_s, str(target_text), str(meta_json or "{}")))
            hash_map[source_hash_s] = new_hash

        progress_updates = 0
        if hash_map:
            placeholders = ",".join("?" for _ in hash_map)
            q = f"SELECT COUNT(*) FROM progress WHERE source_hash IN ({placeholders})"
            progress_updates = int(conn.execute(q, tuple(hash_map.keys())).fetchone()[0] or 0)

        print(f"TM ruleset: {_TM_RULESET_VERSION}")
        print(f"TM rows total: {len(rows)}")
        print(f"TM rows to migrate: {len(migrate_rows)}")
        print(f"Progress rows to update: {progress_updates}")
        print(f"Dry-run: {bool(args.dry_run)}")

        if args.dry_run or not migrate_rows:
            return 0

        backup_path = None
        if not args.no_backup:
            backup_path = _backup_sqlite(db_path)

        now = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        fts_exists = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name='tm_exact_fts'"
            ).fetchone()
        )

        conn.execute("BEGIN")
        try:
            for new_hash, source_norm, target_text, meta_json in migrate_rows:
                conn.execute(
                    """
                    INSERT INTO tm_exact(source_hash, source_norm, target_text, meta_json, created_at, updated_at)
                    VALUES(?,?,?,?,?,?)
                    ON CONFLICT(source_hash) DO UPDATE SET
                      source_norm=excluded.source_norm,
                      target_text=excluded.target_text,
                      meta_json=excluded.meta_json,
                      updated_at=excluded.updated_at
                    """,
                    (new_hash, source_norm, target_text, meta_json, now, now),
                )
                if fts_exists:
                    conn.execute("DELETE FROM tm_exact_fts WHERE source_hash = ?", (new_hash,))
                    conn.execute(
                        "INSERT INTO tm_exact_fts(source_hash, source_norm) VALUES(?, ?)",
                        (new_hash, source_norm),
                    )

            for old_hash, new_hash in hash_map.items():
                conn.execute("UPDATE progress SET source_hash = ?, updated_at = ? WHERE source_hash = ?", (new_hash, now, old_hash))
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        if backup_path is not None:
            print(f"Backup created: {backup_path}")
        print("Migration completed.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
