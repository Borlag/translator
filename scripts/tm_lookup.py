from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

_BRACKET_TOKEN_RE = re.compile(r"(?:🦦[^🧧]*🧧|⟦[^⟧]*⟧)")


def _clean_for_display(text: str) -> str:
    return _BRACKET_TOKEN_RE.sub("", text or "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search translation TM cache by source/target term.")
    p.add_argument("--tm", default="translation_cache.sqlite", help="Path to TM sqlite file.")
    p.add_argument("--term", required=True, help="Search term (EN or RU).")
    p.add_argument("--limit", type=int, default=20, help="Max rows to print.")
    p.add_argument("--json", action="store_true", help="Print JSON records instead of plain text.")
    return p.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    tm_path = Path(args.tm)
    if not tm_path.exists():
        raise SystemExit(f"TM file not found: {tm_path}")

    term = args.term.strip().lower()
    if not term:
        raise SystemExit("Search term is empty.")

    conn = sqlite3.connect(tm_path)
    try:
        cur = conn.execute(
            """
            SELECT source_hash, source_norm, target_text, meta_json, updated_at
            FROM tm_exact
            WHERE LOWER(source_norm) LIKE ? OR LOWER(target_text) LIKE ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (f"%{term}%", f"%{term}%", max(1, int(args.limit))),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        print("No matches.")
        return 0

    for idx, row in enumerate(rows, start=1):
        source_hash, source_norm, target_text, meta_json, updated_at = row
        try:
            meta = json.loads(meta_json) if meta_json else {}
        except json.JSONDecodeError:
            meta = {}
        rec = {
            "n": idx,
            "updated_at": updated_at,
            "source_hash": source_hash,
            "provider": meta.get("provider"),
            "model": meta.get("model"),
            "location": meta.get("location"),
            "section_header": meta.get("section_header"),
            "origin": meta.get("origin"),
            "source_text": source_norm,
            "target_text": target_text,
        }
        if args.json:
            print(json.dumps(rec, ensure_ascii=False))
            continue

        print(f"[{idx}] {updated_at} | {rec['provider']}/{rec['model']} | {rec['location']}")
        if rec["section_header"]:
            print(f"    SECTION: {rec['section_header']}")
        print(f"    EN: {_clean_for_display(source_norm)}")
        print(f"    RU: {_clean_for_display(target_text)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
