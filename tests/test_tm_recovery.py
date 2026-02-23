from __future__ import annotations

import sqlite3
from pathlib import Path

from docxru.tm import TMStore


def test_tmstore_recovers_from_corrupt_sqlite(tmp_path):
    db_path = tmp_path / "tm.sqlite"
    db_path.write_bytes(b"this is not sqlite")

    tm = TMStore(db_path)
    tm.put_exact("h1", "source", "target", {"meta": 1})
    hit = tm.get_exact("h1")
    tm.close()

    assert hit is not None
    assert hit.target_text == "target"
    backups = list(tmp_path.glob("tm.sqlite.corrupt-*"))
    assert backups


def test_tmstore_recovers_progress_after_corrupt_file(tmp_path):
    db_path = Path(tmp_path / "tm2.sqlite")
    db_path.write_bytes(b"broken")

    tm = TMStore(db_path)
    tm.set_progress("seg-1", "ok", source_hash="abc")
    progress = tm.get_progress("seg-1")
    tm.close()

    assert progress is not None
    assert progress["status"] == "ok"


def test_tmstore_recovers_from_runtime_corruption_during_progress_write(tmp_path):
    db_path = tmp_path / "tm3.sqlite"
    tm = TMStore(db_path)

    class _FlakyConn:
        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner
            self._raised = False

        def execute(self, *args: object, **kwargs: object):  # noqa: ANN002, ANN003
            if not self._raised:
                self._raised = True
                raise sqlite3.DatabaseError("database disk image is malformed")
            return self._inner.execute(*args, **kwargs)

        def commit(self) -> None:
            self._inner.commit()

        def close(self) -> None:
            self._inner.close()

    try:
        tm.conn = _FlakyConn(tm.conn)  # type: ignore[assignment]
        tm.set_progress("seg-1", "ok", source_hash="h1")
        progress = tm.get_progress("seg-1")
    finally:
        tm.close()

    assert progress is not None
    assert progress["status"] == "ok"
    backups = list(tmp_path.glob("tm3.sqlite.corrupt-*"))
    assert backups
