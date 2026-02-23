from __future__ import annotations

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
