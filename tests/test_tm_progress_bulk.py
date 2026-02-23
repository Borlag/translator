from __future__ import annotations

from docxru.tm import TMStore


def test_tm_store_get_progress_bulk_returns_matching_rows(tmp_path):
    tm = TMStore(tmp_path / "tm.sqlite")
    try:
        tm.set_progress("seg-1", "ok", source_hash="h1")
        tm.set_progress("seg-2", "error", source_hash="h2", error="boom")

        out = tm.get_progress_bulk(["seg-1", "seg-2", "seg-3"])

        assert set(out.keys()) == {"seg-1", "seg-2"}
        assert out["seg-1"]["status"] == "ok"
        assert out["seg-1"]["source_hash"] == "h1"
        assert out["seg-2"]["status"] == "error"
        assert out["seg-2"]["error"] == "boom"
    finally:
        tm.close()


def test_tm_store_get_progress_bulk_handles_empty_input(tmp_path):
    tm = TMStore(tmp_path / "tm.sqlite")
    try:
        assert tm.get_progress_bulk([]) == {}
    finally:
        tm.close()
