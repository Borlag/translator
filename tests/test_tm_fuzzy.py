from __future__ import annotations

import pytest

from docxru.tm import TMStore


def test_tm_store_fuzzy_ranking_and_threshold(tmp_path):
    tm = TMStore(tmp_path / "tm.sqlite")
    try:
        tm.put_exact(
            source_hash="h1",
            source_norm="install lower bearing subassembly",
            target_text="установите нижний узел подшипника",
            meta={},
        )
        tm.put_exact(
            source_hash="h2",
            source_norm="remove upper torque link",
            target_text="снимите верхний рычаг крутящего момента",
            meta={},
        )
        tm.put_exact(
            source_hash="h3",
            source_norm="install lower bearing assembly",
            target_text="установите нижнюю сборку подшипника",
            meta={},
        )

        if not tm.fts_enabled:
            pytest.skip("SQLite FTS5 is unavailable in this environment")

        hits = tm.get_fuzzy(
            "install lower bearing subassembly",
            top_k=3,
            min_similarity=0.40,
        )
        assert hits
        assert hits[0].source_hash == "h1"
        assert hits[0].similarity >= hits[-1].similarity

        strict_hits = tm.get_fuzzy(
            "install lower bearing subassembly",
            top_k=3,
            min_similarity=1.01,
        )
        assert strict_hits == []
    finally:
        tm.close()


def test_tm_store_fuzzy_returns_empty_when_fts_disabled(tmp_path):
    tm = TMStore(tmp_path / "tm.sqlite")
    try:
        tm.put_exact(
            source_hash="h1",
            source_norm="install lower bearing subassembly",
            target_text="установите нижний узел подшипника",
            meta={},
        )
        tm._fts_enabled = False
        hits = tm.get_fuzzy("install lower bearing", top_k=3, min_similarity=0.1)
        assert hits == []
    finally:
        tm.close()

