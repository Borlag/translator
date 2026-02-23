from __future__ import annotations

import docxru.pipeline as pipeline
from docxru.config import PipelineConfig


def test_translate_plain_chunk_applies_cleanup_for_latin_chunk(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "_translate_shielded_fragment",
        lambda text, llm_client, context: (text, []),  # noqa: ARG005
    )

    source = "Application of Ardrox AV100D to the Upper Diaphragm Tube (15-390) (Sheet 1 of 3)"
    out, issues = pipeline._translate_plain_chunk(
        source,
        PipelineConfig(),
        llm_client=object(),
        context={},
        cache={},
    )

    assert "Нанесение Ardrox AV100D на Upper Diaphragm Tube" in out
    assert "(Лист 1 из 3)" in out
    assert any(issue.code == "plain_chunk_cleanup_applied" for issue in issues)
