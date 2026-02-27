from __future__ import annotations

from docx import Document

import docxru.pipeline as pipeline
from docxru.config import LLMConfig, PipelineConfig


def test_translate_docx_skips_formatting_passes_by_default(monkeypatch, tmp_path):
    input_path = tmp_path / "source.docx"
    output_path = tmp_path / "out.docx"

    doc = Document()
    doc.add_paragraph("Install actuator")
    doc.save(str(input_path))

    def _fail_abbyy_layout(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("_apply_abbyy_and_layout_passes should be skipped by default in translate mode")

    monkeypatch.setattr(pipeline, "_apply_abbyy_and_layout_passes", _fail_abbyy_layout)

    cfg = PipelineConfig(
        llm=LLMConfig(provider="mock", model="mock"),
        mode="reflow",
        # Even with aggressive values present, translate gate should skip the pass by default.
        abbyy_profile="full",
        layout_check=True,
        layout_auto_fix=True,
        font_shrink_table_pt=1.0,
    )

    pipeline.translate_docx(input_path=input_path, output_path=output_path, cfg=cfg)
    assert output_path.exists()
