from __future__ import annotations

import pytest

from docxru.config import LLMConfig, PdfConfig, PipelineConfig, TMConfig
from docxru.pdf_pipeline import translate_pdf

fitz = pytest.importorskip("fitz")


def test_translate_pdf_with_mock_provider_writes_output_and_qa(tmp_path):
    input_pdf = tmp_path / "input.pdf"
    output_pdf = tmp_path / "output.pdf"

    doc = fitz.open()
    page = doc.new_page(width=320, height=200)
    page.insert_text((50, 80), "Install bolt", fontsize=12)
    doc.save(str(input_pdf))
    doc.close()

    cfg = PipelineConfig(
        llm=LLMConfig(provider="mock", glossary_prompt_mode="off"),
        tm=TMConfig(path=str(tmp_path / "tm.sqlite")),
        pdf=PdfConfig(),
        concurrency=1,
        qa_report_path=str(tmp_path / "qa_report.html"),
        qa_jsonl_path=str(tmp_path / "qa.jsonl"),
        log_path=str(tmp_path / "run.log"),
    )

    translate_pdf(input_pdf, output_pdf, cfg, resume=False)

    assert output_pdf.exists()
    out_doc = fitz.open(str(output_pdf))
    text = out_doc[0].get_text()
    out_doc.close()
    assert "Install bolt" in text
    assert (tmp_path / "qa_report.html").exists()
    assert (tmp_path / "qa.jsonl").exists()

