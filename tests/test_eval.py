from __future__ import annotations

import json
from pathlib import Path

from docx import Document

from docxru.config import LLMConfig, PipelineConfig
from docxru.eval import evaluate_batch, evaluate_single, summarize_results, write_eval_report


def _make_docx(path: Path, text: str) -> None:
    doc = Document()
    doc.add_paragraph(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)


def test_evaluate_single_with_mock_provider(tmp_path: Path):
    input_path = tmp_path / "input.docx"
    output_dir = tmp_path / "out"
    _make_docx(input_path, "Remove the bolt.")

    cfg = PipelineConfig(llm=LLMConfig(provider="mock"), concurrency=1)
    result = evaluate_single(input_path=input_path, output_dir=output_dir, cfg=cfg)

    assert result.success
    assert result.error_message is None
    assert Path(result.output_path).exists()
    assert result.segment_count >= 1
    assert result.translated_segments >= 1


def test_evaluate_batch_and_write_report(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "out"
    _make_docx(input_dir / "a.docx", "Install the fitting.")
    _make_docx(input_dir / "b.docx", "Torque the nut.")

    cfg = PipelineConfig(llm=LLMConfig(provider="mock"), concurrency=1)
    results = evaluate_batch(input_dir=input_dir, output_dir=output_dir, cfg=cfg)
    summary = summarize_results(results)

    assert len(results) == 2
    assert summary["files_total"] == 2
    assert summary["files_failed"] == 0

    report_path = output_dir / "eval_report.json"
    report_summary = write_eval_report(results, report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_summary["files_total"] == 2
    assert payload["summary"]["files_success"] == 2
    assert len(payload["results"]) == 2
