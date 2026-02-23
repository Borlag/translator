from __future__ import annotations

from docxru.config import PipelineConfig
from docxru.run_context import resolve_run_paths


def test_resolve_run_paths_keeps_absolute_qa_paths(tmp_path):
    qa_html = tmp_path / "qa_report.html"
    qa_jsonl = tmp_path / "qa.jsonl"
    cfg = PipelineConfig(
        qa_report_path=str(qa_html),
        qa_jsonl_path=str(qa_jsonl),
    )
    paths = resolve_run_paths(cfg, output_path=tmp_path / "out.docx")
    assert paths.qa_report_path == qa_html
    assert paths.qa_jsonl_path == qa_jsonl


def test_resolve_run_paths_places_relative_outputs_under_run_dir(tmp_path):
    cfg = PipelineConfig(
        qa_report_path="qa_report.html",
        qa_jsonl_path="qa.jsonl",
    )
    paths = resolve_run_paths(cfg, output_path=tmp_path / "out.docx")
    assert paths.qa_report_path.parent == paths.run_dir
    assert paths.qa_jsonl_path.parent == paths.run_dir
    assert paths.checker_trace_path.parent == paths.run_dir
