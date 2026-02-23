from __future__ import annotations

from pathlib import Path

from docxru.studio_server import (
    StudioRunManager,
    _build_studio_html,
    _default_model_for_provider,
    _infer_translation_cmd,
    _list_openai_models,
)


def test_infer_translation_cmd_for_docx_and_pdf():
    cmd_docx, out_docx = _infer_translation_cmd(Path("input.docx"))
    cmd_pdf, out_pdf = _infer_translation_cmd(Path("input.pdf"))
    assert cmd_docx == "translate"
    assert out_docx.endswith(".docx")
    assert cmd_pdf == "translate-pdf"
    assert out_pdf.endswith(".pdf")


def test_studio_build_config_payload_includes_checker_settings(tmp_path):
    manager = StudioRunManager(base_dir=tmp_path)
    run_id = "run123"
    run_dir = tmp_path / "runs" / run_id
    payload = manager._build_config_payload(  # noqa: SLF001 - intentional for unit-level validation
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.1,
        max_output_tokens=2000,
        concurrency=3,
        prompt_path=tmp_path / "prompt.md",
        glossary_path=tmp_path / "glossary.md",
        checker_enabled=True,
        checker_provider="openai",
        checker_model="gpt-4o-mini",
        checker_pages_per_chunk=3,
        checker_fallback_segments_per_chunk=80,
        checker_temperature=0.0,
        checker_max_output_tokens=1500,
        checker_openai_batch_enabled=True,
        run_base_dir=tmp_path / "runs",
        run_id=run_id,
        run_dir=run_dir,
    )
    assert payload["llm"]["provider"] == "openai"
    assert payload["llm"]["model"] == "gpt-4o-mini"
    assert payload["llm"]["auto_model_sizing"] is True
    assert payload["checker"]["enabled"] is True
    assert payload["checker"]["provider"] == "openai"
    assert payload["checker"]["model"] == "gpt-4o-mini"
    assert payload["checker"]["pages_per_chunk"] == 3
    assert payload["checker"]["fallback_segments_per_chunk"] == 80
    assert payload["checker"]["openai_batch_enabled"] is True
    assert payload["run"]["run_id"] == run_id
    assert payload["concurrency"] == 3


def test_default_model_for_provider():
    assert _default_model_for_provider("openai") == "gpt-4o-mini"
    assert _default_model_for_provider("ollama") == "qwen2.5:7b"
    assert _default_model_for_provider("google") == "ignored"
    assert _default_model_for_provider("mock") == "mock"


def test_list_openai_models_without_key_returns_fallback():
    models = _list_openai_models("")
    assert "gpt-4o-mini" in models


def test_studio_html_contains_checker_trace_widgets():
    html = _build_studio_html()
    assert "checkerTraceLink" in html
    assert "checkerTraceTail" in html
    assert "checkerOpenaiBatch" in html
