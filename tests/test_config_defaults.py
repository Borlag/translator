from __future__ import annotations

from docxru.config import LLMConfig, PdfConfig, PipelineConfig, load_config


def test_llm_config_default_context_window_chars_is_600():
    assert LLMConfig().context_window_chars == 600
    assert LLMConfig().auto_model_sizing is False


def test_load_config_defaults_context_window_chars_to_600(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("llm:\n  provider: mock\n", encoding="utf-8")

    cfg = load_config(config_path)
    assert cfg.llm.context_window_chars == 600
    assert cfg.llm.auto_model_sizing is False


def test_pdf_config_defaults():
    cfg = PdfConfig()
    assert cfg.max_font_shrink_ratio == 0.6
    assert cfg.table_detection is True
    assert cfg.max_pages is None


def test_load_config_defaults_pdf_section(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("llm:\n  provider: mock\n", encoding="utf-8")

    cfg = load_config(config_path)
    assert cfg.pdf.bilingual_mode is False
    assert cfg.pdf.max_font_shrink_ratio == 0.6


def test_pipeline_config_font_shrink_defaults_are_disabled():
    cfg = PipelineConfig()
    assert cfg.font_shrink_body_pt == 0.0
    assert cfg.font_shrink_table_pt == 0.0


def test_load_config_reads_font_shrink_values(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "llm:\n"
        "  provider: mock\n"
        "font_shrink_body_pt: 1.5\n"
        "font_shrink_table_pt: 2.5\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    assert cfg.font_shrink_body_pt == 1.5
    assert cfg.font_shrink_table_pt == 2.5


def test_pipeline_config_checker_and_pricing_defaults():
    cfg = PipelineConfig()
    assert cfg.checker.enabled is False
    assert cfg.checker.max_output_tokens == 6000
    assert cfg.checker.pages_per_chunk == 3
    assert cfg.checker.fallback_segments_per_chunk == 80
    assert cfg.checker.retries == 0
    assert cfg.checker.safe_output_path == "checker_suggestions_safe.json"
    assert cfg.checker.auto_apply_safe is False
    assert cfg.checker.auto_apply_min_confidence == 0.7
    assert cfg.checker.openai_batch_enabled is False
    assert cfg.checker.openai_batch_completion_window == "24h"
    assert cfg.checker.openai_batch_poll_interval_s == 20.0
    assert cfg.checker.openai_batch_timeout_s == 86400.0
    assert cfg.pricing.enabled is False
    assert cfg.pricing.currency == "USD"
    assert cfg.run.status_flush_every_n_segments == 10
    assert cfg.run.batch_fallback_warn_ratio == 0.08
    assert cfg.run.fail_fast_on_translate_error is True


def test_load_config_reads_checker_pricing_and_run_sections(tmp_path):
    checker_prompt = tmp_path / "checker_prompt.md"
    checker_prompt.write_text("prompt", encoding="utf-8")
    pricing_file = tmp_path / "pricing.yaml"
    pricing_file.write_text("pricing: {}", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "llm:\n"
        "  provider: mock\n"
        "checker:\n"
        "  enabled: true\n"
        "  system_prompt_path: checker_prompt.md\n"
        "  pages_per_chunk: 3\n"
        "  fallback_segments_per_chunk: 80\n"
        "  safe_output_path: checker_safe.json\n"
        "  auto_apply_safe: true\n"
        "  auto_apply_min_confidence: 0.85\n"
        "  openai_batch_enabled: true\n"
        "  openai_batch_completion_window: 24h\n"
        "  openai_batch_poll_interval_s: 10\n"
        "  openai_batch_timeout_s: 7200\n"
        "  only_on_issue_severities: [warn, error]\n"
        "  only_on_issue_codes: [layout_textbox_overflow_risk]\n"
        "pricing:\n"
        "  enabled: true\n"
        "  pricing_path: pricing.yaml\n"
        "  currency: EUR\n"
        "run:\n"
        "  run_dir: runs\n"
        "  status_path: run_status.json\n"
        "  dashboard_html_path: dashboard.html\n"
        "  status_flush_every_n_segments: 5\n"
        "  batch_fallback_warn_ratio: 0.12\n"
        "  fail_fast_on_translate_error: false\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    assert cfg.llm.auto_model_sizing is False
    assert cfg.checker.enabled is True
    assert cfg.checker.system_prompt_path == str(checker_prompt.resolve())
    assert cfg.checker.pages_per_chunk == 3
    assert cfg.checker.fallback_segments_per_chunk == 80
    assert cfg.checker.safe_output_path == "checker_safe.json"
    assert cfg.checker.auto_apply_safe is True
    assert cfg.checker.auto_apply_min_confidence == 0.85
    assert cfg.checker.openai_batch_enabled is True
    assert cfg.checker.openai_batch_completion_window == "24h"
    assert cfg.checker.openai_batch_poll_interval_s == 10.0
    assert cfg.checker.openai_batch_timeout_s == 7200.0
    assert cfg.checker.only_on_issue_severities == ("warn", "error")
    assert cfg.checker.only_on_issue_codes == ("layout_textbox_overflow_risk",)
    assert cfg.pricing.enabled is True
    assert cfg.pricing.pricing_path == str(pricing_file.resolve())
    assert cfg.pricing.currency == "EUR"
    assert cfg.run.run_dir == str((tmp_path / "runs").resolve())
    assert cfg.run.status_path == str((tmp_path / "run_status.json").resolve())
    assert cfg.run.dashboard_html_path == str((tmp_path / "dashboard.html").resolve())
    assert cfg.run.status_flush_every_n_segments == 5
    assert cfg.run.batch_fallback_warn_ratio == 0.12
    assert cfg.run.fail_fast_on_translate_error is False


def test_load_config_reads_auto_model_sizing_flag(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "llm:\n"
        "  provider: openai\n"
        "  model: gpt-5-mini\n"
        "  auto_model_sizing: true\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    assert cfg.llm.auto_model_sizing is True
