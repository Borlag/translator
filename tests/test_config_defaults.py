from __future__ import annotations

from docxru.config import LLMConfig, PdfConfig, PipelineConfig, load_config


def test_llm_config_default_context_window_chars_is_600():
    assert LLMConfig().context_window_chars == 600


def test_load_config_defaults_context_window_chars_to_600(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("llm:\n  provider: mock\n", encoding="utf-8")

    cfg = load_config(config_path)
    assert cfg.llm.context_window_chars == 600


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
