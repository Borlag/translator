from __future__ import annotations

from docxru.config import LLMConfig, load_config


def test_llm_config_default_context_window_chars_is_600():
    assert LLMConfig().context_window_chars == 600


def test_load_config_defaults_context_window_chars_to_600(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("llm:\n  provider: mock\n", encoding="utf-8")

    cfg = load_config(config_path)
    assert cfg.llm.context_window_chars == 600
