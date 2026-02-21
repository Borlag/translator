from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .token_shield import PatternRule, PatternSet, compile_pattern_set


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "mock"  # 'mock' | 'openai' | 'google' | 'ollama'
    model: str = "gpt-4o-mini"
    source_lang: str = "en"
    target_lang: str = "ru"
    base_url: str | None = None
    temperature: float = 0.1
    max_output_tokens: int = 2000
    retries: int = 2
    timeout_s: float = 60.0
    system_prompt_path: str | None = None
    glossary_path: str | None = None


@dataclass(frozen=True)
class TMConfig:
    path: str = "translation_cache.sqlite"


@dataclass(frozen=True)
class PipelineConfig:
    llm: LLMConfig = LLMConfig()
    tm: TMConfig = TMConfig()
    include_headers: bool = False
    include_footers: bool = False
    concurrency: int = 4
    mode: str = "reflow"  # 'reflow' | 'com'
    qa_report_path: str = "qa_report.html"
    qa_jsonl_path: str = "qa.jsonl"
    log_path: str = "run.log"
    # regex patterns:
    pattern_set: PatternSet = PatternSet([])


def _resolve_optional_path(base_dir: Path, value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def load_config(path: str | Path) -> PipelineConfig:
    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    llm_data = data.get("llm", {}) or {}
    tm_data = data.get("tm", {}) or {}

    llm = LLMConfig(
        provider=str(llm_data.get("provider", "mock")),
        model=str(llm_data.get("model", "gpt-4o-mini")),
        source_lang=str(llm_data.get("source_lang", "en")),
        target_lang=str(llm_data.get("target_lang", "ru")),
        base_url=(str(llm_data["base_url"]) if llm_data.get("base_url") is not None else None),
        temperature=float(llm_data.get("temperature", 0.1)),
        max_output_tokens=int(llm_data.get("max_output_tokens", 2000)),
        retries=int(llm_data.get("retries", 2)),
        timeout_s=float(llm_data.get("timeout_s", 60.0)),
        system_prompt_path=_resolve_optional_path(cfg_path.parent, llm_data.get("system_prompt_path")),
        glossary_path=_resolve_optional_path(cfg_path.parent, llm_data.get("glossary_path")),
    )
    tm = TMConfig(path=str(tm_data.get("path", "translation_cache.sqlite")))

    include_headers = bool(data.get("include_headers", False))
    include_footers = bool(data.get("include_footers", False))
    concurrency = int(data.get("concurrency", 4))
    mode = str(data.get("mode", "reflow"))
    qa_report_path = str(data.get("qa_report_path", "qa_report.html"))
    qa_jsonl_path = str(data.get("qa_jsonl_path", "qa.jsonl"))
    log_path = str(data.get("log_path", "run.log"))

    # Patterns can be either inline list under patterns.rules, or a presets yaml path.
    patterns_data = data.get("patterns", {}) or {}
    preset_path = patterns_data.get("preset_file")
    rules_data: list[dict[str, Any]] = patterns_data.get("rules", []) or []
    if preset_path:
        preset_file = (cfg_path.parent / str(preset_path)).resolve()
        preset_data = yaml.safe_load(preset_file.read_text(encoding="utf-8")) or {}
        # preset file may include multiple named presets; choose `preset_name` or default
        preset_name = str(patterns_data.get("preset_name", "default"))
        preset_rules = (preset_data.get("presets", {}) or {}).get(preset_name)
        if preset_rules is None:
            raise ValueError(
                f"Preset '{preset_name}' not found in {preset_file}. Available: "
                f"{list((preset_data.get('presets', {}) or {}).keys())}"
            )
        rules_data = list(preset_rules)

    rules: list[PatternRule] = []
    for rd in rules_data:
        rules.append(
            PatternRule(
                name=str(rd["name"]),
                pattern=str(rd["pattern"]),
                flags=str(rd.get("flags", "")),
                description=str(rd.get("description", "")),
            )
        )
    pattern_set = compile_pattern_set(PatternSet(rules))

    return PipelineConfig(
        llm=llm,
        tm=tm,
        include_headers=include_headers,
        include_footers=include_footers,
        concurrency=concurrency,
        mode=mode,
        qa_report_path=qa_report_path,
        qa_jsonl_path=qa_jsonl_path,
        log_path=log_path,
        pattern_set=pattern_set,
    )
