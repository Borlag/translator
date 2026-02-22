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
    # If false, glossary is not injected into each LLM prompt (saves tokens).
    # If hard_glossary is enabled, pipeline can still enforce glossary via placeholder shielding.
    glossary_in_prompt: bool = True
    # Hard glossary = pre-translation term locking via placeholders.
    # Improves strict term stability but may hurt RU morphology (case/number agreement).
    hard_glossary: bool = False
    # OpenAI reasoning effort hint: none|minimal|low|medium|high|xhigh
    reasoning_effort: str | None = None
    # OpenAI prompt caching controls (Chat Completions / Responses).
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    # Translate several nearby segments in one LLM request for better context coherence.
    batch_segments: int = 1
    # Soft payload limiter for batch mode; very long segments are still translated individually.
    batch_max_chars: int = 12000
    # Structured response contract handling for prompt-based LLMs.
    structured_output_mode: str = "auto"  # 'off' | 'auto' | 'strict'
    # Glossary prompt injection mode.
    glossary_prompt_mode: str = "full"  # 'off' | 'full' | 'matched'
    glossary_match_limit: int = 24
    # Batch eligibility guardrails.
    batch_skip_on_brline: bool = True
    batch_max_style_tokens: int = 16
    # Sliding prompt context for near-neighbor snippets and recent translations.
    # 0 disables sequential recent-translation context mode.
    context_window_chars: int = 0


@dataclass(frozen=True)
class TMConfig:
    path: str = "translation_cache.sqlite"
    fuzzy_enabled: bool = False
    fuzzy_top_k: int = 3
    fuzzy_min_similarity: float = 0.75
    fuzzy_prompt_max_chars: int = 500


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
    # Optional append-only translation history for human review and term/context lookup.
    translation_history_path: str | None = None
    log_path: str = "run.log"
    abbyy_profile: str = "off"  # 'off' | 'safe' | 'aggressive'
    glossary_lemma_check: str = "off"  # 'off' | 'warn' | 'retry'
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


def _normalize_choice(value: Any, *, field_name: str, allowed: set[str], default: str) -> str:
    raw = str(default if value is None else value).strip().lower()
    if raw not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid value for {field_name}: {raw!r}. Allowed: {allowed_list}")
    return raw


def load_config(path: str | Path) -> PipelineConfig:
    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    llm_data = data.get("llm", {}) or {}
    tm_data = data.get("tm", {}) or {}
    llm_legacy_glossary_in_prompt = bool(llm_data.get("glossary_in_prompt", True))
    llm_has_prompt_mode = "glossary_prompt_mode" in llm_data
    glossary_prompt_mode = _normalize_choice(
        llm_data.get("glossary_prompt_mode", "full" if llm_legacy_glossary_in_prompt else "off"),
        field_name="llm.glossary_prompt_mode",
        allowed={"off", "full", "matched"},
        default="full",
    )
    glossary_in_prompt = (
        llm_legacy_glossary_in_prompt if not llm_has_prompt_mode else (glossary_prompt_mode != "off")
    )

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
        glossary_in_prompt=glossary_in_prompt,
        hard_glossary=bool(llm_data.get("hard_glossary", False)),
        reasoning_effort=(
            str(llm_data["reasoning_effort"]).strip()
            if llm_data.get("reasoning_effort") is not None and str(llm_data["reasoning_effort"]).strip()
            else None
        ),
        prompt_cache_key=(
            str(llm_data["prompt_cache_key"]).strip()
            if llm_data.get("prompt_cache_key") is not None and str(llm_data["prompt_cache_key"]).strip()
            else None
        ),
        prompt_cache_retention=(
            str(llm_data["prompt_cache_retention"]).strip()
            if llm_data.get("prompt_cache_retention") is not None
            and str(llm_data["prompt_cache_retention"]).strip()
            else None
        ),
        batch_segments=int(llm_data.get("batch_segments", 1)),
        batch_max_chars=int(llm_data.get("batch_max_chars", 12000)),
        structured_output_mode=_normalize_choice(
            llm_data.get("structured_output_mode", "auto"),
            field_name="llm.structured_output_mode",
            allowed={"off", "auto", "strict"},
            default="auto",
        ),
        glossary_prompt_mode=glossary_prompt_mode,
        glossary_match_limit=int(llm_data.get("glossary_match_limit", 24)),
        batch_skip_on_brline=bool(llm_data.get("batch_skip_on_brline", True)),
        batch_max_style_tokens=int(llm_data.get("batch_max_style_tokens", 16)),
        context_window_chars=max(0, int(llm_data.get("context_window_chars", 0))),
    )
    tm = TMConfig(
        path=str(tm_data.get("path", "translation_cache.sqlite")),
        fuzzy_enabled=bool(tm_data.get("fuzzy_enabled", False)),
        fuzzy_top_k=int(tm_data.get("fuzzy_top_k", 3)),
        fuzzy_min_similarity=float(tm_data.get("fuzzy_min_similarity", 0.75)),
        fuzzy_prompt_max_chars=int(tm_data.get("fuzzy_prompt_max_chars", 500)),
    )

    include_headers = bool(data.get("include_headers", False))
    include_footers = bool(data.get("include_footers", False))
    concurrency = int(data.get("concurrency", 4))
    mode = str(data.get("mode", "reflow"))
    qa_report_path = str(data.get("qa_report_path", "qa_report.html"))
    qa_jsonl_path = str(data.get("qa_jsonl_path", "qa.jsonl"))
    translation_history_path = _resolve_optional_path(cfg_path.parent, data.get("translation_history_path"))
    log_path = str(data.get("log_path", "run.log"))
    abbyy_profile = _normalize_choice(
        data.get("abbyy_profile", "off"),
        field_name="abbyy_profile",
        allowed={"off", "safe", "aggressive"},
        default="off",
    )
    glossary_lemma_check = _normalize_choice(
        data.get("glossary_lemma_check", "off"),
        field_name="glossary_lemma_check",
        allowed={"off", "warn", "retry"},
        default="off",
    )

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
        translation_history_path=translation_history_path,
        log_path=log_path,
        abbyy_profile=abbyy_profile,
        glossary_lemma_check=glossary_lemma_check,
        pattern_set=pattern_set,
    )
