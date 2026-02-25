from __future__ import annotations

from dataclasses import dataclass, field
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
    temperature: float = 0.0
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
    context_window_chars: int = 900
    # Optional compact few-shot examples in default translation prompt.
    prompt_examples_mode: str = "core"  # 'off' | 'core'
    # Batch payload context hints per item.
    batch_tm_hints_per_item: int = 1
    batch_recent_translations_per_item: int = 3
    # Optional external overrides for domain post-processing replacements.
    domain_term_pairs_path: str = "config/domain_term_pairs.yaml"
    # Auto-tune batch/checker payload sizing based on selected model context limits.
    auto_model_sizing: bool = False


@dataclass(frozen=True)
class TMConfig:
    path: str = "translation_cache.sqlite"
    fuzzy_enabled: bool = False
    fuzzy_top_k: int = 3
    fuzzy_min_similarity: float = 0.75
    fuzzy_prompt_max_chars: int = 500
    fuzzy_token_regex: str = r"[A-Za-zА-Яа-яЁё0-9]{2,}"
    fuzzy_rank_mode: str = "hybrid"  # 'sequence' | 'hybrid'


@dataclass(frozen=True)
class PdfConfig:
    bilingual_mode: bool = False
    ocr_fallback: bool = False
    max_pages: int | None = None
    max_font_shrink_ratio: float = 0.6
    block_merge_threshold_pt: float = 12.0
    skip_headers_footers: bool = False
    table_detection: bool = True
    font_map: dict[str, str] = field(default_factory=dict)
    default_sans_font: str = "NotoSans-Regular.ttf"
    default_serif_font: str = "NotoSerif-Regular.ttf"
    default_mono_font: str = "NotoSansMono-Regular.ttf"


@dataclass(frozen=True)
class CheckerConfig:
    enabled: bool = False
    provider: str | None = None
    model: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 6000
    timeout_s: float = 60.0
    retries: int = 0
    system_prompt_path: str | None = None
    glossary_path: str | None = None
    pages_per_chunk: int = 3
    max_segments: int = 0
    only_on_issue_severities: tuple[str, ...] = ("warn", "error")
    only_on_issue_codes: tuple[str, ...] = ()
    output_path: str = "checker_suggestions.json"
    safe_output_path: str = "checker_suggestions_safe.json"
    auto_apply_safe: bool = False
    auto_apply_min_confidence: float = 0.7
    # DOCX does not expose real page numbers; fallback to fixed-size segment windows.
    # Keep this conservative to reduce checker latency and output-limit failures.
    fallback_segments_per_chunk: int = 80
    # Optional async OpenAI Batch API mode for checker (cost-saving, high-latency path).
    openai_batch_enabled: bool = False
    openai_batch_completion_window: str = "24h"
    openai_batch_poll_interval_s: float = 20.0
    openai_batch_timeout_s: float = 86400.0


@dataclass(frozen=True)
class PricingConfig:
    enabled: bool = False
    pricing_path: str | None = None
    currency: str = "USD"


@dataclass(frozen=True)
class RunConfig:
    run_dir: str | None = None
    run_id: str | None = None
    status_path: str | None = None
    dashboard_html_path: str | None = None
    # How often pipelines should flush run status to disk.
    status_flush_every_n_segments: int = 10
    # Warn when fallback share among grouped batch attempts exceeds this ratio.
    batch_fallback_warn_ratio: float = 0.08
    # If grouped batch request times out, split batch recursively (N -> N/2 -> ... -> 1).
    batch_timeout_bisect: bool = True
    # Stop translation immediately on segment/batch translation errors instead of continuing.
    fail_fast_on_translate_error: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    llm: LLMConfig = LLMConfig()
    tm: TMConfig = TMConfig()
    pdf: PdfConfig = PdfConfig()
    checker: CheckerConfig = CheckerConfig()
    pricing: PricingConfig = PricingConfig()
    run: RunConfig = RunConfig()
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
    # Warn-only detector for suspiciously short translations on sufficiently long source.
    short_translation_min_ratio: float = 0.35
    short_translation_min_source_chars: int = 24
    # Warn-only checks for visible EN leftovers and LLM repetition artifacts.
    untranslated_latin_warn_ratio: float = 0.15
    untranslated_latin_min_len: int = 3
    untranslated_latin_allowlist_path: str | None = None
    repeated_words_check: bool = True
    repeated_phrase_ngram_max: int = 3
    context_leakage_check: bool = True
    context_leakage_allowlist_path: str | None = None
    # Optional post-translation layout validation/autofix heuristics.
    layout_check: bool = False
    layout_expansion_warn_ratio: float = 1.5
    layout_auto_fix: bool = False
    layout_font_reduction_pt: float = 0.5
    layout_spacing_factor: float = 0.8
    # Optional unconditional post-writeback font shrink.
    # 0.0 disables shrinking for the corresponding scope.
    font_shrink_body_pt: float = 0.0
    font_shrink_table_pt: float = 0.0
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


def _coerce_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        return (text,)
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return tuple(out)
    text = str(value).strip()
    return (text,) if text else ()


def load_config(path: str | Path) -> PipelineConfig:
    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    llm_data = data.get("llm", {}) or {}
    tm_data = data.get("tm", {}) or {}
    pdf_data = data.get("pdf", {}) or {}
    checker_data = data.get("checker", {}) or {}
    pricing_data = data.get("pricing", {}) or {}
    run_data = data.get("run", {}) or {}
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
    prompt_examples_mode = _normalize_choice(
        llm_data.get("prompt_examples_mode", "core"),
        field_name="llm.prompt_examples_mode",
        allowed={"off", "core"},
        default="core",
    )

    llm = LLMConfig(
        provider=str(llm_data.get("provider", "mock")),
        model=str(llm_data.get("model", "gpt-4o-mini")),
        source_lang=str(llm_data.get("source_lang", "en")),
        target_lang=str(llm_data.get("target_lang", "ru")),
        base_url=(str(llm_data["base_url"]) if llm_data.get("base_url") is not None else None),
        temperature=float(llm_data.get("temperature", 0.0)),
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
        context_window_chars=max(0, int(llm_data.get("context_window_chars", 900))),
        prompt_examples_mode=prompt_examples_mode,
        batch_tm_hints_per_item=max(0, int(llm_data.get("batch_tm_hints_per_item", 1))),
        batch_recent_translations_per_item=max(0, int(llm_data.get("batch_recent_translations_per_item", 3))),
        domain_term_pairs_path=(
            _resolve_optional_path(cfg_path.parent, llm_data.get("domain_term_pairs_path", "config/domain_term_pairs.yaml"))
            or "config/domain_term_pairs.yaml"
        ),
        auto_model_sizing=bool(llm_data.get("auto_model_sizing", False)),
    )
    tm = TMConfig(
        path=str(tm_data.get("path", "translation_cache.sqlite")),
        fuzzy_enabled=bool(tm_data.get("fuzzy_enabled", False)),
        fuzzy_top_k=int(tm_data.get("fuzzy_top_k", 3)),
        fuzzy_min_similarity=float(tm_data.get("fuzzy_min_similarity", 0.75)),
        fuzzy_prompt_max_chars=int(tm_data.get("fuzzy_prompt_max_chars", 500)),
        fuzzy_token_regex=str(tm_data.get("fuzzy_token_regex", r"[A-Za-zА-Яа-яЁё0-9]{2,}")).strip()
        or r"[A-Za-zА-Яа-яЁё0-9]{2,}",
        fuzzy_rank_mode=_normalize_choice(
            tm_data.get("fuzzy_rank_mode", "hybrid"),
            field_name="tm.fuzzy_rank_mode",
            allowed={"sequence", "hybrid"},
            default="hybrid",
        ),
    )
    max_pages_raw = pdf_data.get("max_pages")
    max_pages = None if max_pages_raw is None else max(0, int(max_pages_raw))
    pdf = PdfConfig(
        bilingual_mode=bool(pdf_data.get("bilingual_mode", False)),
        ocr_fallback=bool(pdf_data.get("ocr_fallback", False)),
        max_pages=max_pages,
        max_font_shrink_ratio=float(pdf_data.get("max_font_shrink_ratio", 0.6)),
        block_merge_threshold_pt=float(pdf_data.get("block_merge_threshold_pt", 12.0)),
        skip_headers_footers=bool(pdf_data.get("skip_headers_footers", False)),
        table_detection=bool(pdf_data.get("table_detection", True)),
        font_map={str(k): str(v) for k, v in (pdf_data.get("font_map", {}) or {}).items()},
        default_sans_font=str(pdf_data.get("default_sans_font", "NotoSans-Regular.ttf")),
        default_serif_font=str(pdf_data.get("default_serif_font", "NotoSerif-Regular.ttf")),
        default_mono_font=str(pdf_data.get("default_mono_font", "NotoSansMono-Regular.ttf")),
    )
    checker = CheckerConfig(
        enabled=bool(checker_data.get("enabled", False)),
        provider=(
            str(checker_data["provider"]).strip()
            if checker_data.get("provider") is not None and str(checker_data["provider"]).strip()
            else None
        ),
        model=(
            str(checker_data["model"]).strip()
            if checker_data.get("model") is not None and str(checker_data["model"]).strip()
            else None
        ),
        temperature=float(checker_data.get("temperature", 0.0)),
        max_output_tokens=int(checker_data.get("max_output_tokens", 6000)),
        timeout_s=float(checker_data.get("timeout_s", 60.0)),
        retries=max(0, int(checker_data.get("retries", 0))),
        system_prompt_path=_resolve_optional_path(cfg_path.parent, checker_data.get("system_prompt_path")),
        glossary_path=_resolve_optional_path(cfg_path.parent, checker_data.get("glossary_path")),
        pages_per_chunk=max(1, int(checker_data.get("pages_per_chunk", 3))),
        max_segments=max(0, int(checker_data.get("max_segments", 0))),
        only_on_issue_severities=tuple(
            item.strip().lower() for item in _coerce_str_tuple(checker_data.get("only_on_issue_severities", ()))
        )
        or ("warn", "error"),
        only_on_issue_codes=_coerce_str_tuple(checker_data.get("only_on_issue_codes")),
        output_path=str(checker_data.get("output_path", "checker_suggestions.json")),
        safe_output_path=str(checker_data.get("safe_output_path", "checker_suggestions_safe.json")),
        auto_apply_safe=bool(checker_data.get("auto_apply_safe", False)),
        auto_apply_min_confidence=max(
            0.0,
            min(1.0, float(checker_data.get("auto_apply_min_confidence", 0.7))),
        ),
        fallback_segments_per_chunk=max(1, int(checker_data.get("fallback_segments_per_chunk", 80))),
        openai_batch_enabled=bool(checker_data.get("openai_batch_enabled", False)),
        openai_batch_completion_window=(
            str(checker_data.get("openai_batch_completion_window", "24h")).strip() or "24h"
        ),
        openai_batch_poll_interval_s=max(1.0, float(checker_data.get("openai_batch_poll_interval_s", 20.0))),
        openai_batch_timeout_s=max(30.0, float(checker_data.get("openai_batch_timeout_s", 86400.0))),
    )
    pricing = PricingConfig(
        enabled=bool(pricing_data.get("enabled", False)),
        pricing_path=_resolve_optional_path(cfg_path.parent, pricing_data.get("pricing_path")),
        currency=str(pricing_data.get("currency", "USD")).strip() or "USD",
    )
    run_cfg = RunConfig(
        run_dir=_resolve_optional_path(cfg_path.parent, run_data.get("run_dir")),
        run_id=(
            str(run_data["run_id"]).strip()
            if run_data.get("run_id") is not None and str(run_data["run_id"]).strip()
            else None
        ),
        status_path=_resolve_optional_path(cfg_path.parent, run_data.get("status_path")),
        dashboard_html_path=_resolve_optional_path(cfg_path.parent, run_data.get("dashboard_html_path")),
        status_flush_every_n_segments=max(1, int(run_data.get("status_flush_every_n_segments", 10))),
        batch_fallback_warn_ratio=max(
            0.0,
            min(1.0, float(run_data.get("batch_fallback_warn_ratio", 0.08))),
        ),
        batch_timeout_bisect=bool(run_data.get("batch_timeout_bisect", True)),
        fail_fast_on_translate_error=bool(run_data.get("fail_fast_on_translate_error", True)),
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
    short_translation_min_ratio = max(
        0.0,
        min(1.0, float(data.get("short_translation_min_ratio", 0.35))),
    )
    short_translation_min_source_chars = max(1, int(data.get("short_translation_min_source_chars", 24)))
    untranslated_latin_warn_ratio = max(
        0.0,
        min(1.0, float(data.get("untranslated_latin_warn_ratio", 0.15))),
    )
    untranslated_latin_min_len = max(1, int(data.get("untranslated_latin_min_len", 3)))
    untranslated_latin_allowlist_path = _resolve_optional_path(
        cfg_path.parent,
        data.get("untranslated_latin_allowlist_path"),
    )
    repeated_words_check = bool(data.get("repeated_words_check", True))
    repeated_phrase_ngram_max = max(2, int(data.get("repeated_phrase_ngram_max", 3)))
    context_leakage_check = bool(data.get("context_leakage_check", True))
    context_leakage_allowlist_path = _resolve_optional_path(
        cfg_path.parent,
        data.get("context_leakage_allowlist_path"),
    )
    layout_check = bool(data.get("layout_check", False))
    layout_expansion_warn_ratio = float(data.get("layout_expansion_warn_ratio", 1.5))
    layout_auto_fix = bool(data.get("layout_auto_fix", False))
    layout_font_reduction_pt = float(data.get("layout_font_reduction_pt", 0.5))
    layout_spacing_factor = float(data.get("layout_spacing_factor", 0.8))
    font_shrink_body_pt = max(0.0, float(data.get("font_shrink_body_pt", 0.0)))
    font_shrink_table_pt = max(0.0, float(data.get("font_shrink_table_pt", 0.0)))

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
        pdf=pdf,
        checker=checker,
        pricing=pricing,
        run=run_cfg,
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
        short_translation_min_ratio=short_translation_min_ratio,
        short_translation_min_source_chars=short_translation_min_source_chars,
        untranslated_latin_warn_ratio=untranslated_latin_warn_ratio,
        untranslated_latin_min_len=untranslated_latin_min_len,
        untranslated_latin_allowlist_path=untranslated_latin_allowlist_path,
        repeated_words_check=repeated_words_check,
        repeated_phrase_ngram_max=repeated_phrase_ngram_max,
        context_leakage_check=context_leakage_check,
        context_leakage_allowlist_path=context_leakage_allowlist_path,
        layout_check=layout_check,
        layout_expansion_warn_ratio=layout_expansion_warn_ratio,
        layout_auto_fix=layout_auto_fix,
        layout_font_reduction_pt=layout_font_reduction_pt,
        layout_spacing_factor=layout_spacing_factor,
        font_shrink_body_pt=font_shrink_body_pt,
        font_shrink_table_pt=font_shrink_table_pt,
        pattern_set=pattern_set,
    )
