from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class ModelContextProfile:
    provider: str
    canonical_model: str
    model_prefixes: tuple[str, ...]
    input_context_tokens: int
    output_context_tokens: int
    input_per_million: float | None
    output_per_million: float | None
    tier: str  # economy | balanced | premium | turbo


@dataclass(frozen=True)
class TierLimits:
    batch_chars_cap: int
    batch_segments_cap: int
    translate_output_cap: int
    checker_segments_cap: int
    checker_output_cap: int
    checker_pages_per_chunk: int


@dataclass(frozen=True)
class RuntimeModelSizing:
    profile: ModelContextProfile | None
    checker_profile: ModelContextProfile | None
    batch_segments: int
    batch_max_chars: int
    max_output_tokens: int
    checker_pages_per_chunk: int
    checker_fallback_segments_per_chunk: int
    checker_max_output_tokens: int
    notes: tuple[str, ...]


# Conservative character/token approximations for EN->RU technical translation prompts.
_CHARS_PER_INPUT_TOKEN = 3.2
_CHARS_PER_OUTPUT_TOKEN = 2.2
_RU_EXPANSION_RATIO = 1.35
_CHECKER_INPUT_UTILIZATION = 0.14


def _translate_input_utilization(input_context_tokens: int) -> float:
    if int(input_context_tokens) >= 200_000:
        return 0.15
    if int(input_context_tokens) >= 100_000:
        return 0.10
    return 0.08


def recommend_grouped_timeout_s(
    *,
    timeout_s: float,
    batch_segments: int,
    batch_max_chars: int,
) -> float:
    out = max(30.0, float(timeout_s))
    if int(batch_segments) <= 1:
        return out

    chars = max(0, int(batch_max_chars))
    if chars >= 100_000:
        return max(out, 360.0)
    if chars >= 60_000:
        return max(out, 300.0)
    if chars >= 36_000:
        return max(out, 180.0)
    return out


def recommend_checker_timeout_s(
    *,
    timeout_s: float,
    fallback_segments_per_chunk: int,
) -> float:
    out = max(30.0, float(timeout_s))
    segments = max(1, int(fallback_segments_per_chunk))
    if segments >= 160:
        return max(out, 300.0)
    if segments >= 120:
        return max(out, 240.0)
    if segments >= 80:
        return max(out, 180.0)
    if segments >= 40:
        return max(out, 120.0)
    return out

_TIER_LIMITS: dict[str, TierLimits] = {
    "economy": TierLimits(
        batch_chars_cap=12000,
        batch_segments_cap=6,
        translate_output_cap=6000,
        checker_segments_cap=80,
        checker_output_cap=2200,
        checker_pages_per_chunk=2,
    ),
    "balanced": TierLimits(
        batch_chars_cap=18000,
        batch_segments_cap=10,
        translate_output_cap=9000,
        checker_segments_cap=120,
        checker_output_cap=3000,
        checker_pages_per_chunk=3,
    ),
    "premium": TierLimits(
        batch_chars_cap=24000,
        batch_segments_cap=12,
        translate_output_cap=12000,
        checker_segments_cap=140,
        checker_output_cap=3600,
        checker_pages_per_chunk=4,
    ),
    "turbo": TierLimits(
        # Keep turbo meaningfully faster than balanced/premium, but avoid
        # oversized grouped requests that can stall for long periods.
        batch_chars_cap=48_000,
        batch_segments_cap=24,
        translate_output_cap=64_000,
        checker_segments_cap=120,
        checker_output_cap=8000,
        checker_pages_per_chunk=4,
    ),
}


# Sources (OpenAI official):
# - https://platform.openai.com/docs/models
# - https://platform.openai.com/docs/pricing
_MODEL_PROFILES: tuple[ModelContextProfile, ...] = (
    ModelContextProfile(
        provider="openai",
        canonical_model="gpt-5.2",
        model_prefixes=("gpt-5.2",),
        input_context_tokens=400_000,
        output_context_tokens=128_000,
        input_per_million=1.75,
        output_per_million=14.0,
        tier="turbo",
    ),
    ModelContextProfile(
        provider="openai",
        canonical_model="gpt-5-mini",
        model_prefixes=("gpt-5-mini",),
        input_context_tokens=400_000,
        output_context_tokens=128_000,
        input_per_million=0.25,
        output_per_million=2.0,
        tier="turbo",
    ),
    ModelContextProfile(
        provider="openai",
        canonical_model="gpt-5-nano",
        model_prefixes=("gpt-5-nano",),
        input_context_tokens=400_000,
        output_context_tokens=128_000,
        input_per_million=0.05,
        output_per_million=0.4,
        tier="economy",
    ),
    ModelContextProfile(
        provider="openai",
        canonical_model="gpt-5",
        model_prefixes=("gpt-5",),
        input_context_tokens=400_000,
        output_context_tokens=128_000,
        input_per_million=1.25,
        output_per_million=10.0,
        tier="turbo",
    ),
    ModelContextProfile(
        provider="openai",
        canonical_model="gpt-4.1-mini",
        model_prefixes=("gpt-4.1-mini",),
        input_context_tokens=1_047_576,
        output_context_tokens=32_768,
        input_per_million=0.4,
        output_per_million=1.6,
        tier="balanced",
    ),
    ModelContextProfile(
        provider="openai",
        canonical_model="gpt-4o-mini",
        model_prefixes=("gpt-4o-mini",),
        input_context_tokens=128_000,
        output_context_tokens=16_384,
        input_per_million=0.15,
        output_per_million=0.6,
        tier="economy",
    ),
)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _normalize_model(value: str) -> str:
    return (value or "").strip().lower()


def _model_matches(model_name: str, model_prefix: str) -> bool:
    if model_name == model_prefix:
        return True
    return model_name.startswith(f"{model_prefix}-")


def resolve_model_profile(provider: str, model: str) -> ModelContextProfile | None:
    provider_norm = (provider or "").strip().lower()
    model_norm = _normalize_model(model)
    if not provider_norm or not model_norm:
        return None
    for profile in _MODEL_PROFILES:
        if provider_norm != profile.provider:
            continue
        for prefix in profile.model_prefixes:
            if _model_matches(model_norm, _normalize_model(prefix)):
                return profile
    return None


def _median_source_chars(source_char_lengths: Sequence[int]) -> int:
    cleaned = [max(1, int(v)) for v in source_char_lengths if int(v) > 0]
    if not cleaned:
        return 420
    return int(median(cleaned))


def recommend_runtime_model_sizing(
    *,
    provider: str,
    model: str,
    checker_provider: str,
    checker_model: str,
    source_char_lengths: Sequence[int],
    prompt_chars: int,
    batch_segments: int,
    batch_max_chars: int,
    max_output_tokens: int,
    context_window_chars: int,
    checker_pages_per_chunk: int,
    checker_fallback_segments_per_chunk: int,
    checker_max_output_tokens: int,
) -> RuntimeModelSizing:
    profile = resolve_model_profile(provider, model)
    checker_profile = resolve_model_profile(checker_provider, checker_model) or profile

    out_batch_segments = max(1, int(batch_segments))
    out_batch_max_chars = max(1000, int(batch_max_chars))
    out_max_output_tokens = max(256, int(max_output_tokens))
    out_checker_pages = max(1, int(checker_pages_per_chunk))
    out_checker_fallback = max(1, int(checker_fallback_segments_per_chunk))
    out_checker_max_output = max(256, int(checker_max_output_tokens))
    notes: list[str] = []

    median_chars = _median_source_chars(source_char_lengths)

    if profile is not None:
        limits = _TIER_LIMITS.get(profile.tier, _TIER_LIMITS["balanced"])
        if context_window_chars <= 0 and out_batch_segments > 1:
            translate_input_budget = int(profile.input_context_tokens * _translate_input_utilization(profile.input_context_tokens))
            prompt_tokens = int((max(0, int(prompt_chars)) + 1200) / _CHARS_PER_INPUT_TOKEN)
            usable_input_tokens = max(900, translate_input_budget - prompt_tokens)
            budget_chars = int(usable_input_tokens * _CHARS_PER_INPUT_TOKEN)
            out_batch_max_chars = _clamp(budget_chars, 6000, limits.batch_chars_cap)

            per_segment_chars = max(320, median_chars + 180)
            recommended_segments = _clamp(
                int(out_batch_max_chars / per_segment_chars),
                2,
                limits.batch_segments_cap,
            )
            out_batch_segments = recommended_segments

        effective_translate_chars = out_batch_max_chars
        if out_batch_segments <= 1:
            effective_translate_chars = max(1200, int(median_chars * 1.8))

        estimated_output_tokens = int((effective_translate_chars * _RU_EXPANSION_RATIO) / _CHARS_PER_OUTPUT_TOKEN) + 420
        output_context_cap = max(1024, int(profile.output_context_tokens * 0.55))
        recommended_output_tokens = _clamp(
            estimated_output_tokens,
            1400,
            min(limits.translate_output_cap, output_context_cap),
        )
        out_max_output_tokens = max(out_max_output_tokens, recommended_output_tokens)
        notes.append(
            f"translate profile={profile.canonical_model} "
            f"context={profile.input_context_tokens}/{profile.output_context_tokens} "
            f"batch={out_batch_segments}x{out_batch_max_chars}chars "
            f"max_output_tokens={out_max_output_tokens}"
        )
    else:
        notes.append(f"translate profile not found for {provider}:{model}; runtime limits unchanged")

    if checker_profile is not None:
        checker_limits = _TIER_LIMITS.get(checker_profile.tier, _TIER_LIMITS["balanced"])
        checker_budget_tokens = int(checker_profile.input_context_tokens * _CHECKER_INPUT_UTILIZATION)
        checker_prompt_tokens = int(1400 / _CHARS_PER_INPUT_TOKEN)
        checker_usable_tokens = max(700, checker_budget_tokens - checker_prompt_tokens)
        checker_budget_chars = int(checker_usable_tokens * _CHARS_PER_INPUT_TOKEN)
        checker_chars_per_segment = max(260, int(median_chars * 2.4))
        out_checker_fallback = _clamp(
            int(checker_budget_chars / checker_chars_per_segment),
            24,
            checker_limits.checker_segments_cap,
        )
        out_checker_pages = checker_limits.checker_pages_per_chunk

        estimated_edit_count = max(4, out_checker_fallback // 8)
        estimated_checker_output = 500 + (estimated_edit_count * 55)
        checker_output_context_cap = max(1000, int(checker_profile.output_context_tokens * 0.2))
        recommended_checker_output = _clamp(
            estimated_checker_output,
            900,
            min(checker_limits.checker_output_cap, checker_output_context_cap),
        )
        out_checker_max_output = max(out_checker_max_output, recommended_checker_output)
        notes.append(
            f"checker profile={checker_profile.canonical_model} "
            f"context={checker_profile.input_context_tokens}/{checker_profile.output_context_tokens} "
            f"chunk={out_checker_pages}p/{out_checker_fallback}seg "
            f"checker_max_output_tokens={out_checker_max_output}"
        )
    else:
        notes.append(f"checker profile not found for {checker_provider}:{checker_model}; checker limits unchanged")

    return RuntimeModelSizing(
        profile=profile,
        checker_profile=checker_profile,
        batch_segments=out_batch_segments,
        batch_max_chars=out_batch_max_chars,
        max_output_tokens=out_max_output_tokens,
        checker_pages_per_chunk=out_checker_pages,
        checker_fallback_segments_per_chunk=out_checker_fallback,
        checker_max_output_tokens=out_checker_max_output,
        notes=tuple(notes),
    )
