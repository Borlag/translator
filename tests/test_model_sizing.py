from __future__ import annotations

from docxru.model_sizing import (
    _translate_input_utilization,
    recommend_checker_timeout_s,
    recommend_grouped_timeout_s,
    recommend_runtime_model_sizing,
    resolve_model_profile,
)


def test_resolve_model_profile_supports_snapshot_suffix():
    profile = resolve_model_profile("openai", "gpt-5-mini-2026-01-15")
    assert profile is not None
    assert profile.canonical_model == "gpt-5-mini"


def test_recommend_runtime_model_sizing_for_gpt4o_mini_keeps_conservative_batch():
    sizing = recommend_runtime_model_sizing(
        provider="openai",
        model="gpt-4o-mini",
        checker_provider="openai",
        checker_model="gpt-4o-mini",
        source_char_lengths=[420, 480, 530, 610, 390],
        prompt_chars=7000,
        batch_segments=6,
        batch_max_chars=14000,
        max_output_tokens=2400,
        context_window_chars=0,
        checker_pages_per_chunk=3,
        checker_fallback_segments_per_chunk=120,
        checker_max_output_tokens=2000,
    )
    assert sizing.profile is not None
    assert sizing.profile.canonical_model == "gpt-4o-mini"
    assert 2 <= sizing.batch_segments <= 6
    assert 6000 <= sizing.batch_max_chars <= 12000
    assert sizing.max_output_tokens >= 2400
    assert 24 <= sizing.checker_fallback_segments_per_chunk <= 80
    assert sizing.checker_pages_per_chunk == 2


def test_gpt5_profiles_resolve_to_turbo_tier():
    for model in ("gpt-5-mini", "gpt-5", "gpt-5.2"):
        profile = resolve_model_profile("openai", model)
        assert profile is not None
        assert profile.tier == "turbo"


def test_recommend_runtime_model_sizing_for_gpt5_mini_uses_turbo_limits():
    sizing = recommend_runtime_model_sizing(
        provider="openai",
        model="gpt-5-mini",
        checker_provider="openai",
        checker_model="gpt-5-mini",
        source_char_lengths=[520, 480, 700, 610],
        prompt_chars=6_000,
        batch_segments=6,
        batch_max_chars=12_000,
        max_output_tokens=3_000,
        context_window_chars=0,
        checker_pages_per_chunk=3,
        checker_fallback_segments_per_chunk=120,
        checker_max_output_tokens=2_000,
    )
    assert 36_000 <= sizing.batch_max_chars <= 60_000
    assert 12 <= sizing.batch_segments <= 24
    assert sizing.checker_pages_per_chunk == 4


def test_recommend_runtime_model_sizing_for_gpt52_allows_larger_batch_than_gpt4o():
    economy = recommend_runtime_model_sizing(
        provider="openai",
        model="gpt-4o-mini",
        checker_provider="openai",
        checker_model="gpt-4o-mini",
        source_char_lengths=[520, 480, 700, 610],
        prompt_chars=6000,
        batch_segments=6,
        batch_max_chars=12000,
        max_output_tokens=3000,
        context_window_chars=0,
        checker_pages_per_chunk=3,
        checker_fallback_segments_per_chunk=120,
        checker_max_output_tokens=2000,
    )
    premium = recommend_runtime_model_sizing(
        provider="openai",
        model="gpt-5.2",
        checker_provider="openai",
        checker_model="gpt-5.2",
        source_char_lengths=[520, 480, 700, 610],
        prompt_chars=6000,
        batch_segments=6,
        batch_max_chars=12000,
        max_output_tokens=3000,
        context_window_chars=0,
        checker_pages_per_chunk=3,
        checker_fallback_segments_per_chunk=120,
        checker_max_output_tokens=2000,
    )
    assert premium.batch_max_chars >= economy.batch_max_chars
    assert premium.batch_segments >= economy.batch_segments
    assert premium.checker_fallback_segments_per_chunk >= economy.checker_fallback_segments_per_chunk


def test_recommend_runtime_model_sizing_preserves_single_segment_mode():
    sizing = recommend_runtime_model_sizing(
        provider="openai",
        model="gpt-5-mini",
        checker_provider="openai",
        checker_model="gpt-5-mini",
        source_char_lengths=[300, 420, 500],
        prompt_chars=4000,
        batch_segments=1,
        batch_max_chars=12000,
        max_output_tokens=2200,
        context_window_chars=0,
        checker_pages_per_chunk=3,
        checker_fallback_segments_per_chunk=120,
        checker_max_output_tokens=2000,
    )
    assert sizing.batch_segments == 1


def test_translate_input_utilization_is_dynamic():
    assert _translate_input_utilization(400_000) == 0.15
    assert _translate_input_utilization(128_000) == 0.10
    assert _translate_input_utilization(96_000) == 0.08


def test_recommend_grouped_timeout_s_scales_with_batch_size():
    assert recommend_grouped_timeout_s(timeout_s=60.0, batch_segments=20, batch_max_chars=36_000) == 180.0
    assert recommend_grouped_timeout_s(timeout_s=60.0, batch_segments=40, batch_max_chars=60_000) == 300.0
    assert recommend_grouped_timeout_s(timeout_s=60.0, batch_segments=80, batch_max_chars=120_000) == 360.0


def test_recommend_grouped_timeout_s_keeps_single_segment_timeout():
    assert recommend_grouped_timeout_s(timeout_s=75.0, batch_segments=1, batch_max_chars=200_000) == 75.0


def test_recommend_checker_timeout_s_scales_with_checker_chunk_size():
    assert recommend_checker_timeout_s(timeout_s=60.0, fallback_segments_per_chunk=40) == 120.0
    assert recommend_checker_timeout_s(timeout_s=60.0, fallback_segments_per_chunk=80) == 180.0
    assert recommend_checker_timeout_s(timeout_s=60.0, fallback_segments_per_chunk=120) == 240.0
    assert recommend_checker_timeout_s(timeout_s=60.0, fallback_segments_per_chunk=160) == 300.0


def test_recommend_checker_timeout_s_keeps_higher_user_timeout():
    assert recommend_checker_timeout_s(timeout_s=360.0, fallback_segments_per_chunk=160) == 360.0


def test_recommend_runtime_model_sizing_unknown_model_keeps_translate_limits():
    sizing = recommend_runtime_model_sizing(
        provider="openai",
        model="unknown-model",
        checker_provider="openai",
        checker_model="unknown-model",
        source_char_lengths=[400, 500],
        prompt_chars=1000,
        batch_segments=4,
        batch_max_chars=8000,
        max_output_tokens=2000,
        context_window_chars=0,
        checker_pages_per_chunk=3,
        checker_fallback_segments_per_chunk=90,
        checker_max_output_tokens=1800,
    )
    assert sizing.profile is None
    assert sizing.batch_segments == 4
    assert sizing.batch_max_chars == 8000
    assert sizing.max_output_tokens == 2000
