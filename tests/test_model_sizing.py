from __future__ import annotations

from docxru.model_sizing import recommend_runtime_model_sizing, resolve_model_profile


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
