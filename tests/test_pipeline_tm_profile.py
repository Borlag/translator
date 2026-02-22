from __future__ import annotations

from docxru.config import LLMConfig, PipelineConfig
from docxru.models import Segment
from docxru.pipeline import _build_tm_profile_key, _should_attach_neighbor_context


def _make_segment(text: str, *, in_table: bool = False) -> Segment:
    return Segment(
        segment_id="s1",
        location="body/p1",
        context={"part": "body", "in_table": in_table} if in_table else {"part": "body"},
        source_plain=text,
        paragraph_ref=None,
    )


def test_tm_profile_key_changes_for_provider_and_prompt():
    cfg_openai = PipelineConfig(
        llm=LLMConfig(
            provider="openai",
            model="gpt-5-mini",
            source_lang="en",
            target_lang="ru",
            batch_segments=6,
            reasoning_effort="minimal",
        )
    )
    cfg_ollama = PipelineConfig(
        llm=LLMConfig(
            provider="ollama",
            model="gpt-5-mini",
            source_lang="en",
            target_lang="ru",
            batch_segments=6,
            reasoning_effort="minimal",
        )
    )

    k_openai = _build_tm_profile_key(
        cfg_openai,
        custom_system_prompt="Prompt A",
        glossary_text="Term A - Термин А",
    )
    k_ollama = _build_tm_profile_key(
        cfg_ollama,
        custom_system_prompt="Prompt A",
        glossary_text="Term A - Термин А",
    )
    k_prompt_changed = _build_tm_profile_key(
        cfg_openai,
        custom_system_prompt="Prompt B",
        glossary_text="Term A - Термин А",
    )

    assert k_openai != k_ollama
    assert k_openai != k_prompt_changed


def test_should_attach_neighbor_context_for_short_caps_fragment():
    assert _should_attach_neighbor_context(_make_segment("WITH"))


def test_should_attach_neighbor_context_skips_short_lower_fragment():
    assert not _should_attach_neighbor_context(_make_segment("with"))


def test_should_attach_neighbor_context_skips_table_segment():
    assert not _should_attach_neighbor_context(_make_segment("MAIN LANDING GEAR LEG", in_table=True))

