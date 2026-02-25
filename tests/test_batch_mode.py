from __future__ import annotations

import logging

import pytest

from docxru.config import LLMConfig, PipelineConfig, RunConfig
from docxru.models import Segment
from docxru.pipeline import (
    _batch_ineligibility_reasons,
    _build_batch_translation_prompt,
    _chunk_translation_jobs,
    _effective_manual_max_output_tokens,
    _parse_batch_translation_output,
    _recommended_grouped_batch_workers,
    _translate_batch_group,
    _translate_batch_once,
)


def _make_job(seg_id: str, text: str) -> tuple[Segment, str, str]:
    seg = Segment(
        segment_id=seg_id,
        location=f"body/p{seg_id}",
        context={"part": "body"},
        source_plain=text,
        paragraph_ref=None,
        shielded_tagged=text,
    )
    return seg, f"hash-{seg_id}", f"norm-{seg_id}"


def test_parse_batch_translation_output_accepts_translations_object():
    raw = '{"translations":[{"id":"s1","text":"T1"},{"id":"s2","text":"T2"}]}'
    out = _parse_batch_translation_output(raw, ["s1", "s2"])
    assert out == {"s1": "T1", "s2": "T2"}


def test_parse_batch_translation_output_accepts_fenced_json():
    raw = """Here is result:
```json
{"translations":[{"id":"a","text":"AA"},{"id":"b","text":"BB"}]}
```"""
    out = _parse_batch_translation_output(raw, ["a", "b"])
    assert out == {"a": "AA", "b": "BB"}


def test_parse_batch_translation_output_accepts_id_to_text_object():
    raw = '{"s1":"T1","s2":"T2"}'
    out = _parse_batch_translation_output(raw, ["s1", "s2"])
    assert out == {"s1": "T1", "s2": "T2"}


def test_parse_batch_translation_output_accepts_json_with_outer_text():
    raw = 'Result follows: {"s1":"T1","s2":"T2"} -- end.'
    out = _parse_batch_translation_output(raw, ["s1", "s2"])
    assert out == {"s1": "T1", "s2": "T2"}


def test_parse_batch_translation_output_rejects_missing_ids():
    raw = '{"translations":[{"id":"s1","text":"Only one"}]}'
    with pytest.raises(RuntimeError, match="missing ids"):
        _parse_batch_translation_output(raw, ["s1", "s2"])


def test_chunk_translation_jobs_respects_segment_limit():
    jobs = [_make_job("1", "aaaa"), _make_job("2", "bbbb"), _make_job("3", "cccc")]
    chunks = _chunk_translation_jobs(jobs, max_segments=2, max_chars=10000)
    assert [len(chunk) for chunk in chunks] == [2, 1]


def test_chunk_translation_jobs_respects_char_limit():
    jobs = [_make_job("1", "x" * 50), _make_job("2", "y" * 50), _make_job("3", "z" * 50)]
    chunks = _chunk_translation_jobs(jobs, max_segments=10, max_chars=300)
    assert [len(chunk) for chunk in chunks] == [1, 1, 1]


def test_batch_ineligibility_detects_brline():
    seg = Segment(
        segment_id="s1",
        location="body/p1",
        context={"part": "body"},
        source_plain="A",
        paragraph_ref=None,
        shielded_tagged="A⟦BRLINE_1⟧B",
    )
    cfg = PipelineConfig(llm=LLMConfig(batch_skip_on_brline=True))
    reasons = _batch_ineligibility_reasons(seg, cfg)
    assert "contains_brline" in reasons


def test_batch_ineligibility_detects_many_style_tokens():
    seg = Segment(
        segment_id="s1",
        location="body/p1",
        context={"part": "body"},
        source_plain="A",
        paragraph_ref=None,
        shielded_tagged="⟦S_1⟧a⟦/S_1⟧⟦S_2⟧b⟦/S_2⟧",
    )
    cfg = PipelineConfig(llm=LLMConfig(batch_max_style_tokens=1))
    reasons = _batch_ineligibility_reasons(seg, cfg)
    assert "too_many_style_tokens" in reasons


def test_batch_ineligibility_prefers_pre_hard_glossary_text_from_context():
    seg = Segment(
        segment_id="s1",
        location="body/p1",
        context={"part": "body", "_batch_eligibility_text": "A⟦BRLINE_1⟧B"},
        source_plain="A",
        paragraph_ref=None,
        shielded_tagged="A⟦GLS_1⟧B",
    )
    cfg = PipelineConfig(llm=LLMConfig(batch_skip_on_brline=True))
    reasons = _batch_ineligibility_reasons(seg, cfg)
    assert "contains_brline" in reasons


def test_batch_ineligibility_detects_toc_segments():
    seg = Segment(
        segment_id="s1",
        location="body/p1",
        context={"part": "body", "is_toc_entry": True},
        source_plain="A",
        paragraph_ref=None,
        shielded_tagged="A",
    )
    cfg = PipelineConfig(llm=LLMConfig(batch_skip_on_brline=True))
    reasons = _batch_ineligibility_reasons(seg, cfg)
    assert "toc_entry" in reasons


def test_translate_batch_group_marks_json_schema_violation():
    class FakeClient:
        supports_repair = True

        def __init__(self) -> None:
            self.calls = 0

        def translate(self, text: str, context: dict[str, str]) -> str:
            self.calls += 1
            if context.get("task") == "batch_translate":
                # Missing one of expected ids -> batch schema contract violation.
                return '{"translations":[{"id":"1","text":"T1"}]}'
            return "OK"

    jobs = [_make_job("1", "Alpha"), _make_job("2", "Beta")]
    cfg = PipelineConfig(
        llm=LLMConfig(retries=1),
        run=RunConfig(fail_fast_on_translate_error=False),
    )

    results = _translate_batch_group(jobs, cfg, FakeClient(), logging.getLogger("test"))

    assert len(results) == 2
    for _, _, _, _, issues in results:
        codes = {issue.code for issue in issues}
        assert "batch_fallback_single" in codes
        assert "batch_json_schema_violation" in codes


def test_translate_batch_group_retries_timeout_once_before_fallback():
    class FakeClient:
        supports_repair = True

        def __init__(self) -> None:
            self.batch_calls = 0

        def translate(self, text: str, context: dict[str, str]) -> str:
            del text
            if context.get("task") == "batch_translate":
                self.batch_calls += 1
                if self.batch_calls == 1:
                    raise RuntimeError("OpenAI request failed: The read operation timed out")
                return '{"1":"T1","2":"T2"}'
            return "UNUSED"

    jobs = [_make_job("1", "Alpha"), _make_job("2", "Beta")]
    cfg = PipelineConfig(
        llm=LLMConfig(retries=1),
        run=RunConfig(fail_fast_on_translate_error=False),
    )
    fake = FakeClient()

    results = _translate_batch_group(jobs, cfg, fake, logging.getLogger("test"))

    assert fake.batch_calls == 2
    assert len(results) == 2
    for seg, _, _, out, issues in results:
        assert out == ("T1" if seg.segment_id == "1" else "T2")
        codes = {issue.code for issue in issues}
        assert "batch_ok" in codes
        assert "batch_fallback_single" not in codes


def test_translate_batch_group_fail_fast_raises_on_batch_error():
    class FakeClient:
        supports_repair = True

        def translate(self, text: str, context: dict[str, str]) -> str:
            del text
            del context
            raise RuntimeError("The read operation timed out")

    jobs = [_make_job("1", "Alpha"), _make_job("2", "Beta")]
    cfg = PipelineConfig(
        llm=LLMConfig(retries=1),
        run=RunConfig(fail_fast_on_translate_error=True),
    )

    with pytest.raises(RuntimeError, match="Batch translate failed"):
        _translate_batch_group(jobs, cfg, FakeClient(), logging.getLogger("test"))


def test_build_batch_translation_prompt_mentions_context_and_glossary():
    prompt = _build_batch_translation_prompt(
        [
            {
                "id": "s1",
                "text": "Install Main Fitting",
                "context": "SECTION=32-10-00 | TABLE_CELL",
                "glossary": [{"source": "Main Fitting", "target": "Корпус стойки"}],
            }
        ]
    )
    assert "Use item.context for disambiguation" in prompt
    assert "item.glossary" in prompt
    assert "item.tm_hints/recent_translations" in prompt
    assert '"context"' in prompt
    assert '"glossary"' in prompt


def test_translate_batch_once_includes_per_item_context_glossary_and_tm_hints_in_prompt():
    class FakeClient:
        supports_repair = True

        def __init__(self) -> None:
            self.prompt = ""
            self.context = {}

        def translate(self, text: str, context: dict[str, str]) -> str:
            self.prompt = text
            self.context = context
            return '{"1":"T1","2":"T2"}'

    jobs = [_make_job("1", "Alpha"), _make_job("2", "Beta")]
    jobs[0][0].context.update(
        {
            "section_header": "32-10-00",
            "in_table": True,
            "matched_glossary_terms": [{"source": "Main Fitting", "target": "Корпус стойки"}],
            "tm_references": [{"source": "Install Main Fitting", "target": "Установите корпус стойки"}],
            "recent_translations": [{"source": "Install panel.", "target": "Установите панель."}],
        }
    )
    jobs[1][0].context.update({"section_header": "32-20-00"})

    fake = FakeClient()
    cfg = PipelineConfig(
        llm=LLMConfig(
            batch_tm_hints_per_item=1,
            batch_recent_translations_per_item=2,
        )
    )
    out = _translate_batch_once(fake, jobs, cfg)

    assert out == {"1": "T1", "2": "T2"}
    assert fake.context.get("task") == "batch_translate"
    assert '"context"' in fake.prompt
    assert "SECTION=32-10-00" in fake.prompt
    assert "TABLE_CELL" in fake.prompt
    assert '"glossary"' in fake.prompt
    assert '"tm_hints"' in fake.prompt
    assert '"recent_translations"' in fake.prompt
    assert "Main Fitting" in fake.prompt


def test_effective_manual_max_output_tokens_auto_raises_for_large_grouped_batches():
    raised = _effective_manual_max_output_tokens(
        auto_model_sizing=False,
        batch_segments=20,
        batch_max_chars=36_000,
        max_output_tokens=2_400,
        source_char_lengths=[450, 500, 520, 470, 610, 390],
    )
    assert raised > 2_400


def test_effective_manual_max_output_tokens_keeps_single_segment_budget():
    kept = _effective_manual_max_output_tokens(
        auto_model_sizing=False,
        batch_segments=1,
        batch_max_chars=36_000,
        max_output_tokens=2_400,
        source_char_lengths=[450, 500, 520],
    )
    assert kept == 2_400


def test_recommended_grouped_batch_workers_caps_huge_batches():
    workers = _recommended_grouped_batch_workers(
        concurrency=4,
        grouped_jobs_count=10,
        batch_max_chars=120_000,
    )
    assert workers == 2


def test_recommended_grouped_batch_workers_caps_60k_batches_to_two_workers():
    workers = _recommended_grouped_batch_workers(
        concurrency=4,
        grouped_jobs_count=10,
        batch_max_chars=60_000,
    )
    assert workers == 2


def test_recommended_grouped_batch_workers_respects_default_for_regular_batches():
    workers = _recommended_grouped_batch_workers(
        concurrency=4,
        grouped_jobs_count=3,
        batch_max_chars=18_000,
    )
    assert workers == 3
