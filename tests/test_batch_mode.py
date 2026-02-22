from __future__ import annotations

import logging

import pytest

from docxru.config import LLMConfig, PipelineConfig
from docxru.models import Segment
from docxru.pipeline import (
    _batch_ineligibility_reasons,
    _chunk_translation_jobs,
    _parse_batch_translation_output,
    _translate_batch_group,
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
    cfg = PipelineConfig(llm=LLMConfig(retries=1))

    results = _translate_batch_group(jobs, cfg, FakeClient(), logging.getLogger("test"))

    assert len(results) == 2
    for _, _, _, _, issues in results:
        codes = {issue.code for issue in issues}
        assert "batch_fallback_single" in codes
        assert "batch_json_schema_violation" in codes
