from __future__ import annotations

import pytest

from docxru.models import Segment
from docxru.pipeline import _chunk_translation_jobs, _parse_batch_translation_output


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
