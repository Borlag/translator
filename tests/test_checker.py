from __future__ import annotations

import json
import logging

from docx import Document

from docxru.checker import apply_checker_suggestions_to_segments, filter_checker_suggestions, run_llm_checker
from docxru.config import CheckerConfig
from docxru.models import Issue, Segment, Severity
from docxru.tagging import paragraph_to_tagged


class _FakeCheckerClient:
    def __init__(self, responses: list[object]):
        self._responses = list(responses)
        self.calls = 0

    def translate(self, text: str, context: dict[str, str]) -> str:
        self.calls += 1
        payload = self._responses.pop(0) if self._responses else {"chunk_id": "x", "edits": []}
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False)


class _FakeBatchCheckerClient(_FakeCheckerClient):
    def __init__(
        self,
        responses: list[object],
        *,
        batch_response: dict[str, object] | None = None,
        batch_error: Exception | None = None,
    ):
        super().__init__(responses)
        self._batch_response = batch_response or {}
        self._batch_error = batch_error
        self.batch_calls = 0
        self.batch_requests: list[tuple[str, str]] = []

    def run_checker_batch(
        self,
        requests: list[tuple[str, str]],
        *,
        completion_window: str,
        poll_interval_s: float,
        timeout_s: float,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, object]:
        del completion_window, poll_interval_s, timeout_s, metadata
        self.batch_calls += 1
        self.batch_requests = list(requests)
        if self._batch_error is not None:
            raise self._batch_error
        return dict(self._batch_response)


def _seg(seg_id: str, page: int) -> Segment:
    return Segment(
        segment_id=seg_id,
        location=f"pdf/p{page}/{seg_id}",
        context={"part": "pdf"},
        source_plain=f"SOURCE {seg_id}",
        paragraph_ref=None,
        target_tagged=f"TARGET {seg_id}",
    )


def _docx_seg(seg_id: str, paragraph_idx: int) -> Segment:
    return Segment(
        segment_id=seg_id,
        location=f"body/p{paragraph_idx}",
        context={"part": "body"},
        source_plain=f"SOURCE {seg_id}",
        paragraph_ref=None,
        target_tagged=f"TARGET {seg_id}",
    )


def test_run_llm_checker_adds_machine_edit_issues_in_page_chunks():
    segments = [_seg("s1", 1), _seg("s2", 2), _seg("s3", 3), _seg("s4", 4)]
    segments[0].issues.append(
        Issue(code="length_ratio_high", severity=Severity.WARN, message="warn", details={})
    )

    client = _FakeCheckerClient(
        [
            {
                "chunk_id": "pages_1_3",
                "edits": [
                    {
                        "segment_id": "s1",
                        "location": "pdf/p1/s1",
                        "severity": "warn",
                        "issue_type": "meaning",
                        "source_excerpt": "SOURCE s1",
                        "current_target": "TARGET s1",
                        "suggested_target": "TARGET s1 FIXED",
                        "instruction": "Replace with precise wording.",
                        "confidence": 0.88,
                    }
                ],
            },
            {
                "chunk_id": "pages_4_4",
                "edits": [
                    {
                        "segment_id": "s4",
                        "location": "pdf/p4/s4",
                        "severity": "error",
                        "issue_type": "terminology",
                        "source_excerpt": "SOURCE s4",
                        "current_target": "TARGET s4",
                        "suggested_target": "TARGET s4 FIXED",
                        "instruction": "Replace term.",
                        "confidence": 0.91,
                    }
                ],
            },
        ]
    )

    cfg = CheckerConfig(
        enabled=True,
        pages_per_chunk=3,
        only_on_issue_severities=("warn", "error"),
    )
    edits = run_llm_checker(
        segments=segments,
        checker_cfg=cfg,
        checker_client=client,
        logger=logging.getLogger("test_checker"),
    )

    assert client.calls == 2
    assert len(edits) == 2
    codes = [issue.code for seg in segments for issue in seg.issues if issue.code.startswith("llm_check_")]
    assert "llm_check_meaning" in codes
    assert "llm_check_terminology" in codes


def test_run_llm_checker_docx_locations_fallback_to_segment_windows(tmp_path):
    segments = [_docx_seg("s1", 10), _docx_seg("s2", 11), _docx_seg("s3", 12), _docx_seg("s4", 13), _docx_seg("s5", 14)]
    client = _FakeCheckerClient(
        [
            {"chunk_id": "segments_1_2", "edits": []},
            {"chunk_id": "segments_3_4", "edits": []},
            {"chunk_id": "segments_5_5", "edits": []},
        ]
    )
    trace_path = tmp_path / "checker_trace.jsonl"
    cfg = CheckerConfig(
        enabled=True,
        pages_per_chunk=3,
        fallback_segments_per_chunk=2,
        only_on_issue_severities=(),
        only_on_issue_codes=(),
    )
    edits = run_llm_checker(
        segments=segments,
        checker_cfg=cfg,
        checker_client=client,
        logger=logging.getLogger("test_checker"),
        trace_path=trace_path,
    )

    assert edits == []
    assert client.calls == 3
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    request_chunk_ids = [str(item.get("chunk_id") or "") for item in events if item.get("event") == "request"]
    assert request_chunk_ids == ["segments_1_2", "segments_3_4", "segments_5_5"]


def test_run_llm_checker_ignores_unknown_segment_ids():
    segments = [_seg("s1", 1)]
    client = _FakeCheckerClient(
        [
            {
                "chunk_id": "pages_1_1",
                "edits": [
                    {
                        "segment_id": "missing",
                        "location": "x",
                        "severity": "warn",
                        "issue_type": "other",
                        "source_excerpt": "x",
                        "current_target": "y",
                        "suggested_target": "z",
                        "instruction": "replace",
                        "confidence": 0.6,
                    }
                ],
            }
        ]
    )
    cfg = CheckerConfig(enabled=True, pages_per_chunk=3, only_on_issue_severities=(), only_on_issue_codes=())
    edits = run_llm_checker(
        segments=segments,
        checker_cfg=cfg,
        checker_client=client,
        logger=logging.getLogger("test_checker"),
    )
    assert edits == []
    assert not [issue for issue in segments[0].issues if issue.code.startswith("llm_check_")]


def test_run_llm_checker_splits_chunk_after_retries_exhausted():
    segments = [_seg("s1", 1), _seg("s2", 1)]
    for seg in segments:
        seg.issues.append(
            Issue(code="length_ratio_high", severity=Severity.WARN, message="warn", details={})
        )

    client = _FakeCheckerClient(
        [
            RuntimeError("Empty checker response"),
            {
                "chunk_id": "pages_1_1.a",
                "edits": [
                    {
                        "segment_id": "s1",
                        "location": "pdf/p1/s1",
                        "severity": "warn",
                        "issue_type": "meaning",
                        "source_excerpt": "SOURCE s1",
                        "current_target": "TARGET s1",
                        "suggested_target": "TARGET s1 FIXED",
                        "instruction": "replace s1",
                        "confidence": 0.9,
                    }
                ],
            },
            {
                "chunk_id": "pages_1_1.b",
                "edits": [
                    {
                        "segment_id": "s2",
                        "location": "pdf/p1/s2",
                        "severity": "warn",
                        "issue_type": "terminology",
                        "source_excerpt": "SOURCE s2",
                        "current_target": "TARGET s2",
                        "suggested_target": "TARGET s2 FIXED",
                        "instruction": "replace s2",
                        "confidence": 0.8,
                    }
                ],
            },
        ]
    )

    cfg = CheckerConfig(
        enabled=True,
        pages_per_chunk=3,
        retries=0,
        only_on_issue_severities=("warn", "error"),
    )
    edits = run_llm_checker(
        segments=segments,
        checker_cfg=cfg,
        checker_client=client,
        logger=logging.getLogger("test_checker"),
    )

    assert client.calls == 3
    assert len(edits) == 2
    codes = [issue.code for seg in segments for issue in seg.issues if issue.code.startswith("llm_check_")]
    assert "llm_check_meaning" in codes
    assert "llm_check_terminology" in codes


def test_run_llm_checker_writes_trace_and_stats(tmp_path):
    segments = [_seg("s1", 1)]
    client = _FakeCheckerClient(
        [
            {
                "chunk_id": "pages_1_1",
                "edits": [
                    {
                        "segment_id": "s1",
                        "location": "pdf/p1/s1",
                        "severity": "warn",
                        "issue_type": "meaning",
                        "source_excerpt": "SOURCE s1",
                        "current_target": "TARGET s1",
                        "suggested_target": "TARGET s1 FIXED",
                        "instruction": "replace s1",
                        "confidence": 0.9,
                    }
                ],
            }
        ]
    )
    trace_path = tmp_path / "checker_trace.jsonl"
    stats: dict[str, object] = {}
    cfg = CheckerConfig(enabled=True, pages_per_chunk=3, only_on_issue_severities=(), only_on_issue_codes=())

    edits = run_llm_checker(
        segments=segments,
        checker_cfg=cfg,
        checker_client=client,
        logger=logging.getLogger("test_checker"),
        trace_path=trace_path,
        stats_out=stats,
    )

    assert len(edits) == 1
    assert trace_path.exists()
    events = [json.loads(line).get("event") for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    assert events[0] == "start"
    assert "request" in events
    assert "response" in events
    assert events[-1] == "summary"
    assert stats["requests_total"] == 1
    assert stats["requests_succeeded"] == 1
    assert stats["requests_failed"] == 0
    assert stats["suggestions_total"] == 1


def test_run_llm_checker_openai_batch_mode_success():
    segments = [_seg("s1", 1), _seg("s2", 2)]
    client = _FakeBatchCheckerClient(
        responses=[],
        batch_response={
            "pages_1_2": {
                "content": json.dumps(
                    {
                        "chunk_id": "pages_1_2",
                        "edits": [
                            {
                                "segment_id": "s1",
                                "location": "pdf/p1/s1",
                                "severity": "warn",
                                "issue_type": "meaning",
                                "source_excerpt": "SOURCE s1",
                                "current_target": "TARGET s1",
                                "suggested_target": "TARGET s1 FIXED",
                                "instruction": "replace s1",
                                "confidence": 0.9,
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            }
        },
    )
    cfg = CheckerConfig(
        enabled=True,
        pages_per_chunk=3,
        openai_batch_enabled=True,
        only_on_issue_severities=(),
        only_on_issue_codes=(),
    )
    stats: dict[str, object] = {}
    edits = run_llm_checker(
        segments=segments,
        checker_cfg=cfg,
        checker_client=client,
        logger=logging.getLogger("test_checker"),
        stats_out=stats,
    )

    assert len(edits) == 1
    assert client.batch_calls == 1
    assert client.calls == 0
    assert stats["checker_mode"] == "openai_batch"
    assert stats["requests_total"] == 1
    assert any(issue.code == "llm_check_meaning" for issue in segments[0].issues)


def test_run_llm_checker_openai_batch_mode_falls_back_to_sync():
    segments = [_seg("s1", 1)]
    client = _FakeBatchCheckerClient(
        responses=[
            {
                "chunk_id": "pages_1_1",
                "edits": [
                    {
                        "segment_id": "s1",
                        "location": "pdf/p1/s1",
                        "severity": "warn",
                        "issue_type": "terminology",
                        "source_excerpt": "SOURCE s1",
                        "current_target": "TARGET s1",
                        "suggested_target": "TARGET s1 FIXED",
                        "instruction": "replace s1",
                        "confidence": 0.9,
                    }
                ],
            }
        ],
        batch_error=RuntimeError("batch temporary failure"),
    )
    cfg = CheckerConfig(
        enabled=True,
        pages_per_chunk=3,
        openai_batch_enabled=True,
        only_on_issue_severities=(),
        only_on_issue_codes=(),
    )
    stats: dict[str, object] = {}
    edits = run_llm_checker(
        segments=segments,
        checker_cfg=cfg,
        checker_client=client,
        logger=logging.getLogger("test_checker"),
        stats_out=stats,
    )

    assert len(edits) == 1
    assert client.batch_calls == 1
    assert client.calls == 1
    assert stats.get("batch_mode_fallback") is True
    assert "batch temporary failure" in str(stats.get("batch_mode_error") or "")


def test_filter_checker_suggestions_drops_noop_and_token_unsafe_edits():
    edits = [
        {
            "segment_id": "s1",
            "location": "body/p0",
            "current_target": "⟦S_1⟧A⟦/S_1⟧",
            "suggested_target": "⟦S_1⟧A⟦/S_1⟧",
            "instruction": "No change needed.",
            "confidence": 0.9,
        },
        {
            "segment_id": "s2",
            "location": "body/p1",
            "current_target": "⟦S_1⟧B⟦/S_1⟧",
            "suggested_target": "B",
            "instruction": "remove tags",
            "confidence": 0.9,
        },
        {
            "segment_id": "s3",
            "location": "body/p2",
            "current_target": "⟦S_1⟧C⟦/S_1⟧",
            "suggested_target": "⟦S_1⟧C-fixed⟦/S_1⟧",
            "instruction": "replace C with C-fixed",
            "confidence": 0.9,
        },
    ]
    safe, skipped = filter_checker_suggestions(edits, safe_only=True, min_confidence=0.7)
    assert len(safe) == 1
    assert safe[0]["segment_id"] == "s3"
    assert len(skipped) == 2


def test_apply_checker_suggestions_to_segments_applies_safe_edit():
    doc = Document()
    paragraph = doc.add_paragraph("SOURCE")
    tagged, spans, inline_run_map = paragraph_to_tagged(paragraph)
    seg = Segment(
        segment_id="s1",
        location="body/p0",
        context={"part": "body"},
        source_plain="SOURCE",
        paragraph_ref=paragraph,
        target_tagged=tagged,
        spans=spans,
        inline_run_map=inline_run_map,
    )
    edit = {
        "segment_id": "s1",
        "location": "body/p0",
        "current_target": tagged,
        "suggested_target": tagged.replace("SOURCE", "TARGET"),
        "instruction": "Replace SOURCE with TARGET.",
        "confidence": 0.95,
    }
    summary = apply_checker_suggestions_to_segments(
        segments=[seg],
        edits=[edit],
        safe_only=True,
        min_confidence=0.7,
        require_current_match=True,
    )
    assert summary["applied"] == 1
    assert paragraph.text == "TARGET"
