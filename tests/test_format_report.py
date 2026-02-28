from __future__ import annotations

from pathlib import Path

from docxru.format_report import write_format_report
from docxru.models import Issue, Segment, Severity


def _seg(seg_id: str, *, issues: list[Issue]) -> Segment:
    return Segment(
        segment_id=seg_id,
        location=f"body/p{seg_id}",
        context={"part": "body"},
        source_plain="source",
        paragraph_ref=None,
        target_tagged="target",
        issues=issues,
    )


def test_write_format_report_emits_expected_sections(tmp_path: Path):
    segments = [
        _seg(
            "1",
            issues=[
                Issue(
                    code="layout_auto_fix_applied",
                    severity=Severity.INFO,
                    message="fixed",
                    details={"spacing_factor": 0.9},
                ),
                Issue(
                    code="layout_textbox_overflow_risk",
                    severity=Severity.WARN,
                    message="overflow",
                    details={},
                ),
                Issue(
                    code="writeback_inplace_fallback",
                    severity=Severity.INFO,
                    message="fallback",
                    details={},
                ),
            ],
        )
    ]

    out = tmp_path / "format_report.html"
    write_format_report(segments, out)

    text = out.read_text(encoding="utf-8")
    assert "Formatting Report" in text
    assert "Remaining Overflow Risks" in text
    assert "writeback_inplace_fallback" in text
