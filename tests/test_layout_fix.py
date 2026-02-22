from __future__ import annotations

import pytest
from docx import Document
from docx.shared import Pt

from docxru.config import PipelineConfig
from docxru.layout_fix import fix_expansion_issues
from docxru.models import Issue, Segment, Severity


def _issue(seg_id: str, code: str = "length_ratio_high") -> Issue:
    return Issue(
        code=code,
        severity=Severity.WARN,
        message="overflow risk",
        details={"segment_id": seg_id},
    )


def test_fix_expansion_issues_reduces_font_and_emits_issue():
    doc = Document()
    paragraph = doc.add_paragraph()
    run = paragraph.add_run("Long translated text")
    run.font.size = Pt(12)

    seg = Segment(
        segment_id="s1",
        location="body/p1",
        context={"part": "body"},
        source_plain="text",
        paragraph_ref=paragraph,
        target_tagged="Long translated text",
    )
    cfg = PipelineConfig(layout_auto_fix=True, layout_font_reduction_pt=0.5)

    applied = fix_expansion_issues([seg], [_issue("s1")], cfg)
    assert applied == 1
    assert run.font.size is not None
    assert run.font.size.pt == pytest.approx(11.5)
    assert any(issue.code == "layout_auto_fix_applied" for issue in seg.issues)


def test_fix_expansion_issues_reduces_non_table_spacing():
    doc = Document()
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(14)
    paragraph.paragraph_format.space_after = Pt(8)
    run = paragraph.add_run("Long translated text")
    run.font.size = Pt(11)

    seg = Segment(
        segment_id="s1a",
        location="body/p2",
        context={"part": "body", "in_textbox": True},
        source_plain="text",
        paragraph_ref=paragraph,
        target_tagged="Long translated text",
    )
    cfg = PipelineConfig(layout_auto_fix=True, layout_spacing_factor=0.5, layout_font_reduction_pt=0.5)

    applied = fix_expansion_issues([seg], [_issue("s1a", code="layout_textbox_overflow_risk")], cfg)
    assert applied == 1
    assert paragraph.paragraph_format.space_before is not None
    assert paragraph.paragraph_format.space_before.pt < 14
    assert paragraph.paragraph_format.space_after is not None
    assert paragraph.paragraph_format.space_after.pt < 8
    assert run.font.size is not None
    assert run.font.size.pt == pytest.approx(10.5)


def test_fix_expansion_issues_reduces_table_spacing():
    doc = Document()
    table = doc.add_table(rows=1, cols=1)
    paragraph = table.cell(0, 0).paragraphs[0]
    paragraph.text = "Translated"
    paragraph.paragraph_format.space_before = Pt(12)
    paragraph.paragraph_format.space_after = Pt(10)

    seg = Segment(
        segment_id="s2",
        location="body/t0/r0/c0/p0",
        context={"part": "body", "in_table": True},
        source_plain="text",
        paragraph_ref=paragraph,
        target_tagged="Translated",
    )
    cfg = PipelineConfig(layout_auto_fix=True, layout_spacing_factor=0.5, layout_font_reduction_pt=0.5)

    applied = fix_expansion_issues([seg], [_issue("s2", code="layout_table_overflow_risk")], cfg)
    assert applied == 1
    before = paragraph.paragraph_format.space_before
    after = paragraph.paragraph_format.space_after
    assert before is not None and before.pt < 12
    assert after is not None and after.pt < 10
