from __future__ import annotations

from docx import Document

from docxru.models import Segment
from docxru.pipeline import _apply_final_run_level_cleanup


def test_final_cleanup_skips_untranslated_segments():
    doc = Document()
    paragraph = doc.add_paragraph("Table")
    seg = Segment(
        segment_id="s1",
        location="body/p1",
        context={"part": "body"},
        source_plain="Table",
        paragraph_ref=paragraph,
    )

    changed = _apply_final_run_level_cleanup([seg])

    assert changed == 0
    assert paragraph.text == "Table"


def test_final_cleanup_applies_to_translated_segments():
    doc = Document()
    paragraph = doc.add_paragraph("Table")
    seg = Segment(
        segment_id="s1",
        location="body/p1",
        context={"part": "body"},
        source_plain="Table",
        paragraph_ref=paragraph,
        target_shielded_tagged="table-ru",
        target_tagged="table-ru",
    )

    changed = _apply_final_run_level_cleanup([seg])

    assert changed == 1
    assert paragraph.text != "Table"
