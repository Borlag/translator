from __future__ import annotations

from docx import Document

from docxru.models import Segment
from docxru.pipeline import _apply_final_run_level_cleanup, _should_translate_segment_text


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


def test_should_translate_segment_text_skips_russian_only():
    assert _should_translate_segment_text("Руководство по техническому обслуживанию компонентов") is False


def test_should_translate_segment_text_skips_russian_dominant_with_sparse_latin():
    text = "Руководство по техническому обслуживанию PN 201587001 ATA 32-12-22"
    assert _should_translate_segment_text(text) is False


def test_should_translate_segment_text_translates_english_text():
    text = "Install the charging valve and measure the nitrogen pressure."
    assert _should_translate_segment_text(text) is True
