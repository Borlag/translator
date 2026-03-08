from __future__ import annotations

from unittest.mock import patch

import pytest

from docxru.pdf_models import FontSpec, PdfSpan, PdfSpanStyle, PdfTextBlock
from docxru.pdf_writer import build_bilingual_ocg, replace_block_text

fitz = pytest.importorskip("fitz")


def test_replace_block_text_inserts_translated_htmlbox(tmp_path):
    out_path = tmp_path / "out.pdf"

    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((50, 60), "Install bolt", fontsize=12)
    block = PdfTextBlock(
        block_id=0,
        bbox=(45, 40, 220, 90),
        text="Install bolt",
        spans=[
            PdfSpan(
                text="Install bolt",
                bbox=(45, 40, 220, 90),
                style=PdfSpanStyle(font_name="Arial", font_size_pt=12),
            )
        ],
    )
    ok, scale = replace_block_text(
        page,
        block,
        "Установите болт",
        FontSpec(family="Noto Sans"),
        max_font_shrink_ratio=0.6,
    )
    assert ok is True
    assert 0 < scale <= 1.0
    doc.save(str(out_path))
    doc.close()

    out_doc = fitz.open(str(out_path))
    text = out_doc[0].get_text()
    out_doc.close()
    assert "Установите болт" in text


def test_build_bilingual_ocg_creates_layer():
    doc = fitz.open()
    doc.new_page(width=100, height=100)
    ocg_xref = build_bilingual_ocg(doc)
    assert isinstance(ocg_xref, int)
    assert ocg_xref > 0
    doc.close()


def test_replace_block_text_retries_full_bbox_after_inner_bbox_overflow():
    class _FakePage:
        def __init__(self):
            self.calls = []

        def add_redact_annot(self, rect, fill=None):  # noqa: ANN001, ANN202
            self.redact_rect = tuple(rect)
            self.redact_fill = fill

        def apply_redactions(self):  # noqa: ANN201
            self.redacted = True

        def insert_htmlbox(self, rect, html_text, **kwargs):  # noqa: ANN001, ANN003, ANN202
            self.calls.append((tuple(rect), html_text, kwargs))
            return (4.0, 0.91)

    page = _FakePage()
    block = PdfTextBlock(
        block_id=0,
        bbox=(10.0, 20.0, 210.0, 120.0),
        text="Install bolt",
        spans=[
            PdfSpan(
                text="Install bolt",
                bbox=(20.0, 30.0, 180.0, 60.0),
                style=PdfSpanStyle(font_name="Arial", font_size_pt=12.0),
            )
        ],
    )

    with patch("docxru.pdf_writer._probe_insert_htmlbox", side_effect=[(-1.0, 0.72), (4.0, 0.91)]):
        ok, scale = replace_block_text(
            page,
            block,
            "Translated text",
            FontSpec(family="Arial"),
            inner_bbox=(20.0, 30.0, 180.0, 60.0),
            text_align="center",
            rotation_deg=90.0,
            line_height_factor=0.95,
        )

    assert ok is True
    assert scale == 0.91
    assert len(page.calls) == 1
    assert page.calls[0][0] == (10.0, 20.0, 210.0, 120.0)
    assert page.calls[0][2]["rotate"] == 90


def test_replace_block_text_keeps_original_when_probe_fails():
    class _FakePage:
        def __init__(self):
            self.calls = []
            self.rect = type("_Rect", (), {"width": 240.0, "height": 160.0})()
            self.redacted = False

        def add_redact_annot(self, rect, fill=None):  # noqa: ANN001, ANN202
            self.redacted = True

        def apply_redactions(self):  # noqa: ANN201
            self.redacted = True

        def insert_htmlbox(self, rect, html_text, **kwargs):  # noqa: ANN001, ANN003, ANN202
            self.calls.append((tuple(rect), html_text, kwargs))
            return (-1.0, 0.72)

    page = _FakePage()
    block = PdfTextBlock(
        block_id=0,
        bbox=(10.0, 20.0, 210.0, 120.0),
        text="Install bolt",
        spans=[
            PdfSpan(
                text="Install bolt",
                bbox=(20.0, 30.0, 180.0, 60.0),
                style=PdfSpanStyle(font_name="Arial", font_size_pt=12.0),
            )
        ],
    )

    with patch("docxru.pdf_writer._probe_insert_htmlbox", return_value=(-1.0, 0.72)):
        ok, scale = replace_block_text(
            page,
            block,
            "Translated text",
            FontSpec(family="Arial"),
            inner_bbox=(20.0, 30.0, 180.0, 60.0),
        )

    assert ok is False
    assert scale == 0.72
    assert page.redacted is False
    assert page.calls == []
