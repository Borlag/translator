from __future__ import annotations

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
