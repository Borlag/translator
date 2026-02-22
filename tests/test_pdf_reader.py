from __future__ import annotations

import pytest

from docxru.pdf_reader import extract_pdf_pages

fitz = pytest.importorskip("fitz")


def test_extract_pdf_pages_reads_text_blocks(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((50, 80), "Install bolt", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()

    pages = extract_pdf_pages(pdf_path)
    assert len(pages) == 1
    assert pages[0].has_text is True
    assert pages[0].blocks
    assert "Install bolt" in pages[0].blocks[0].text

