from __future__ import annotations

import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from docxru.oxml_table_fix import normalize_abbyy_oxml


def _append_exact_tr_height(doc: Document):
    table = doc.add_table(rows=1, cols=1)
    tr_pr = table.rows[0]._tr.get_or_add_trPr()
    tr_height = OxmlElement("w:trHeight")
    tr_height.set(qn("w:val"), "240")
    tr_height.set(qn("w:hRule"), "exact")
    tr_pr.append(tr_height)
    return tr_pr


def _append_frame_pr(doc: Document):
    p = doc.add_paragraph("Sample")
    p_pr = p._p.get_or_add_pPr()
    frame_pr = OxmlElement("w:framePr")
    frame_pr.set(qn("w:w"), "100")
    p_pr.append(frame_pr)
    return p_pr


def test_normalize_abbyy_oxml_safe_removes_exact_trheight_only():
    doc = Document()
    tr_pr = _append_exact_tr_height(doc)
    p_pr = _append_frame_pr(doc)

    stats = normalize_abbyy_oxml(doc, profile="safe")

    assert stats["tr_height_exact_removed"] == 1
    assert stats["frame_pr_removed"] == 0
    assert tr_pr.find(qn("w:trHeight")) is None
    assert p_pr.find(qn("w:framePr")) is not None


def test_normalize_abbyy_oxml_aggressive_removes_framepr_too():
    doc = Document()
    tr_pr = _append_exact_tr_height(doc)
    p_pr = _append_frame_pr(doc)

    stats = normalize_abbyy_oxml(doc, profile="aggressive")

    assert stats["tr_height_exact_removed"] == 1
    assert stats["frame_pr_removed"] == 1
    assert tr_pr.find(qn("w:trHeight")) is None
    assert p_pr.find(qn("w:framePr")) is None


def test_normalize_abbyy_oxml_rejects_invalid_profile():
    with pytest.raises(ValueError, match="Unsupported ABBYY profile"):
        normalize_abbyy_oxml(Document(), profile="unknown")

