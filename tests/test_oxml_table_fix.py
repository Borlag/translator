from __future__ import annotations

import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from docxru.oxml_table_fix import normalize_abbyy_oxml, normalize_table_cell_margins, set_textbox_autofit


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
    frame_pr.set(qn("w:h"), "240")
    frame_pr.set(qn("w:hRule"), "exact")
    p_pr.append(frame_pr)
    return p_pr


def _append_exact_line_spacing(doc: Document):
    p = doc.add_paragraph("Line spacing sample")
    p_pr = p._p.get_or_add_pPr()
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:line"), "240")
    spacing.set(qn("w:lineRule"), "exact")
    p_pr.append(spacing)
    return spacing


def _append_textbox(doc: Document, *, text: str) -> tuple[object, object]:
    host = doc.add_paragraph("Host")
    run = host.add_run("")

    shape = OxmlElement("w:shape")
    body_pr = OxmlElement("a:bodyPr")
    body_pr.append(OxmlElement("a:noAutofit"))

    txbx_content = OxmlElement("w:txbxContent")
    if text:
        p = OxmlElement("w:p")
        r = OxmlElement("w:r")
        t = OxmlElement("w:t")
        t.text = text
        r.append(t)
        p.append(r)
        txbx_content.append(p)

    shape.append(body_pr)
    shape.append(txbx_content)
    run._r.append(shape)
    return body_pr, txbx_content


def _has_child(node, local_name: str) -> bool:
    for child in list(node):
        tag = str(getattr(child, "tag", ""))
        if tag.endswith("}" + local_name) or tag.endswith(":" + local_name) or tag == local_name:
            return True
    return False


def test_set_textbox_autofit_replaces_noautofit_for_non_empty_textbox():
    doc = Document()
    body_pr, _ = _append_textbox(doc, text="Textbox text")

    changed = set_textbox_autofit(doc)

    assert changed == 1
    assert _has_child(body_pr, "noAutofit") is False
    assert _has_child(body_pr, "normAutofit") is True


def test_set_textbox_autofit_skips_empty_textbox():
    doc = Document()
    body_pr, _ = _append_textbox(doc, text="")

    changed = set_textbox_autofit(doc)

    assert changed == 0
    assert _has_child(body_pr, "noAutofit") is True
    assert _has_child(body_pr, "normAutofit") is False


def test_normalize_abbyy_oxml_safe_removes_exact_trheight_only():
    doc = Document()
    tr_pr = _append_exact_tr_height(doc)
    p_pr = _append_frame_pr(doc)
    spacing = _append_exact_line_spacing(doc)

    stats = normalize_abbyy_oxml(doc, profile="safe")

    assert stats["tr_height_exact_removed"] == 1
    assert stats["frame_pr_removed"] == 0
    assert stats["frame_pr_exact_relaxed"] == 0
    assert stats["line_spacing_exact_relaxed"] == 0
    assert stats["textbox_autofit_updated"] == 0
    assert stats["table_cell_margins_normalized"] == 0
    assert tr_pr.find(qn("w:trHeight")) is None
    assert p_pr.find(qn("w:framePr")) is not None
    assert spacing.get(qn("w:lineRule")) == "exact"


def test_normalize_abbyy_oxml_aggressive_relaxes_framepr_height_rule():
    doc = Document()
    tr_pr = _append_exact_tr_height(doc)
    p_pr = _append_frame_pr(doc)
    spacing = _append_exact_line_spacing(doc)

    stats = normalize_abbyy_oxml(doc, profile="aggressive")

    assert stats["tr_height_exact_removed"] == 1
    assert stats["frame_pr_removed"] == 0
    assert stats["frame_pr_exact_relaxed"] == 1
    assert stats["line_spacing_exact_relaxed"] == 1
    assert stats["textbox_autofit_updated"] == 0
    assert stats["table_cell_margins_normalized"] == 0
    assert tr_pr.find(qn("w:trHeight")) is None
    frame_pr = p_pr.find(qn("w:framePr"))
    assert frame_pr is not None
    assert frame_pr.get(qn("w:hRule")) == "atLeast"
    assert spacing.get(qn("w:lineRule")) == "atLeast"


def test_normalize_abbyy_oxml_full_applies_textbox_autofit():
    doc = Document()
    tr_pr = _append_exact_tr_height(doc)
    p_pr = _append_frame_pr(doc)
    spacing = _append_exact_line_spacing(doc)
    body_pr, _ = _append_textbox(doc, text="Overflowing textbox")

    stats = normalize_abbyy_oxml(doc, profile="full")

    assert stats["tr_height_exact_removed"] == 1
    assert stats["frame_pr_removed"] == 0
    assert stats["frame_pr_exact_relaxed"] == 1
    assert stats["line_spacing_exact_relaxed"] == 1
    assert stats["textbox_autofit_updated"] == 1
    assert stats["table_cell_margins_normalized"] == 0
    assert tr_pr.find(qn("w:trHeight")) is None
    frame_pr = p_pr.find(qn("w:framePr"))
    assert frame_pr is not None
    assert frame_pr.get(qn("w:hRule")) == "atLeast"
    assert spacing.get(qn("w:lineRule")) == "atLeast"
    assert _has_child(body_pr, "noAutofit") is False
    assert _has_child(body_pr, "normAutofit") is True


def test_normalize_table_cell_margins_caps_excessive_values():
    doc = Document()
    cell = doc.add_table(rows=1, cols=1).cell(0, 0)
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement("w:tcMar")
    left = OxmlElement("w:left")
    left.set(qn("w:type"), "dxa")
    left.set(qn("w:w"), "480")
    tc_mar.append(left)
    tc_pr.append(tc_mar)

    changed = normalize_table_cell_margins(doc, max_margin_twips=108)

    assert changed == 1
    assert left.get(qn("w:w")) == "108"


def test_normalize_abbyy_oxml_rejects_invalid_profile():
    with pytest.raises(ValueError, match="Unsupported ABBYY profile"):
        normalize_abbyy_oxml(Document(), profile="unknown")
