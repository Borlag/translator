from __future__ import annotations

from docxru.pdf_font_map import select_replacement_font


def test_select_replacement_font_uses_family_heuristics():
    spec = select_replacement_font("TimesNewRomanPSMT", is_bold=True, is_italic=False)
    assert "Noto Serif" in spec.family
    assert spec.bold is True
    assert spec.italic is False
    assert spec.font_file is not None


def test_select_replacement_font_respects_explicit_mapping():
    spec = select_replacement_font(
        "FrutigerLTStd-Roman",
        is_bold=False,
        is_italic=True,
        font_map={"FrutigerLTStd-Roman": "PTSans-Regular.ttf"},
    )
    assert spec.family == "PTSans-Regular"
    assert spec.italic is True
    assert spec.font_file is not None

