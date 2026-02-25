from __future__ import annotations

import re
from pathlib import Path

from docxru.config import load_config
from docxru.token_shield import PatternRule, PatternSet, shield, shield_terms, unshield


def test_shield_unshield_preserves_bracket_tokens():
    patterns = PatternSet(
        [
            PatternRule(name="PN", pattern="\\b\\d{6,}\\b"),
            PatternRule(name="REF", pattern="\\(\\s*\\d+\\s*-\\s*\\d+\\s*\\)"),
        ]
    )

    text = "Remove the ⟦S_1|B⟧bolt⟦/S_1⟧ (1-40) and washer PN 201587001."
    shielded, token_map = shield(text, patterns)

    # Style tags must remain unchanged
    assert "⟦S_1|B⟧" in shielded
    assert "⟦/S_1⟧" in shielded

    # Protected items replaced
    assert "⟦REF_1⟧" in shielded
    assert "⟦PN_1⟧" in shielded

    restored = unshield(shielded, token_map)
    assert restored == text


def test_shield_terms_can_preserve_break_boundaries():
    replacements = (
        (
            re.compile(r"lower(?:\s+|⟦BRLINE_\d+⟧)+bearing(?:\s+|⟦BRLINE_\d+⟧)+subassembly", flags=re.IGNORECASE),
            "нижний узел подшипника",
        ),
    )

    wrapped = "lower⟦BRLINE_1⟧bearing⟦BRLINE_2⟧subassembly"
    shielded_wrapped, token_map_wrapped = shield_terms(
        wrapped,
        replacements,
        token_prefix="GLS",
        bridge_break_tokens=False,
    )
    assert shielded_wrapped == wrapped
    assert token_map_wrapped == {}

    plain = "lower bearing subassembly"
    shielded_plain, token_map_plain = shield_terms(
        plain,
        replacements,
        token_prefix="GLS",
        bridge_break_tokens=False,
    )
    assert shielded_plain == "⟦GLS_1⟧"
    assert token_map_plain == {"⟦GLS_1⟧": "нижний узел подшипника"}


def test_default_regex_preset_shields_revision_and_service_references():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.example.yaml")
    text = "Rev. A per SB 1234-1 and AD 2023-15-12 in AMM 32-00-01 Figure 5A."
    shielded, token_map = shield(text, cfg.pattern_set)

    assert shielded != text
    values = list(token_map.values())
    assert any("Rev. A" in value for value in values)
    assert any("SB 1234-1" in value for value in values)
    assert any("AD 2023-15-12" in value for value in values)
    assert any("AMM 32-00-01" in value for value in values)
    assert any("Figure 5A" in value for value in values)


def test_default_regex_preset_reduces_false_positives_for_5a_and_5m():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.example.yaml")
    text = "Label 5A, gap 5m, current 5 A."
    shielded, token_map = shield(text, cfg.pattern_set)

    assert "5A" in shielded
    assert "5m" in shielded
    assert any(value == "5 A" for value in token_map.values())
