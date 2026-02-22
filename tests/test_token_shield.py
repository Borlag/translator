from __future__ import annotations

import re

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
