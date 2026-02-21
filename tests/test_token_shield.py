from __future__ import annotations

from docxru.token_shield import PatternRule, PatternSet, shield, unshield


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
