from __future__ import annotations

from docxru.validator import validate_placeholders, validate_style_tokens


def test_validator_placeholder_mismatch():
    src = "Hello ⟦PN_1⟧ ⟦S_1⟧x⟦/S_1⟧"
    tgt = "Hello ⟦PN_2⟧ ⟦S_1⟧x⟦/S_1⟧"
    issues = validate_placeholders(src, tgt)
    assert issues and issues[0].code == "placeholders_mismatch"


def test_validator_style_token_mismatch():
    src = "⟦S_1⟧a⟦/S_1⟧⟦S_2⟧b⟦/S_2⟧"
    tgt = "⟦S_2⟧b⟦/S_2⟧⟦S_1⟧a⟦/S_1⟧"
    issues = validate_style_tokens(src, tgt)
    assert issues and issues[0].code == "style_tags_mismatch"
