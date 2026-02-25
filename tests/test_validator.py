from __future__ import annotations

from docxru.validator import (
    validate_all,
    validate_context_leakage,
    validate_placeholders,
    validate_repeated_words,
    validate_style_tokens,
    validate_untranslated_fragments,
)


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


def test_validate_untranslated_fragments_warns_on_visible_english_leftovers():
    issues = validate_untranslated_fragments(
        "Remove the bolt and nut.",
        "Снимите bolt и nut.",
        warn_ratio=0.2,
        min_len=3,
    )
    assert issues and issues[0].code == "untranslated_fragments"


def test_validate_untranslated_fragments_allows_codes_and_common_abbreviations():
    issues = validate_untranslated_fragments(
        "Install PN123 on MLG.",
        "Установите PN123 на MLG.",
        warn_ratio=0.2,
        min_len=3,
    )
    assert issues == []


def test_validate_repeated_words_detects_word_or_phrase_repetition():
    issues = validate_repeated_words("Снимите снимите болт. Момент затяжки момент затяжки.")
    assert issues and issues[0].code == "repeated_words"


def test_validate_context_leakage_detects_prompt_tokens():
    issues = validate_context_leakage("Перевод SECTION=Hydraulic system")
    assert issues and issues[0].code == "context_leakage"


def test_validate_all_includes_new_warn_checks():
    issues = validate_all(
        source_shielded_tagged="Remove bolt.",
        target_shielded_tagged="TABLE_CELL bolt болт болт",
        source_unshielded_plain="Remove bolt.",
        target_unshielded_plain="TABLE_CELL bolt болт болт",
    )
    codes = {issue.code for issue in issues}
    assert "untranslated_fragments" in codes
    assert "repeated_words" in codes
    assert "context_leakage" in codes
