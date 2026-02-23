from __future__ import annotations

from docxru.llm import build_glossary_matchers, build_user_prompt, select_matched_glossary_terms


def test_select_matched_glossary_terms_respects_limit():
    glossary = """
Main Fitting - Main fitting RU
Lower Torque Link - Lower torque link RU
Sliding Tube - Sliding tube RU
"""
    matchers = build_glossary_matchers(glossary)
    matched = select_matched_glossary_terms(
        "Install Main Fitting and Lower Torque Link.",
        matchers,
        limit=1,
    )
    assert len(matched) == 1
    assert matched[0] in {
        ("Main Fitting", "Main fitting RU"),
        ("Lower Torque Link", "Lower torque link RU"),
    }


def test_select_matched_glossary_terms_handles_brline_wrapped_phrase():
    glossary = "lower bearing subassembly - lower bearing assembly RU"
    matchers = build_glossary_matchers(glossary)
    matched = select_matched_glossary_terms("lower bearing⟦BRLINE_1⟧subassembly", matchers, limit=5)
    assert matched == [("lower bearing subassembly", "lower bearing assembly RU")]


def test_build_user_prompt_includes_context_blocks():
    prompt = build_user_prompt(
        "Install Main Fitting.",
        {
            "part": "body",
            "matched_glossary_terms": [{"source": "Main Fitting", "target": "Main fitting RU"}],
            "document_glossary": [{"source": "Bearing", "target": "Bearing RU"}],
            "tm_references": [
                {
                    "source": "Install the main fitting",
                    "target": "Install main fitting RU",
                    "similarity": 0.91,
                }
            ],
            "tm_references_max_chars": 500,
            "recent_translations": [{"source": "Remove the bolt.", "target": "Remove bolt RU."}],
            "recent_translations_max_chars": 500,
        },
    )
    assert "MATCHED_GLOSSARY (EN -> RU)" in prompt
    assert "- Main Fitting -> Main fitting RU" in prompt
    assert "DOCUMENT_GLOSSARY (EN -> RU)" in prompt
    assert "- Bearing -> Bearing RU" in prompt
    assert "TM_REFERENCES:" in prompt
    assert "Install the main fitting => Install main fitting RU" in prompt
    assert "RECENT_TRANSLATIONS (EN => RU):" in prompt
    assert "Remove the bolt. => Remove bolt RU." in prompt