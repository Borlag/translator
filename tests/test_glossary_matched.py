from __future__ import annotations

from docxru.llm import build_glossary_matchers, build_user_prompt, select_matched_glossary_terms


def test_select_matched_glossary_terms_respects_limit():
    glossary = """
Main Fitting — Корпус стойки
Lower Torque Link — Нижний рычаг крутящего момента
Sliding Tube — Скользящая трубка
"""
    matchers = build_glossary_matchers(glossary)
    matched = select_matched_glossary_terms(
        "Install Main Fitting and Lower Torque Link.",
        matchers,
        limit=1,
    )
    assert len(matched) == 1
    assert matched[0] in {
        ("Main Fitting", "Корпус стойки"),
        ("Lower Torque Link", "Нижний рычаг крутящего момента"),
    }


def test_select_matched_glossary_terms_handles_brline_wrapped_phrase():
    glossary = "lower bearing subassembly — нижний узел подшипника"
    matchers = build_glossary_matchers(glossary)
    matched = select_matched_glossary_terms("lower bearing⟦BRLINE_1⟧subassembly", matchers, limit=5)
    assert matched == [("lower bearing subassembly", "нижний узел подшипника")]


def test_build_user_prompt_includes_matched_glossary_and_tm_references():
    prompt = build_user_prompt(
        "Install Main Fitting.",
        {
            "part": "body",
            "matched_glossary_terms": [{"source": "Main Fitting", "target": "Корпус стойки"}],
            "tm_references": [
                {
                    "source": "Install the main fitting",
                    "target": "Установите корпус стойки",
                    "similarity": 0.91,
                }
            ],
            "tm_references_max_chars": 500,
        },
    )
    assert "MATCHED_GLOSSARY (EN -> RU)" in prompt
    assert "- Main Fitting -> Корпус стойки" in prompt
    assert "TM_REFERENCES:" in prompt
    assert "Install the main fitting => Установите корпус стойки" in prompt

