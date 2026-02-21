from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import patch

from docxru.llm import (
    GoogleFreeTranslateClient,
    MockLLMClient,
    OllamaChatClient,
    OpenAIChatCompletionsClient,
    apply_glossary_replacements,
    build_glossary_replacements,
    build_hard_glossary_replacements,
    build_llm_client,
    build_domain_replacements,
    build_translation_system_prompt,
    parse_glossary_pairs,
    supports_repair,
)
from docxru.token_shield import shield_terms


@dataclass
class _FakeResponse:
    body: str

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self.body.encode("utf-8")


def test_build_llm_client_supports_expected_providers():
    mock = build_llm_client(
        provider="mock",
        model="irrelevant",
        temperature=0.1,
        timeout_s=10.0,
        max_output_tokens=100,
    )
    assert isinstance(mock, MockLLMClient)
    assert supports_repair(mock)

    openai = build_llm_client(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.1,
        timeout_s=10.0,
        max_output_tokens=100,
    )
    assert isinstance(openai, OpenAIChatCompletionsClient)
    assert supports_repair(openai)

    google = build_llm_client(
        provider="google",
        model="ignored",
        temperature=0.1,
        timeout_s=10.0,
        max_output_tokens=100,
        source_lang="en",
        target_lang="ru",
    )
    assert isinstance(google, GoogleFreeTranslateClient)
    assert not supports_repair(google)

    ollama = build_llm_client(
        provider="ollama",
        model="qwen2.5:7b",
        temperature=0.1,
        timeout_s=10.0,
        max_output_tokens=100,
    )
    assert isinstance(ollama, OllamaChatClient)
    assert supports_repair(ollama)


def test_google_free_client_translate_with_mocked_http():
    urls: list[str] = []

    def fake_urlopen(req, timeout=0):
        urls.append(req.full_url)
        return _FakeResponse('[[["Привет","Hello",null,null,1]],null,"en"]')

    client = GoogleFreeTranslateClient(source_lang="en", target_lang="ru")
    with patch("docxru.llm.urllib.request.urlopen", side_effect=fake_urlopen):
        out = client.translate("Hello", {"task": "translate"})

    assert out == "Привет"
    assert urls and "sl=en" in urls[0] and "tl=ru" in urls[0]


def test_google_free_client_repair_mode_passthrough():
    client = GoogleFreeTranslateClient()
    text = "TASK: REPAIR_MARKERS\n\nSOURCE:\na\n\nOUTPUT:\nbad output"
    assert client.translate(text, {"task": "repair"}) == "bad output"


def test_parse_glossary_pairs_extracts_dash_separated_terms():
    glossary = """
## Glossary
Downlocking Spring — Пружина фиксации
1) Bearing — Подшипник
"""
    pairs = parse_glossary_pairs(glossary)
    assert ("Downlocking Spring", "Пружина фиксации") in pairs
    assert ("Bearing", "Подшипник") in pairs


def test_google_free_client_applies_glossary_replacements():
    def fake_urlopen(req, timeout=0):
        return _FakeResponse('[[["Downlocking Spring","Downlocking Spring",null,null,1]],null,"en"]')

    replacements = build_glossary_replacements("Downlocking Spring — Пружина фиксации")
    client = GoogleFreeTranslateClient(source_lang="en", target_lang="ru", glossary_replacements=replacements)
    with patch("docxru.llm.urllib.request.urlopen", side_effect=fake_urlopen):
        out = client.translate("Downlocking Spring", {"task": "translate"})

    assert out == "Пружина фиксации"


def test_build_translation_system_prompt_includes_optional_sections():
    prompt = build_translation_system_prompt(
        "BASE",
        custom_system_prompt="CUSTOM",
        glossary_text="TERM — ТЕРМИН",
    )
    assert "BASE" in prompt
    assert "CUSTOM" in prompt
    assert "TERM — ТЕРМИН" in prompt


def test_domain_replacements_include_repair_no():
    replacements = build_domain_replacements()
    out = "Repair No. 11-34"
    for pattern, repl in replacements:
        out = pattern.sub(repl, out)
    assert "Ремонт №" in out


def test_domain_replacements_normalize_common_sb_phrases():
    replacements = build_domain_replacements()
    out = (
        "Subject Reference | Insert New/Revised | "
        "Added fig-item (18-80A) in para 1. "
        "Updated Messier-Dowty Limited to Safran Landing Systems."
    )
    for pattern, repl in replacements:
        out = pattern.sub(repl, out)
    assert "Тема/ссылка" in out
    assert "Вставить новые/пересмотренные" in out
    assert "элемент рисунка" in out
    assert "Наименование Messier-Dowty Limited изменено на Safran Landing Systems" in out


def test_apply_glossary_replacements_fixes_common_google_artifacts():
    out = apply_glossary_replacements(
        "MLG Нога — добавлен элемент риса (18-80A). Обновленная ценность конверсии в рисунке 602.",
        (),
    )
    assert "Стойка MLG" in out
    assert "элемент рисунка" in out
    assert "Обновлено значение пересчета" in out


def test_glossary_replacements_match_wrapped_phrases_and_hyphens():
    replacements = build_glossary_replacements(
        "Introduction of new — Введение новых\nfig-item — элемент рисунка"
    )
    out = apply_glossary_replacements("Introduction of\nnew fig - item", replacements)
    assert "Введение новых" in out
    assert "элемент рисунка" in out


def test_hard_glossary_matches_across_brline_tokens():
    text = "lower bearing⟦BRLINE_1⟧subassembly"
    terms = build_hard_glossary_replacements("lower bearing subassembly — нижний узел подшипника")
    _, token_map = shield_terms(text, terms, token_prefix="GLS")
    assert "нижний узел подшипника" in token_map.values()


def test_ollama_client_translate_with_mocked_http():
    payloads: list[dict] = []

    def fake_urlopen(req, timeout=0):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse('{"message":{"content":"Привет"}}')

    client = OllamaChatClient(model="qwen2.5:7b")
    with patch("docxru.llm.urllib.request.urlopen", side_effect=fake_urlopen):
        out = client.translate("Hello", {"task": "translate"})

    assert out == "Привет"
    assert payloads and payloads[0]["model"] == "qwen2.5:7b"
    assert payloads[0]["stream"] is False
