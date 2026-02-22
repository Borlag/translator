from __future__ import annotations

import io
import json
import urllib.error
from dataclasses import dataclass
from unittest.mock import patch

from docxru.llm import (
    GoogleFreeTranslateClient,
    MockLLMClient,
    OllamaChatClient,
    OpenAIChatCompletionsClient,
    apply_glossary_replacements,
    build_domain_replacements,
    build_glossary_replacements,
    build_hard_glossary_replacements,
    build_llm_client,
    build_translation_system_prompt,
    build_user_prompt,
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


def test_parse_glossary_pairs_extracts_markdown_table_rows():
    glossary = """
| English | Русский термин |
|---|---|
| Main Fitting | Корпус стойки |
| Sliding Tube | Скользящая трубка |
"""
    pairs = parse_glossary_pairs(glossary)
    assert ("Main Fitting", "Корпус стойки") in pairs
    assert ("Sliding Tube", "Скользящая трубка") in pairs


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


def test_domain_replacements_do_not_apply_single_word_rules_by_default():
    out = "Upper diaphragm tube"
    for pattern, repl in build_domain_replacements():
        out = pattern.sub(repl, out)
    assert out == "Upper diaphragm tube"

    out_with_single = "Upper diaphragm tube"
    for pattern, repl in build_domain_replacements(include_single_words=True):
        out_with_single = pattern.sub(repl, out_with_single)
    assert out_with_single != "Upper diaphragm tube"


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


def test_hard_glossary_skips_single_word_entries():
    terms = build_hard_glossary_replacements(
        "Repair — Ремонт\nMain Landing Gear Leg — Стойка основного шасси"
    )
    _, token_map = shield_terms("Repair Main Landing Gear Leg", terms, token_prefix="GLS")
    assert "Стойка основного шасси" in token_map.values()
    assert "Ремонт" not in token_map.values()


def test_apply_glossary_replacements_normalizes_with_and_pn_and():
    out = apply_glossary_replacements("WITH\n⟦PN_1⟧ AND ⟦PN_2⟧", ())
    assert "С" in out
    assert "⟦PN_1⟧ И ⟦PN_2⟧" in out


def test_apply_glossary_replacements_fixes_repair_procedure_conditions():
    out = apply_glossary_replacements("Repair Procedure Conditions 602", ())
    assert out == "Условия выполнения процедуры ремонта 602"


def test_apply_glossary_replacements_fixes_repair_no_merges():
    out = apply_glossary_replacements("Ремонт № 1-1 Ремонт узла нижнего подшипника №1-1601", ())
    assert "Ремонт № 1-1 узла нижнего подшипника № 1-1 601" == out


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


def test_ollama_client_repair_extracts_output_payload():
    def fake_urlopen(req, timeout=0):
        return _FakeResponse(
            '{"message":{"content":"TASK: REPAIR_MARKERS\\n\\nSOURCE:\\nX\\n\\nOUTPUT:\\nfixed text"}}'
        )

    client = OllamaChatClient(model="qwen2.5:7b")
    with patch("docxru.llm.urllib.request.urlopen", side_effect=fake_urlopen):
        out = client.translate("TASK: REPAIR_MARKERS\n\nSOURCE:\nX\n\nOUTPUT:\nbad", {"task": "repair"})

    assert out == "fixed text"


def test_build_user_prompt_passthrough_for_batch_task():
    text = '{"translations":[{"id":"1","text":"T1"}]}'
    out = build_user_prompt(text, {"task": "batch_translate", "part": "body"})
    assert out == text


def test_openai_client_batch_uses_json_mode_and_skips_temperature_for_gpt5():
    payloads: list[dict] = []

    def fake_urlopen(req, timeout=0):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse('{"choices":[{"message":{"content":"{\\"translations\\":[{\\"id\\":\\"s1\\",\\"text\\":\\"T1\\"}]}"}}]}')

    client = OpenAIChatCompletionsClient(
        model="gpt-5-mini",
        reasoning_effort="minimal",
        prompt_cache_key="docxru-test",
        prompt_cache_retention="24h",
    )
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
        "docxru.llm.urllib.request.urlopen",
        side_effect=fake_urlopen,
    ):
        out = client.translate("{}", {"task": "batch_translate"})

    assert '"translations"' in out
    assert payloads and payloads[0]["response_format"] == {"type": "json_object"}
    assert "temperature" not in payloads[0]
    assert payloads[0]["max_completion_tokens"] == client.max_output_tokens
    assert "max_tokens" not in payloads[0]
    assert payloads[0]["prompt_cache_key"] == "docxru-test"
    assert payloads[0]["prompt_cache_retention"] == "24h"


def test_openai_client_non_gpt5_uses_max_tokens():
    payloads: list[dict] = []

    def fake_urlopen(req, timeout=0):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse('{"choices":[{"message":{"content":"ok"}}]}')

    client = OpenAIChatCompletionsClient(model="gpt-4o-mini")
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
        "docxru.llm.urllib.request.urlopen",
        side_effect=fake_urlopen,
    ):
        out = client.translate("hello", {"task": "translate"})

    assert out == "ok"
    assert payloads and payloads[0]["max_tokens"] == client.max_output_tokens
    assert "max_completion_tokens" not in payloads[0]


def test_openai_client_repair_extracts_output_payload():
    def fake_urlopen(req, timeout=0):
        return _FakeResponse(
            '{"choices":[{"message":{"content":"TASK: REPAIR_MARKERS\\n\\nSOURCE:\\nX\\n\\nOUTPUT:\\nrestored"}}]}'
        )

    client = OpenAIChatCompletionsClient(model="gpt-4o-mini")
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
        "docxru.llm.urllib.request.urlopen",
        side_effect=fake_urlopen,
    ):
        out = client.translate("TASK: REPAIR_MARKERS\n\nSOURCE:\nX\n\nOUTPUT:\nbad", {"task": "repair"})

    assert out == "restored"


def test_openai_client_retries_without_prompt_cache_retention_when_unsupported():
    payloads: list[dict] = []
    call_n = {"n": 0}

    def fake_urlopen(req, timeout=0):
        payloads.append(json.loads(req.data.decode("utf-8")))
        call_n["n"] += 1
        if call_n["n"] == 1:
            body = json.dumps(
                {
                    "error": {
                        "message": "prompt_cache_retention is not supported on this model",
                        "type": "invalid_request_error",
                        "param": "prompt_cache_retention",
                        "code": "invalid_parameter",
                    }
                }
            )
            raise urllib.error.HTTPError(
                url=req.full_url,
                code=400,
                msg="Bad Request",
                hdrs=None,
                fp=io.BytesIO(body.encode("utf-8")),
            )
        return _FakeResponse('{"choices":[{"message":{"content":"ok"}}]}')

    client = OpenAIChatCompletionsClient(
        model="gpt-5-mini",
        prompt_cache_key="docxru-cache",
        prompt_cache_retention="24h",
    )
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
        "docxru.llm.urllib.request.urlopen",
        side_effect=fake_urlopen,
    ):
        out = client.translate("hello", {"task": "translate"})

    assert out == "ok"
    assert len(payloads) == 2
    assert "prompt_cache_retention" in payloads[0]
    assert "prompt_cache_retention" not in payloads[1]
