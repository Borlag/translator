from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Pattern, Protocol


class LLMClient(Protocol):
    supports_repair: bool

    def translate(self, text: str, context: dict[str, Any]) -> str: ...


SYSTEM_PROMPT_TEMPLATE = """Ты — профессиональный переводчик авиационной технической документации (CMM/AMM/IPC).

КРИТИЧЕСКИ ВАЖНО:
1) Строки/маркеры вида ⟦...⟧ НЕЛЬЗЯ менять, удалять, переставлять, переводить или добавлять новые.
   Это включает:
   - плейсхолдеры: ⟦PN_1⟧, ⟦ATA_2⟧, ⟦DIM_3⟧, ⟦REF_4⟧ и т.п.
   - маркеры форматирования: ⟦S_1|B|I⟧ ... ⟦/S_1⟧ и т.п.
   Оставь их В ТОЧНОСТИ как в исходном тексте.

2) Числа, знаки ±, диапазоны, единицы измерения, скобки и пунктуацию — сохраняй как в исходном.

3) Переводи техническим русским:
   - процедуры: повелительное наклонение (“Снимите”, “Установите”, “Проверьте”)
   - без художественности, без добавления объяснений от себя
   - термины по авиационному контексту (bolt = болт/винт по смыслу, torque = момент затяжки, washer = шайба, etc.)

4) Не добавляй предупреждений/допущений. Перевод должен быть максимально буквальным и корректным.

Если встречается непереводимый код/PN/номер/ссылка — оставь как есть (обычно это уже плейсхолдер).
"""


REPAIR_SYSTEM_PROMPT_TEMPLATE = """Ты — строгий валидатор/ремонтник маркеров для DOCX перевода.

ТВОЯ ЗАДАЧА:
- Восстановить маркеры вида ⟦...⟧ в OUTPUT так, чтобы они совпали с SOURCE.
- Менять обычный текст (не маркеры) ЗАПРЕЩЕНО, кроме минимальных правок пробелов вокруг маркеров,
  если это необходимо для корректной вставки/вложения маркеров.

ОГРАНИЧЕНИЯ:
- НЕЛЬЗЯ переводить, перефразировать или дополнять текст.
- НЕЛЬЗЯ добавлять новые маркеры, которых нет в SOURCE.
- Результат должен быть ТОЛЬКО исправленный OUTPUT (без комментариев).

Входной формат:
TASK: REPAIR_MARKERS

SOURCE:
...

OUTPUT:
...
"""

GlossaryReplacement = tuple[Pattern[str], str]


def build_user_prompt(text: str, context: dict[str, Any]) -> str:
    task = str(context.get("task", "translate")).lower()
    if task == "repair":
        return text

    # Context is kept short to reduce hallucination risk.
    ctx_parts: list[str] = []
    if context.get("section_header"):
        ctx_parts.append(f"SECTION: {context['section_header']}")
    if context.get("in_table"):
        ctx_parts.append(f"TABLE: r{context.get('row_index')} c{context.get('col_index')}")
    if context.get("part"):
        ctx_parts.append(f"PART: {context.get('part')}")
    ctx = " | ".join(ctx_parts) if ctx_parts else "(no context)"
    return f"Context: {ctx}\n\nText:\n{text}"


def supports_repair(client: LLMClient) -> bool:
    return bool(getattr(client, "supports_repair", False))


def _extract_repair_output(text: str) -> str:
    m = re.search(r"^OUTPUT:\s*\n(.*)\Z", text, flags=re.DOTALL | re.MULTILINE)
    return m.group(1) if m else text


def build_translation_system_prompt(
    base_system_prompt: str,
    *,
    custom_system_prompt: str | None = None,
    glossary_text: str | None = None,
) -> str:
    sections = [base_system_prompt.strip()]
    custom = (custom_system_prompt or "").strip()
    if custom:
        sections.append(f"Дополнительные инструкции:\n{custom}")
    glossary = (glossary_text or "").strip()
    if glossary:
        sections.append(f"Глоссарий (EN -> RU, приоритетно соблюдать):\n{glossary}")
    return "\n\n".join(sections)


def parse_glossary_pairs(glossary_text: str | None) -> list[tuple[str, str]]:
    if not glossary_text:
        return []

    pairs: list[tuple[str, str]] = []
    for raw_line in glossary_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Accept lines like: "Downlocking Spring — Пружина фиксации"
        m = re.match(r"^(.+?)\s+[—–-]\s+(.+)$", line)
        if not m:
            continue
        source_term = m.group(1).strip()
        target_term = m.group(2).strip()
        source_term = re.sub(r"^\d+[.)]?\s*", "", source_term)
        source_term = source_term.lstrip("•*- ").strip()
        if not source_term or not target_term:
            continue
        pairs.append((source_term, target_term))

    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return pairs


def _compile_term_pattern(source_term: str) -> Pattern[str]:
    escaped = re.escape(source_term)
    if re.search(r"[A-Za-z0-9]", source_term):
        # Restrict replacements to standalone ASCII terms to avoid accidental partial matches.
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", flags=re.IGNORECASE)
    return re.compile(escaped, flags=re.IGNORECASE)


def build_domain_replacements() -> tuple[GlossaryReplacement, ...]:
    """Static aviation-doc replacements to reduce EN leftovers in free-provider output."""
    term_pairs = [
        ("Record of Temporary Revisions", "Запись временных изменений"),
        ("Record of Revisions", "Запись изменений"),
        ("Revision Record", "Запись изменений"),
        ("List of Effective Pages", "Список действующих страниц"),
        ("List of Effective Pages (Continued)", "Список действующих страниц (продолжение)"),
        ("List of Service Bulletins", "Список сервисных бюллетеней"),
        ("New/Revised Pages", "Новые/пересмотренные страницы"),
        ("Table of Contents", "Содержание"),
        ("Table of Contents (Continued)", "Содержание (продолжение)"),
        ("Fig. Page", "Рис. Страница"),
        ("Fig Page", "Рис. Страница"),
        ("Subject Reference", "Тема Ссылка"),
        ("Main Landing Gear Leg", "Основная стойка шасси"),
        ("Lower Torque Link", "Нижний рычаг крутящего момента"),
        ("Upper Torque Link", "Верхний рычаг крутящего момента"),
        ("Sliding Tube", "Скользящая трубка"),
        ("Transfer Block", "Переходной блок"),
        ("Spherical Bearing", "Сферический подшипник"),
        ("Pivot Bracket", "Кронштейн шарнира"),
        ("Harness Support Bracket", "Кронштейн крепления жгута"),
        ("Retaining Pin", "Фиксирующий штифт"),
        ("Pintle Pin", "Шкворневой штифт"),
        ("Forward Pintle Pin", "Передний шкворневой штифт"),
        ("Uplock Pin", "Штифт аплока"),
        ("Main Fitting", "Основной фитинг"),
        ("Protective Treatment", "Защитная обработка"),
        ("Repair No.", "Ремонт №"),
        ("Lower", "Нижний"),
        ("Upper", "Верхний"),
        ("Torque", "Крутящий момент"),
        ("Link", "Рычаг"),
        ("Repair", "Ремонт"),
        ("Sheet", "Лист"),
        ("Withdrawn", "Аннулировано"),
        ("Illustrations", "Иллюстрации"),
        ("Blank", "Пусто"),
        ("Fig.", "Рис."),
        ("Fig", "Рис"),
        ("Subject", "Тема"),
        ("Table", "Таблица"),
        ("List", "Список"),
        ("Part No.", "№ детали"),
        ("Part", "Деталь"),
        ("Revision", "Изменений"),
        ("New/Revised", "Новые/пересмотренные"),
        ("Uplock", "Аплок"),
        ("No.", "№"),
    ]
    term_pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return tuple((_compile_term_pattern(source), target) for source, target in term_pairs)


def build_glossary_replacements(glossary_text: str | None) -> tuple[GlossaryReplacement, ...]:
    replacements: list[GlossaryReplacement] = []
    for source_term, target_term in parse_glossary_pairs(glossary_text):
        replacements.append((_compile_term_pattern(source_term), target_term))
    return tuple(replacements)


def apply_glossary_replacements(text: str, replacements: tuple[GlossaryReplacement, ...]) -> str:
    out = text
    for pattern, replacement in replacements:
        out = pattern.sub(replacement, out)

    # Mixed heading cleanup for partial machine translations.
    out = re.sub(r"\bSubject\s+Ссылка\b", "Тема Ссылка", out, flags=re.IGNORECASE)
    out = re.sub(r"\bNEW/REVISED\s+СТРАНИЦЫ\b", "НОВЫЕ/ПЕРЕСМОТРЕННЫЕ СТРАНИЦЫ", out, flags=re.IGNORECASE)
    out = re.sub(r"\bREVISION\s+ЗАПИСЬ\b", "ЗАПИСЬ ИЗМЕНЕНИЙ", out, flags=re.IGNORECASE)
    out = re.sub(r"\bLIST\s+СЕРВИСНЫХ\s+БЮЛЛЕТЕНЕЙ\b", "СПИСОК СЕРВИСНЫХ БЮЛЛЕТЕНЕЙ", out, flags=re.IGNORECASE)
    out = re.sub(r"\bPART\s+№\b", "№ ДЕТАЛИ", out, flags=re.IGNORECASE)
    out = re.sub(r"\btable\s+(\d+)\b", r"таблица \1", out, flags=re.IGNORECASE)
    out = re.sub(r"\(Table\s+(\d+)\)", r"(таблица \1)", out, flags=re.IGNORECASE)
    out = re.sub(r"\bFig\.\s+Страница\b", "Рис. Страница", out, flags=re.IGNORECASE)

    # TOC artifact cleanup: merged "Repair No. X-Y601" -> "Ремонт № X-Y 601"
    out = re.sub(r"Ремонт\s*№\s*(\d+-\d+)(\d{3,4})\b", r"Ремонт № \1 \2", out)
    # Normalize occasional machine-translated abbreviation variants.
    out = out.replace("Ремонт Ном.", "Ремонт №")
    out = out.replace("Ремонт Нет.", "Ремонт №")
    out = out.replace("Нет.", "№")
    out = out.replace("Штифт Штифт", "Штифт")
    return out


@dataclass(frozen=True)
class MockLLMClient:
    """Deterministic mock for tests and offline runs."""

    supports_repair: bool = True

    def translate(self, text: str, context: dict[str, Any]) -> str:
        task = str(context.get("task", "translate")).lower()
        if task == "repair":
            # Best-effort: return the provided OUTPUT block unchanged.
            return _extract_repair_output(text)
        return text


@dataclass(frozen=True)
class OpenAIChatCompletionsClient:
    """Minimal OpenAI Chat Completions client (template).

    Requires env:
      - OPENAI_API_KEY
    Optional:
      - OPENAI_BASE_URL (default https://api.openai.com)
    """

    model: str
    temperature: float = 0.1
    timeout_s: float = 60.0
    max_output_tokens: int = 2000
    base_url: str | None = None
    translation_system_prompt: str = SYSTEM_PROMPT_TEMPLATE
    glossary_replacements: tuple[GlossaryReplacement, ...] = ()
    supports_repair: bool = True

    def translate(self, text: str, context: dict[str, Any]) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        task = str(context.get("task", "translate")).lower()
        system_prompt = REPAIR_SYSTEM_PROMPT_TEMPLATE if task == "repair" else self.translation_system_prompt

        base = (self.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")).rstrip("/")
        url = f"{base}/v1/chat/completions"

        payload = {
            "model": self.model,
            "temperature": self.temperature if task != "repair" else 0.0,
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_prompt(text, context)},
            ],
        }
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            raise RuntimeError(f"OpenAI HTTPError {e.code}: {body}") from e
        except Exception as e:
            raise RuntimeError(f"OpenAI request failed: {e}") from e

        try:
            content = data["choices"][0]["message"]["content"]
            if task != "repair":
                content = apply_glossary_replacements(content, self.glossary_replacements)
            return content
        except Exception as e:
            raise RuntimeError(f"Unexpected OpenAI response schema: {data}") from e


@dataclass(frozen=True)
class GoogleFreeTranslateClient:
    """Google Translate free web endpoint client (no API key).

    This endpoint is unofficial and may change/rate-limit.
    """

    source_lang: str = "en"
    target_lang: str = "ru"
    timeout_s: float = 30.0
    base_url: str = "https://translate.googleapis.com"
    max_chunk_chars: int = 3800
    glossary_replacements: tuple[GlossaryReplacement, ...] = ()
    supports_repair: bool = False

    def _split_chunks(self, text: str) -> list[str]:
        if len(text) <= self.max_chunk_chars:
            return [text]

        chunks: list[str] = []
        cur = ""
        for part in text.splitlines(keepends=True):
            if len(part) > self.max_chunk_chars:
                if cur:
                    chunks.append(cur)
                    cur = ""
                for i in range(0, len(part), self.max_chunk_chars):
                    chunks.append(part[i : i + self.max_chunk_chars])
                continue

            if len(cur) + len(part) > self.max_chunk_chars:
                chunks.append(cur)
                cur = part
            else:
                cur += part

        if cur:
            chunks.append(cur)
        return chunks

    def _translate_chunk(self, text: str) -> str:
        query = urllib.parse.urlencode(
            {
                "client": "gtx",
                "sl": self.source_lang,
                "tl": self.target_lang,
                "dt": "t",
                "q": text,
            }
        )
        url = f"{self.base_url.rstrip('/')}/translate_a/single?{query}"
        req = urllib.request.Request(
            url=url,
            headers={"User-Agent": "docxru/0.1 (+free-google-endpoint)"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            raise RuntimeError(f"Google free translate HTTPError {e.code}: {body}") from e
        except Exception as e:
            raise RuntimeError(f"Google free translate request failed: {e}") from e

        try:
            parts = data[0]
            return "".join(p[0] for p in parts if isinstance(p, list) and p and isinstance(p[0], str))
        except Exception as e:
            raise RuntimeError(f"Unexpected Google free translate response schema: {data}") from e

    def translate(self, text: str, context: dict[str, Any]) -> str:
        task = str(context.get("task", "translate")).lower()
        if task == "repair":
            # This provider cannot perform structural marker repair.
            return _extract_repair_output(text)
        translated = "".join(self._translate_chunk(chunk) for chunk in self._split_chunks(text))
        return apply_glossary_replacements(translated, self.glossary_replacements)


@dataclass(frozen=True)
class OllamaChatClient:
    """Local Ollama chat client for on-device translation."""

    model: str
    temperature: float = 0.1
    timeout_s: float = 60.0
    max_output_tokens: int = 2000
    base_url: str = "http://localhost:11434"
    translation_system_prompt: str = SYSTEM_PROMPT_TEMPLATE
    glossary_replacements: tuple[GlossaryReplacement, ...] = ()
    supports_repair: bool = True

    def translate(self, text: str, context: dict[str, Any]) -> str:
        task = str(context.get("task", "translate")).lower()
        system_prompt = REPAIR_SYSTEM_PROMPT_TEMPLATE if task == "repair" else self.translation_system_prompt
        url = f"{self.base_url.rstrip('/')}/api/chat"

        options: dict[str, Any] = {"num_predict": self.max_output_tokens}
        options["temperature"] = 0.0 if task == "repair" else self.temperature

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_prompt(text, context)},
            ],
            "options": options,
        }
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            raise RuntimeError(f"Ollama HTTPError {e.code}: {body}") from e
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

        try:
            content = data["message"]["content"]
            if task != "repair":
                content = apply_glossary_replacements(content, self.glossary_replacements)
            return content
        except Exception as e:
            raise RuntimeError(f"Unexpected Ollama response schema: {data}") from e


def build_llm_client(
    provider: str,
    model: str,
    temperature: float,
    timeout_s: float,
    max_output_tokens: int,
    *,
    source_lang: str = "en",
    target_lang: str = "ru",
    base_url: str | None = None,
    custom_system_prompt: str | None = None,
    glossary_text: str | None = None,
) -> LLMClient:
    provider_norm = provider.strip().lower()
    translation_prompt = build_translation_system_prompt(
        SYSTEM_PROMPT_TEMPLATE,
        custom_system_prompt=custom_system_prompt,
        glossary_text=glossary_text,
    )
    glossary_replacements = build_domain_replacements() + build_glossary_replacements(glossary_text)
    if provider_norm == "mock":
        return MockLLMClient()
    if provider_norm == "openai":
        return OpenAIChatCompletionsClient(
            model=model,
            temperature=temperature,
            timeout_s=timeout_s,
            max_output_tokens=max_output_tokens,
            base_url=base_url,
            translation_system_prompt=translation_prompt,
            glossary_replacements=glossary_replacements,
        )
    if provider_norm == "google":
        return GoogleFreeTranslateClient(
            source_lang=source_lang,
            target_lang=target_lang,
            timeout_s=timeout_s,
            base_url=base_url or "https://translate.googleapis.com",
            glossary_replacements=glossary_replacements,
        )
    if provider_norm == "ollama":
        return OllamaChatClient(
            model=model,
            temperature=temperature,
            timeout_s=timeout_s,
            max_output_tokens=max_output_tokens,
            base_url=base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            translation_system_prompt=translation_prompt,
            glossary_replacements=glossary_replacements,
        )
    raise ValueError(f"Unknown LLM provider: {provider}")
