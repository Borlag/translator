from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


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
   - термины из глоссария можно склонять и согласовывать по правилам русского языка
     (сохраняй значение и терминологическую основу, не «замораживай» форму)

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

BATCH_SYSTEM_PROMPT_TEMPLATE = """You translate technical aviation text from English to Russian.
Return ONLY valid JSON in the requested schema.
Do not add commentary.
Preserve all marker tokens exactly (for example: ⟦...⟧ / 🦦...🧧).
Preserve numbers, units, and punctuation.
"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", flags=re.IGNORECASE)

GlossaryReplacement = tuple[re.Pattern[str], str]
GlossaryMatcher = tuple[str, str, re.Pattern[str]]


def _compact_prompt_snippet(value: Any, *, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _coerce_glossary_pairs(value: Any) -> list[tuple[str, str]]:
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]

    pairs: list[tuple[str, str]] = []
    for item in items:
        source: Any = None
        target: Any = None
        if isinstance(item, dict):
            source = item.get("source")
            target = item.get("target")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            source, target = item[0], item[1]

        if not isinstance(source, str) or not isinstance(target, str):
            continue
        source_text = source.strip()
        target_text = target.strip()
        if not source_text or not target_text:
            continue
        pairs.append((source_text, target_text))
    return pairs


def _format_matched_glossary_block(value: Any) -> str:
    pairs = _coerce_glossary_pairs(value)
    if not pairs:
        return ""
    lines = [f"- {source} -> {target}" for source, target in pairs]
    return "\n".join(lines)


def _format_recent_translations_block(value: Any, *, max_chars: int = 500) -> str:
    pairs = _coerce_glossary_pairs(value)
    if not pairs:
        return ""

    lines: list[str] = []
    consumed = 0
    budget = max(0, int(max_chars))
    for source, target in pairs:
        src = _compact_prompt_snippet(source, max_chars=120)
        tgt = _compact_prompt_snippet(target, max_chars=120)
        if not src or not tgt:
            continue
        line = f"- {src} => {tgt}"
        line_cost = len(line) + 1
        if budget > 0 and consumed + line_cost > budget:
            break
        lines.append(line)
        consumed += line_cost
    return "\n".join(lines)


def _format_tm_references_block(value: Any, *, max_chars: int = 500) -> str:
    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]

    lines: list[str] = []
    consumed = 0
    budget = max(0, int(max_chars))
    for item in items:
        source: Any = None
        target: Any = None
        similarity: Any = None
        if isinstance(item, dict):
            source = item.get("source")
            target = item.get("target")
            similarity = item.get("similarity")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            source, target = item[0], item[1]
            similarity = item[2] if len(item) > 2 else None
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        src = _compact_prompt_snippet(source, max_chars=120)
        tgt = _compact_prompt_snippet(target, max_chars=120)
        if not src or not tgt:
            continue
        sim_txt = ""
        if isinstance(similarity, (int, float)):
            sim_txt = f" ({float(similarity):.2f})"
        line = f"-{sim_txt} {src} => {tgt}"
        line_cost = len(line) + 1
        if budget > 0 and consumed + line_cost > budget:
            break
        lines.append(line)
        consumed += line_cost
    return "\n".join(lines)


def build_user_prompt(text: str, context: dict[str, Any]) -> str:
    task = str(context.get("task", "translate")).lower()
    if task in {"repair", "batch_translate"}:
        return text

    # Context is kept short to reduce hallucination risk.
    ctx_parts: list[str] = []
    if context.get("section_header"):
        ctx_parts.append(f"SECTION: {context['section_header']}")
    if context.get("in_table"):
        ctx_parts.append("TABLE_CELL")
    if context.get("part"):
        ctx_parts.append(f"PART: {context.get('part')}")
    if not context.get("in_table"):
        prev_text = _compact_prompt_snippet(context.get("prev_text"))
        next_text = _compact_prompt_snippet(context.get("next_text"))
        if prev_text:
            ctx_parts.append(f"PREV: {prev_text}")
        if next_text:
            ctx_parts.append(f"NEXT: {next_text}")
    ctx = " | ".join(ctx_parts) if ctx_parts else "(no context)"

    extra_blocks: list[str] = []
    matched_glossary = _format_matched_glossary_block(context.get("matched_glossary_terms"))
    if matched_glossary:
        extra_blocks.append(f"MATCHED_GLOSSARY (EN -> RU):\n{matched_glossary}")

    document_glossary = _format_matched_glossary_block(context.get("document_glossary"))
    if document_glossary:
        extra_blocks.append(f"DOCUMENT_GLOSSARY (EN -> RU):\n{document_glossary}")

    raw_tm_max_chars = context.get("tm_references_max_chars", 500)
    try:
        tm_max_chars = int(raw_tm_max_chars)
    except (TypeError, ValueError):
        tm_max_chars = 500
    tm_references = _format_tm_references_block(context.get("tm_references"), max_chars=tm_max_chars)
    if tm_references:
        extra_blocks.append(f"TM_REFERENCES:\n{tm_references}")

    raw_recent_max_chars = context.get("recent_translations_max_chars", 500)
    try:
        recent_max_chars = int(raw_recent_max_chars)
    except (TypeError, ValueError):
        recent_max_chars = 500
    recent_translations = _format_recent_translations_block(
        context.get("recent_translations"),
        max_chars=recent_max_chars,
    )
    if recent_translations:
        extra_blocks.append(f"RECENT_TRANSLATIONS (EN => RU):\n{recent_translations}")

    extra_section = ""
    if extra_blocks:
        extra_section = "\n\n" + "\n\n".join(extra_blocks)
    return (
        "Use context only for disambiguation. Do not translate or repeat context in output.\n"
        f"Context: {ctx}{extra_section}\n\n"
        "Translate ONLY the text below:\n"
        f"{text}"
    )


def supports_repair(client: LLMClient) -> bool:
    return bool(getattr(client, "supports_repair", False))


def _extract_repair_output(text: str) -> str:
    raw = text or ""
    # Primary path: extract everything after the last OUTPUT: marker.
    marker = "OUTPUT:"
    idx = raw.upper().rfind(marker)
    if idx >= 0:
        return raw[idx + len(marker) :].lstrip("\r\n ").rstrip()

    # Fallback for markdown wrappers like ``` ... OUTPUT: ... ```
    m = re.search(r"```(?:text|md|markdown)?\s*([\s\S]*?)\s*```", raw, flags=re.IGNORECASE)
    if m:
        inner = m.group(1)
        idx2 = inner.upper().rfind(marker)
        if idx2 >= 0:
            return inner[idx2 + len(marker) :].lstrip("\r\n ").rstrip()
        return inner.strip()

    return raw


def _normalize_structured_output_mode(mode: str | None) -> str:
    value = (mode or "auto").strip().lower()
    if value not in {"off", "auto", "strict"}:
        raise ValueError(f"Unsupported structured_output_mode: {mode!r}")
    return value


def _extract_json_payload(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        raise RuntimeError("Empty JSON response")

    candidates: list[str] = [text]
    for m in _JSON_FENCE_RE.finditer(text):
        fenced = (m.group(1) or "").strip()
        if fenced:
            candidates.append(fenced)

    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        candidates.append(text[obj_start : obj_end + 1].strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("Response is not valid JSON")


def _extract_structured_text(raw: str, *, field_name: str) -> str:
    payload = _extract_json_payload(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("Structured response must be a JSON object")
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise RuntimeError(f"Structured response missing string field: {field_name}")
    return value


def _is_gpt5_family(model: str) -> bool:
    return model.strip().lower().startswith("gpt-5")


def _supports_temperature(model: str, reasoning_effort: str | None) -> bool:
    m = model.strip().lower()
    if not _is_gpt5_family(m):
        return True
    if m.startswith("gpt-5.1") or m.startswith("gpt-5.2"):
        return (reasoning_effort or "").strip().lower() in {"", "none"}
    return False


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

    def _strip_source_prefix(term: str) -> str:
        cleaned = term
        cleaned = re.sub(r"^\d+[.)]?\s*", "", cleaned)
        cleaned = cleaned.lstrip("•*- ").strip()
        return cleaned

    def _is_md_separator(cell: str) -> bool:
        return bool(re.fullmatch(r":?-{3,}:?", cell.strip()))

    pairs: list[tuple[str, str]] = []
    for raw_line in glossary_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Accept markdown table rows: | English | Русский термин |
        if line.startswith("|") and line.endswith("|"):
            cols = [c.strip() for c in line.strip("|").split("|")]
            if len(cols) >= 2:
                source_term = cols[0]
                target_term = cols[1]
                if (
                    source_term
                    and target_term
                    and not (source_term.lower() == "english" and target_term.lower().startswith("рус"))
                    and not (_is_md_separator(source_term) and _is_md_separator(target_term))
                ):
                    source_term = _strip_source_prefix(source_term)
                    if source_term and target_term:
                        pairs.append((source_term, target_term))
            continue

        # Accept lines like: "Downlocking Spring — Пружина фиксации"
        m = re.match(r"^(.+?)\s+[—–-]\s+(.+)$", line)
        if not m:
            continue
        source_term = _strip_source_prefix(m.group(1).strip())
        target_term = m.group(2).strip()
        if not source_term or not target_term:
            continue
        pairs.append((source_term, target_term))

    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return pairs


def _compile_term_pattern(source_term: str) -> re.Pattern[str]:
    escaped = re.escape(source_term.strip())
    gap = r"(?:\s+|⟦BR(?:LINE|COL|PAGE)_\d+⟧)+"
    optional_gap = r"(?:\s+|⟦BR(?:LINE|COL|PAGE)_\d+⟧)*"
    # OCR/Word converted manuals often insert hard line-wraps and variable spaces around hyphens.
    # Make term matching tolerant to those artifacts so phrase-level glossary enforcement still works.
    escaped = escaped.replace(r"\ ", gap)
    escaped = escaped.replace(r"\-", rf"{optional_gap}-{optional_gap}")
    if re.search(r"[A-Za-z0-9]", source_term):
        # Restrict replacements to standalone ASCII terms to avoid accidental partial matches.
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", flags=re.IGNORECASE)
    return re.compile(escaped, flags=re.IGNORECASE)


def build_glossary_matchers(glossary_text: str | None) -> tuple[GlossaryMatcher, ...]:
    matchers: list[GlossaryMatcher] = []
    for source_term, target_term in parse_glossary_pairs(glossary_text):
        matchers.append((source_term, target_term, _compile_term_pattern(source_term)))
    return tuple(matchers)


def select_matched_glossary_terms(
    text: str,
    matchers: tuple[GlossaryMatcher, ...],
    *,
    limit: int = 24,
) -> list[tuple[str, str]]:
    if not text or not matchers:
        return []

    max_terms = max(0, int(limit))
    if max_terms == 0:
        return []

    seen_sources: set[str] = set()
    matched: list[tuple[str, str]] = []
    for source_term, target_term, pattern in matchers:
        source_key = source_term.strip().lower()
        if not source_key or source_key in seen_sources:
            continue
        if not pattern.search(text):
            continue
        seen_sources.add(source_key)
        matched.append((source_term, target_term))
        if len(matched) >= max_terms:
            break
    return matched


DOMAIN_TERM_PAIRS: tuple[tuple[str, str], ...] = (
    # Cover-page legal small print: keep these translations concise to avoid layout overflow.
    (
        "This document and all information contained herein is the sole property of Safran Landing Systems (and/or its affiliated companies).",
        "Документ и его содержание - собственность Safran Landing Systems (и/или аффилированных компаний).",
    ),
    (
        "No intellectual property rights are granted by the delivery of this document or the disclosure of its content.",
        "Права ИС не предоставляются.",
    ),
    (
        "This document shall not be reproduced to a third party without the express written consent of Safran Landing Systems (and/or the appropriate affiliated company).",
        "Воспроизведение третьим лицам - только с письменного согласия Safran Landing Systems.",
    ),
    # Frequent SB-description lines in ABBYY-converted manuals.
    (
        "MLG - Installation of stub bolt subassembly for the forward pintle pin in place of the cross bolt.",
        "MLG - Установка подсборки болта-заглушки для переднего шкворневого штифта вместо поперечного болта.",
    ),
    (
        "MLG - To allow an increase in aircraft maximum take-off weight to 93 tonne.",
        "MLG - Для увеличения максимальной взлетной массы самолета до 93 т.",
    ),
    (
        "MLG -To add tracking numbers to parts listed in Airbus Airworthiness Limitations Section (ALS).",
        "MLG - Добавлены номера отслеживания к деталям, перечисленным в разделе Airbus Airworthiness Limitations Section (ALS).",
    ),
    (
        "MLG - Installation of a 201585 series MLG Leg and Dressings where a 201387 MLG Leg and Dressings has been installed.",
        "MLG - Установка стойки MLG серии 201585 и комплектов dressings вместо ранее установленной стойки MLG серии 201387 и комплектов dressings.",
    ),
    (
        "MLG -To add tracking numbers to parts listed in Airbus Maintenance Planning Document, Section 9-1. (Torque link apex pin nut)",
        "MLG - Добавлены номера отслеживания к деталям, перечисленным в Airbus Maintenance Planning Document, раздел 9-1. (Гайка шкворневого штифта вершины рычага крутящего момента)",
    ),
    ("MLG - Introduction of a new lower bearing subassembly.", "MLG - Введение нового нижнего узла подшипника."),
    ("MLG - Introduction of new charging labels", "MLG - Введение новых маркировочных табличек."),
    ("MLG - Introduction of new 1M and 2M Axle harnesses", "MLG - Введение новых жгутов оси 1M и 2M."),
    (
        "MLG - Introduction of new 1M and 2M Leg Harness and of new 1M and 2M Axle Harnesses",
        "MLG - Введение новых жгутов стойки 1M и 2M, а также новых жгутов оси 1M и 2M.",
    ),
    (
        "MLG Leg-Introduction of new retaining pins and a new lower bearing subassembly with a new self lubricating liner",
        "Стойка MLG - Введение новых стопорных штифтов и нового нижнего узла подшипника с новым самосмазывающимся вкладышем.",
    ),
    (
        "MLG Leg - Introduction of new retaining pins for the lower bearing subassembly",
        "Стойка MLG - Введение новых стопорных штифтов для нижнего узла подшипника.",
    ),
    (
        "MLG Leg - Introduction of a new lower bearing subassembly with a new low friction inner liner",
        "Стойка MLG - Введение нового нижнего узла подшипника с новым внутренним вкладышем с низким коэффициентом трения.",
    ),
    (
        "MLG Leg - Barkhausen Noise Inspection of Main Landing Gear Sliding Tube Axles.",
        "Стойка MLG - Контроль шума Баркхаузена осей скользящей трубы основной стойки шасси.",
    ),
    ("MLG Leg - Introduction of a new Main Fitting", "Стойка MLG - Введение нового корпуса стойки."),
    ("MLG Leg - Introduction of a new torque link damper unit", "Стойка MLG - Введение нового демпферного узла рычага крутящего момента."),
    (
        "MLG Leg - Introduction of a new main fitting subassembly and related parts",
        "Стойка MLG - Введение новой подсборки корпуса стойки и связанных деталей.",
    ),
    ("MLG - Introduction of a new upper pivot bracket", "MLG - Введение нового верхнего кронштейна шарнира."),
    ("MLG - Introduction of a new changeover valve stem and housing", "MLG - Введение нового штока и корпуса переключающего клапана."),
    ("MLG complete - Introduction of a new transfer block subassembly", "Стойка MLG в сборе - Введение новой подсборки переходного блока."),
    ("MLG Complete - Modification of the transfer block subassembly", "Стойка MLG в сборе - Модификация подсборки переходного блока."),
    (
        "MLG - Conversion of low - friction lower - bearing MLG to standard lower - bearing MLG",
        "MLG - Переход от нижнего подшипника MLG с низким коэффициентом трения к стандартному нижнему подшипнику MLG.",
    ),
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
    ("Subject Reference", "Тема/ссылка"),
    ("Remove and Destroy Pages", "Удалить заменяемые страницы"),
    ("Insert New/Revised", "Вставить новые/пересмотренные"),
    ("Reason for Change", "Причина изменения"),
    ("Added fig-item", "Добавлен элемент рисунка"),
    ("Updated fig-items", "Обновлены элементы рисунка"),
    ("fig-item", "элемент рисунка"),
    ("fig-items", "элементы рисунка"),
    ("Updated Messier-Dowty Limited to Safran Landing Systems", "Наименование Messier-Dowty Limited изменено на Safran Landing Systems"),
    ("Updated conversion value in figure", "Обновлено значение пересчета на рисунке"),
    ("MLG Leg", "Стойка MLG"),
    ("MLG Complete", "Стойка MLG в сборе"),
    ("MLG complete", "Стойка MLG в сборе"),
    ("Introduction of new", "Введение новых"),
    ("Introduction of a new", "Введение нового"),
    ("transfer block subassembly", "подсборка переходного блока"),
    ("lower bearing subassembly", "нижний узел подшипника"),
    ("Barkhausen Noise Inspection", "контроль шума Баркхаузена"),
    ("Main Landing Gear Sliding Tube Axles", "оси скользящей трубы основной стойки шасси"),
    ("torque link damper unit", "демпферный узел рычага крутящего момента"),
    ("changeover valve stem and housing", "шток и корпус переключающего клапана"),
    ("Airworthiness Limitations Section", "раздел ограничений летной годности"),
    ("Maintenance Planning Document", "документ планирования технического обслуживания"),
    ("tracking numbers", "номера отслеживания"),
    ("stub bolt subassembly", "подсборка болта-заглушки"),
    ("self lubricating liner", "самосмазывающийся вкладыш"),
    ("low friction inner liner", "внутренний вкладыш с низким коэффициентом трения"),
    ("Axle Harnesses", "жгуты оси"),
    ("Axle harnesses", "жгуты оси"),
    ("Leg Harness", "жгут стойки"),
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
    ("Part No.", "Номер детали"),
    ("Part", "Деталь"),
    ("Revision", "Ревизия"),
    ("New/Revised", "Новые/пересмотренные"),
    ("Uplock", "Аплок"),
    ("No.", "№"),
)


def build_domain_replacements(*, include_single_words: bool = False) -> tuple[GlossaryReplacement, ...]:
    """Static aviation-doc replacements to reduce EN leftovers in translated output."""
    term_pairs = list(DOMAIN_TERM_PAIRS)
    if not include_single_words:
        term_pairs = [item for item in term_pairs if re.search(r"\s", item[0])]
    term_pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return tuple((_compile_term_pattern(source), target) for source, target in term_pairs)


def build_hard_glossary_replacements(glossary_text: str | None) -> tuple[GlossaryReplacement, ...]:
    """Build replacements intended for pre-translation shielding (hard glossary).

    For quality/safety, this includes:
    - user glossary pairs (exact EN -> RU)
    - selected domain phrases (multi-word only), to fix common headings in free providers

    User glossary entries override domain defaults for the same source term.
    """
    domain_pairs = [item for item in DOMAIN_TERM_PAIRS if re.search(r"\s", item[0])]
    glossary_pairs = [item for item in parse_glossary_pairs(glossary_text) if re.search(r"\s", item[0])]

    # Deduplicate source terms case-insensitively; user glossary overrides domain defaults.
    merged: dict[str, tuple[str, str]] = {}
    for src, tgt in domain_pairs:
        key = src.strip().lower()
        if key:
            merged.setdefault(key, (src, tgt))
    for src, tgt in glossary_pairs:
        key = src.strip().lower()
        if key:
            merged[key] = (src, tgt)

    pairs = list(merged.values())
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return tuple((_compile_term_pattern(source), target) for source, target in pairs)


def build_glossary_replacements(glossary_text: str | None) -> tuple[GlossaryReplacement, ...]:
    replacements: list[GlossaryReplacement] = []
    for source_term, target_term, pattern in build_glossary_matchers(glossary_text):
        replacements.append((pattern, target_term))
    return tuple(replacements)


def apply_glossary_replacements(text: str, replacements: tuple[GlossaryReplacement, ...]) -> str:
    out = text
    for pattern, replacement in replacements:
        out = pattern.sub(replacement, out)

    # Canonical EN heading fallback (for partially untranslated outputs).
    out = re.sub(r"\bMain Landing Gear Leg\b", "Стойка основного шасси", out, flags=re.IGNORECASE)
    out = re.sub(r"\bRepair Procedure Conditions\b", "Условия выполнения процедуры ремонта", out, flags=re.IGNORECASE)
    out = re.sub(r"\bMLG\s+Leg\b", "Стойка MLG", out, flags=re.IGNORECASE)
    out = re.sub(r"\bList of Effective Pages\b", "Перечень действующих страниц", out, flags=re.IGNORECASE)
    out = re.sub(r"\bRevision Record\b", "Запись изменений", out, flags=re.IGNORECASE)
    out = re.sub(r"\bDate Incorporated Into Manual\b", "Дата включения в руководство", out, flags=re.IGNORECASE)
    out = re.sub(r"\bInsert New/Revised Pages\b", "Вставить новые/пересмотренные страницы", out, flags=re.IGNORECASE)
    out = re.sub(r"\bRemove and Destroy Pages\b", "Удалить и уничтожить страницы", out, flags=re.IGNORECASE)
    out = re.sub(r"\bReason for Change\b", "Причина изменения", out, flags=re.IGNORECASE)
    out = re.sub(r"\bSubject/?Reference\b", "Тема/ссылка", out, flags=re.IGNORECASE)
    out = re.sub(r"\bService Bulletin List\b", "Список сервисных бюллетеней", out, flags=re.IGNORECASE)
    out = re.sub(r"\bMain Fitting Subassembly\b", "Подсборка корпуса стойки", out, flags=re.IGNORECASE)
    out = re.sub(r"\bSubassembly\b", "подсборка", out, flags=re.IGNORECASE)
    out = re.sub(r"\bIPL\s+FIGURE\b", "IPL РИСУНОК", out, flags=re.IGNORECASE)
    out = re.sub(r"\bIPL\s+fig\b", "IPL рис.", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(?:fig\.?|figure)\b", "рис.", out, flags=re.IGNORECASE)
    out = re.sub(r"\bfig\s*\(", "рис. (", out, flags=re.IGNORECASE)

    # Mixed heading cleanup for partial machine translations.
    out = re.sub(r"\bSubject\s+Ссылка\b", "Тема/ссылка", out, flags=re.IGNORECASE)
    out = re.sub(r"\bТема\s+Ссылка\b", "Тема/ссылка", out, flags=re.IGNORECASE)
    out = re.sub(r"\bВставить\s+новый/исправленный\b", "Вставить новые/пересмотренные", out, flags=re.IGNORECASE)
    out = re.sub(r"\bВставить\s+новый/пересмотренный\b", "Вставить новые/пересмотренные", out, flags=re.IGNORECASE)
    out = re.sub(r"\bNEW/REVISED\s+СТРАНИЦЫ\b", "НОВЫЕ/ПЕРЕСМОТРЕННЫЕ СТРАНИЦЫ", out, flags=re.IGNORECASE)
    out = re.sub(r"\bREVISION\s+ЗАПИСЬ\b", "ЗАПИСЬ ИЗМЕНЕНИЙ", out, flags=re.IGNORECASE)
    out = re.sub(r"\bLIST\s+СЕРВИСНЫХ\s+БЮЛЛЕТЕНЕЙ\b", "СПИСОК СЕРВИСНЫХ БЮЛЛЕТЕНЕЙ", out, flags=re.IGNORECASE)
    out = re.sub(r"\bPART\s+№\b", "№ ДЕТАЛИ", out, flags=re.IGNORECASE)
    out = re.sub(r"\btable\s+(\d+)\b", r"таблица \1", out, flags=re.IGNORECASE)
    out = re.sub(r"\(Table\s+(\d+)\)", r"(таблица \1)", out, flags=re.IGNORECASE)
    out = re.sub(r"\bFig\.\s+Страница\b", "Рис. Страница", out, flags=re.IGNORECASE)
    out = re.sub(r"\bэлемент\s+риса\b", "элемент рисунка", out, flags=re.IGNORECASE)
    out = re.sub(r"\bMLG\s+Нога\b", "Стойка MLG", out, flags=re.IGNORECASE)
    out = re.sub(r"\bНога\s+MLG\b", "Стойка MLG", out, flags=re.IGNORECASE)
    out = re.sub(r"\bОпора\s+MLG\b", "Стойка MLG", out, flags=re.IGNORECASE)
    out = re.sub(r"\bВетка\s+MLG\b", "Стойка MLG", out, flags=re.IGNORECASE)
    out = re.sub(r"\bMLG\s+завершено\b", "Стойка MLG в сборе", out, flags=re.IGNORECASE)
    out = re.sub(
        r"\bОбновлена\s+компания\s+Messier-Dowty\s+Limited\s+для\s+Safran\s+Landing\s+Systems\b",
        "Наименование Messier-Dowty Limited изменено на Safran Landing Systems",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\bКомпания\s+Messier-Dowty\s+Limited\s+обновлена\s+до\s+Safran\s+Landing\s+System[s]?\b",
        "Наименование Messier-Dowty Limited изменено на Safran Landing Systems",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\bОбновленн[а-я]+\s+ценност[а-я]+\s+конверсии\b",
        "Обновлено значение пересчета",
        out,
        flags=re.IGNORECASE,
    )
    # Keep heading joiners consistent in OCR/PDF-converted manuals.
    out = re.sub(r"(?m)^\s*WITH\s*$", "С", out, flags=re.IGNORECASE)
    out = re.sub(r"(⟦PN_\d+⟧)\s+AND\s+(⟦PN_\d+⟧)", r"\1 И \2", out, flags=re.IGNORECASE)

    # TOC artifact cleanup: merged "Repair No. X-Y601" -> "Ремонт № X-Y 601"
    out = re.sub(r"Ремонт\s*№\s*(\d+-\d+)(\d{3,4})\b", r"Ремонт № \1 \2", out)
    out = re.sub(r"(Ремонт\s*№\s*\d+-\d+)\s+Ремонт\s+", r"\1 ", out, flags=re.IGNORECASE)
    out = re.sub(r"№\s*(\d+-\d+)(\d{3,4})\b", r"№ \1 \2", out)
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
    reasoning_effort: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    structured_output_mode: str = "auto"
    supports_repair: bool = True

    def translate(self, text: str, context: dict[str, Any]) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        task = str(context.get("task", "translate")).lower()
        structured_mode = _normalize_structured_output_mode(self.structured_output_mode)
        structured_for_text = task in {"translate", "repair"} and structured_mode != "off"
        if task == "repair":
            system_prompt = REPAIR_SYSTEM_PROMPT_TEMPLATE
            if structured_for_text:
                system_prompt += '\n\nReturn ONLY JSON object: {"repaired_text":"..."}'
        elif task == "batch_translate":
            system_prompt = BATCH_SYSTEM_PROMPT_TEMPLATE
        else:
            system_prompt = self.translation_system_prompt
            if structured_for_text:
                system_prompt += '\n\nReturn ONLY JSON object: {"translated_text":"..."}'

        base = (self.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")).rstrip("/")
        url = f"{base}/v1/chat/completions"

        model_reasoning_effort = self.reasoning_effort
        temperature: float | None = None
        if task == "repair":
            temperature = 0.0
        elif task == "batch_translate":
            temperature = 0.0 if _supports_temperature(self.model, model_reasoning_effort) else None
        elif _supports_temperature(self.model, model_reasoning_effort):
            temperature = self.temperature

        include_cache_key = bool(self.prompt_cache_key)
        include_cache_retention = bool(self.prompt_cache_retention)
        include_response_format = task == "batch_translate" or structured_for_text
        data: dict[str, Any] | None = None

        for _ in range(3):
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_user_prompt(text, context)},
                ],
            }
            if _is_gpt5_family(self.model):
                payload["max_completion_tokens"] = self.max_output_tokens
            else:
                payload["max_tokens"] = self.max_output_tokens
            if temperature is not None:
                payload["temperature"] = temperature
            if model_reasoning_effort:
                payload["reasoning_effort"] = model_reasoning_effort
            if include_response_format:
                payload["response_format"] = {"type": "json_object"}
            if include_cache_key and self.prompt_cache_key:
                payload["prompt_cache_key"] = self.prompt_cache_key
            if include_cache_retention and self.prompt_cache_retention:
                payload["prompt_cache_retention"] = self.prompt_cache_retention

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
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
                body_l = body.lower()
                param: str | None = None
                try:
                    err_data = json.loads(body) if body else {}
                    param_raw = ((err_data.get("error", {}) or {}).get("param"))
                    param = str(param_raw) if param_raw is not None else None
                except Exception:
                    param = None

                if include_cache_retention and (
                    param == "prompt_cache_retention" or "prompt_cache_retention" in body_l
                ):
                    include_cache_retention = False
                    continue
                if include_cache_key and (param == "prompt_cache_key" or "prompt_cache_key" in body_l):
                    include_cache_key = False
                    continue
                if (
                    include_response_format
                    and structured_mode == "auto"
                    and (param == "response_format" or "response_format" in body_l)
                ):
                    include_response_format = False
                    continue
                raise RuntimeError(f"OpenAI HTTPError {e.code}: {body}") from e
            except Exception as e:
                raise RuntimeError(f"OpenAI request failed: {e}") from e

        if data is None:
            raise RuntimeError("OpenAI request failed after retries")

        try:
            content = data["choices"][0]["message"]["content"]
            if task == "batch_translate":
                return content
            if task == "repair":
                if structured_for_text or include_response_format:
                    try:
                        return _extract_structured_text(content, field_name="repaired_text")
                    except Exception as e:
                        if structured_mode == "strict":
                            raise RuntimeError(f"Strict structured repair parse failed: {e}") from e
                return _extract_repair_output(content)
            if structured_for_text or include_response_format:
                try:
                    content = _extract_structured_text(content, field_name="translated_text")
                except Exception as e:
                    if structured_mode == "strict":
                        raise RuntimeError(f"Strict structured translate parse failed: {e}") from e
            content = apply_glossary_replacements(content, self.glossary_replacements)
            return content
        except (KeyError, IndexError, TypeError) as e:
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
    structured_output_mode: str = "auto"
    supports_repair: bool = True

    def translate(self, text: str, context: dict[str, Any]) -> str:
        task = str(context.get("task", "translate")).lower()
        structured_mode = _normalize_structured_output_mode(self.structured_output_mode)
        structured_for_text = task in {"translate", "repair"} and structured_mode != "off"
        if task == "repair":
            system_prompt = REPAIR_SYSTEM_PROMPT_TEMPLATE
            if structured_for_text:
                system_prompt += '\n\nReturn ONLY JSON object: {"repaired_text":"..."}'
        elif task == "batch_translate":
            system_prompt = BATCH_SYSTEM_PROMPT_TEMPLATE
        else:
            system_prompt = self.translation_system_prompt
            if structured_for_text:
                system_prompt += '\n\nReturn ONLY JSON object: {"translated_text":"..."}'
        url = f"{self.base_url.rstrip('/')}/api/chat"

        options: dict[str, Any] = {"num_predict": self.max_output_tokens}
        options["temperature"] = 0.0 if task in {"repair", "batch_translate"} else self.temperature

        include_json_format = task == "batch_translate" or structured_for_text
        data: dict[str, Any] | None = None
        for _ in range(2):
            payload = {
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_user_prompt(text, context)},
                ],
                "options": options,
            }
            if include_json_format:
                payload["format"] = "json"
            req = urllib.request.Request(
                url=url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
                body_l = body.lower()
                if include_json_format and structured_mode == "auto" and "format" in body_l:
                    include_json_format = False
                    continue
                raise RuntimeError(f"Ollama HTTPError {e.code}: {body}") from e
            except Exception as e:
                raise RuntimeError(f"Ollama request failed: {e}") from e

        if data is None:
            raise RuntimeError("Ollama request failed after retries")

        try:
            content = data["message"]["content"]
            if task == "batch_translate":
                return content
            if task == "repair":
                if structured_for_text or include_json_format:
                    try:
                        return _extract_structured_text(content, field_name="repaired_text")
                    except Exception as e:
                        if structured_mode == "strict":
                            raise RuntimeError(f"Strict structured repair parse failed: {e}") from e
                return _extract_repair_output(content)
            if structured_for_text or include_json_format:
                try:
                    content = _extract_structured_text(content, field_name="translated_text")
                except Exception as e:
                    if structured_mode == "strict":
                        raise RuntimeError(f"Strict structured translate parse failed: {e}") from e
            content = apply_glossary_replacements(content, self.glossary_replacements)
            return content
        except (KeyError, TypeError) as e:
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
    glossary_prompt_text: str | None = None,
    reasoning_effort: str | None = None,
    prompt_cache_key: str | None = None,
    prompt_cache_retention: str | None = None,
    structured_output_mode: str = "auto",
) -> LLMClient:
    provider_norm = provider.strip().lower()
    prompt_glossary_text = glossary_text if glossary_prompt_text is None else glossary_prompt_text
    translation_prompt = build_translation_system_prompt(
        SYSTEM_PROMPT_TEMPLATE,
        custom_system_prompt=custom_system_prompt,
        glossary_text=prompt_glossary_text,
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
            reasoning_effort=reasoning_effort,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            structured_output_mode=structured_output_mode,
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
            structured_output_mode=structured_output_mode,
        )
    raise ValueError(f"Unknown LLM provider: {provider}")
