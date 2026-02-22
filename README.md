# docxru — Автоперевод CMM/AMM/IPC DOCX → RU с сохранением форматирования

Проект: **параграф/ячейка‑уровневый** перевод авиационной техдокументации (DOCX, часто после PDF→DOCX) на русский
с **инлайн‑защитой** PN/ATA/размеров/ссылок, **строгой валидацией** маркеров и **QA‑отчётом**.

> Важно: `python-docx` **не пересчитывает поля Word** (TOC/PAGEREF).
> Если `mode: com`, pipeline теперь автоматически запускает COM-проход Word:
> обновляет поля/TOC и делает автофит текстбоксов (с уменьшением шрифта до 1-2 шагов при переполнении).

---

## Установка (dev)

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e ".[dev]"
pre-commit install
```

---

## Быстрый старт

```bash
docxru translate --input source.docx --output target_ru.docx --config config/config.example.yaml
```

## PDF перевод (new)

Установка PDF-зависимостей:

```bash
pip install -e ".[pdf]"
```

Базовый запуск:

```bash
docxru translate-pdf --input source.pdf --output target_ru.pdf --config config/config.example.yaml
```

Дополнительно:

- `--bilingual` -> включает OCG-слой `Russian Translation` (EN+RU переключаемые слои).
- `--ocr-fallback` -> запускает `ocrmypdf` для сканированных страниц.
- `--max-pages N` -> ограничивает перевод первыми `N` страницами.
- `--resume` -> повторно использует TM/progress кэш.

Выходные файлы:
- `target_ru.docx` — переведённый документ
- `qa_report.html` — QA отчёт (ошибки/предупреждения)
- `translation_cache.sqlite` — кэш (translation memory)
- `run.log` — лог

---

## Конфиг

Смотри `config/config.example.yaml` и `config/regex_presets.yaml`.

---

## Архитектура

- `docx_reader.py` — извлечение сегментов (paragraph/table cell)
- `tagging.py` — runs → tagged string с маркерами `⟦S_...⟧...⟦/S_...⟧` и восстановление runs
- `token_shield.py` — инлайн‑плейсхолдеры `⟦PN_1⟧`, `⟦ATA_2⟧`, …
- `tm.py` — SQLite translation memory + progress
- `llm.py` — адаптер LLM (Mock по умолчанию)
- `validator.py` — строгая проверка плейсхолдеров/тегов/чисел
- `qa_report.py` — HTML отчёт

---

## Примечания по качеству перевода

По умолчанию включён `MockLLMClient` (без реального перевода) — чтобы пайплайн и форматирование работали из коробки.
Для боевого перевода подключи реальный LLM провайдер в `llm.py` или реализуй свой адаптер.

---

## Лицензия

MIT

---

## Provider options (new)

- `mock` (default): no real translation, safe dry runs.
- `openai`: real LLM translation via `OPENAI_API_KEY`.
- `google`: free public Google web endpoint (no API key, unofficial/rate-limited).
- `ollama`: local model via `http://localhost:11434` (or `llm.base_url`).

Example provider switch in config:

```yaml
llm:
  provider: google
  source_lang: en
  target_lang: ru
  glossary_path: glossary.md
  timeout_s: 60
```

```yaml
llm:
  provider: ollama
  model: qwen2.5:7b
  base_url: http://localhost:11434
  system_prompt_path: general_prompt.md
  glossary_path: glossary.md
  temperature: 0.1
```

Notes:

- `llm.system_prompt_path` is applied for `openai` and `ollama` providers.
- `llm.glossary_path` is applied for all providers.
- `llm.glossary_in_prompt: false` disables sending full glossary in every request (saves tokens).
- `llm.hard_glossary: true` enables strict placeholder-based term locking.
  Scope is adaptive in pipeline (TOC/table/short labels), but for natural prose keep it `false` unless strict locking is required.
- `llm.reasoning_effort` can tune OpenAI reasoning spend (`none|minimal|low|medium|high|xhigh`).
- `llm.prompt_cache_key` / `llm.prompt_cache_retention` can reduce cost for repeated prompt prefixes in OpenAI calls.
- Grouped translation mode is available for `openai` and `ollama`:
  - `llm.batch_segments` (default `1`) controls how many nearby segments are translated in one LLM request.
  - `llm.batch_max_chars` (default `12000`) is a soft per-request payload cap.
  - CLI overrides: `--batch-segments` and `--batch-max-chars`.
  - If grouped output fails marker validation, the pipeline automatically falls back to single-segment translation for safety.
- Optional translation memory history:
  - `translation_history_path` writes append-only JSONL with source/target/context for successful segments.
  - Use `python scripts/tm_lookup.py --term "your term"` to search prior decisions in `translation_cache.sqlite`.

Reliability and terminology controls:

- Structured output for prompt-based providers:
  - `llm.structured_output_mode: "off" | "auto" | "strict"`
  - CLI: `--structured-output off|auto|strict`
- Glossary prompt scope:
  - `llm.glossary_prompt_mode: "off" | "full" | "matched"`
  - `llm.glossary_match_limit: 24` (used in `matched` mode)
  - CLI: `--glossary-prompt-mode off|full|matched`
- Batch guardrails:
  - `llm.batch_skip_on_brline: true`
  - `llm.batch_max_style_tokens: 16`
  - `llm.context_window_chars: 600` by default (sequential mode with recent EN=>RU context); set `0` to allow grouped batch mode.
- Fuzzy TM hints in prompt context:
  - `tm.fuzzy_enabled`, `tm.fuzzy_top_k`, `tm.fuzzy_min_similarity`, `tm.fuzzy_prompt_max_chars`
  - CLI switch: `--fuzzy-tm`
- Optional ABBYY normalization:
  - `abbyy_profile: "off" | "safe" | "aggressive"`
  - CLI: `--abbyy-profile off|safe|aggressive`
- Optional glossary morphology check (`pymorphy3`, if installed):
  - `glossary_lemma_check: "off" | "warn" | "retry"`
  - In `retry` mode, pipeline performs one additional glossary-focused rewrite when matched terms are missing.
- Optional consistency and layout checks:
  - `layout_check: true|false`
  - `layout_expansion_warn_ratio: 1.5` (warn on high RU/EN expansion ratio)
  - `layout_auto_fix: true|false` (apply optional spacing/font reductions on risky segments)
  - `layout_font_reduction_pt`, `layout_spacing_factor`

New QA/diagnostic issue codes:

- `batch_json_schema_violation`: grouped batch response failed JSON/schema contract.
- `batch_validation_fallback`: grouped output failed marker validation, single-segment fallback used.
- `batch_fallback_single`: grouped request failed, translated segment-by-segment.
- `consistency_term_variation`: same glossary source term observed with multiple RU variants.
- `length_ratio_high`: translated segment expansion ratio exceeds configured threshold.
- `layout_table_overflow_risk`: translated table-cell text likely exceeds available width.
- `layout_textbox_overflow_risk`: translated textbox text likely exceeds available area.
- `layout_auto_fix_applied`: auto-fix (spacing/font reduction) was applied to a segment.

YAML note:

- Enum-like values such as `"off"` should be quoted in YAML. Unquoted `off` may be parsed as boolean `false`.

Safety behavior:

- If translation/marker validation has hard errors, write-back for that segment is skipped and source text is preserved.
- `--resume` now reuses recorded progress + TM entries when source hash matches.

OpenAI "agent translator" profile:

```bash
export OPENAI_API_KEY=...
docxru translate --input source.docx --output target_ru.docx --config config/config.agent_openai.yaml
```

---

## Быстрый прогон на первых страницах

Для отладки качества и верстки удобно переводить только первые сегменты (примерно первые страницы):

```bash
docxru translate --input source.docx --output target_ru.docx --config config/config.example.yaml --max-segments 120
```

---

## Визуальная проверка страниц (опционально)

Рендер: DOCX -> PDF -> PNG, чтобы глазами сравнить оригинал и перевод.

Установка зависимостей:

```bash
pip install -e ".[render]"
```

Рендер одного файла:

```bash
python scripts/render_docx_pages.py path/to/file.docx --output-dir tmp/docs/pages --backend word
```

Сравнение двух файлов (HTML index):

```bash
python scripts/compare_docx_pages.py --left path/to/original.docx --right path/to/translated.docx --output-dir tmp/docs/compare --backend word
```
