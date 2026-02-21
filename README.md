# docxru — Автоперевод CMM/AMM/IPC DOCX → RU с сохранением форматирования

Проект: **параграф/ячейка‑уровневый** перевод авиационной техдокументации (DOCX, часто после PDF→DOCX) на русский
с **инлайн‑защитой** PN/ATA/размеров/ссылок, **строгой валидацией** маркеров и **QA‑отчётом**.

> Важно: `python-docx` **не пересчитывает поля Word** (TOC/PAGEREF). В cross‑platform режиме документ
> может потребовать ручного “Update Fields” в Word.

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
- `llm.glossary_path` is applied for all providers; for `google` it is hard-enforced via placeholder shielding (so glossary terms cannot drift).
- `llm.glossary_in_prompt: false` disables sending full glossary in every request (saves tokens); glossary can still be enforced via hard shielding.
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
