Да, начать **с LLM-чекера + минимального “дашборда”** — самый правильный MVP. Ты получишь быстрый профит (контроль качества и экономики) без того, чтобы сразу строить тяжёлый page-review UI.

## Что бы я сделал в MVP (минимум кода, максимум пользы)

### 1) Форматы “системный промт” и “глоссарий”

* **System prompt**: обычный текстовый файл (`.md` или `.txt`). В твоём проекте это уже поддержано через `llm.system_prompt_path`  и в README явно сказано, что он применяется для `openai`/`ollama` .
  Я бы писал его **на русском**, т.к. у тебя правила/термины RU-ориентированные и так меньше “шансов перепонять”.
* **Glossary**:

  * Для человека — **Markdown** (таблица или пары), потому что у тебя это уже основной путь через `llm.glossary_path`  .
  * Для машины/автоматизации — можно держать “источник истины” в JSON, а MD генерировать из него (чтобы меньше ломалось при ручных правках). Это не обязательно в MVP.

Отдельно: “technical aviation EN→RU” лучше фиксировать **в системном промте**, а в конфиге оставить просто `source_lang/target_lang` (они уже есть) .

---

### 2) LLM-чекер (второй проход)

Идея: после перевода прогоняем второй моделью **audit** по сегментам и добавляем issues в общий QA отчёт. У тебя уже есть инфраструктура Issue/Severity  и генерация `qa.jsonl` + `qa_report.html`  — чекер просто должен дописать ещё issues.

**Критично для цены**: чекер НЕ должен проверять всё подряд. В MVP пусть проверяет только:

* сегменты с уже найденными warn/error (детерминированные правила),
* либо “рискованные” по эвристикам (высокий length ratio, table/textbox overflow risk — у тебя это уже есть в проекте по README) .

**Какую модель на чекер**: бери более дешёвую/быструю, чем переводная (например “mini” класс) — чекеру не нужно генерировать много текста, он должен отдавать компактный JSON.

---

### 3) Токены и стоимость

Тебе нужно 2 вещи:

1. **Учёт usage** на каждом LLM-запросе (prompt/completion/total).
2. **Калькулятор стоимости** по таблице цен (config/YAML/JSON).

Почему так: цены меняются, и без web-доступа их нельзя “зашить” навечно. Поэтому делай **pricing таблицу редактируемой**.

Формула:
[
Cost = \frac{T_{in}}{10^6} \cdot P_{in} + \frac{T_{out}}{10^6} \cdot P_{out}
]
(или per-1K — не важно, главное единообразно).

⚠️ Не все провайдеры возвращают usage одинаково. Поэтому:

* если usage нет — показывай “N/A” и **оценку** (fallback) только если очень надо.

---

### 4) “Супер простой визуал” без тяжёлого фронта

У тебя уже есть HTML QA отчёт — это уже “визуал” для качества . Для прогресса/стоимости добавь ещё один:

* `run_status.json` (обновляется в процессе),
* `dashboard.html` (статический, с JS polling `run_status.json`).

Чтобы polling работал в браузере без CORS/file:// ада — добавь команду:

* `docxru dashboard --dir <run_dir>`
  которая поднимает **локальный** server на стандартной библиотеке (`http.server`) и открывает браузер.

Кнопка “Open folder”:

* в MVP можно просто показывать путь + кнопку “Copy path”.
* если хочешь прям открывать Explorer/Finder — добавь endpoint `/api/open-output` (в том же лёгком сервере), который вызывает `explorer.exe` / `open` / `xdg-open`.

Важно: сервер слушает **только 127.0.0.1**, иначе это дырка.

---

## Серия промтов (EN) для агента кодинга

Ниже — готовые промты “как ТЗ” (копируй по одному).

### Prompt 1 — Repo survey + design

**Goal:** implement an MVP “LLM checker + minimal dashboard with tokens/cost/progress and open output folder”

**Instructions:**

1. Inspect the current CLI (`src/docxru/cli.py`) and pipelines (`src/docxru/pipeline.py`, `src/docxru/pdf_pipeline.py`), TM (`src/docxru/tm.py`), QA (`src/docxru/qa_report.py`), and LLM adapter (`src/docxru/llm.py`).
2. Propose a minimal design that:

   * Adds a second-pass “checker LLM” that appends `Issue`s to segments and is included in existing `qa.jsonl` and `qa_report.html`.
   * Tracks LLM usage tokens and computes cost using a configurable pricing table.
   * Writes a `run_status.json` periodically and provides a minimal `dashboard.html` that shows progress, tokens, cost, ETA, and output paths.
   * Adds a new CLI subcommand `docxru dashboard --dir <run_dir>` that serves the directory on localhost and optionally offers an endpoint to open the output folder.
3. Output: a short design doc + file-by-file change plan + acceptance criteria.

Constraints:

* Do not add heavy dependencies (prefer Python stdlib).
* Preserve backward compatibility of existing CLI and config.
* Keep all network-facing features local-only (127.0.0.1).

---

### Prompt 2 — Config extensions (checker + pricing + run dir)   

Implement config extensions in `src/docxru/config.py`:

1. Add `CheckerConfig` dataclass with fields: `enabled`, `provider`, `model`, `temperature`, `max_output_tokens`, `timeout_s`, `retries`, `system_prompt_path`, `glossary_path`, `max_segments`, `only_on_issue_severities` (e.g., ["warn","error"]), and `only_on_issue_codes` (optional).
2. Add `PricingConfig` dataclass with fields: `enabled`, `pricing_path` (YAML/JSON), `currency` (default "USD").  
3. Add `RunConfig` dataclass with fields: `run_dir` (base), `run_id` (auto), `status_path` (default `<run_dir>/run_status.json`), `dashboard_html_path`.
4. Update `load_config()` to read these sections from YAML, resolve relative paths like existing `_resolve_optional_path()` does .
5. Update `config/config.example.yaml` accordingly and update README notes about quoting enum strings.

Acceptance criteria:

* Old configs still load.
* New configs load and paths resolve correctly on Windows/Linux.

---

### Prompt 3 — Usage tracking + cost calculator

Implement token usage tracking:

1. Create `src/docxru/usage.py` with:

   * `UsageRecord` (provider, model, phase, input_tokens, output_tokens, total_tokens, cost, ts, extra dict).
   * `UsageTotals` accumulator.
2. Create `src/docxru/pricing.py`:

   * Load pricing table from YAML/JSON: mapping `{provider: {model: {input_per_million, output_per_million}}}`.
   * `estimate_cost(provider, model, in_tokens, out_tokens)`.
3. Modify `src/docxru/llm.py` clients (at least OpenAI) to:

   * Parse `usage` from API responses.
   * Call optional callback `on_usage(record)` or update a `UsageTotals` object.
   * Keep the public `translate()` signature unchanged (still returns text).
4. Modify `pipeline.py` and `pdf_pipeline.py` to:

   * Instantiate pricing + totals and pass callback into the LLM client.
   * Periodically write `run_status.json` including: done/total segments, counts by status, tokens, cost, ETA.

Tests:

* Add unit tests mocking OpenAI HTTP responses containing a `usage` object to ensure tokens are recorded correctly (pattern similar to existing tests).

---

### Prompt 4 — LLM checker (second pass)

Implement a “checker” pass:

1. Create `src/docxru/checker.py`:

   * Build a strict JSON-only prompt: given SOURCE, TARGET, glossary snippet, and rules, return `{"issues":[...]}`
   * Parse JSON robustly (handle fenced code blocks).
   * Convert results into `Issue` objects and attach to `Segment.issues` .
2. Integrate into `pipeline.py` and `pdf_pipeline.py`:

   * After translation and deterministic validations, run checker only for segments matching the configured filters (severity/code) and up to `max_segments`.
   * Add issue codes prefixed `llm_check_*` and include details with suggestions.
3. Ensure checker never modifies document text; it only adds QA issues.

Acceptance criteria:

* When enabled, `qa_report.html` includes checker findings automatically.
* When disabled, behavior unchanged.

---

### Prompt 5 — Minimal dashboard server (stdlib)

Implement `docxru dashboard`:

1. Extend `src/docxru/cli.py` to add a new subcommand:

   * `docxru dashboard --dir <run_dir> [--port 0] [--open-browser]`
2. Implement `src/docxru/dashboard_server.py` using `http.server`:

   * Serve static files in `<run_dir>` (including `dashboard.html`, `run_status.json`, `qa_report.html`).
   * Add `/api/status` that returns `run_status.json`.
   * Add `/api/open-output` that opens the output folder (Windows: explorer.exe, macOS: open, Linux: xdg-open).
   * Bind to 127.0.0.1 only.
3. Add a minimal `dashboard.html` (no build tools) that:

   * Polls `/api/status` every 1s.
   * Shows: phase, progress %, done/total, ETA, tokens, cost, links to QA report, and a button “Open output folder”.

Acceptance criteria:

* Works without extra pip dependencies.
* Does not expose server to LAN.

---

### Prompt 6 — Docs + examples

Update documentation:

1. README: add a “MVP dashboard + checker” section:

   * How to enable checker in YAML.
   * How to run `docxru dashboard`.
   * Explain that progress is segment-based (approximate pages).
2. Add example pricing file `config/pricing.example.yaml`.
3. Ensure all new outputs are written into a run directory and paths are printed at the end of `docxru translate`.

---

## Риски/неприятные моменты (чтобы не удивляться)

* Прогресс будет **по сегментам**, не по страницам (это нормально для MVP; страницы можно прикрутить позже через рендер/compare пайплайн ).
* Токены/стоимость будут точными только там, где провайдер реально возвращает usage.
* “Open folder” нужно строго локально, иначе можно случайно получить RCE-подобную дыру.

---

## Next steps

* [ ] Завести `pricing.example.yaml` и включить учёт usage/cost.
* [ ] Сделать checker только на warn/error сегменты.
* [ ] Добавить `run_status.json` + `docxru dashboard` (stdlib server).
* [ ] Уже после MVP: page-level review (render + side-by-side), интерактивные правки и “resume from page N”.

ИМХО: MVP “чекер + прогресс/стоимость/папка” даст 80% ценности и резко снизит риск слива времени на большой UI раньше, чем качество/экономика будут стабилизированы.
