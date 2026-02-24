# Единый Большой План: Качество Перевода + Устойчивость `docxru`

## Краткое резюме
Цель плана: одновременно повысить фактическое качество EN->RU перевода и снизить вероятность падений и нестабильности рантайма.

План объединяет:
- сильные идеи из `plan_claude.md` (prompt/checker/validator/consistency),
- инженерные меры надежности (Studio upload safety, отказ от `cgi`, CI-гейты),
- снижение технического долга (`ruff`/`mypy`, декомпозиция крупных модулей).

## Что берем из `plan_claude.md`
1. Усиление batch prompt до уровня основного переводческого prompt.
2. Усиление checker prompt доменными правилами авиационной документации.
3. Few-shot примеры в переводчике и checker.
4. Новые валидации: не переведенные латинские фрагменты и повторы слов/фраз.
5. Увеличение полезного контекста предыдущих переводов.
6. Снижение стохастичности (temperature в дефолтах).

## Что добавляем сверх `plan_claude.md`
1. Замена deprecated `cgi` в Studio (поддержка Python 3.13+).
2. Лимиты и стриминг multipart-загрузок (защита от OOM/DoS).
3. Оптимизация чтения логов в Studio status polling.
4. Реально работающие quality gates: `ruff`, `mypy`, `pytest` в CI.
5. Снижение архитектурного риска из-за крупных модулей (`pipeline.py`, `studio_server.py`, `llm.py`, `checker.py`).
6. Интеграция ключевых правил из `general_prompt.md` в дефолтный `SYSTEM_PROMPT_TEMPLATE` (без зависимости от ручного `custom_system_prompt_file`).
7. Усиление batch context injection: добавить в batch элементы `tm_hints` и `recent_translations`, чтобы batch режим не терял контекст single режима.
8. Вынести `DOMAIN_TERM_PAIRS` и пост-обработочные regex-правила из hardcode в конфигурируемый `YAML`.
9. Добавить отдельную QA-проверку `context_leakage` для артефактов вида `PART/SECTION/TABLE_CELL` в выходном тексте.
10. Ввести обязательную оценку token-cost и риска truncation до rollout обогащенных prompt'ов.

## Scope / Out of scope
### In scope
- DOCX/PDF перевод, batch/single режимы, checker, Studio/Dashboard, QA-артефакты.

### Out of scope (текущая волна)
- Новый frontend/UI-фреймворк.
- Полный редизайн Word COM режима.
- Миграция на иной storage/queue.

Примечание: эти три направления переносятся в отдельный этап после стабилизации quality/reliability метрик, чтобы не вносить высокорисковые регрессии в перевод.

## Изменения интерфейсов и конфигов (decision-complete)
### `src/docxru/config.py`
1. Изменить default `LLMConfig.temperature` с `0.1` на `0.0`.
2. Изменить default `LLMConfig.context_window_chars` с `600` на `900`.
3. Добавить в `PipelineConfig`:
- `untranslated_latin_warn_ratio: float = 0.15`
- `untranslated_latin_min_len: int = 3`
- `untranslated_latin_allowlist_path: str | None = None`
- `repeated_words_check: bool = True`
- `repeated_phrase_ngram_max: int = 3`
- `context_leakage_check: bool = True`
- `context_leakage_allowlist_path: str | None = None`
4. Добавить в `LLMConfig`:
- `prompt_examples_mode: str = "core"` (`off|core`).
- `batch_tm_hints_per_item: int = 1`
- `batch_recent_translations_per_item: int = 3`
- `domain_term_pairs_path: str = "config/domain_term_pairs.yaml"`

### `src/docxru/cli.py`
Для `studio` команды добавить:
- `--max-request-mb` (default `130`)
- `--max-upload-mb` (default `128`)
- `--log-tail-kb` (default `256`)

### Новые QA issue codes
- `untranslated_fragments`
- `repeated_words`
- `context_leakage`
- `studio_request_too_large`
- `studio_multipart_invalid`

### Документация
- Обновить `README.md` и `config/config.example.yaml` под новые поля и дефолты.
- Добавить и описать `config/domain_term_pairs.yaml` и allowlist для `untranslated/context_leakage` проверок.

## План реализации по фазам

## Фаза 0 (1-2 дня): Бейзлайн и метрики
1. Зафиксировать baseline:
- `pytest -q`, `ruff check .`, `mypy src`.
- QA-метрики по `qa.jsonl` на `samples/test_1.docx`, `samples/test_2.docx`, `samples/32-12-22...abby_short.docx` с `--max-segments 120`.
2. Обновить `docs/benchmark_baseline.md`:
- таблица по кодам `placeholders_mismatch`, `style_tags_mismatch`, `batch_fallback_single`, `writeback_skipped_due_to_errors`, `untranslated_fragments`, `context_leakage`.
3. Зафиксировать token baseline:
- измерить средние input/output tokens для single/batch/checker (на 100 сегментов),
- зафиксировать текущую оценку стоимости и долю truncation (`finish_reason`).
4. Критерий выхода:
- baseline зафиксирован и воспроизводим,
- утвержден допустимый budget роста token-cost для Фазы 1.

## Фаза 1 (4-6 дней): Качество перевода (ядро)
1. `src/docxru/llm.py`:
- интегрировать ключевые части `general_prompt.md` в дефолтный `SYSTEM_PROMPT_TEMPLATE` (роль эксперта, принципы, правила по заголовкам/предупреждениям/таблицам),
- переписать `BATCH_SYSTEM_PROMPT_TEMPLATE`, чтобы batch всегда использовал тот же core-rules builder, что и single (без silent деградации при отсутствии custom prompt),
- включить few-shot при `prompt_examples_mode=core` (минимум 3 примера: procedural imperative, табличный заголовок, сегмент с `⟦...⟧`; целевой бюджет 200-300 токенов),
- вынести `DOMAIN_TERM_PAIRS` и regex-правила из `llm.py` в загрузку из `YAML` (generic post-processing без OEM hardcode),
- пересмотреть `max_output_tokens` для batch после обогащения prompt'ов и зафиксировать анти-truncation пороги.
2. `src/docxru/checker.py`:
- переписать `CHECKER_SYSTEM_PROMPT` с доменными критериями (CMM/AMM/IPC, терминология, стиль, дефект/не-дефект),
- добавить явные правила использования `glossary_terms_used` и границы "не быть креативным" (не предлагать стилистические правки без дефекта),
- сохранить strict JSON контракт checker и существующий confidence gating.
3. `src/docxru/validator.py`:
- добавить `validate_untranslated_fragments(...)`,
- добавить `validate_repeated_words(...)`,
- добавить `validate_context_leakage(...)`,
- подключить все три в `validate_all(...)` как `WARN`,
- расширить allowlist для ATA/PN/брендов и стандартных аббревиатур, чтобы снизить false positives.
4. `src/docxru/pipeline.py` и `src/docxru/pdf_pipeline.py`:
- увеличить окно recent context до нового дефолта (`900`),
- увеличить очередь recent переводов до `maxlen=6`,
- усилить `_build_batch_item_context(...)`: добавить `tm_hints` (top-1) и последние 2-3 перевода (`recent`) в batch item payload,
- добавить легкий `term_memory` (`EN_term -> RU_first_used`) для proactive consistency на уровне документа,
- прокинуть новые настройки validator из `PipelineConfig`.
5. Критерий выхода:
- нет роста `placeholders_mismatch`/`style_tags_mismatch` относительно baseline,
- появляются корректные сигналы `untranslated_fragments`/`repeated_words`/`context_leakage` на синтетических тестах,
- batch не уступает single на контрольной выборке по видимым языковым дефектам,
- рост token-cost и latency находится в утвержденном budget, доля truncation не растет.

## Фаза 2 (3-5 дней): Устойчивость Studio и снижение падений
1. `src/docxru/studio_server.py`:
- убрать `cgi.FieldStorage`,
- внедрить multipart parsing через `python-multipart` с явными лимитами размера,
- перейти на стриминговую запись upload-файлов на диск.
2. `src/docxru/studio_server.py` и `src/docxru/cli.py`:
- поддержать `--max-request-mb`, `--max-upload-mb`, `--log-tail-kb`,
- возвращать `413` при превышении лимитов и структурированный JSON error.
3. `src/docxru/studio_server.py`:
- переписать `_tail_lines` на чтение с конца файла по байтам (без полного чтения в память).
4. Тесты:
- проверить и обновить моки/тесты, если они зависят от `cgi`-поведения.
5. Критерий выхода:
- сервер не падает на malformed multipart и oversized payload,
- upload/log не читаются целиком в память в горячем пути.

## Фаза 3 (4-6 дней): Quality gates и технический долг
1. `pyproject.toml`:
- мигрировать Ruff-настройки на актуальный формат (`tool.ruff.lint`).
2. Кодовая база:
- довести `ruff check .` до `0` ошибок,
- довести `mypy src` до `0` ошибок, включая корректные исключения для unstubbed зависимостей.
3. CI:
- добавить `.github/workflows/ci.yml`,
- запускать Python 3.11/3.12 и проверки `ruff check .`, `mypy src`, `pytest -q`.
4. Критерий выхода:
- зеленый CI на PR,
- локальные проверки совпадают с CI.

## Фаза 4 (5-8 дней, после стабилизации): Архитектурная декомпозиция
1. Разделить `pipeline.py`:
- `pipeline_translate.py`,
- `pipeline_checker.py`,
- `pipeline_writeback.py`,
- `pipeline_status.py`.
2. Разделить `studio_server.py`:
- `studio_http.py`,
- `studio_manager.py`,
- `studio_uploads.py`.
3. Разделить `llm.py`:
- `llm_prompts.py`,
- `llm_clients_openai.py`,
- `llm_clients_ollama.py`,
- `llm_contracts.py`.
4. Совместимость:
- сохранить backward-compatible import paths через re-export модули и `DeprecationWarning`.
5. Критерий выхода:
- поведение эквивалентно (contract tests + integration tests зеленые),
- целевой размер файлов: soft-limit 800-1200 строк для orchestration-модулей и до 800 для leaf-модулей.

## Риски и митигации (добавлено по ревью)
1. Рост token-cost из-за enriched prompt'ов.
- Митигировать через baseline в Фазе 0, лимит роста бюджета и compact few-shot (200-300 токенов).
2. Рост latency и риск batch truncation.
- Митигировать пересмотром `max_output_tokens`, мониторингом `finish_reason` и canary rollout batch режима.
3. Ложноположительные срабатывания `untranslated_fragments`.
- Митигировать расширяемым allowlist (ATA/PN/OEM/бренды), WARN-only rollout и калибровкой порогов после 1 итерации.
4. Checker может стать "too creative" и предлагать спорные правки.
- Митигировать жесткими правилами дефект/не-дефект, few-shot "не-дефект" примерами и confidence gating.
5. Риск регрессии импортов при декомпозиции.
- Митигировать re-export слоями и проверкой обратной совместимости в integration тестах.

## Тесты и сценарии (обязательные)
### Unit
1. `tests/test_validator.py`:
- кейсы untranslated/repeated/context leakage и allowlist.
2. `tests/test_llm_clients.py`:
- prompt contract для main/batch/checker и JSON-режимов.
3. `tests/test_checker.py`:
- критерии дефектов/не-дефектов в checker и фильтрация низкоуверенных стилистических правок.
4. `tests/test_studio_server.py`:
- multipart parsing, 413 лимиты, malformed формы, tail-reading.

### Integration
1. `tests/test_pipeline_*.py` и `tests/test_pdf_pipeline.py`:
- отсутствие регрессий по маркерам и fallback.
2. Смоук-прогон:
- `docxru translate` и `docxru translate-pdf` на samples.

### Non-functional
1. Стресс upload (файл > лимита) не валит сервер.
2. Большой лог не увеличивает latency status API существенно.
3. Рост token-cost после Фазы 1 находится в утвержденном budget.
4. Доля batch truncation (`finish_reason=length`) не выше baseline.

### Acceptance
1. `pytest -q` полностью green.
2. `ruff check .` и `mypy src` green.
3. По baseline:
- нет ухудшения структурных QA-кодов,
- есть снижение видимых англоязычных "утечек" в ручной выборке,
- нет роста `context_leakage` и batch truncation относительно baseline.

## Rollout / rollback
1. Внедрять фазами: P0 -> P1 -> P2 -> P3 -> P4.
2. Для P1:
- новые validator checks сначала `WARN-only`,
- после одной итерации фиксируем рабочие пороги.
3. Rollback:
- quality-checks отключаются конфигом,
- prompt examples отключаются `prompt_examples_mode=off`,
- Studio лимиты регулируются CLI без правок кода.

## Явные предположения и выбранные дефолты
1. Приоритет: сначала надежность и предсказуемое качество, затем архитектурная чистка.
2. Для техстиля RU выбирается `temperature=0.0` как дефолт.
3. Новые validator-проверки не блокируют writeback, а сигналят через `WARN`.
4. Поддержка Python 3.13+ обязательна для Studio, поэтому `cgi` заменяется в этой волне.
5. COM-mode и UI-редизайн не входят в текущий обязательный контур.
6. Ключевые доменные правила из `general_prompt.md` становятся частью дефолтного prompt-пути.
7. Уровень токенов и стоимость фиксируются как управляемый quality/risk KPI, а не побочный эффект.
