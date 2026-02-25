# План: Повышение качества перевода `docxru`

## Краткое резюме
Цель: повысить терминологическую точность и устойчивость перевода без регрессий по структуре DOCX.

Приоритет:
1. Исправления с высоким ROI и низким риском.
2. Улучшение retrieval/validator слоя.
3. Точечное усиление checker и consistency.
4. Опциональный R&D (QE/local recheck) только после метрик.

## Изменения интерфейсов (backward-compatible)
Добавить поля:
- `PipelineConfig.short_translation_min_ratio: float = 0.35`
- `PipelineConfig.short_translation_min_source_chars: int = 24`
- `TMConfig.fuzzy_token_regex: str = "[A-Za-zА-Яа-яЁё0-9]{2,}"`
- `TMConfig.fuzzy_rank_mode: str = "hybrid"` (`sequence|hybrid`)
- `RunConfig.batch_timeout_bisect: bool = true`

## Волна 1: критичные исправления
1. Cleanup rule для `Table`.
- Заменить глобальный `\bTable\b` на контекстный вариант для ссылок вида `Table 1`.
- Добавить защиту от `Table of Contents`.

2. Batch timeout recovery.
- При timeout в grouped batch внедрить бисекцию пачки (`N -> N/2 -> ... -> 1`) вместо повтора той же пачки.

3. Глоссарий.
- Исправить `Deoxydine`.
- Развести "таблицу стандартов" и enforce-глоссарий (чтобы код стандарта не заменялся описанием).
- Зафиксировать DPI/FPI формулировки в целевом стиле проекта.

## Волна 2: качество генерации + retrieval
1. Новый validator `short_translation`.
- Выдавать `WARN`, если перевод подозрительно короткий при достаточно длинном source.

2. Token Shield паттерны.
- Добавить защиту для `Revision/Rev`, `SB`, `AD`, `AMM`, `Figure 5A` и др.
- Сделать PN alnum-pattern case-insensitive.
- Снизить false positives для `A`/`m` через порядок/контекст правил.

3. Fuzzy TM.
- Перейти на unicode-токенизацию.
- Оставить FTS-кандидатов, но ранжировать `hybrid`:
  - token Jaccard + SequenceMatcher.
- Сохранить API `get_fuzzy`; меняется только внутренняя метрика ранжирования.

## Волна 3: checker + consistency
1. Checker prompt hardening.
- Явно прописать, когда считать дефектом (`meaning`, `omission`, `number_error`, `terminology`) и когда не трогать сегмент.
- Явно использовать `glossary_terms_used` как обязательный ориентир.

2. Issue taxonomy (additive).
- Добавить: `omission`, `addition`, `number_error`, `untranslated`, `register`.
- Сохранить совместимость существующих `llm_check_*`.

3. Consistency frequency weighting.
- Для вариативных переводов термина помечать миноритарный вариант как отклонение с указанием `majority/minority`.

## Волна 4 (опционально, после метрик)
1. Встроить COMETKiwi сначала только в `eval`-контур (без боевого gating).
2. Если метрики подтверждают пользу, включить как условный триггер checker/recheck.
3. Local LLM recheck — только как opt-in профиль, без изменения дефолтного пайплайна.

## Тесты и сценарии
### Unit
- cleanup `Table` positive/negative cases.
- batch bisect on timeout (`N=8,4,2,1`).
- `short_translation` threshold checks.
- новые Token Shield паттерны + false-positive регрессии.
- fuzzy TM ranking на смешанной кириллице/латинице.
- consistency majority/minority detection.

### Integration
- `docxru translate` на baseline-срезе 120 сегментов (OpenAI/Ollama профили).
- checker pass + safe apply.
- grouped batch mode с искусственными timeout.

### Acceptance KPI
- `placeholders_mismatch == 0`
- `style_tags_mismatch == 0`
- `batch_fallback_single` ratio не хуже baseline более чем на +2 п.п.
- снижение `untranslated_fragments` минимум на 20%
- снижение `glossary_lemma_mismatch` минимум на 15%
- без роста `writeback_skipped_due_to_errors`

## Предположения
- GPU не обязателен; базовый путь должен работать на текущем профиле.
- Все новые проверки сначала `WARN-only`.
- Никаких breaking changes в CLI/форматах артефактов.
- При конфликте "качество vs latency" приоритет у качества; рост latency ограничить до ~15% на baseline-срезе.
