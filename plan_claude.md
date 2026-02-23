Контекст проблемы
Перевод 70 страниц (395 сегментов) занимает >1 часа. Три корневые причины:
Причина 1: max_output_tokens не масштабируется при auto_model_sizing=False
Режим grouped_aggressive устанавливает auto_model_sizing=False, из-за чего max_output_tokens остаётся на 2,400 (конфиг по умолчанию). Для батча из 20 сегментов нужно ~8,600 output-токенов. Модель не может уместить ответ → "Empty batch response" → fallback на 20 одиночных вызовов.
Это не модель не справляется — ей не дали достаточно output-токенов.
Код проблемы (pipeline.py:1565-1567):
pythoneffective_llm_max_output_tokens = cfg.llm.max_output_tokens  # 2400
if cfg.llm.auto_model_sizing:  # False для aggressive → пропускается!
    effective_llm_max_output_tokens = sizing.max_output_tokens  # было бы ~9000
Причина 2: Контекстное окно используется на 1.4%
Этап ограниченияЗначениеПотеряКонтекст GPT-5-mini400,000 токенов—_TRANSLATE_INPUT_UTILIZATION = 0.0832,000 токенов92% потеряноТир "balanced" batch_chars_cap=18,000~5,625 токеновещё 82% обрезаноИтого используется~5,625 из 400,00098.6% окна пустует
Причина 3: Prepare фаза — 7 минут (81 TOC-сегмент последовательно)
81 TOC-сегмент переводится поштучно в цикле Prepare (pipeline.py:1795-1808). Каждый — отдельный API-вызов, без параллелизма. 81 × ~5 сек = ~7 минут ещё до начала основного перевода.

План изменений
1. Исправить max_output_tokens для aggressive-режима (критический баг)
Файл: src/docxru/pipeline.py
Добавить safety net перед созданием LLM-клиента (~строка 1614). Если auto_model_sizing=False и batch_segments > 1, вычислить минимальный max_output_tokens по формуле из содержимого:
pythonif not cfg.llm.auto_model_sizing and effective_batch_segments > 1:
    median_chars = _median_source_chars([len(s.source_plain or "") for s in segments])
    estimated_batch_output_chars = effective_batch_max_chars * 1.35  # RU expansion
    estimated_output_tokens = int(estimated_batch_output_chars / 2.2) + 420
    if estimated_output_tokens > effective_llm_max_output_tokens:
        effective_llm_max_output_tokens = estimated_output_tokens
        logger.info(
            "Auto-raised max_output_tokens to %d (batch content requires more than configured %d)",
            effective_llm_max_output_tokens, cfg.llm.max_output_tokens,
        )
Использовать вспомогательную функцию _median_source_chars из model_sizing.py (импортировать).
Также: в grouped_aggressive профиле в studio_server.py (строка 181) переключить auto_model_sizing на True:
python"auto_model_sizing": True,  # Было False, что приводило к batch failures
2. Добавить тир "turbo" в model_sizing.py
Файл: src/docxru/model_sizing.py
Добавить в _TIER_LIMITS:
python"turbo": TierLimits(
    batch_chars_cap=120_000,
    batch_segments_cap=80,
    translate_output_cap=64_000,
    checker_segments_cap=250,
    checker_output_cap=8000,
    checker_pages_per_chunk=8,
),
Обоснование:

120K символов ≈ 37,500 input-токенов → 10% от 400K (безопасно для внимания модели)
64K output-токенов = 50% от 128K (достаточный запас)

3. Сделать _TRANSLATE_INPUT_UTILIZATION динамическим
Файл: src/docxru/model_sizing.py
Заменить константу 0.08 на функцию:
pythondef _translate_input_utilization(input_context_tokens: int) -> float:
    if input_context_tokens >= 200_000:
        return 0.15
    if input_context_tokens >= 100_000:
        return 0.10
    return 0.08
Обновить строку 212: _translate_input_utilization(profile.input_context_tokens)
4. Переназначить модели GPT-5 на тир "turbo"
Файл: src/docxru/model_sizing.py

gpt-5-mini (400K/128K): "balanced" → "turbo"
gpt-5 (400K/128K): "premium" → "turbo"
gpt-5.2 (400K/128K): "premium" → "turbo"
gpt-5-nano, gpt-4.1-mini, gpt-4o-mini — без изменений

5. Добавить режим "grouped_turbo" в студию
Файл: src/docxru/studio_server.py
В _translation_grouping_profile добавить:
pythonif key == "grouped_turbo":
    return (
        "grouped_turbo",
        {
            "batch_segments": 80,
            "batch_max_chars": 120_000,
            "context_window_chars": 0,
            "auto_model_sizing": True,
        },
        0.20,
    )
Добавить <option value="grouped_turbo">Grouped (turbo, large-context models)</option> в HTML-форму.
auto_model_sizing=True гарантирует безопасность: при gpt-4o-mini тир "economy" обрежет до 12K/6 сегментов.
6. Масштабировать таймауты и ETA
Файл: src/docxru/studio_server.py
В _estimate_request_latency_bounds_seconds — добавить batch_max_chars параметр:
pythonif grouped_mode and batch_max_chars > 36000:
    scale = min(4.0, batch_max_chars / 36000)
    low *= scale; high *= scale
Передать batch_max_chars из estimate_from_form (строка 1028).
При формировании конфига для запуска: если effective_batch_max_chars > 36000, установить timeout_s = 180.
7. Параллелизировать TOC-перевод в Prepare
Файл: src/docxru/pipeline.py
В цикле Prepare (строка 1742) TOC- и complex-сегменты сейчас переводятся последовательно. Изменить подход:

Во время Prepare-цикла собирать TOC/complex сегменты в отдельный список вместо немедленного перевода
После основного Prepare-цикла — перевести собранные TOC/complex сегменты параллельно через ThreadPoolExecutor(max_workers=cfg.concurrency)
Обновить прогресс-бар после параллельной обработки

Это ускорит Prepare с ~7 мин до ~2 мин (81 вызов / 4 воркера × ~5 сек).
8. Добавить тесты
Файл: tests/test_model_sizing.py

GPT-5-mini резолвится в тир "turbo"
Turbo-тир даёт batch_max_chars >= 60_000 и batch_segments >= 30
gpt-4o-mini остаётся "economy" с batch_max_chars <= 12_000
Динамический utilization: 400K → 0.15, 128K → 0.10
Single-segment mode (batch_segments=1) сохраняется при любом тире
Safety net: при auto_model_sizing=False и большом батче max_output_tokens автоподнимается


Ожидаемый результат
МетрикаСейчас (aggressive)После (turbo)Prepare фаза~7 мин (81 TOC sequential)~2 мин (81 TOC parallel)Batch failures8 из 15 (53%!)0 (output tokens достаточно)Символов на запрос36,000 (но ответ обрезается)120,000Сегментов на запрос20 (но падает)80API-вызовов (285 LLM сегм.)~15 батчей + ~160 fallback = ~1754-6 батчейВремя перевода (70 стр)>1 час~3-5 минУтилизация input~1.4%~10%Утилизация outputобрезается до 2400~50%
Порядок реализации

Шаг 1 — Safety net для max_output_tokens в pipeline.py + auto_model_sizing=True в aggressive (исправляет текущий баг)
Шаг 2 — Тир "turbo" + динамический utilization в model_sizing.py (увеличивает батчи)
Шаг 3 — Переназначить GPT-5 модели на turbo (активирует новые лимиты)
Шаг 4 — Режим "grouped_turbo" в студии (UI для нового режима)
Шаг 5 — Таймауты и ETA (корректные оценки)
Шаг 6 — Параллелизация TOC в Prepare (ускорение Prepare фазы)
Шаг 7 — Тесты

Ключевые файлы

src/docxru/model_sizing.py — тиры, utilization, профили моделей
src/docxru/pipeline.py — safety net для output tokens, параллелизация TOC
src/docxru/studio_server.py — grouped_turbo профиль, таймауты, ETA, aggressive fix
tests/test_model_sizing.py — тесты

Верификация

pytest tests/test_model_sizing.py — новые + старые тесты проходят
Запустить студию → aggressive mode → убедиться что batch failures исчезли
Запустить студию → turbo mode → перевести 70 стр → время < 5 мин
Сравнить качество turbo vs grouped_fast на одном документе
Запустить grouped_fast → убедиться что обратная совместимость сохранена