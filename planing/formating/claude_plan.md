
Plan: Максимальное качество форматирования DOCX на выходе

Контекст / Проблема
Пайплайн перевода EN→RU создает DOCX-файлы, требующие значительных ручных правок форматирования перед передачей заказчику. Причины:

Write-back уничтожает runs — tagged_to_runs() удаляет ВСЕ runs параграфа и пересоздает их через paragraph.add_run(). Теряются character styles, тонкие свойства rPr, нумерация runs.
Все фиксы отключены по умолчанию — layout_check: false, layout_auto_fix: false, abbyy_profile: "off", translate_enable_formatting_fixes: false. Код есть, но не работает.
ABBYY-ограничения — FineReader ставит noAutofit, hRule="exact" → текст обрезается при расширении RU.
Нет учёта ограничений контейнера при переводе — LLM не знает, что текст в узком textbox, и делает длинный перевод.
Порядок обработки субоптимален — в postformat_docx font shrink до layout check.
Фазы реализации
Фаза 1: Умные пресеты + правильный порядок обработки
Импакт: ВЫСОКИЙ | Риск: НИЗКИЙ | Усилия: 1-2 дня

Самый быстрый выигрыш — просто включить существующий код.

1A. Пресеты в конфиге
Файл: src/docxru/config.py

Добавить поле formatting_preset в PipelineConfig:

formatting_preset: str = "off"   # "off" | "native_docx" | "abbyy_standard" | "abbyy_aggressive" | "auto"
Таблица пресетов (значения-по-умолчанию, переопределяемые YAML):

Параметр	off	native_docx	abbyy_standard	abbyy_aggressive
translate_enable_formatting_fixes	false	true	true	true
abbyy_profile	off	off	aggressive	full
layout_check	false	true	true	true
layout_auto_fix	false	false	true	true
layout_auto_fix_passes	1	1	2	3
font_shrink_body_pt	0.0	0.0	0.0	0.5
font_shrink_table_pt	0.0	0.0	0.5	1.0
mode	reflow	reflow	reflow	com
В load_config(): применить пресет как базовые значения, потом перезаписать явные ключи из YAML.

1B. Автодетекция источника документа
Файл: src/docxru/pipeline.py — новая функция _detect_document_origin(doc) -> str:

Проверить core_properties.creator / last_modified_by на "ABBYY"
Посчитать <w:framePr> — если >30% параграфов, вероятно ABBYY
Проверить наличие <a:noAutofit/> в textbox'ах
При formatting_preset: "auto" автоматически выбрать abbyy_standard или native_docx
1C. Исправить порядок в postformat_docx()
Файл: src/docxru/pipeline.py (строки 3468-3489)

Текущий порядок (НЕВЕРНЫЙ):

apply_global_font_shrink ← font shrink ПЕРВЫЙ
_apply_abbyy_and_layout_passes ← layout check ВТОРОЙ
Правильный порядок:

_apply_abbyy_and_layout_passes ← сначала убрать ограничения + найти и пофиксить overflow
apply_global_font_shrink ← потом уменьшить шрифт для оставшихся сегментов
Фаза 2: In-place write-back (сохранение runs)
Импакт: ОЧЕНЬ ВЫСОКИЙ | Риск: СРЕДНИЙ | Усилия: 4-5 дней

Самое фундаментальное улучшение — не уничтожать runs, а заменять текст на месте.

2A. Запись границ runs в Span
Файл: src/docxru/models.py — добавить поле в Span:

python
original_run_lengths: tuple[int, ...] = ()  # длины текста каждого run, слитого в этот span
2B. Захват границ в paragraph_to_tagged()
Файл: src/docxru/tagging.py (строки 222-256)

При мерже runs (строка 253-254) вести список длин каждого run:

Новый run добавляется к span → добавить len(text) в список
При создании Span (строка 265) передать original_run_lengths=tuple(lengths)
2C. Новая функция tagged_to_runs_inplace()
Файл: src/docxru/tagging.py

Алгоритм:

Распарсить translated tagged text → pieces: list[(text, span_id)] (та же логика что в tagged_to_runs, строки 393-417)
Проверить feasibility: количество span_id совпадает, нет структурных изменений → если нет, return False
Для каждого span: найти исходные runs в параграфе по тексту
Распределить переведённый текст пропорционально original_run_lengths:
Span покрывал 3 runs с длинами [10, 5, 15] → перевод длиной 36 символов → runs получают [12, 6, 18]
Заменить run.text напрямую — все rPr, character styles, language metadata сохраняются
Return True
Fallback на tagged_to_runs() при:

Количество span_id в переводе ≠ в оригинале
Есть inline OBJ токены, переместившиеся между spans
Гиперссылки изменили границы
2D. Интеграция в pipeline
Файл: src/docxru/pipeline.py (строки 2824-2830)

python
# Пробуем in-place, если не получилось — fallback на полный rebuild
inplace_ok = tagged_to_runs_inplace(seg.paragraph_ref, target_tagged_unshielded, seg.spans, ...)
if not inplace_ok:
    tagged_to_runs(seg.paragraph_ref, target_tagged_unshielded, seg.spans, ...)
Ожидание: 60-70% сегментов пойдут через in-place путь. Для них ВСЁ форматирование сохраняется идеально.

Фаза 3: LLM учитывает ограничения контейнера
Импакт: ВЫСОКИЙ | Риск: НИЗКИЙ | Усилия: 1-2 дня

3A. Предварительный анализ контейнеров
Файл: src/docxru/pipeline.py — новая функция _attach_container_constraints(segments, cfg):

Для сегментов в table/textbox/frame: вычислить max_target_chars из размеров контейнера
Использовать функции из layout_check.py для получения размеров
Добавить в seg.context["max_target_chars"] только если ожидаемый RU текст близок к лимиту
3B. Подсказка в LLM промпте
Файл: src/docxru/llm.py (функция build_user_prompt, строка 215)

После сборки ctx_parts добавить:

python
max_chars = context.get("max_target_chars")
if max_chars:
    ctx_parts.append(f"SPACE_LIMIT: ~{max_chars} chars max, keep translation concise")
LLM будет стараться делать более компактный перевод для ограниченных контейнеров.

Фаза 4: Адаптивное уменьшение шрифта
Импакт: СРЕДНИЙ | Риск: НИЗКИЙ | Усилия: 1 день

Файл: src/docxru/layout_fix.py

Вместо фиксированного layout_font_reduction_pt для всех — пропорциональное уменьшение:

python
def _calculate_adaptive_reduction(overflow_ratio, current_font_pt, max_reduction=3.0):
    if overflow_ratio <= 1.0: return 0.0
    needed = current_font_pt * (1.0 - 1.0/overflow_ratio)
    return min(max_reduction, max(0.2, needed))
20% overflow при 10pt → ~1.7pt уменьшение
50% overflow при 10pt → ~3pt (capped)
Нет overflow → 0pt
Применить в _fix_table_overflow, _fix_textbox_overflow, _fix_frame_overflow.

Фаза 5: Эскалация фиксов в multi-pass
Импакт: СРЕДНИЙ | Риск: НИЗКИЙ | Усилия: 0.5 дня

Файл: src/docxru/layout_fix.py

Передавать pass_number в fix_expansion_issues():

Pass 1: spacing + пропорциональный font shrink
Pass 2: character spacing -15tw + aggressive spacing
Pass 3: убрать frame constraints + минимальный line spacing
Файл: src/docxru/pipeline.py — передавать номер прохода из цикла в _apply_abbyy_and_layout_passes.

Фаза 6: Отчёт о форматировании
Импакт: СРЕДНИЙ (workflow) | Риск: НИЗКИЙ | Усилия: 1 день

Новый файл: src/docxru/format_report.py

Генерация отчёта рядом с qa_report.html:

Список сегментов с уменьшенным шрифтом (было → стало)
Сегменты с изменённым spacing
Сегменты с ABBYY-нормализацией
Сегменты с ОСТАЮЩИМСЯ overflow risk после всех фиксов ← самое полезное
Сегменты где in-place write-back не удался (fallback на rebuild)
Это позволяет ревьюеру сразу видеть, какие места проверять вручную.

Критические файлы для изменения
Файл	Что меняется
src/docxru/config.py	formatting_preset, пресеты в load_config()
src/docxru/models.py	Span.original_run_lengths
src/docxru/tagging.py	paragraph_to_tagged() — захват длин; tagged_to_runs_inplace() — новая функция
src/docxru/pipeline.py	Интеграция in-place, _detect_document_origin(), _attach_container_constraints(), порядок в postformat_docx()
src/docxru/llm.py	SPACE_LIMIT hint в build_user_prompt()
src/docxru/layout_fix.py	Адаптивный shrink, эскалация по pass_number
src/docxru/format_report.py	Новый: генерация отчёта
Верификация
Unit тесты: расширить test_tagging_roundtrip.py для in-place path; добавить test_config_presets.py; расширить test_layout_fix.py для adaptive shrink
Интеграционный тест: обработать известный ABBYY DOCX с formatting_preset: abbyy_standard, сравнить overflow count до и после
Визуальная проверка: открыть результат в Word, проверить что текст не обрезан в textbox'ах и таблицах
A/B сравнение: перевести тот же документ с formatting_preset: off vs abbyy_standard, подсчитать количество мест требующих ручной правки