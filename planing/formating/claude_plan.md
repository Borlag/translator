Plan: Автоматическое исправление форматирования переведённых DOCX

Контекст
Переведённые документы (EN→RU, из ABBYY FineReader) содержат массовые проблемы форматирования:

Скрытый/обрезанный текст в текстбоксах — русский текст длиннее, не помещается в фиксированные текстбоксы
Переполнение ячеек таблиц — текст выходит за границы
Слишком крупный шрифт для контейнеров после перевода
Смещение текста — абсолютное позиционирование от ABBYY
Разрывы предложений — неудачные переносы строк
Артефакты ABBYY — exact row heights, framePr, exact line spacing блокирующие авто-подгонку
Главная причина: ABBYY FineReader ставит <a:noAutofit/> в текстбоксах, что запрещает Word автоматически уменьшать шрифт для вмещения текста. Плюс жёсткие constraints (exact heights, framePr).

Текущее состояние — в проекте уже есть layout_check/layout_fix/com_word/oxml_table_fix, но:

layout_check и layout_auto_fix выключены по умолчанию
Фикс шрифта плоский (−0.5pt), без учёта геометрии контейнера
com_word.py ограничен 2 шагами shrink
oxml_table_fix.py не трогает текстбоксы вообще
Нет инъекции normAutofit в bodyPr
Подход: Гибридный (python-docx/lxml + Word COM)
Pass 1: XML-level фиксы (быстро, детерминировано, ~5 сек) Pass 2: Word COM верификация + точечные фиксы (30-120 сек, Windows-only)

Волна 1: Инъекция autofit в текстбоксы (наибольший эффект)
Файл: src/docxru/oxml_table_fix.py
Добавить функцию set_textbox_autofit(document, mode="normAutofit"):

Итерировать все <a:bodyPr> элементы в документе
Заменить <a:noAutofit/> на <a:normAutofit/> (Word будет авто-уменьшать шрифт)
Обрабатывать как DrawingML (wsp:bodyPr), так и legacy VML (v:textbox)
Применять только к текстбоксам с непустым w:txbxContent
Расширить normalize_abbyy_oxml() новым профилем "full":

full = aggressive + textbox autofit + нормализация отступов таблиц
Файл: src/docxru/config.py
Новые поля в PipelineConfig:

layout_textbox_autofit: bool = True           # инъекция normAutofit
layout_textbox_min_font_pt: float = 7.0       # мин. размер шрифта для textbox
layout_com_max_shrink_steps: int = 4          # шаги уменьшения в COM
layout_textbox_expand_height: bool = False    # разрешить рост высоты (off по умолчанию)
layout_textbox_max_height_growth: float = 1.5 # макс. рост высоты
layout_table_normalize_margins: bool = True   # уменьшить избыточные margins ячеек
Файл: src/docxru/pipeline.py (строки ~2769-2810)
Вставить после ABBYY normalization, перед save:

python
if cfg.layout_textbox_autofit:
    autofit_count = set_textbox_autofit(doc, mode="normAutofit")
```

---

## Волна 2: Умные контейнер-специфичные фиксы

### Файл: `src/docxru/layout_fix.py`

Рефакторинг `fix_expansion_issues()` — диспетчеризация по типу контейнера:

**Для текстбоксов** (`fix_textbox_overflow`):
1. Вычислить площадь текстбокса из `wp:extent`
2. Оценить нужную площадь из длины текста + метрик шрифта
3. Рассчитать коэффициент масштабирования (sqrt от отношения площадей)
4. Пропорционально уменьшить шрифт всех runs (min 7pt)
5. Обнулить spacing before/after внутри текстбокса

**Для ячеек таблиц** (`fix_table_cell_overflow`):
1. Обнулить paragraph spacing
2. Сжать character spacing (−15 twips вместо −10)
3. Установить line spacing = 1.0 × font size
4. Пропорционально уменьшить шрифт (до −2pt, min 6pt)

### Файл: `src/docxru/oxml_table_fix.py`

Добавить `normalize_table_cell_margins(document)`:
- Уменьшить избыточные `w:tcMar` в ячейках с переполнением
- Минимальный margin = 29 twips (~0.5mm)

---

## Волна 3: Расширение текстбоксов по высоте (опционально)

### Файл: `src/docxru/layout_fix.py` или `src/docxru/oxml_table_fix.py`

Добавить `expand_textbox_extent(segment, max_height_growth)`:
- Увеличить `wp:extent cy` и `a:ext cy` синхронно
- Только высота, не ширина (ширина ломает колонки)
- Максимальный рост ограничен `max_height_growth` (1.5x)
- **Выключено по умолчанию** — может повлиять на общую компоновку страницы

---

## Волна 4: Усиленный COM pass

### Файл: `src/docxru/com_word.py`

1. Увеличить `max_shrink_steps` с 2 до 4
2. Добавить `_expand_overflowing_shapes(doc)`:
   - Для shape'ов где `TextFrame.Overflowing == True` после autofit+shrink
   - Увеличивать `shape.Height` на 7.2 points (0.1 дюйма) итеративно
   - Лимит: `max_height_growth` от оригинала
3. Обновить `update_fields_via_com()`:
   - Новые параметры: `expand_overflowing`, `max_height_growth`
   - Передать настройки из конфига

---

## Интеграция в pipeline (итоговый порядок)
```
1. Write-back перевода в параграфы
2. Global font shrink (если font_shrink_*_pt > 0)
3. Layout check → layout auto-fix (контейнер-специфичный)
4. ABBYY OXML normalization (профиль из конфига)
5. Textbox autofit injection (layout_textbox_autofit)
6. Table margin normalization (layout_table_normalize_margins)
7. Final run-level cleanup
8. Save DOCX
9. Word COM pass (mode="com"): fields + TOC + autofit + expand
Тестирование
Unit-тесты для каждой новой функции (test_layout_fixer.py):
test_set_textbox_autofit_replaces_noAutofit
test_set_textbox_autofit_skips_already_autofit
test_fix_textbox_overflow_proportional_shrink
test_normalize_table_cell_margins
test_expand_textbox_extent
Интеграционный тест с ABBYY sample:
Загрузить samples/...abby_short.docx
Применить полную нормализацию
Проверить что нет <a:noAutofit/> в результате
Визуальная проверка — прогнать translated.docx через новый pipeline и сравнить с оригиналом
Критические файлы
Файл	Изменения
src/docxru/oxml_table_fix.py	+set_textbox_autofit(), +normalize_table_cell_margins(), расширить normalize_abbyy_oxml() профилем "full"
src/docxru/layout_fix.py	Рефакторинг fix_expansion_issues(), +fix_textbox_overflow(), +fix_table_cell_overflow() усиленный, +expand_textbox_extent()
src/docxru/config.py	+6 новых полей в PipelineConfig + load_config()
src/docxru/pipeline.py	Вставить вызовы новых функций (строки ~2769-2810)
src/docxru/com_word.py	Усилить shrink steps, +_expand_overflowing_shapes(), обновить update_fields_via_com()
tests/test_layout_fixer.py	Новый файл с тестами
Оценка затрат
Стоимость: $0 (все инструменты бесплатные — python-docx, lxml, pywin32)
Время реализации: 4 волны, каждая независима
Волна 1 даёт ~70% эффекта — одна замена noAutofit → normAutofit решает основную проблему текстбоксов