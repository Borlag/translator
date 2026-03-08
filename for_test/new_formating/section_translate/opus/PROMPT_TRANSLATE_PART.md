# Промт для перевода части CMM-документа (EN→RU)

Скопируй весь текст ниже и вставь в новый чат Claude (Opus или Sonnet).
Замени `N` на номер нужной части (2, 3, 4 ... 10).

---

## ПРОМТ (начало) ─────────────────────────────────────────

Ты — инженер-переводчик авиационной технической документации. Твоя задача — перевести часть N документа CMM (Component Maintenance Manual) на стойку основного шасси (Main Landing Gear Leg) Safran Landing Systems с английского на русский.

### Контекст проекта

Документ разделён на 10 частей (`original_new_part1.docx` ... `original_new_part10.docx`).
Часть 1 уже переведена и является эталоном терминологии.
Все части используют **единую** библиотеку перевода `cmm_translation_lib.py`, которая содержит:
- Глоссарии терминов (компоненты, заголовки, секции, SB-титулы, причины изменений)
- Функции перевода (параграфы, таблицы, TOC, колонтитулы, текстбоксы)
- XML-обработку Word-файлов с сохранением форматирования
- Постобработку (шрифт таблиц 9pt, очистка пустых runs)

### Расположение файлов

```
C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\
├── glossary.md                          ← Авиационный глоссарий EN→RU
├── general_prompt.md                    ← Принципы перевода
├── section\
│   ├── original_new_part1.docx          ← Исходники (НЕ ТРОГАТЬ)
│   ├── original_new_part2.docx
│   └── ... original_new_part10.docx
└── section_translate\opus\
    ├── cmm_translation_lib.py           ← ОБЩАЯ БИБЛИОТЕКА (глоссарии + логика)
    ├── translate_part1.py               ← Обёртка для части 1 (уже переведена)
    ├── translate_part2.py               ← Обёртка для части 2
    └── ... translate_part10.py
```

### Порядок работы

#### Шаг 1. Изучи эталон и материалы
Прочитай файлы:
- `cmm_translation_lib.py` — вся логика и словари
- `glossary.md` — авиационный глоссарий
- `general_prompt.md` — принципы перевода

#### Шаг 2. Запусти перевод
```bash
cd C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\section_translate\opus
python translate_partN.py
```

#### Шаг 3. Проанализируй результат
Скрипт выведет:
- Количество переведённых элементов (параграфы, таблицы, колонтитулы, текстбоксы)
- Автоматическую верификацию на оставшийся английский текст

Если верификация показывает `All paragraphs appear to be translated!` и `All table cells appear to be translated!` — перевод завершён.

#### Шаг 4. Если остались непереведённые элементы
Запусти глубокий скан выходного файла:
```python
python -c "
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

doc = Document('original_new_partN.docx')

# Скан параграфов
for i, para in enumerate(doc.paragraphs):
    text = para.text.strip()
    if text and re.search(r'[A-Za-z]{4,}', text):
        if not any(s in text for s in ['Safran','MLG','NLG','ASTM','AMS-','201587','EDES2','Cheltenham','32-12-22']):
            if not re.search(r'[\u0400-\u04FF]{3,}', text):
                print(f'P{i}: {text[:120]}')

# Скан таблиц
for ti, table in enumerate(doc.tables):
    for ri, row in enumerate(table.rows):
        for ci, cell in enumerate(row.cells):
            for para in cell.paragraphs:
                text = para.text.strip()
                if text and re.search(r'[A-Za-z]{4,}', text):
                    if not any(s in text for s in ['SAFRAN','Safran','Messier','MLG','SERVICE BULLETIN','ASTM','AMS-','201587']):
                        if not re.search(r'[\u0400-\u04FF]{3,}', text):
                            print(f'T{ti+1}R{ri+1}C{ci+1}: {text[:120]}')

# Скан текстбоксов
for txbx in doc.element.body.iter(qn('w:txbxContent')):
    for p in txbx.iter(qn('w:p')):
        para = Paragraph(p, doc)
        text = para.text.strip()
        if text and re.search(r'[A-Za-z]{4,}', text):
            if not any(s in text for s in ['Safran','MLG','NLG','ASTM','AMS-','201587','EDES2','Cheltenham']):
                if not re.search(r'[\u0400-\u04FF]{3,}', text):
                    print(f'TB: {text[:120]}')

# Скан колонтитулов
for si, section in enumerate(doc.sections):
    for hf in [section.header, section.even_page_header, section.first_page_header,
               section.footer, section.even_page_footer, section.first_page_footer]:
        if hf.is_linked_to_previous: continue
        for txbx in hf._element.iter(qn('w:txbxContent')):
            for p in txbx.iter(qn('w:p')):
                para = Paragraph(p, doc)
                text = para.text.strip()
                if text and re.search(r'[A-Za-z]{4,}', text):
                    if not any(s in text for s in ['SAFRAN','Safran','Cheltenham','Gloucester','CAGE','Landing Systems','www.']):
                        if not re.search(r'[\u0400-\u04FF]{3,}', text):
                            print(f'HF S{si}: {text[:120]}')
"
```

#### Шаг 5. Добавь недостающие переводы в библиотеку
Для каждого непереведённого элемента определи, к какому типу он относится, и добавь перевод в `cmm_translation_lib.py`:

| Тип текста | Куда добавлять | Пример |
|---|---|---|
| Название компонента | `COMPONENT_NAMES` | `"Axle Sleeve": "Втулка оси"` |
| Заголовок секции / фиксированная фраза | `FIXED` | `"DESCRIPTION AND OPERATION": "ОПИСАНИЕ И РАБОТА"` |
| Заголовок таблицы | `TABLE_HEADERS` | `"ITEM": "ПОЗИЦИЯ"` |
| Название секции в ревизионной таблице | `SECTION_NAMES` | `"Description and Operation": "Описание и работа"` |
| Фраза причины изменения | `REASON_PHRASES` | `"Deleted para": "Удалён пункт"` |
| Заголовок SB | `SB_TITLE_PARTS` | `"MLG - New feature...": "ОШ — Новая функция..."` |

Для новых **паттернов** (не простых замен), которые не покрываются существующими обработчиками — добавь логику в соответствующую функцию:
- `translate_text()` — основной переводчик параграфов/текстбоксов
- `translate_table_cell_text()` — переводчик ячеек таблиц
- `translate_hf_text()` — переводчик колонтитулов
- `translate_repair_description()` — описания ремонтов
- `translate_component_name()` — компонентные имена с суффиксами

#### Шаг 6. Перезапусти и проверь
После каждого изменения `cmm_translation_lib.py`:
```bash
python translate_partN.py
```
Повторяй шаги 3-6 пока верификация не покажет 0 непереведённых.

### Критически важные правила

1. **НЕ МЕНЯЙ исходные файлы** в папке `section\` — только `cmm_translation_lib.py`
2. **Терминология из глоссария** — приоритет. Используй `glossary.md` как источник истины
3. **Согласованность** — те же компоненты должны переводиться одинаково во всех частях
4. **Короткие слова** (`to`, `and`, `in`, `or`) заменяй только через `\b` (word-boundary regex), иначе испортишь слова типа `Landing` → `Lиing`
5. **Таблицы** — шрифт 9pt (18 half-points), нормализуется автоматически
6. **Не переводить**: названия компаний (Safran, Messier-Dowty), адреса, даты (Mar, Sep), ATA-коды (32-12-22), номера деталей (201587xxx), стандарты (AMS-, ASTM)
7. **Точки-лидеры** (`. . . . .`) — в regex требуй минимум 2 точки (`\.\s+\.`), иначе совпадёт с точкой сокращения `No.`
8. **Скрипт детерминистичен** — никаких API вызовов, только словарные замены. Модель (ты) нужна только для анализа непереведённого текста и добавления новых терминов в библиотеку.

### Пример рабочего процесса

```
> python translate_part2.py
Translated 180 paragraphs
Translated 500 table cells
...
Found 3 potentially untranslated paragraphs:
  P42: Description and Operation
  P105: WARNING: Do not use...
  P210: Lubrication adapter

→ Добавляешь в FIXED: "Description and Operation": "Описание и работа"
→ Добавляешь в FIXED: "WARNING:..." перевод
→ Добавляешь в COMPONENT_NAMES: "Lubrication adapter": "Смазочный адаптер"

> python translate_part2.py
All paragraphs appear to be translated!
All table cells appear to be translated!
✓ Готово
```

## ПРОМТ (конец) ───────────────────────────────────────────

Начни с шага 1: прочитай `cmm_translation_lib.py`, `glossary.md` и `general_prompt.md`.
Затем запусти `python translate_partN.py` и работай итеративно до полного перевода.
