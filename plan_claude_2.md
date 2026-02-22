Контекст
Проект docxru — CLI-инструмент для перевода DOCX-документов (авиационная техдокументация EN→RU) с сохранением форматирования. Сейчас PDF не поддерживается вообще — ожидается внешняя конвертация PDF→DOCX. Нужно:

Добавить полноценный PDF→PDF перевод с прямой заменой текстовых блоков через PyMuPDF
Двуязычный режим (bilingual) — EN/RU слои с переключением в PDF-ридере
OCR-fallback для сканированных страниц
Улучшить форматирование существующего DOCX-пайплайна

Основная идея: на каждой странице PDF определить текстовые блоки → извлечь текст с метаданными (шрифт, размер, позиция, цвет) → перевести LLM (переиспользуя существующую инфраструктуру) → «закрасить» оригинал белым (redaction) → вставить переведённый текст insert_htmlbox с подбором шрифта → при необходимости автоматически уменьшить шрифт.

Архитектура: подход C — прямая работа с PDF через PyMuPDF
Почему не PDF→DOCX→перевод→PDF:

Двойная конвертация теряет позиционирование и вёрстку
pdf2docx плохо справляется со сложными layout (авиамануалы)
Лицензия AGPL у pdf2docx

Почему прямой PyMuPDF:

Точное извлечение текста с метаданными шрифта/цвета/позиции
insert_htmlbox — автоматический перенос строк, CSS-стили, fallback на Noto шрифты для кириллицы
Redaction API для чистого удаления оригинала
OCG (Optional Content Groups) для двуязычных слоёв
Уже является зависимостью проекта (в [project.optional-dependencies] render)


Новые модули
src/docxru/pdf_models.py — Модели данных
PdfSpanStyle    — шрифт, размер, цвет, bold/italic флаги
PdfSpan         — отрезок текста с bbox и стилем
PdfTextBlock    — логический текстовый блок (параграф) на странице
PdfPage         — все блоки одной страницы + размеры + has_text флаг
PdfSegment      — единица перевода (один или несколько сгруппированных блоков)
src/docxru/pdf_reader.py — Извлечение текста

extract_pdf_pages(pdf_path) -> list[PdfPage]
Использует page.get_text("dict") для блоков/строк/спанов
Определяет has_text для детекции сканированных страниц
Обрабатывает повёрнутый текст, пропускает image-блоки

src/docxru/pdf_layout.py — Анализ layout

group_blocks_into_segments(page) -> list[PdfSegment] — группировка смежных блоков
detect_table_regions(page) -> list[TableRegion] — через page.get_drawings() + grid-анализ bbox
detect_columns(page) -> list[ColumnRegion] — кластеризация по x-координате
classify_block() — "body" | "header" | "footer" | "table_cell"
Эвристики: вертикальная близость (1.5x line-height), однородность шрифта, выравнивание по левому краю

src/docxru/pdf_font_map.py — Подбор шрифтов

select_replacement_font(original_font, is_bold, is_italic) -> FontSpec
Маппинг: serif → Noto Serif / PT Serif, sans-serif → Noto Sans / PT Sans, mono → Noto Mono
Конфигурируемая таблица замен в YAML (pdf.font_map)
Fallback на встроенный Helvetica-Cyrillic

src/docxru/pdf_writer.py — Запись переведённого текста

replace_block_text(page, block, translated_text, font_spec, *, ocg_xref) — основная функция замены
Алгоритм на каждый блок:

page.add_redact_annot(block.bbox, fill=white) — закрасить оригинал
page.apply_redactions() — применить
Собрать HTML из переведённого текста с CSS (font-family, font-size, color, bold/italic)
page.insert_htmlbox(rect, html, css, archive, oc=ocg_xref) — вставить перевод


Автоподбор размера шрифта: если текст не влезает → уменьшать до max_font_shrink_ratio (по умолчанию 0.6x)
build_bilingual_pdf() — создание OCG-слоя "Russian Translation" для переключения EN/RU
doc.subset_fonts() — оптимизация размера файла

src/docxru/pdf_pipeline.py — Оркестрация
Stage 0: Загрузка PDF + извлечение страниц (pdf_reader)
Stage 1: Layout анализ + создание сегментов (pdf_layout)
  ↓ для скан-страниц: OCR через ocrmypdf/tesseract → повторное извлечение
Stage 2: Перевод (ПЕРЕИСПОЛЬЗОВАНИЕ существующего):
  - token_shield.shield() → shielded + token_map
  - tm.get_exact() → TM hit или miss
  - llm_client.translate() → перевод через LLM
  - validator.validate_*() → валидация маркеров/чисел
  - token_shield.unshield() → финальный текст
  - Поддержка: concurrency, batch mode, context window, glossary
Stage 3: Подбор шрифтов (pdf_font_map)
Stage 4: Запись PDF (pdf_writer)
  - Redaction + insert_htmlbox для каждого блока
  - Опционально: OCG bilingual overlay
  - doc.subset_fonts() + ez_save()
Stage 5: QA отчёт (qa_report — переиспользование)

Изменения в существующих файлах
src/docxru/cli.py — новая подкоманда

docxru translate-pdf --input input.pdf --output output.pdf --config config.yaml
Флаги: --bilingual, --max-pages N, --ocr-fallback, --resume

src/docxru/config.py — новый PdfConfig
python@dataclass(frozen=True)
class PdfConfig:
    bilingual_mode: bool = False
    ocr_fallback: bool = False
    max_font_shrink_ratio: float = 0.6
    block_merge_threshold_pt: float = 12.0
    skip_headers_footers: bool = False
    table_detection: bool = True
    font_map: dict[str, str] = field(default_factory=dict)
    default_sans_font: str = "NotoSans-Regular.ttf"
    default_serif_font: str = "NotoSerif-Regular.ttf"
pyproject.toml — зависимости

Новая группа pdf: PyMuPDF>=1.24.0, ocrmypdf>=16.0.0 (опционально)
Или перенести PyMuPDF в основные зависимости


Улучшения DOCX-пайплайна
1. Font metric-based layout check (layout_check.py)

Сейчас используется константа _APPROX_CHAR_WIDTH_TWIPS = 120 — неточно
Извлекать font_name + font_size_pt из RunStyleSnapshot
Использовать lookup-таблицу средних ширин символов для популярных шрифтов
Fallback на текущую константу если шрифт неизвестен

2. Улучшенное авто-исправление layout (layout_fix.py)

Каскад: уменьшение spacing → уменьшение character spacing (w:spacing) → уменьшение font size → вставка переносов
Для табличных ячеек: учитывать ширину ячейки при подборе шрифта
Настраиваемые шаги через конфиг

3. Поддержка гиперссылок в tagging (tagging.py)

Сейчас is_supported_paragraph() пропускает параграфы с w:hyperlink
Расширить: извлекать run-ы из w:hyperlink, добавить HREF флаг
Сохранять структуру гиперссылки при write-back

4. Улучшенная обработка таблиц (layout_fix.py)

Character spacing reduction (w:spacing w:val="-10") до уменьшения шрифта
Для multi-paragraph cells: агрессивнее уменьшать spacing между параграфами


Порядок реализации
Фаза 1: Ядро PDF (pdf_models, pdf_reader, pdf_layout, pdf_font_map)

Модели данных
Извлечение текстовых блоков с метаданными
Группировка блоков в сегменты
Подбор шрифтов

Фаза 2: PDF-пайплайн (pdf_pipeline, pdf_writer)

Основной цикл перевода (переиспользуя llm, tm, token_shield, validator)
Redaction + insert_htmlbox замена текста
Автоподбор размера шрифта

Фаза 3: CLI и конфиг (cli, config, pyproject.toml)

Подкоманда translate-pdf
PdfConfig в конфиге
Зависимости

Фаза 4: Продвинутые фичи

Bilingual OCG-слои (--bilingual)
Детекция таблиц через page.get_drawings()
Детекция колонок
OCR fallback для сканированных страниц

Фаза 5: Улучшения DOCX

Font metric layout check
Каскадное авто-исправление
Поддержка гиперссылок в tagging
Улучшенная обработка табличных ячеек

Фаза 6: Тесты

Unit-тесты: pdf_reader, pdf_layout, pdf_font_map, pdf_writer
Интеграционные тесты с реальными авиа-PDF
QA-отчёты для PDF-сегментов
Визуальное сравнение оригинал vs перевод


Переиспользование существующей инфраструктуры (без изменений)
МодульЧто переиспользуетсяllm.pyВсе LLM-провайдеры, промпты, глоссарий, batch-режимtm.pySQLite TM (exact + fuzzy), progress trackingtoken_shield.pyshield/unshield, паттерны PN/ATA/DIMvalidator.pyВалидация маркеров, чисел, глоссарных леммconsistency.pyПроверка единообразия терминовqa_report.pyHTML/JSONL отчётыconfig.pyload_config(), базовые LLM/TM конфиги

Ключевые файлы для реализации

src/docxru/pipeline.py — образец оркестрации для pdf_pipeline.py
src/docxru/llm.py — LLM-клиенты для переиспользования
src/docxru/config.py — расширение PdfConfig
src/docxru/cli.py — добавление подкоманды
src/docxru/layout_check.py — улучшение метрик шрифтов
src/docxru/layout_fix.py — каскадное авто-исправление
src/docxru/tagging.py — поддержка гиперссылок


Верификация

Unit-тесты: запуск pytest tests/ после каждой фазы
PDF-перевод: ручной тест на реальном авиа-PDF → визуальная проверка
Bilingual: проверка переключения слоёв в Adobe Reader/Foxit
OCR: тест на сканированном PDF → проверка распознавания + перевода
DOCX-улучшения: запуск на существующих samples/ → сравнение QA-отчётов с baseline
Регрессия: docxru verify на существующих тестовых документах