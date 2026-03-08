"""
CMM Translation Library — shared module for translating aviation CMM documents (EN→RU).

Provides glossary-based terminology, translation functions, XML-level text replacement,
font handling, and the main translation pipeline. All 10 document parts use this library
for consistent terminology and formatting.

Usage:
    from cmm_translation_lib import translate_document
    translate_document(
        src="path/to/original_new_partN.docx",
        dst="path/to/translated_partN.docx",
    )
"""
import copy
import os
import re
import sys
from pathlib import Path
from lxml import etree
from docx import Document
from docx.shared import Pt, Emu
from docx.oxml.ns import qn


# ══════════════════════════════════════════════════════════════════════════════
#  GLOSSARIES  (EN → RU)
# ══════════════════════════════════════════════════════════════════════════════

# ── Component name glossary ──────────────────────────────────────────────────
COMPONENT_NAMES = {
    "Main Landing Gear Leg": "Стойка основного шасси",
    "Main Landing Gear": "Основное шасси",
    "MLG Shock Strut": "Стойка амортизатора основного шасси",
    "Main Fitting Subassembly": "Сборка корпуса стойки",
    "Main Fitting": "Корпус стойки",
    "Sliding Tube": "Скользящая труба",
    "Lower Torque Link": "Нижний шлиц-шарнир",
    "Upper Torque Link": "Верхний шлиц-шарнир",
    "Upper Slave Link": "Верхнее ведомое звено",
    "Lower Slave Link": "Нижнее ведомое звено",
    "Upper Diaphragm Tube Subassembly": "Сборка верхней диафрагменной трубы",
    "Upper Diaphragm Tube": "Верхняя диафрагменная труба",
    "Pivot Pin": "Штифт вращения",
    "Pivot Bracket": "Поворотный кронштейн",
    "Upper Pivot Bracket": "Верхний поворотный кронштейн",
    "Harness Support Bracket": "Кронштейн крепления жгута",
    "Lock Stay Cardan": "Кардан фиксатора",
    "Transfer Block": "Переходный блок",
    "Transfer block": "Переходный блок",
    "Spherical Bearing": "Сферический подшипник",
    "Pintle pin": "Штифт навеса стойки",
    "Pintle Pin": "Штифт навеса стойки",
    "Forward Pintle Pin": "Передний штифт навеса стойки",
    "Retaining Pin": "Стопорный штифт",
    "Uplock Pin ain Fitting": "Штифт замка убранного положения корпуса стойки",
    "Cylinder": "Цилиндр",
    "Bracket": "Кронштейн",
    "Pin": "Штифт",
    "Spacer": "Проставка",
    "Drag-arm Spacer": "Проставка тяги",
    "Slave Link": "Ведомое звено",
    "Valve Stem": "Шток клапана",
    "Torque Link": "Шлиц-шарнир",
    "Uplock Pin": "Штифт замка убранного положения",
    "Uplock": "Замок убранного положения",
    "Inflation Valve": "Клапан зарядки",
    "Lower Bearing Subassembly": "Сборка нижнего подшипника",
    "Lower Bearing": "Нижний подшипник",
    # Lowercase variants
    "Sliding tube": "Скользящая труба",
    "Pintle pin": "Штифт навеса стойки",
    # Repair procedure descriptions
    "Bush": "Втулка",
    "Bushes": "Втулки",
    "Machining": "Механическая обработка",
    "Installation": "Установка",
    "Machining and Installation": "Механическая обработка и установка",
    "Machining and installation": "Механическая обработка и установка",
}

# ── Fixed phrase translations ────────────────────────────────────────────────
FIXED = {
    # Title page
    "MAIN LANDING GEAR LEG": "СТОЙКА ОСНОВНОГО ШАССИ",
    "PART NUMBER": "НОМЕР ДЕТАЛИ",
    "COMPONENT MAINTENANCE MANUAL WITH": "РУКОВОДСТВО ПО ТЕХНИЧЕСКОМУ ОБСЛУЖИВАНИЮ КОМПОНЕНТОВ С",
    "ILLUSTRATED PARTS LIST": "ИЛЛЮСТРИРОВАННЫМ ПЕРЕЧНЕМ ДЕТАЛЕЙ",
    "STATEMENT OF INITIAL CERTIFICATION": "СВИДЕТЕЛЬСТВО О ПЕРВИЧНОЙ СЕРТИФИКАЦИИ",
    "INTENTIONALLY BLANK": "НАМЕРЕННО ОСТАВЛЕНО ПУСТЫМ",

    # Structural headings
    "NEW/REVISED PAGES": "НОВЫЕ/ПЕРЕСМОТРЕННЫЕ СТРАНИЦЫ",
    "REVISION RECORD": "ЗАПИСЬ ИЗМЕНЕНИЙ",
    "RECORD OF REVISIONS": "ЗАПИСЬ ИЗМЕНЕНИЙ",
    "RECORD OF TEMPORARY REVISIONS": "ЗАПИСЬ ВРЕМЕННЫХ ИЗМЕНЕНИЙ",
    "LIST OF SERVICE BULLETINS": "СПИСОК СЕРВИСНЫХ БЮЛЛЕТЕНЕЙ",
    "LIST OF SERVICE BULLETINS (Continued)": "СПИСОК СЕРВИСНЫХ БЮЛЛЕТЕНЕЙ (Продолжение)",
    "LIST OF EFFECTIVE PAGES": "ПЕРЕЧЕНЬ ДЕЙСТВУЮЩИХ СТРАНИЦ",
    "LIST OF EFFECTIVE PAGES (Continued)": "ПЕРЕЧЕНЬ ДЕЙСТВУЮЩИХ СТРАНИЦ (Продолжение)",
    "TABLE OF CONTENTS": "СОДЕРЖАНИЕ",
    "TABLE OF CONTENTS (Continued)": "СОДЕРЖАНИЕ (Продолжение)",
    "ILLUSTRATIONS": "ИЛЛЮСТРАЦИИ",
    "ILLUSTRATIONS (Continued)": "ИЛЛЮСТРАЦИИ (Продолжение)",
    "UNIT IDENTIFICATION CHART": "ТАБЛИЦА ИДЕНТИФИКАЦИИ ИЗДЕЛИЯ",
    "UNIT IDENTIFICATION CHART (Continued)": "ТАБЛИЦА ИДЕНТИФИКАЦИИ ИЗДЕЛИЯ (Продолжение)",
    "TITLE PAGE": "ТИТУЛЬНАЯ СТРАНИЦА",
    "Issued by": "Выпущено",
    "REV NO.": "НОМ. РЕД.",
    "ISSUE DATE": "ДАТА ВЫПУСКА",
    "INSERTION DATE": "ДАТА ВСТАВКИ",
    "BY": "КЕМ",
    "REMOVAL DATE": "ДАТА УДАЛЕНИЯ",

    # TOC section headings
    "Assembly (Including Storage)": "Сборка (включая хранение)",
    "General": "Общие сведения",
    "Procedure": "Процедура",
    "Storage": "Хранение",
    "Fits and Clearances": "Посадки и зазоры",
    "Fits and Clearances Definitions": "Определения посадок и зазоров",
    "Remarks": "Примечания",
    "Torque Data": "Данные моментов затяжки",
    "Special Tools, Fixtures and Equipment": "Специальные инструменты, приспособления и оборудование",
    "Illustrated Parts List": "Иллюстрированный перечень деталей",
    "Introduction": "Введение",
    "How To Use This Illustrated Parts List": "Как пользоваться иллюстрированным перечнем деталей",
    "Vendor Codes, Names and Addresses": "Коды, наименования и адреса поставщиков",
    "Numerical Index": "Числовой указатель",
    "Detailed Parts List": "Подробный список деталей",

    # Figure captions
    "Diagram of Operation": "Схема работы",
    "Electrical Bonding Resistance Test Points": "Точки проверки сопротивления электрического соединения",
    "Protective Treatment": "Защитная обработка",

    # Section/TOC headings (textbox labels)
    "Title Page": "Титульная страница",
    "Record of Temporary": "Запись временных",
    "Revisions": "Изменений",
    "Unit Identification": "Идентификация изделия",
    "Chart": "Таблица",
    "List of Effective": "Перечень действующих",
    "Pages": "Страниц",
    "Pages (Continued)": "Страниц (Продолжение)",
    "Table of Contents": "Содержание",
    "Testing and Fault": "Проверка и поиск неисправностей",
    "Isolation": "Локализация",
    "Subject": "Раздел",
    "Page": "Страница",
    "Date": "Дата",
    "Operation": "Работа",
    "INCORPORATED": "ВКЛЮЧЕНО",
    "Description and": "Описание и",
    "Disassembly": "Разборка",
    "Cleaning": "Очистка",
    "Repair": "Ремонт",
    "Repair Procedure Conditions": "Условия выполнения процедуры ремонта",
    "Approved Repairs": "Утверждённые ремонты",
    "Check": "Проверка",
    "Equipment and Materials": "Оборудование и материалы",
    "Test Conditions": "Условия испытаний",
    "Detailed Inspection": "Детальный осмотр",
    "Special Detailed Inspection": "Специальный детальный осмотр",
    "Fault Isolation": "Поиск неисправностей",
    "Description and Operation": "Описание и работа",
    "Description": "Описание",
    "Data": "Данные",
    "Superseded": "Заменён",
    "Assembly": "Сборка",
    "Assembly (Including": "Сборка (включая",
    "Storage)": "хранение)",
    "Storage) (Continued)": "хранение) (Продолжение)",
    "Tables 101": "Таблицы 101",
    "Tables": "Таблицы",
    "Only": "Только",
    "Conditions": "Условия",
    "Inspection": "Осмотр",
    "Materials": "Материалы",
    "Special Tools, Fixtures": "Специальные инструменты, приспособления",
    "and Equipment": "и оборудование",
    "and Equipment (Continued)": "и оборудование (Продолжение)",
    "Repair (Continued)": "Ремонт (Продолжение)",
    "Cleaning (Continued)": "Очистка (Продолжение)",
    "Check (Continued)": "Проверка (Продолжение)",
    "Disassembly (Continued)": "Разборка (Продолжение)",
    "Assembly (Continued)": "Сборка (Продолжение)",
    "Illustrations": "Иллюстрации",
    "(Continued)": "(Продолжение)",
    "Isolation (Continued)": "Локализация (Продолжение)",
    "REV": "РЕД",

    # Mixed-case variants for textbox labels
    "Record of Revisions": "Запись изменений",
    "List of Service Bulletins": "Список сервисных бюллетеней",
    "List of Effective Pages": "Перечень действующих страниц",
    "Record of Temporary Revisions": "Запись временных изменений",
    "Testing and Fault Isolation": "Проверка и поиск неисправностей",
    "Unit Identification Chart": "Таблица идентификации изделия",
    "Assembly (Including Storage)": "Сборка (включая хранение)",
    "Special Tools, Fixtures and Equipment": "Специальные инструменты, приспособления и оборудование",
    "Illustrated Parts List": "Иллюстрированный перечень деталей",
    "Fits and Clearances": "Посадки и зазоры",
}

# ── Table header translations ────────────────────────────────────────────────
TABLE_HEADERS = {
    "Subject Reference": "Ссылка на раздел",
    "Remove and Destroy Pages": "Удалить и уничтожить страницы",
    "Insert New/Revised": "Вставить новые/пересмотренные",
    "Pages": "Страницы",
    "Dated": "Дата",
    "Reason for Change": "Причина изменения",
    "REV.\nNo.": "НОМ.\nРЕД.",
    "ISSUE DATE": "ДАТА ВЫПУСКА",
    "DATE INSERTED": "ДАТА ВСТАВКИ",
    "PAGE NUMBER": "НОМЕР СТРАНИЦЫ",
    "BY": "КЕМ",
    "DATE REMOVED": "ДАТА УДАЛЕНИЯ",
    "SB NUMBER": "НОМЕР SB",
    "SB TITLE": "НАЗВАНИЕ SB",
    "SB REVISION NUMBER": "НОМЕР РЕДАКЦИИ SB",
    "DATE INCORPORATED INTO MANUAL": "ДАТА ВКЛЮЧЕНИЯ В РУКОВОДСТВО",
    "COVER SB NO.": "НОМЕР ОХВАТ. SB",
    "PART NUMBER": "НОМЕР ДЕТАЛИ",
    "DASH NO.": "НОМЕР ТИРЕ",
    "MOD. STRIKE NO.": "НОМЕР СНЯТИЯ МОД.",
    "SAFRAN LANDING SYSTEMS MODIFICATION NUMBER": "НОМЕР МОДИФИКАЦИИ SAFRAN LANDING SYSTEMS",
    "SAFRAN LANDING SYSTEMS\nSERVICE BULLETIN NUMBER": "НОМЕР СЕРВИСНОГО БЮЛЛЕТЕНЯ\nSAFRAN LANDING SYSTEMS",
    "SAFRAN LANDING SYSTEMS SERVICE BULLETIN NUMBER": "НОМЕР СЕРВИСНОГО БЮЛЛЕТЕНЯ SAFRAN LANDING SYSTEMS",
    "Page": "Страница",
    "Date": "Дата",
    "Subject": "Раздел",
    "Initial Issue": "Первоначальный выпуск",
    "No effect": "Без изменений",
}

# ── Revision table section name translations ─────────────────────────────────
SECTION_NAMES = {
    "Record of Revisions": "Запись изменений",
    "Unit Identification Chart": "Таблица идентификации изделия",
    "List of Effective Pages": "Перечень действующих страниц",
    "Table of Contents": "Содержание",
    "Disassembly": "Разборка",
    "Check": "Проверка",
    "Repair": "Ремонт",
    "Assembly (Including Storage)": "Сборка (включая хранение)",
    "Fits and Clearances": "Посадки и зазоры",
    "Special Tools, Fixtures and Equipment": "Специальные инструменты, приспособления и оборудование",
    "Illustrated Parts List": "Иллюстрированный перечень деталей",
    # List of Effective Pages section names
    "Subject": "Раздел",
    "(Continued)": "(Продолжение)",
    "Testing and Fault": "Проверка и поиск неисправностей",
    "Isolation": "Локализация",
    "Isolation (Continued)": "Локализация (Продолжение)",
    "Cleaning": "Очистка",
    "Repair (Continued)": "Ремонт (Продолжение)",
    "Assembly (Including": "Сборка (включая",
    "Storage)": "хранение)",
    "Storage) (Continued)": "хранение) (Продолжение)",
    "Special Tools, Fixtures": "Специальные инструменты, приспособления",
    "and Equipment": "и оборудование",
}

# ── Reason for change phrase translations ────────────────────────────────────
REASON_PHRASES = {
    "Updated revision status": "Обновлён статус редакции",
    "Updated pages": "Обновлены страницы",
    "Added content": "Добавлено содержание",
    "Updated page numbers": "Обновлены номера страниц",
    "Updated figures": "Обновлены рисунки",
    "Updated figure numbers": "Обновлены номера рисунков",
    "Updated tables": "Обновлены таблицы",
    "Added paras": "Добавлены пункты",
    "Added para": "Добавлен пункт",
    "Updated fig-items": "Обновлены поз. рисунков",
    "Updated fig-item": "Обновлена поз. рисунка",
    "Added fig-item": "Добавлена поз. рисунка",
    "in para": "в пункте",
    "in paras": "в пунктах",
    "and": "и",
    "only": "только",
    "only in para": "только в пункте",
    "in": "в",
    "Updated Messier-Dowty Limited to Safran Landing Systems": "Обновлено Messier-Dowty Limited на Safran Landing Systems",
    "Updated Messier-Dowty Limited to Safran Landing System": "Обновлено Messier-Dowty Limited на Safran Landing Systems",
    "Updated Messier-Dowty Limited to Safran Landing Sy": "Обновлено Messier-Dowty Limited на Safran Landing Sy",
    "Updated Messier-Dowty": "Обновлено Messier-Dowty",
    "Limited to Safran Landing Systems": "Limited на Safran Landing Systems",
    "Added repair no": "Добавлен ремонт №",
    "Added repair no.": "Добавлен ремонт №",
    "Added caution at para": "Добавлено предупреждение в пункте",
    "Updated material specification": "Обновлена спецификация материала",
    "Updated materia": "Обновлены материа",
    "Updated paras": "Обновлены пункты",
    "Updated IPL figs": "Обновлены рисунки ИПД",
    "to include Ref": "для включения ссылок",
    "Updated figure numbers": "Обновлены номера рисунков",
    "Updated figure": "Обновлён рисунок",
    "Updated figures": "Обновлены рисунки",
    "Added figure": "Добавлен рисунок",
    "Added Ref. Codes": "Добавлены коды ссылок",
    "Updated tables": "Обновлены таблицы",
    "Updated table": "Обновлена таблица",
    # Standalone "Updated" at end of line (continuation to next paragraph)
    ". Updated": ". Обновлены",
    "Deleted": "Удалены",
    "titles": "заголовки",
    "conversion value": "значение пересчёта",
    "caution at para": "предупреждение в пункте",
    "caution": "предупреждение",
    "Updated IPL fig": "Обновлён рисунок ИПД",
    "figure": "рисунок",
    "figures": "рисунки",
    "Codes": "Коды",
    "details": "детали",
    "to": "до",
}

# ── SB title translations ───────────────────────────────────────────────────
SB_TITLE_PARTS = {
    # Full SB title translations (sorted longest-first for matching)
    "MLG - Installation of stub bolt subassembly for the forward pintle pin in place of the cross bolt.":
        "ОШ — Установка сборки болта-вставки для переднего штифта навеса стойки взамен болта.",
    "MLG - To allow an increase in aircraft maximum take-off weight to 93 tonne.":
        "ОШ — Для обеспечения увеличения максимальной взлётной массы воздушного судна до 93 тонн.",
    "MLG -To add tracking numbers to parts listed in Airbus Airworthiness Limitations Section (ALS).":
        "ОШ — Добавление учётных номеров к деталям, указанным в разделе ограничений лётной годности Airbus (ALS).",
    "MLG - Installation of a 201585 series MLG Leg and Dressings where a 201387 MLG Leg and Dressings has been installed":
        "ОШ — Установка стойки ОШ серии 201585 и обвязки вместо установленной стойки ОШ серии 201387 и обвязки",
    "MLG -To add tracking numbers to parts listed in Airbus Maintenance Planning Document, Section 9-1. (Torque link apex pin nut)":
        "ОШ — Добавление учётных номеров к деталям, указанным в документе планирования ТО Airbus, раздел 9-1. (Гайка штифта вершины шлиц-шарнира)",
    "MLG -To add tracking numbers to parts listed in Airbus Maintenance Planning Document, Section 9-":
        "ОШ — Добавление учётных номеров к деталям, указанным в документе планирования ТО Airbus, раздел 9-",
    "MLG - Introduction of a new lower bearing subassembly":
        "ОШ — Введение новой сборки нижнего подшипника",
    "MLG - Introduction of new charging labels":
        "ОШ — Введение новых этикеток зарядки",
    "MLG - Introduction of new 1M and 2M Axle harnesses":
        "ОШ — Введение новых жгутов осей 1М и 2М",
    "MLG - Introduction of new 1M and 2M Leg Harness and of new 1M and 2M Axle Harnesses":
        "ОШ — Введение новых жгутов стойки 1М и 2М и новых жгутов осей 1М и 2М",
    "MLG - Introduction of new 1M and 2M Leg Harness an":
        "ОШ — Введение новых жгутов стойки 1М и 2М и",
    "MLG Leg-Introduction of new retaining pins and a new lower bearing subassembly with a new self lubricating liner":
        "Стойка ОШ — Введение новых стопорных штифтов и новой сборки нижнего подшипника с новым самосмазывающимся вкладышем",
    "MLG Leg - Introduction of new retaining pins for the lower bearing subassembly":
        "Стойка ОШ — Введение новых стопорных штифтов для сборки нижнего подшипника",
    "MLG Leg - Introduction of a new lower bearing subassembly with a new low friction inner liner":
        "Стойка ОШ — Введение новой сборки нижнего подшипника с новым низкофрикционным внутренним вкладышем",
    "MLG Leg - Barkhausen Noise Inspection of Main Landing Gear Sliding Tube Axles.":
        "Стойка ОШ — Контроль методом шума Баркгаузена осей скользящей трубы основного шасси.",
    "MLG Leg - Introduction of a new Main Fitting":
        "Стойка ОШ — Введение нового корпуса стойки",
    "MLG Leg - Introduction of a new torque link damper unit":
        "Стойка ОШ — Введение нового демпфера шлиц-шарнира",
    "MLG Leg - Introduction of a new torque link damper":
        "Стойка ОШ — Введение нового демпфера шлиц-шарнира",
    "MLG Leg - Introduction of a new main fitting subassembly and related parts":
        "Стойка ОШ — Введение новой сборки корпуса стойки и связанных деталей",
    "MLG Leg - Introduction of a new main fitting subas":
        "Стойка ОШ — Введение новой сборки корпуса стойки",
    "MLG - Introduction of a new upper pivot bracket":
        "ОШ — Введение нового верхнего поворотного кронштейна",
    "MLG - Introduction of a new changeover valve stem and housing":
        "ОШ — Введение нового штока и корпуса переключающего клапана",
    "MLG - Introduction of a new changeover valve stem ":
        "ОШ — Введение нового штока переключающего клапана",
    "MLG Complete - Modification of the transfer block subassembly":
        "ОШ в сборе — Модификация сборки переходного блока",
    "MLG Complete - Modification of the transfer block ":
        "ОШ в сборе — Модификация переходного блока",
    "MLG - Conversion of low - friction lower - bearing MLG to standard lower - bearing MLG":
        "ОШ — Замена низкофрикционного ОШ с нижним подшипником на стандартный ОШ с нижним подшипником",
    "MLG - Conversion of low - friction lower - bearing":
        "ОШ — Замена низкофрикционного нижнего подшипника",
    "MLG complete - Introduction of a new transfer block subassembly":
        "ОШ в сборе — Введение новой сборки переходного блока",
    "MLG complete - Introduction of a new transfer bloc":
        "ОШ в сборе — Введение нового переходного блока",
}


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSLATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def is_only_numbers_or_codes(text: str) -> bool:
    """Return True if text is just numbers, part numbers, dates, page refs."""
    stripped = text.strip()
    if not stripped:
        return True
    # Pure numbers, dates, page refs, codes
    if re.match(r'^[\d\s,\-/\.\t©®]+$', stripped):
        return True
    # Part numbers like 201587001
    if re.match(r'^[\d\s,]+$', stripped):
        return True
    # Dates like "Dec 6/2019", "Mar 18/2025"
    if re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+/\d+$', stripped):
        return True
    # Tab-separated page ref + date: "601\tDec 6/2019" or "602\tMar 18/2025"
    if re.match(r'^\d+\t(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+/\d+$', stripped):
        return True
    # ATA codes like 32-12-22
    if re.match(r'^\d{2}-\d{2}-\d{2}$', stripped):
        return True
    # Single short codes
    if re.match(r'^[A-Z]?\d+[\-\.]\d+', stripped) and len(stripped) < 15:
        return True
    return False


def translate_toc_entry(text: str) -> str:
    """Translate a TOC entry preserving dot leaders and page numbers."""
    # Handle "Repair No." pattern in TOC (with dot leaders: ". . . .")
    m = re.match(r'^(Repair No\.\s*\d+-\d+)\s+(.*?)(\s*\.\s+\.[\s\.]+.*)', text)
    if m:
        prefix = m.group(1)
        comp_name = m.group(2).strip()
        rest = m.group(3)
        translated_comp = translate_component_name(comp_name)
        prefix_ru = prefix.replace("Repair No.", "Ремонт №")
        rest_ru = rest.replace("Repair No.", "Ремонт №")
        return f"{prefix_ru} {translated_comp}{rest_ru}"

    # Handle "Page Repair No." pattern
    m = re.match(r'^Page\s+(Repair No\.\s*\d+-\d+)\s+(.*?)(\s*\.\s+\.[\s\.]+.*)', text)
    if m:
        prefix = m.group(1)
        comp_name = m.group(2).strip()
        rest = m.group(3)
        translated_comp = translate_component_name(comp_name)
        prefix_ru = prefix.replace("Repair No.", "Ремонт №")
        rest_ru = rest.replace("Repair No.", "Ремонт №")
        return f"Стр. {prefix_ru} {translated_comp}{rest_ru}"

    # Handle figure TOC entries (component name with fig-item refs)
    # Require at least 2 dots (". .") to distinguish from abbreviation dots like "No."
    m = re.match(r'^(.*?)(\s*\.\s+\.[\s\.]+\s*\d*.*)$', text)
    if m:
        name_part = m.group(1).strip()
        rest = m.group(2)
        if name_part in FIXED:
            return f"{FIXED[name_part]}{rest}"
        translated_name = translate_component_name(name_part)
        return f"{translated_name}{rest}"

    # Handle tab-separated entries like "Storage\t798.12"
    m = re.match(r'^(.*?)\t(.*)$', text)
    if m:
        name_part = m.group(1).strip()
        page = m.group(2)
        if name_part in FIXED:
            page_ru = page.replace("Repair No.", "Ремонт №")
            return f"{FIXED[name_part]}\t{page_ru}"
        m_repair = re.match(r'^(Repair No\.\s*\d+-\d+)\s+(.*)$', name_part)
        if m_repair:
            prefix = m_repair.group(1).replace("Repair No.", "Ремонт №")
            comp = translate_component_name(m_repair.group(2).strip())
            page_ru = page.replace("Repair No.", "Ремонт №")
            return f"{prefix} {comp}\t{page_ru}"
        translated_name = translate_component_name(name_part)
        page_ru = page.replace("Repair No.", "Ремонт №")
        return f"{translated_name}\t{page_ru}"

    return text


def _translate_suffix(suffix: str) -> str:
    """Translate common suffixes like 'Protective Treatment (Sheet 1)'."""
    result = suffix
    if "Protective Treatment" in result:
        result = result.replace("Protective Treatment", "Защитная обработка")
    if "Sheet" in result:
        result = re.sub(r'Sheet\s+(\d+)', r'Лист \1', result)
    if result.strip() == "Withdrawn" or result.strip() == "(Withdrawn)":
        result = result.replace("Withdrawn", "Отозвано")
    if "Superseded" in result:
        result = result.replace("Superseded", "Заменён")
    if "Only" in result:
        result = result.replace("Only", "Только")
    if "Tables" in result:
        result = result.replace("Tables", "Таблицы")
    if "Table" in result and "Таблицы" not in result:
        result = result.replace("Table", "Таблица")
    return result


def _lookup_base_name(name: str) -> str:
    """Look up base component name in glossaries."""
    if name in COMPONENT_NAMES:
        return COMPONENT_NAMES[name]
    if name in FIXED:
        return FIXED[name]
    return name


def translate_component_name(name: str) -> str:
    """Translate a component name from the glossary."""
    if name in COMPONENT_NAMES:
        return COMPONENT_NAMES[name]
    if name in FIXED:
        return FIXED[name]

    # ── Pattern: "Name (refs) ... or/and (refs) ... - Suffix" ──
    dash_idx = name.rfind(" - ")
    if dash_idx > 0:
        name_part = name[:dash_idx].strip()
        suffix_part = name[dash_idx + 3:].strip()
        suffix_ru = _translate_suffix(suffix_part)

        paren_idx = name_part.find("(")
        if paren_idx > 0:
            base = name_part[:paren_idx].strip()
            refs = name_part[paren_idx:]
            base_ru = _lookup_base_name(base)
            refs_ru = refs.replace("and", "и").replace("or", "или")
            refs_ru = _translate_suffix(refs_ru)
            return f"{base_ru} {refs_ru} — {suffix_ru}"
        else:
            base_ru = _lookup_base_name(name_part)
            return f"{base_ru} — {suffix_ru}"

    # ── Pattern: "Name (refs)" without suffix ──
    paren_idx = name.find("(")
    if paren_idx > 0:
        base = name[:paren_idx].strip()
        refs = name[paren_idx:]
        base_ru = _lookup_base_name(base)
        refs_ru = _translate_suffix(refs)
        refs_ru = refs_ru.replace(" and ", " и ").replace(" or ", " или ")
        if base_ru != base:
            return f"{base_ru} {refs_ru}"

    # ── Pattern: "(Withdrawn)" suffix ──
    if name.endswith("(Withdrawn)"):
        base = name.replace("(Withdrawn)", "").strip()
        base_ru = translate_component_name(base)
        return f"{base_ru} (Отозвано)"

    return name


def translate_sb_title(text: str) -> str:
    """Translate SB title text."""
    for en, ru in sorted(SB_TITLE_PARTS.items(), key=lambda x: -len(x[0])):
        if text.startswith(en):
            remainder = text[len(en):]
            if remainder:
                return ru + remainder
            return ru
    return text


def translate_reason(text: str) -> str:
    """Translate reason-for-change text in revision tables."""
    result = text
    normalized = re.sub(r'\n', ' ', result)
    # Fix garbled text like "Safran Lиing Systems" -> "Safran Landing Systems"
    normalized = re.sub(r'Safran L[а-яёА-ЯЁ\w]*ing Systems', 'Safran Landing Systems', normalized)
    sorted_phrases = sorted(REASON_PHRASES.keys(), key=len, reverse=True)
    for en in sorted_phrases:
        ru = REASON_PHRASES[en]
        if len(en) <= 4:
            # Word-boundary replacement for short words
            normalized = re.sub(r'\b' + re.escape(en) + r'\b', ru, normalized)
        else:
            normalized = normalized.replace(en, ru)
    return normalized


def translate_repair_description(text: str) -> str:
    """Translate 'Repair to Component — Process' descriptions."""
    result = text
    parts = result.split('\t')
    desc = parts[0]
    rest_parts = '\t'.join(parts[1:]) if len(parts) > 1 else ""

    rest_parts = rest_parts.replace("Repair No.", "Ремонт №")

    desc = desc.replace("Oversize Bushes", "Ремонтные (увеличенные) втулки")
    desc = desc.replace("Oversize Bush(es)", "Ремонтные (увеличенные) втулки")
    desc = desc.replace("Oversize Bush (es)", "Ремонтные (увеличенные) втулки")
    desc = desc.replace("Oversize Bush", "Ремонтная (увеличенная) втулка")
    desc = desc.replace("Oversize Lubrication adapter", "Ремонтный (увеличенный) смазочный адаптер")
    desc = desc.replace("Oversize Transfer Dowel", "Ремонтный (увеличенный) переходной штифт")
    desc = desc.replace("Lower Bearing Subassembly", "Сборка нижнего подшипника")
    desc = desc.replace("Machining and Inner Liner Installation", "Механическая обработка и установка внутреннего вкладыша")
    desc = desc.replace("Machining and Liner Installation", "Механическая обработка и установка вкладыша")
    desc = desc.replace("Repair Bearing", "Ремонт подшипника")
    desc = desc.replace("Repair to ", "Ремонт ")
    desc = desc.replace("Repair Bushes", "Ремонт втулок")
    desc = desc.replace("Repair Bush", "Ремонт втулки")

    for en, ru in sorted(COMPONENT_NAMES.items(), key=lambda x: -len(x[0])):
        desc = desc.replace(en, ru)

    desc = desc.replace("Machining and Installation", "Механическая обработка и установка")
    desc = desc.replace("Machining and installation", "Механическая обработка и установка")
    desc = desc.replace("Machining", "Механическая обработка")
    desc = desc.replace("Installation", "Установка")
    desc = desc.replace("Cadmium Plating", "Кадмирование")
    desc = desc.replace("Chromium Plate Termination", "Граница хромового покрытия")
    desc = re.sub(r'Sheet\s+(\d+)', r'Лист \1', desc)
    desc = desc.replace("Rework", "Доработка")
    desc = desc.replace(" — ", " — ")  # already correct
    desc = desc.replace(" and ", " и ")

    if rest_parts:
        return f"{desc}\t{rest_parts}"
    return desc


def translate_long_text(text: str) -> str:
    """Translate longer text blocks (certification, copyright, etc.)."""
    if text.startswith("This manual complies with British Civil Airworthiness"):
        return "Данное руководство соответствует требованиям Британских авиационных правил лётной годности, раздел A, глава A5-3."

    if text.startswith("NOTE: The above certification does not apply"):
        return ("ПРИМЕЧАНИЕ: Вышеуказанная сертификация не распространяется на изменения или "
                "дополнения, внесённые после даты первичной сертификации другими "
                "одобренными организациями. Изменения или дополнения, внесённые другими "
                "одобренными организациями, должны быть отдельно сертифицированы и "
                "зарегистрированы на отдельных учётных листах.")

    if text.startswith("This document and all information contained herein"):
        return ("Настоящий документ и вся содержащаяся в нём информация являются "
                "исключительной собственностью Safran Landing Systems (и/или аффилированных компаний).")

    if text.startswith("No intellectual property rights are granted"):
        return ("Передача данного документа или раскрытие его содержания не предоставляет "
                "никаких прав на интеллектуальную собственность. Воспроизведение данного "
                "документа для третьих лиц запрещено без письменного согласия Safran Landing Systems "
                "(и/или соответствующей аффилированной компании).")

    if text.startswith("Record the issue date and insertion date"):
        return ("Запишите дату выпуска и дату вставки данной редакции в Запись изменений "
                "и сохраните данное Письмо о рассылке.")

    if text.startswith("The technical data in this document"):
        return ("Технические данные в настоящем документе (или файле) могут содержать данные США "
                "и подлежать экспортному контролю в соответствии с")

    return text


def translate_header_line(text: str) -> str:
    """Translate header/footer running text."""
    m = re.match(r'^PART\s+No\.\s+([\d\s]+)AND\s+([\d]+)\s+COMPONENT MAINTENANCE MANUAL\s+MAIN LANDING GEAR LEG$', text)
    if m:
        return f"ДЕТАЛЬ № {m.group(1)}И {m.group(2)} РУКОВОДСТВО ПО ТЕХНИЧЕСКОМУ ОБСЛУЖИВАНИЮ КОМПОНЕНТОВ СТОЙКА ОСНОВНОГО ШАССИ"

    if "SAFRAN LANDING SYSTEMS" in text and "SUBSEQUENT REVISION PAGE DATES" in text:
        m2 = re.match(r'[©®\s]*SAFRAN LANDING SYSTEMS\s+(\d+)\s+\(AND SUBSEQUENT REVISION PAGE DATES\)', text)
        if m2:
            return f"© SAFRAN LANDING SYSTEMS {m2.group(1)} (И ПОСЛЕДУЮЩИЕ ДАТЫ ПЕРЕСМОТРА СТРАНИЦ)"
        return text.replace("AND SUBSEQUENT REVISION PAGE DATES", "И ПОСЛЕДУЮЩИЕ ДАТЫ ПЕРЕСМОТРА СТРАНИЦ")

    return text


def translate_hf_text(text: str) -> str:
    """Translate header/footer textbox text.

    Handles patterns specific to headers/footers:
    - Section labels (RECORD OF REVISIONS, LIST OF EFFECTIVE PAGES, etc.)
    - "Page X of Y" references
    - "Page X Mar 18/2025" combined page+date
    - "Letter of Transmittal No. N"
    - "TITLE PAGE Sep 16/2016"
    - "COMPONENT MAINTENANCE MANUAL 32-12-22 MAIN LANDING GEAR LEG"
    - Running header "PART No. ... COMPONENT MAINTENANCE MANUAL ..."
    Falls back to translate_text() for anything else.
    """
    stripped = text.strip()
    if not stripped:
        return text

    # Skip already-Russian content
    if re.search(r'[А-Яа-яЁё]{3,}', stripped):
        return text

    # Skip pure dates/numbers/codes
    if is_only_numbers_or_codes(stripped):
        return text

    leading = text[:len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()):]
    core = stripped

    # 1. Exact match in FIXED (handles most section labels)
    if core in FIXED:
        return leading + FIXED[core] + trailing

    # 2. Running header: "PART No. ... COMPONENT MAINTENANCE MANUAL MAIN LANDING GEAR LEG"
    result = translate_header_line(core)
    if result != core:
        return leading + result + trailing

    # 3. "COMPONENT MAINTENANCE MANUAL 32-12-22 MAIN LANDING GEAR LEG"
    m = re.match(r'^COMPONENT\s+MAINTENANCE\s+MANUAL\s+(\d{2}-\d{2}-\d{2})\s+MAIN LANDING GEAR LEG$', core)
    if m:
        return leading + f"РУКОВОДСТВО ПО ТЕХНИЧЕСКОМУ ОБСЛУЖИВАНИЮ КОМПОНЕНТОВ {m.group(1)} СТОЙКА ОСНОВНОГО ШАССИ" + trailing

    # 4. "TITLE PAGE Sep 16/2016" — section label + date
    m = re.match(r'^(TITLE PAGE|RECORD OF REVISIONS|RECORD OF TEMPORARY REVISIONS|'
                 r'LIST OF SERVICE BULLETINS|LIST OF EFFECTIVE PAGES|'
                 r'UNIT IDENTIFICATION CHART|TABLE OF CONTENTS|ILLUSTRATIONS)'
                 r'(?:\s+\(Continued\))?\s+'
                 r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+/\d+)$', core)
    if m:
        label = m.group(1)
        # Check for (Continued) in the original
        has_cont = "(Continued)" in core
        date = m.group(2)
        label_full = label + (" (Continued)" if has_cont else "")
        label_ru = FIXED.get(label_full, FIXED.get(label, label))
        return leading + f"{label_ru} {date}" + trailing

    # 5. "Letter of Transmittal No. N"
    m = re.match(r'^Letter of Transmittal No\.\s*(\d+)$', core)
    if m:
        return leading + f"Письмо о рассылке № {m.group(1)}" + trailing

    # 6. "Page X of Y" (page count in footer)
    m = re.match(r'^Page\s+(\d+)\s+of\s+(\d+)$', core)
    if m:
        return leading + f"Стр. {m.group(1)} из {m.group(2)}" + trailing

    # 7. "Page X Mar 18/2025" — page + date combined
    m = re.match(r'^Page\s+(\d+)\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+/\d+)$', core)
    if m:
        return leading + f"Стр. {m.group(1)} {m.group(2)}" + trailing

    # 8. Standalone "Page X"
    m = re.match(r'^Page\s+(\d+)$', core)
    if m:
        return leading + f"Стр. {m.group(1)}" + trailing

    # 9. "32-12-22 Mar 18/2025" — ATA code + date (keep as-is)
    m = re.match(r'^\d{2}-\d{2}-\d{2}\s+', core)
    if m:
        return text

    # 10. Fallback: use main translate_text()
    return translate_text(text)


def translate_text(text: str) -> str:
    """Main translation function. Returns translated text or original if untranslatable."""
    stripped = text.strip()
    if not stripped:
        return text

    if is_only_numbers_or_codes(stripped):
        return text

    # Skip already-Russian content
    if re.search(r'[А-Яа-яЁё]{3,}', stripped):
        return text

    # Preserve leading/trailing whitespace
    leading = text[:len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()):]
    core = text.strip()

    # 1. Exact match in FIXED
    if core in FIXED:
        return leading + FIXED[core] + trailing

    # 2. Long text translations
    for start_phrase in [
        "This manual complies",
        "NOTE: The above certification",
        "This document and all information",
        "No intellectual property rights",
        "Record the issue date",
        "The technical data in this document",
    ]:
        if core.startswith(start_phrase):
            return leading + translate_long_text(core) + trailing

    # 3. Header line
    if "COMPONENT MAINTENANCE MANUAL" in core or "SUBSEQUENT REVISION PAGE DATES" in core:
        result = translate_header_line(core)
        if result != core:
            return leading + result + trailing

    # 4. Protective Treatment patterns (check BEFORE dot leaders)
    # Skip digit+tab entries which should be handled by the tab handler
    if "Protective Treatment" in core and not re.match(r'^\d+\t', core):
        m_cont = re.match(r'^-\s*(Protective Treatment.*?)(\s*\.[\s\.]+.*)$', core)
        if m_cont:
            suffix_ru = _translate_suffix(m_cont.group(1))
            return leading + f"— {suffix_ru}{m_cont.group(2)}" + trailing
        m_pt = re.match(r'^(Protective Treatment\s*-?\s*.*?)(\s*\.[\s\.]+.*)$', core)
        if m_pt:
            suffix_ru = _translate_suffix(m_pt.group(1))
            return leading + f"{suffix_ru}{m_pt.group(2)}" + trailing
        result = translate_component_name(core)
        if result != core:
            return leading + result + trailing

    # 4b. Tab-separated entries
    if "\t" in core:
        m_lep = re.match(r'^(Repair No\.\s*\d+-\d+)\t(.*)$', core)
        if m_lep:
            prefix = m_lep.group(1).replace("Repair No.", "Ремонт №")
            return leading + f"{prefix}\t{m_lep.group(2)}" + trailing

        m_blank = re.match(r'^(\d+)\tBlank$', core)
        if m_blank:
            return leading + f"{m_blank.group(1)}\tПусто" + trailing

        if core.startswith("Fig."):
            return leading + core.replace("Fig.", "Рис.").replace("Page", "Страница") + trailing

        m_fig = re.match(r'^(\d+)\t(.*)$', core)
        if m_fig:
            fig_num = m_fig.group(1)
            rest = m_fig.group(2)
            if ". . ." in rest:
                m_dots = re.match(r'^(.*?)(\s*\.[\s\.]+\s*)(\t?\d+.*)$', rest)
                if m_dots:
                    comp_part = m_dots.group(1).strip()
                    dots = m_dots.group(2)
                    page = m_dots.group(3)
                    translated_comp = translate_component_name(comp_part)
                    return leading + f"{fig_num}\t{translated_comp}{dots}{page}" + trailing
                m_dots2 = re.match(r'^(.*?)(\s*\.[\s\.]+\s*)$', rest)
                if m_dots2:
                    comp_part = m_dots2.group(1).strip()
                    dots = m_dots2.group(2)
                    m_trailing_dash = re.match(r'^(.*?)\s*-\s*$', comp_part)
                    if m_trailing_dash:
                        inner = m_trailing_dash.group(1).strip()
                        translated_inner = translate_component_name(inner)
                        return leading + f"{fig_num}\t{translated_inner} —{dots}" + trailing
                    translated_comp = translate_component_name(comp_part)
                    return leading + f"{fig_num}\t{translated_comp}{dots}" + trailing
            else:
                translated_rest = translate_component_name(rest.strip())
                return leading + f"{fig_num}\t{translated_rest}" + trailing

        if not re.match(r'^\d', core):
            return leading + translate_toc_entry(core) + trailing

        return text

    # 5. TOC entries (contain dot leaders, no tabs)
    if ". . ." in core:
        return leading + translate_toc_entry(core) + trailing

    # 6. "Fig.\tPage" header (no-tab variant)
    if core.startswith("Fig."):
        return leading + core.replace("Fig.", "Рис.").replace("Page", "Страница") + trailing

    # 7. Component name with Protective Treatment (fallback for non-dot cases)
    if "Protective Treatment" in core:
        result = translate_component_name(core)
        if result != core:
            return leading + result + trailing

    # 8. "Repair No." entries (outside TOC)
    if core.startswith("Repair No."):
        m = re.match(r'^(Repair No\.\s*\d+-\d+)\s+(.*?)$', core)
        if m:
            comp = m.group(2).strip()
            translated_comp = translate_component_name(comp)
            prefix = m.group(1).replace("Repair No.", "Ремонт №")
            return leading + f"{prefix} {translated_comp}" + trailing
        return leading + core.replace("Repair No.", "Ремонт №") + trailing

    # 9. "Page X" reference
    if re.match(r'^Page\s+\d+', core):
        return leading + core.replace("Page", "Стр.") + trailing

    # 10. Exact match in component names
    if core in COMPONENT_NAMES:
        return leading + COMPONENT_NAMES[core] + trailing

    # 10b. Component name with pattern matching
    result = translate_component_name(core)
    if result != core:
        return leading + result + trailing

    # 10c. Component name + "Only" suffix
    if core.endswith(" Only"):
        base = core[:-5].strip()
        base_ru = translate_component_name(base)
        if base_ru != base:
            return leading + f"{base_ru} Только" + trailing

    # 10d. "- ComponentName" continuation lines
    m_cont = re.match(r'^-\s+(.+)$', core)
    if m_cont:
        inner = m_cont.group(1).strip()
        inner_ru = translate_component_name(inner)
        if inner_ru != inner:
            return leading + f"— {inner_ru}" + trailing

    # 11. "Blank" as page status
    if core == "Blank":
        return leading + "Пусто" + trailing

    return text


def translate_table_cell_text(text: str) -> str:
    """Translate text within a table cell."""
    stripped = text.strip()
    if not stripped:
        return text

    clean = stripped.strip('\n').strip()

    # Table headers
    if clean in TABLE_HEADERS:
        return text.replace(clean, TABLE_HEADERS[clean])

    # Section names in revision tables
    if clean in SECTION_NAMES:
        return text.replace(clean, SECTION_NAMES[clean])

    # "Repair No." section refs
    if clean.startswith("Repair No."):
        return text.replace("Repair No.", "Ремонт №")

    # SB titles
    for en_prefix in sorted(SB_TITLE_PARTS.keys(), key=len, reverse=True):
        if clean.startswith(en_prefix):
            return text.replace(clean, translate_sb_title(clean))

    # SB revision numbers
    if clean == "Initial Issue":
        return text.replace(clean, "Первоначальный выпуск")
    if clean == "No effect":
        return text.replace(clean, "Без изменений")

    # "Updated paras X.Y" type entries
    if clean.startswith("Updated paras"):
        return text.replace("Updated paras", "Обновлены пункты")
    if clean.startswith("Updated para"):
        return text.replace("Updated para", "Обновлён пункт")

    # Standalone continuation fragments
    if clean.startswith("para ") or clean.startswith("paras "):
        result = clean.replace("paras ", "пунктов ").replace("para ", "пункта ")
        result = translate_reason(result)
        return text.replace(clean, result)

    # Reason for change
    if any(phrase in clean for phrase in ["Updated", "Added", "Updated fig", "Updated Messier"]):
        return text.replace(clean, translate_reason(clean))

    # Standalone "figure NNN" or "figures NNN" (continuation lines)
    m_fig = re.match(r'^(figures?)\s+(.*)$', clean)
    if m_fig:
        word = "рисунки" if m_fig.group(1) == "figures" else "рисунок"
        return text.replace(clean, f"{word} {m_fig.group(2)}")

    # Standalone continuation fragments from multi-paragraph reason cells
    if clean.startswith("in para"):
        return text.replace("in para", "в пункте")
    if clean.startswith("in paras"):
        return text.replace("in paras", "в пунктах")

    # "Sliding tube" entries in figure tables
    if "Sliding tube" in clean or "sliding tube" in clean:
        result = clean
        result = result.replace("Sliding tube", "Скользящая труба")
        result = result.replace("sliding tube", "Скользящая труба")
        result = _translate_suffix(result)
        result = result.replace(" and ", " и ").replace(" or ", " или ")
        return text.replace(clean, result)

    # Repair/Oversize descriptions
    if (clean.startswith("Repair to ") or clean.startswith("Repair Bush") or
        clean.startswith("Repair Bearing") or clean.startswith("Oversize ") or
        clean.startswith("Lower Bearing")):
        fixed = re.sub(r'Installation(Repair No\.)', r'Installation\t\1', clean)
        result = translate_repair_description(fixed)
        return text.replace(clean, result)

    # "Component Repairs - Key Diagram" pattern
    if "Repairs - Key Diagram" in clean or "Key Diagram" in clean:
        result = clean
        result = result.replace("Approved Repairs - Key Diagram", "Утверждённые ремонты — Ключевая схема")
        result = result.replace("Repairs - Key Diagram", "Ремонты — Ключевая схема")
        for en, ru in sorted(COMPONENT_NAMES.items(), key=lambda x: -len(x[0])):
            result = result.replace(en, ru)
        return text.replace(clean, result)

    # Continuation fragments from multi-paragraph reason cells
    if clean.startswith("Limited to Safran"):
        result = clean
        result = re.sub(r'Safran L[а-яёА-ЯЁ\w]*ing Systems', 'Safran Landing Systems', result)
        result = result.replace("Limited to Safran Landing Systems", "Limited на Safran Landing Systems")
        return text.replace(clean, result)

    # Numbers with "and" (continuation lines): "814, 818 and 823"
    if re.match(r'^[\d,\s]+ and [\d,\s]+$', clean):
        return text.replace(" and ", " и ")

    # Standalone short words in table cells
    if clean == "and":
        return text.replace("and", "и")
    if clean == "REV":
        return text.replace("REV", "РЕД")
    if clean == "Only":
        return text.replace("Only", "Только")
    if clean == "Tables":
        return text.replace("Tables", "Таблицы")

    # "NNN to NNN and" continuation pattern
    if re.match(r'^[\d\s]+to\s+[\d\s]+and$', clean):
        return text.replace(" to ", " до ").replace(" and", " и")

    # Page number ranges with "to": "601 to 603", "504, 507 to 512", "704 to" (continuation)
    if re.search(r'\bto\b', clean) and re.search(r'\d', clean):
        result = re.sub(r'\bto\b', 'до', text)
        result = re.sub(r'\band\b', 'и', result)
        return result

    # Page number ranges with "and" only: "709 and 729"
    if re.match(r'^[\d,\.\s]+(and\s+[\d,\.\s]*)+$', clean):
        return re.sub(r'\band\b', 'и', text)

    # Generic fallback to main translator
    result = translate_text(text)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  XML-LEVEL TEXT REPLACEMENT WITH FORMATTING PRESERVATION
# ══════════════════════════════════════════════════════════════════════════════

def replace_paragraph_text(para, new_text: str):
    """Replace all text in a paragraph while preserving formatting.

    Works at XML level to handle text inside hyperlinks and other structures.
    Handles tabs properly: if original XML has <w:tab/> elements, splits text
    at tab positions so each segment goes into the correct w:t element.
    Removes empty runs after replacement to keep XML clean.
    """
    element = para._element

    all_t_elements = list(element.iter(qn('w:t')))
    if not all_t_elements:
        return

    # Handle tabs
    tab_runs = []
    for r_elem in element.iter(qn('w:r')):
        if r_elem.find(qn('w:tab')) is not None:
            tab_runs.append(r_elem)

    if tab_runs and '\t' in new_text:
        segments = new_text.split('\t')
        t_groups = [[]]
        for child in element.iter():
            if child.tag == qn('w:r') and child in tab_runs:
                t_groups.append([])
            elif child.tag == qn('w:t'):
                t_groups[-1].append(child)

        for seg_idx, segment in enumerate(segments):
            if seg_idx < len(t_groups) and t_groups[seg_idx]:
                t_groups[seg_idx][0].text = segment
                t_groups[seg_idx][0].set(qn('xml:space'), 'preserve')
                for t_elem in t_groups[seg_idx][1:]:
                    _remove_run_if_empty(t_elem)
            elif seg_idx < len(t_groups):
                if seg_idx > 0 and seg_idx - 1 < len(tab_runs) and segment.strip():
                    tab_run = tab_runs[seg_idx - 1]
                    new_t = etree.SubElement(tab_run, qn('w:t'))
                    new_t.text = segment
                    new_t.set(qn('xml:space'), 'preserve')

        for gi in range(len(segments), len(t_groups)):
            for t_elem in t_groups[gi]:
                _remove_run_if_empty(t_elem)

        needed_tabs = len(segments) - 1
        if len(tab_runs) > needed_tabs:
            for extra_tab_run in tab_runs[needed_tabs:]:
                parent = extra_tab_run.getparent()
                if parent is not None:
                    parent.remove(extra_tab_run)
        return

    # Strategy: put all text in the first non-empty w:t, clear others
    ref_idx = 0
    for i, t_elem in enumerate(all_t_elements):
        if t_elem.text and t_elem.text.strip():
            ref_idx = i
            break

    all_t_elements[ref_idx].text = new_text
    all_t_elements[ref_idx].set(qn('xml:space'), 'preserve')

    for i, t_elem in enumerate(all_t_elements):
        if i != ref_idx:
            _remove_run_if_empty(t_elem)


def _remove_run_if_empty(t_elem):
    """Remove a w:t element's parent run if it's empty, otherwise clear text."""
    run_elem = t_elem.getparent()
    if run_elem is not None and run_elem.tag == qn('w:r'):
        children = [c for c in run_elem if c.tag != qn('w:rPr')]
        if len(children) == 1 and children[0] is t_elem:
            run_parent = run_elem.getparent()
            if run_parent is not None:
                run_parent.remove(run_elem)
                return
    t_elem.text = ""


def process_paragraph(para):
    """Translate a paragraph's text content."""
    original = para.text
    if not original or not original.strip():
        return
    translated = translate_text(original)
    if translated != original:
        replace_paragraph_text(para, translated)


def process_table_cell(cell):
    """Translate all paragraphs within a table cell."""
    for para in cell.paragraphs:
        original = para.text
        if not original or not original.strip():
            continue
        translated = translate_table_cell_text(original)
        if translated != original:
            replace_paragraph_text(para, translated)


# ══════════════════════════════════════════════════════════════════════════════
#  FONT SIZE ADJUSTMENT
# ══════════════════════════════════════════════════════════════════════════════

def _get_font_size_from_element(element):
    """Get font size from a w:r element's w:rPr/w:sz."""
    rPr = element.find(qn('w:rPr'))
    if rPr is not None:
        sz = rPr.find(qn('w:sz'))
        if sz is not None:
            val = sz.get(qn('w:val'))
            if val:
                return int(val)  # half-points
    return None


def _set_font_size_on_element(element, half_points: int):
    """Set font size on a w:r element."""
    rPr = element.find(qn('w:rPr'))
    if rPr is None:
        rPr = etree.SubElement(element, qn('w:rPr'))
        element.insert(0, rPr)
    sz = rPr.find(qn('w:sz'))
    if sz is None:
        sz = etree.SubElement(rPr, qn('w:sz'))
    sz.set(qn('w:val'), str(half_points))
    szCs = rPr.find(qn('w:szCs'))
    if szCs is None:
        szCs = etree.SubElement(rPr, qn('w:szCs'))
    szCs.set(qn('w:val'), str(half_points))


def adjust_font_if_needed(para, original_text: str, new_text: str,
                          default_half_pts: int = 18, in_table: bool = False):
    """Reduce font size if Russian text is significantly longer."""
    if not original_text.strip() or not new_text.strip():
        return
    if new_text == original_text:
        return

    ratio = len(new_text) / max(len(original_text), 1)
    threshold = 1.2 if in_table else 1.3
    if ratio <= threshold:
        return

    element = para._element
    runs = list(element.iter(qn('w:r')))
    if not runs:
        return

    for run_elem in runs:
        current_hp = _get_font_size_from_element(run_elem)
        if current_hp is None:
            current_hp = default_half_pts

        if ratio > 1.7:
            factor = 0.78
        elif ratio > 1.5:
            factor = 0.82
        elif ratio > 1.3:
            factor = 0.88
        else:
            factor = 0.92

        new_hp = max(int(current_hp * factor), 12)  # minimum 6pt
        _set_font_size_on_element(run_elem, new_hp)


# ══════════════════════════════════════════════════════════════════════════════
#  POST-PROCESSING: FORMATTING FIXES
# ══════════════════════════════════════════════════════════════════════════════

def post_process_formatting(doc, table_font_half_pts: int = 18) -> int:
    """Fix formatting issues after translation. Returns count of fixes applied.

    table_font_half_pts: target font size for ALL table cell text in half-points.
                         18 = 9pt. Set to 0 to skip font normalization.
    """
    fix_count = 0

    for ti, table in enumerate(doc.tables):
        for ri, row in enumerate(table.rows):
            for ci, cell in enumerate(row.cells):
                for para in cell.paragraphs:
                    text = para.text
                    if not text:
                        continue

                    # Fix 1: Remove trailing tabs in TOC repair entries
                    if text.rstrip() != text.rstrip('\t') and '\t' in text:
                        stripped = text.rstrip('\t')
                        if stripped != text:
                            t_elems = list(para._element.iter(qn('w:t')))
                            for t_elem in t_elems:
                                if t_elem.text and t_elem.text.endswith('\t'):
                                    t_elem.text = t_elem.text.rstrip('\t')
                                    fix_count += 1

                    # Fix 2: Normalize table font size to target
                    if table_font_half_pts > 0:
                        element = para._element
                        for run_elem in element.iter(qn('w:r')):
                            current_hp = _get_font_size_from_element(run_elem)
                            if current_hp != table_font_half_pts:
                                _set_font_size_on_element(run_elem, table_font_half_pts)
                                fix_count += 1

    return fix_count


# ══════════════════════════════════════════════════════════════════════════════
#  VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def verify_translation(dst_path: str, verbose: bool = True) -> dict:
    """Verify translated document for remaining English text.

    Returns dict with 'untranslated_paragraphs' and 'untranslated_cells' lists.
    """
    doc = Document(str(dst_path))
    result = {"untranslated_paragraphs": [], "untranslated_cells": []}

    # Check paragraphs
    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if not text:
            continue
        if re.search(r'[A-Za-z]{4,}', text) and not is_only_numbers_or_codes(text):
            if any(skip in text for skip in [
                "Safran Landing Systems UK",
                "Cheltenham Road",
                "201587", "EDES2", "MAF1",
                "(c)AC", "A320-", "A321-",
                "32-12-22", "AMS-", "ASTM",
                "MLG", "NLG",
            ]):
                continue
            if re.search(r'[А-Яа-яЁё]{3,}', text):
                continue
            result["untranslated_paragraphs"].append((i, text[:100]))

    if verbose:
        if result["untranslated_paragraphs"]:
            print(f"Found {len(result['untranslated_paragraphs'])} potentially untranslated paragraphs:")
            for idx, txt in result["untranslated_paragraphs"][:20]:
                print(f"  P{idx}: {txt}")
        else:
            print("All paragraphs appear to be translated!")

    # Check table cells
    for ti, table in enumerate(doc.tables):
        for ri, row in enumerate(table.rows):
            for ci, cell in enumerate(row.cells):
                for para in cell.paragraphs:
                    text = para.text.strip()
                    if not text:
                        continue
                    if re.search(r'[A-Za-z]{4,}', text) and not is_only_numbers_or_codes(text):
                        if any(skip in text for skip in [
                            "SAFRAN LANDING SYSTEMS",
                            "Safran Landing Systems",
                            "201587", "EDES2", "MAF1",
                            "(c)AC", "A320-", "A321-",
                            "32-12-22", "AMS-", "ASTM",
                            "MLG", "NLG", "Messier",
                            "SERVICE BULLETIN",
                        ]):
                            continue
                        if re.search(r'[А-Яа-яЁё]{3,}', text):
                            continue
                        result["untranslated_cells"].append((f"T{ti+1}R{ri+1}C{ci+1}", text[:120]))

    if verbose:
        if result["untranslated_cells"]:
            print(f"\nFound {len(result['untranslated_cells'])} potentially untranslated table cells:")
            for loc, txt in result["untranslated_cells"][:20]:
                print(f"  {loc}: {txt}")
        else:
            print("All table cells appear to be translated!")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRANSLATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def translate_document(src: str, dst: str, table_font_half_pts: int = 18):
    """Translate a CMM document from English to Russian.

    Args:
        src: Path to the source .docx file
        dst: Path for the output translated .docx file
        table_font_half_pts: Target font size for table cells (18 = 9pt).
                             Set to 0 to skip font normalization.
    """
    src_path = Path(src)
    dst_path = Path(dst)

    print(f"Loading: {src_path}")
    doc = Document(str(src_path))

    # ── Process paragraphs ──
    translated_count = 0
    for i, para in enumerate(doc.paragraphs):
        original = para.text
        if not original or not original.strip():
            continue
        translated = translate_text(original)
        if translated != original:
            replace_paragraph_text(para, translated)
            adjust_font_if_needed(para, original, translated)
            translated_count += 1

    print(f"Translated {translated_count} paragraphs")

    # ── Process tables ──
    table_translated = 0
    for ti, table in enumerate(doc.tables):
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    original = para.text
                    if not original or not original.strip():
                        continue
                    translated = translate_table_cell_text(original)
                    if translated != original:
                        replace_paragraph_text(para, translated)
                        adjust_font_if_needed(para, original, translated, in_table=True)
                        table_translated += 1

    print(f"Translated {table_translated} table cells")

    # ── Post-processing: fix formatting issues ──
    fix_count = post_process_formatting(doc, table_font_half_pts)
    print(f"Applied {fix_count} formatting fixes")

    # ── Process headers and footers ──
    # Text in headers/footers is typically inside textboxes (w:txbxContent),
    # not in simple paragraphs. We process both levels.
    hf_count = 0
    from docx.text.paragraph import Paragraph as _Paragraph
    for section in doc.sections:
        hf_elements = [
            section.header, section.first_page_header, section.even_page_header,
            section.footer, section.first_page_footer, section.even_page_footer,
        ]
        for hf in hf_elements:
            if hf.is_linked_to_previous:
                continue
            # Process textboxes inside header/footer
            for txbx in hf._element.iter(qn('w:txbxContent')):
                for p_elem in txbx.iter(qn('w:p')):
                    para = _Paragraph(p_elem, doc)
                    original = para.text
                    if original and original.strip():
                        translated = translate_hf_text(original)
                        if translated != original:
                            replace_paragraph_text(para, translated)
                            hf_count += 1
            # Also check direct paragraphs (fallback)
            for para in hf.paragraphs:
                original = para.text
                if original and original.strip():
                    translated = translate_hf_text(original)
                    if translated != original:
                        replace_paragraph_text(para, translated)
                        hf_count += 1

    print(f"Translated {hf_count} header/footer elements")

    # ── Process textboxes ──
    tb_count = 0
    body = doc.element.body
    for txbx in body.iter(qn('w:txbxContent')):
        for p in txbx.iter(qn('w:p')):
            from docx.text.paragraph import Paragraph
            para = Paragraph(p, doc)
            original = para.text
            if original and original.strip():
                translated = translate_text(original)
                if translated != original:
                    replace_paragraph_text(para, translated)
                    adjust_font_if_needed(para, original, translated, in_table=True)
                    tb_count += 1

    print(f"Translated {tb_count} textbox elements")

    # ── Save ──
    os.makedirs(str(dst_path.parent), exist_ok=True)
    doc.save(str(dst_path))
    print(f"Saved: {dst_path}")

    # ── Verification ──
    print("\n=== Verification ===")
    verify_translation(str(dst_path))
