from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from docx import Document
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
DOCXRU_SRC = ROOT / ".claude" / "worktrees" / "competent-jones" / "src"
if str(DOCXRU_SRC) not in sys.path:
    sys.path.insert(0, str(DOCXRU_SRC))

from docxru.com_word import update_fields_via_com
from docxru.config import PipelineConfig
from docxru.docx_reader import collect_segments
from docxru.layout_check import validate_layout
from docxru.layout_fix import fix_expansion_issues
from docxru.tagging import _apply_style, _clear_paragraph_runs, paragraph_to_tagged


LATIN_RE = re.compile(r"[A-Za-z]")
CODE_LIKE_RE = re.compile(
    r"^(?:"
    r"[0-9.,/ -]+|"
    r"[A-Z]{1,6}[0-9A-Z./-]*|"
    r"\([a-z]\)[A-Z0-9-]+|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}/\d{4}|"
    r"Page\s+\d+\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}/\d{4}|"
    r"[0-9]+\s+to\s+[0-9.]+(?:\s+and\s+[0-9.]+)?|"
    r"[0-9]+\s+and\s+[0-9]+|"
    r"[0-9]+\s+\t\s+[0-9A-Za-z./ -]+"
    r")$"
)
RESERVED_UNCHANGED = {
    "©",
    "SAFRAN",
    "SAFRAN LANDING SYSTEMS",
    "CAGE: K0654",
    "M-D",
    "Safran Landing Systems UK Ltd",
    "SAFRAN LANDING SYSTEMS UK Ltd",
    "Cheltenham Road, Gloucester, GL2 9QH, England",
    "EDES2-0005-2253",
    "EDES2-0005-2255",
    "EDES2-0005-2542",
    "MAF1-0005-2155",
    "MAF1-0005-2156",
    "MAF1-0005-2157",
    "MAF1-0005-2160",
    "MAF1-0005-2161",
    "MAF1-0005-2162",
    "MAF1-0005-2164",
    "MAF1-0005-2180",
    "MAF1-0005-2207",
}

MONTH_MAP: dict[str, str] = {
    "Jan": "янв.",
    "Feb": "февр.",
    "Mar": "мар.",
    "Apr": "апр.",
    "May": "мая",
    "Jun": "июн.",
    "Jul": "июл.",
    "Aug": "авг.",
    "Sep": "сент.",
    "Oct": "окт.",
    "Nov": "нояб.",
    "Dec": "дек.",
}

MONTH_NUMBER: dict[str, str] = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12",
}


EXACT_MAP: dict[str, str] = {
    "MAIN LANDING GEAR LEG": "СТОЙКА ОСНОВНОГО ШАССИ",
    "PART NUMBER": "НОМЕР ДЕТАЛИ",
    "COMPONENT MAINTENANCE MANUAL WITH": "РУКОВОДСТВО ПО ТЕХНИЧЕСКОМУ ОБСЛУЖИВАНИЮ КОМПОНЕНТА С",
    "ILLUSTRATED PARTS LIST": "ИЛЛЮСТРИРОВАННЫЙ ПЕРЕЧЕНЬ ДЕТАЛЕЙ",
    "STATEMENT OF INITIAL CERTIFICATION": "ЗАЯВЛЕНИЕ О ПЕРВОНАЧАЛЬНОЙ СЕРТИФИКАЦИИ",
    "This manual complies with British Civil Airworthiness Requirements, Section A, Chapter A5-3.": (
        "Настоящее руководство соответствует Британским требованиям летной годности гражданской авиации, раздел A, глава A5-3."
    ),
    "NOTE: The above certification does not apply to revisions or amendments made after the date of initial certification by other Approved Organisations. Revisions or Amendments made by other Approved Organisations must each be separately certified and recorded on separate record sheets.": (
        "ПРИМЕЧАНИЕ. Указанная выше сертификация не распространяется на ревизии или изменения, внесенные после даты первоначальной сертификации другими одобренными организациями. Каждая ревизия или изменение, выполненные другими одобренными организациями, должны быть сертифицированы отдельно и зарегистрированы на отдельных листах учета."
    ),
    "SAFRAN LANDING SYSTEMS 2016 (AND SUBSEQUENT REVISION PAGE DATES)": "SAFRAN LANDING SYSTEMS 2016 (И ПОСЛЕДУЮЩИЕ ДАТЫ РЕВИЗИОННЫХ СТРАНИЦ)",
    "SAFRAN LANDING SYSTEMS 2025 (AND SUBSEQUENT REVISION PAGE DATES)": "SAFRAN LANDING SYSTEMS 2025 (И ПОСЛЕДУЮЩИЕ ДАТЫ РЕВИЗИОННЫХ СТРАНИЦ)",
    "© SAFRAN LANDING SYSTEMS 2016 (AND SUBSEQUENT REVISION PAGE DATES)": "© SAFRAN LANDING SYSTEMS 2016 (И ПОСЛЕДУЮЩИЕ ДАТЫ РЕВИЗИОННЫХ СТРАНИЦ)",
    "© SAFRAN LANDING SYSTEMS 2025 (AND SUBSEQUENT REVISION PAGE DATES)": "© SAFRAN LANDING SYSTEMS 2025 (И ПОСЛЕДУЮЩИЕ ДАТЫ РЕВИЗИОННЫХ СТРАНИЦ)",
    "This document and all information contained herein is the sole property of Safran Landing Systems (and/or its affiliated companies).": (
        "Настоящий документ и вся содержащаяся в нем информация являются исключительной собственностью Safran Landing Systems (и/или ее аффилированных компаний)."
    ),
    "No intellectual property rights are granted by the delivery of this document or the disclosure of its content. This document shall not be reproduced to a third party without the express written consent of Safran Landing Systems (and/or the appropriate affiliated company).": (
        "Передача настоящего документа или раскрытие его содержания не предоставляет никаких прав интеллектуальной собственности. Воспроизведение документа для третьих лиц допускается только при наличии прямого письменного согласия Safran Landing Systems (и/или соответствующей аффилированной компании)."
    ),
    "PART No. 201587001 AND 201587002 COMPONENT MAINTENANCE MANUAL MAIN LANDING GEAR LEG": (
        "ДЕТАЛЬ № 201587001 И 201587002 РУКОВОДСТВО ПО ТЕХНИЧЕСКОМУ ОБСЛУЖИВАНИЮ КОМПОНЕНТА СТОЙКА ОСНОВНОГО ШАССИ"
    ),
    "INTENTIONALLY BLANK": "ПРЕДНАМЕРЕННО ОСТАВЛЕНО ПУСТЫМ",
    "NEW/REVISED PAGES": "НОВЫЕ/ПЕРЕСМОТРЕННЫЕ СТРАНИЦЫ",
    "REVISION RECORD": "ЖУРНАЛ РЕВИЗИЙ",
    "RECORD OF REVISIONS": "ЖУРНАЛ РЕВИЗИЙ",
    "RECORD OF TEMPORARY REVISIONS": "ЖУРНАЛ ВРЕМЕННЫХ РЕВИЗИЙ",
    "LIST OF SERVICE BULLETINS": "СПИСОК СЕРВИСНЫХ БЮЛЛЕТЕНЕЙ",
    "LIST OF EFFECTIVE PAGES": "ПЕРЕЧЕНЬ ДЕЙСТВУЮЩИХ СТРАНИЦ",
    "TABLE OF CONTENTS": "СОДЕРЖАНИЕ",
    "ILLUSTRATIONS": "ИЛЛЮСТРАЦИИ",
    "Record the issue date and insertion date of this revision in the Record of Revisions and retain this Letter of Transmittal.": (
        "Зарегистрируйте дату выпуска и дату внесения настоящей ревизии в журнале ревизий и сохраните данное сопроводительное письмо."
    ),
    "Issued by": "ВЫПУЩЕНО",
    "Cheltenham Road, Gloucester, GL2 9QH, England Telephone: +44 (0) 1452 712424 Fax: +44 (0) 1452 713821": (
        "Cheltenham Road, Gloucester, GL2 9QH, England. Тел.: +44 (0) 1452 712424. Факс: +44 (0) 1452 713821"
    ),
    "Telephone: +44 (0) 1452 712424 www.safran-landing-systems.com": "Тел.: +44 (0) 1452 712424  www.safran-landing-systems.com",
    "UNIT IDENTIFICATION CHART": "ТАБЛИЦА ИДЕНТИФИКАЦИИ АГРЕГАТА",
    "COMPONENT MAINTENANCE MANUAL 32-12-22 MAIN LANDING GEAR LEG": (
        "РУКОВОДСТВО ПО ТЕХНИЧЕСКОМУ ОБСЛУЖИВАНИЮ КОМПОНЕНТА 32-12-22 СТОЙКА ОСНОВНОГО ШАССИ"
    ),
    "Fig.\tPage": "Рис.\tСтр.",
    "Fig.	Page": "Рис.\tСтр.",
    "Fig. Page": "Рис.\tСтр.",
    "Рис.\tPage": "Рис.\tСтр.",
    "Рис. Page": "Рис.\tСтр.",
    "Page": "Стр.",
    "Date": "Дата",
    "Blank": "Пусто",
    "604 Blank": "604\tПусто",
    "Pages (Continued)": "Страницы (Продолжение)",
    "Pages (Продолжение)": "Страницы (Продолжение)",
    "Initial Issue": "Первоначальный выпуск",
    "No effect": "Без влияния",
    "This document and all information contained herein is the sole property of Safran Landing Systems (and/or its affiliated companies).": (
        "Настоящий документ и вся содержащаяся в нем информация являются исключительной собственностью Safran Landing Systems (и/или ее аффилированных компаний)."
    ),
    "The technical data in this document (or file) may contain US data and be controlled for export under the Export Administration Regulations (EAR), 15 CFR Parts 730-774, ECCN: 9E991. Violations of these laws may be subject to fines and penalties under the Export Administration Act.": (
        "Технические данные, содержащиеся в настоящем документе (или файле), могут включать данные США и подпадать под экспортный контроль в соответствии с Правилами экспортного администрирования (EAR), 15 CFR части 730-774, ECCN: 9E991. Нарушение этих требований может повлечь штрафы и иные санкции в соответствии с Законом об экспортном администрировании."
    ),
    "MLG - Installation of stub bolt subassembly for the forward pintle pin in place of the cross bolt.": (
        "MLG - Установка подсборки короткого болта для переднего штифта навеса стойки вместо поперечного болта."
    ),
    "MLG - To allow an increase in aircraft maximum take-off weight to 93 tonne.": (
        "MLG - Обеспечение увеличения максимальной взлетной массы самолета до 93 т."
    ),
    "MLG -To add tracking numbers to parts listed in Airbus Airworthiness Limitations Section (ALS).": (
        "MLG - Добавление номеров отслеживания к деталям, перечисленным в разделе ограничений летной годности Airbus (ALS)."
    ),
    "MLG - Installation of a 201585 series MLG Leg and Dressings where a 201387 MLG Leg and Dressings has been installed.": (
        "MLG - Установка стойки MLG серии 201585 и ее навесных элементов вместо ранее установленной стойки MLG серии 201387 с навесными элементами."
    ),
    "MLG -To add tracking numbers to parts listed in Airbus Maintenance Planning Document, Section 9-1. (Torque link apex pin nut)": (
        "MLG - Добавление номеров отслеживания к деталям, перечисленным в разделе 9-1 документа планирования технического обслуживания Airbus (гайка штифта вершины шлиц-шарнира)."
    ),
    "MLG - Introduction of a new lower bearing subassembly.": "MLG - Введение новой сборки нижнего подшипника.",
    "MLG - Introduction of new charging labels": "MLG - Введение новых табличек заправки.",
    "MLG - Introduction of new 1M and 2M Axle harnesses": "MLG - Введение новых жгутов оси 1M и 2M.",
    "MLG - Introduction of new 1M and 2M Leg Harness and of new 1M and 2M Axle Harnesses": (
        "MLG - Введение новых жгутов стойки 1M и 2M и новых жгутов оси 1M и 2M."
    ),
    "MLG Leg-Introduction of new retaining pins and a new lower bearing subassembly with a new self lubricating liner": (
        "Стойка MLG - Введение новых стопорных штифтов и новой сборки нижнего подшипника с новым самосмазывающимся вкладышем."
    ),
    "MLG Leg - Introduction of new retaining pins for the lower bearing subassembly": (
        "Стойка MLG - Введение новых стопорных штифтов для сборки нижнего подшипника."
    ),
    "MLG Leg - Introduction of a new lower bearing subassembly with a new low friction inner liner": (
        "Стойка MLG - Введение новой сборки нижнего подшипника с новым внутренним вкладышем с низким коэффициентом трения."
    ),
    "MLG Leg - Barkhausen Noise Inspection of Main Landing Gear Sliding Tube Axles.": (
        "Стойка MLG - Контроль методом шума Баркгаузена осей скользящей трубы основной стойки шасси."
    ),
    "MLG Leg - Introduction of a new Main Fitting": "Стойка MLG - Введение нового корпуса стойки.",
    "MLG Leg - Introduction of a new torque link damper unit": (
        "Стойка MLG - Введение нового демпферного узла шлиц-шарнира."
    ),
    "MLG Leg - Introduction of a new main fitting subassembly and related parts": (
        "Стойка MLG - Введение новой сборки корпуса стойки и сопутствующих деталей."
    ),
    "MLG - Introduction of a new upper pivot bracket": "MLG - Введение нового верхнего кронштейна шарнира.",
    "MLG - Introduction of a new changeover valve stem and housing": (
        "MLG - Введение нового штока и корпуса переключающего клапана."
    ),
    "MLG Complete - Modification of the transfer block subassembly": (
        "MLG в сборе - Модификация подсборки переходного блока."
    ),
    "MLG - Conversion of low - friction lower - bearing MLG to standard lower - bearing MLG": (
        "MLG - Переоборудование MLG с нижним подшипником пониженного трения в стандартную конфигурацию нижнего подшипника."
    ),
    "MLG complete - Introduction of a new transfer block subassembly": (
        "MLG в сборе - Введение новой подсборки переходного блока."
    ),
    "Updated revision status": "Обновлен статус ревизии",
    "Added Ref. Codes 2253 and 2255 details": "Добавлены сведения по кодам ссылок 2253 и 2255",
    "Updated pages": "Обновлены страницы",
    "Added content. Updated page numbers. Updated figure numbers": (
        "Добавлено содержимое. Обновлены номера страниц. Обновлены номера рисунков"
    ),
    "Added para 2.P.(31)": "Добавлен п. 2.P.(31)",
    "Updated tables 501 and 502": "Обновлены таблицы 501 и 502",
    "Updated table 601. Updated caution at para": "Обновлена таблица 601. Обновлено предупреждение в п.",
    "3.C. Updated figure titles. Deleted figures 626, 627, 649, 650, 653": (
        "3.C. Обновлены наименования рисунков. Удалены рисунки 626, 627, 649, 650, 653"
    ),
    "and 654. Updated figure 626. Added figures 642": "и 654. Обновлен рисунок 626. Добавлены рисунки 642",
    "648. Updated table 602. Updated figure numbers": "648. Обновлена таблица 602. Обновлены номера рисунков",
    "Added fig-item (18-80A) in para 1. Updated Messier-Dowty Limited to Safran Landing Systems": (
        "Добавлена позиция рисунка (18-80A) в п. 1. Наименование Messier-Dowty Limited заменено на Safran Landing Systems"
    ),
    "Added fig-item (18-80A) in paras 1. Updated Messier-Dowty Limited to Safran Landing Systems": (
        "Добавлена позиция рисунка (18-80A) в пп. 1. Наименование Messier-Dowty Limited заменено на Safran Landing Systems"
    ),
    "Added fig-item (18-80A) in paras 1. and 1.A.(2)": "Добавлена позиция рисунка (18-80A) в пп. 1 и 1.A.(2)",
    "Added fig-item (18-80A) in para 1. Updated Messier-Dowty Limited to Safran Landing Systems. Updated conversion value in figure 602": (
        "Добавлена позиция рисунка (18-80A) в п. 1. Наименование Messier-Dowty Limited заменено на Safran Landing Systems. Обновлено значение пересчета на рисунке 602"
    ),
    "Added fig-item (18-80A) in para 1. Updated material specification in para 1.D.(1). Updated Messier-Dowty Limited to Safran Landing System. Updated": (
        "Добавлена позиция рисунка (18-80A) в п. 1. Обновлена спецификация материала в п. 1.D.(1). Наименование Messier-Dowty Limited заменено на Safran Landing Systems. Обновлен"
    ),
    "figure 603": "рисунок 603",
    "Added fig-item (18-80A)": "Добавлена позиция рисунка (18-80A)",
    "in para 1. Updated Messier-Dowty Limited to Safran Landing Systems": (
        "в п. 1. Наименование Messier-Dowty Limited заменено на Safran Landing Systems"
    ),
    "para 1. Updated Messier-Dowty Limited to Safran Landing Systems": (
        "п. 1. Наименование Messier-Dowty Limited заменено на Safran Landing Systems"
    ),
    "Updated Messier-Dowty Limited to Safran Landing Systems": (
        "Наименование Messier-Dowty Limited заменено на Safran Landing Systems"
    ),
    "Updated fig-items in para 1. Updated Messier-Dowty Limited to Safran Landing Systems": (
        "Обновлены позиции рисунков в п. 1. Наименование Messier-Dowty Limited заменено на Safran Landing Systems"
    ),
    "Updated fig-items in para 1": "Обновлены позиции рисунков в п. 1",
    "Updated fig-items in para 1. Added caution at para 1.E. Updated Messier-Dowty Limited to Safran Landing Systems": (
        "Обновлены позиции рисунков в п. 1. Добавлено предупреждение в п. 1.E. Наименование Messier-Dowty Limited заменено на Safran Landing Systems"
    ),
    "Updated Messier-Dowty": "Наименование Messier-Dowty",
    "Limited to Safran Landing Systems. Updated table 601": (
        "изменено на Safran Landing Systems. Обновлена таблица 601"
    ),
    "Updated fig-item (2-340) only in para 1": "Обновлена только позиция рисунка (2-340) в п. 1",
    "Updated fig-item (2-350) only in para 1": "Обновлена только позиция рисунка (2-350) в п. 1",
    "Updated paras 1.H.(1),": "Обновлены пп. 1.H.(1),",
    "1.I.(1), 2.E, 2.F, 2.G,": "1.I.(1), 2.E, 2.F, 2.G,",
    "2.H, 2.J, 2.K, 2.M, 2.N": "2.H, 2.J, 2.K, 2.M, 2.N",
    "and 2.O. Added paras": "и 2.O. Добавлены пп.",
    "2.I and 2.L. Added figure 713. Updated": "2.I и 2.L. Добавлен рисунок 713. Обновлены",
    "figures 705, 706, 707,": "рисунки 705, 706, 707,",
    "708 and 710. Updated figure numbers": "708 и 710. Обновлены номера рисунков",
    "Updated figure 815.": "Обновлен рисунок 815.",
    "Updated tables 813,": "Обновлены таблицы 813,",
    "814, 818 and 823": "814, 818 и 823",
    "Updated para 1.A": "Обновлен п. 1.A",
    "Updated IPL figs 13 to 18 to include Ref.": "Обновлены рисунки IPL 13-18 с включением кодов ссылок",
    "Codes: 2253 and 2255. Updated IPL fig 15.": "2253 и 2255. Обновлен рисунок IPL 15.",
    "Subject Reference": "Тема/ссылка",
    "Remove and Destroy Pages": "Изъять и уничтожить страницы",
    "Insert New/Revised": "Вставить новые/пересмотренные",
    "Reason for Change": "Причина изменения",
    "Pages": "Страницы",
    "Dated": "Дата",
    "REV. No.": "№ РЕВ.",
    "REV.": "РЕВ.",
    "No.": "№",
    "ISSUE DATE": "ДАТА ВЫПУСКА",
    "DATE INSERTED": "ДАТА ВНЕСЕНИЯ",
    "PAGE NUMBER": "НОМЕР СТРАНИЦЫ",
    "BY": "КЕМ",
    "DATE REMOVED": "ДАТА УДАЛЕНИЯ",
    "SB NUMBER": "№ SB",
    "SB TITLE": "НАИМЕНОВАНИЕ SB",
    "SB REVISION NUMBER": "№ РЕВИЗИИ SB",
    "DATE INCORPORATED INTO MANUAL": "ДАТА ВКЛЮЧЕНИЯ В РУКОВОДСТВО",
    "COVER SB NO.": "№ ОХВАТЫВАЮЩЕГО SB",
    "SAFRAN LANDING SYSTEMS SERVICE BULLETIN NUMBER": "НОМЕР СЕРВИСНОГО БЮЛЛЕТЕНЯ SAFRAN LANDING SYSTEMS",
    "SERVICE BULLETIN NUMBER": "НОМЕР СЕРВИСНОГО БЮЛЛЕТЕНЯ",
    "INSERTION DATE": "ДАТА ВНЕСЕНИЯ",
    "REV NO.": "№ РЕВ.",
    "DASH NO.": "№ ПОЗ.",
    "SAFRAN LANDING SYSTEMS MODIFICATION NUMBER": "НОМЕР МОДИФИКАЦИИ SAFRAN LANDING SYSTEMS",
    "MOD. STRIKE NO.": "№ ВНЕДРЕНИЯ МОДИФ.",
    "List of Effective": "Перечень действующих",
    "Unit Identification": "Идентификация агрегата",
    "Record of Temporary": "Журнал временных",
    "Revisions": "ревизий",
    "Assembly (Including": "Сборка (включая",
    "Storage)": "хранение)",
    "Storage) (Continued)": "хранение) (Продолжение)",
    "Assembly (Including Storage)": "Сборка (включая хранение)",
    "Updated fig-items in": "Обновлены позиции рисунков в",
    "Description and": "Описание и",
    "Operation": "работа",
    "Testing and Fault": "Испытания и поиск",
    "Isolation": "неисправностей",
    "Isolation (Continued)": "неисправностей (Продолжение)",
    "Isolation (Продолжение)": "неисправностей (Продолжение)",
    "List of Service Bulletins": "Список сервисных бюллетеней",
    "Upper Pivot Bracket (10-160) Only - Protective Treatment": (
        "Только верхний кронштейн шарнира (10-160) - Защитная обработка"
    ),
    "Spacer (4-180) Only - Protective Treatment": "Только проставка (4-180) - Защитная обработка",
    "Uplock Pin ain Fitting (5-400A) - Protective Treatment": (
        "Штифт замка убранного положения корпуса стойки (5-400A) - Защитная обработка"
    ),
    "Штифт замка убранного положения ain Fitting (5-400A) - Защитная обработка": (
        "Штифт замка убранного положения корпуса стойки (5-400A) - Защитная обработка"
    ),
    "Torque Link Repairs - Key Diagram": "Ремонты шлиц-шарнира - Схема расположения",
    "Torque Link Ремонтs - Схема расположения": "Ремонты шлиц-шарнира - Схема расположения",
    "Main Fitting Repairs - Key Diagram": "Ремонты корпуса стойки - Схема расположения",
    "Sliding Tube Repairs - Key Diagram": "Ремонты скользящей трубы - Схема расположения",
    "Upper Diaphragm Tube Repairs - Key Diagram": "Ремонты верхней диафрагменной трубы - Схема расположения",
    "Cylinder Repairs - Key Diagram": "Ремонты цилиндра - Схема расположения",
    "Transfer Block Repairs - Key Diagram.": "Ремонты переходного блока - Схема расположения.",
    "Harness Support Bracket Repairs - Key Diagram": "Ремонты кронштейна крепления жгута - Схема расположения",
    "Upper Pivot Bracket Repairs - Key Diagram": "Ремонты верхнего кронштейна шарнира - Схема расположения",
    "Корпус стойки Ремонтs - Схема расположения": "Ремонты корпуса стойки - Схема расположения",
    "Скользящая труба Ремонтs - Схема расположения": "Ремонты скользящей трубы - Схема расположения",
    "Верхняя диафрагменная труба Ремонтs - Схема расположения": (
        "Ремонты верхней диафрагменной трубы - Схема расположения"
    ),
    "Цилиндр Ремонтs - Схема расположения": "Ремонты цилиндра - Схема расположения",
    "Переходной блок Ремонтs - Схема расположения.": "Ремонты переходного блока - Схема расположения.",
    "Кронштейн крепления жгута Ремонтs - Схема расположения": (
        "Ремонты кронштейна крепления жгута - Схема расположения"
    ),
    "Верхний кронштейн шарнира Ремонтs - Схема расположения": (
        "Ремонты верхнего кронштейна шарнира - Схема расположения"
    ),
    "Верхний кронштейн шарнира (10-160) Только - Защитная обработка": (
        "Только верхний кронштейн шарнира (10-160) - Защитная обработка"
    ),
    "Проставка (4-180) Только - Защитная обработка": "Только проставка (4-180) - Защитная обработка",
    "Oversize Bush(es) - Machining and Installation Repair No. 11-2": (
        "Ремонтная (увеличенная) втулка - Механическая обработка и установка\tРемонт № 11-2"
    ),
    "Oversize Bush(es) - Механическая обработка и установка Ремонт № 11-2": (
        "Ремонтная (увеличенная) втулка - Механическая обработка и установка\tРемонт № 11-2"
    ),
    "Repair to Bushes - Machining and Installation Repair No. 11-13": (
        "Ремонт втулок - Механическая обработка и установка\tРемонт № 11-13"
    ),
    "Ремонт Bushes - Механическая обработка и установка Ремонт № 11-13": (
        "Ремонт втулок - Механическая обработка и установка\tРемонт № 11-13"
    ),
    "Page Repair No. 11-37 Main Fitting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . Repair No. 11-37 601": (
        "Стр. Ремонт № 11-37 Корпус стойки . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . Ремонт № 11-37\t601"
    ),
    "Page Ремонт № 11-37 Корпус стойки . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . Ремонт № 11-37 601": (
        "Стр. Ремонт № 11-37 Корпус стойки . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . Ремонт № 11-37\t601"
    ),
    "CAUTION: DO NOT USE CHLORINATED SOLVENTS. CHLORINATED SOLVENTS CAN MIX WITH VERY SMALL QUANTITIES OF WATER IN HYDRAULIC SYSTEMS TO MAKE HYDROCHLORIC ACID. HYDROCHLORIC ACID WILL CAUSE CORROSION ON METAL SURFACES.": (
        "ОСТОРОЖНО: НЕ ИСПОЛЬЗУЙТЕ ХЛОРИРОВАННЫЕ РАСТВОРИТЕЛИ. ХЛОРИРОВАННЫЕ РАСТВОРИТЕЛИ МОГУТ СМЕШИВАТЬСЯ С ОЧЕНЬ МАЛЫМИ КОЛИЧЕСТВАМИ ВОДЫ В ГИДРАВЛИЧЕСКИХ СИСТЕМАХ С ОБРАЗОВАНИЕМ СОЛЯНОЙ КИСЛОТЫ. СОЛЯНАЯ КИСЛОТА ВЫЗЫВАЕТ КОРРОЗИЮ МЕТАЛЛИЧЕСКИХ ПОВЕРХНОСТЕЙ."
    ),
    "All the materials in this manual have a Ref. Item identification. This is the reference item number of the material in the Aircraft Manufacturer’s Consumable Materials List.": (
        "Все материалы, приведенные в настоящем руководстве, имеют код ссылки. Это номер ссылки материала в перечне расходных материалов изготовителя воздушного судна."
    ),
    "All references in this manual are to the left configuration of the unit unless the instructions tell you differently.": (
        "Если в указаниях не оговорено иное, все ссылки в настоящем руководстве относятся к левому исполнению агрегата."
    ),
    "NOTE: Nitrogen will be released through the charging valve (13-60) as the piston (17-200) moves.": (
        "ПРИМЕЧАНИЕ. При перемещении поршня (17-200) азот будет выходить через заправочный клапан (13-60)."
    ),
    "NOTE: The charging valve (17-20) must be open to let the unit extend fully.": (
        "ПРИМЕЧАНИЕ. Заправочный клапан (17-20) должен быть открыт, чтобы агрегат полностью выдвинулся."
    ),
    "NOTE: The thread size is M142 x 1.5 pitch - 5h6h to BS3643.": (
        "ПРИМЕЧАНИЕ.\tРазмер резьбы: M142 x 1.5, поле допуска 5h6h по BS3643."
    ),
    "NOTE: Alternative equivalents are permitted.": "ПРИМЕЧАНИЕ. Допускаются альтернативные эквиваленты.",
    "CAUTION: DO NOT USE A PRESSURE OF MORE THAN 7,58 BAR (110 LBF/IN2).": (
        "ОСТОРОЖНО: НЕ ИСПОЛЬЗУЙТЕ ДАВЛЕНИЕ БОЛЕЕ 7,58 БАР (110 ФУНТ/ДЮЙМ2)."
    ),
    "CAUTION: DO NOT CAUSE DAMAGE TO THE PAINT FINISH.": (
        "ОСТОРОЖНО: НЕ ДОПУСТИТЕ ПОВРЕЖДЕНИЯ ЛАКОКРАСОЧНОГО ПОКРЫТИЯ."
    ),
    "AECMA Simplified English to PSC-85-16598 is used in this manual.": (
        "В настоящем руководстве используется упрощенный английский язык AECMA в соответствии с PSC-85-16598."
    ),
    "Parts of permanent assemblies that are not correctly attached.": (
        "Детали постоянных сборок, закрепленные ненадлежащим образом."
    ),
    "There is an error because of the pressure gauge capacity.": (
        "Из-за предела измерения манометра возникает погрешность."
    ),
    "If there is an error because of the gauge capacity:": (
        "Если возникает погрешность из-за диапазона прибора:"
    ),
    "Discard parts that you must not use again. These include:": (
        "Забракуйте детали, повторное использование которых не допускается. К ним относятся:"
    ),
    "Examine the unit for damage before you start the tests.": (
        "Перед началом испытаний осмотрите агрегат на наличие повреждений."
    ),
    "The test fluid must be clean: refer to M-DLPS910-1.": (
        "Испытательная жидкость должна быть чистой: см. M-DLPS910-1."
    ),
    "Examine each part for these types of damage:": (
        "Осмотрите каждую деталь на наличие следующих видов повреждений:"
    ),
    "Where K = 273 for temperatures in C": "Где K = 273 для температур в °C",
    "(459 for temperatures in F)": "(459 для температур в °F)",
    "They are a set: keep them together.": "Они образуют комплект: храните их вместе.",
    "This equipment is necessary:": "Необходимо следующее оборудование:",
    "These special tools are necessary:": "Необходимы следующие специальные инструменты:",
    "These materials are necessary:": "Необходимы следующие материалы:",
    "Clean the part: refer to para 2.A.": "Очистите деталь: см. п. 2.A.",
    "Unless instructions are different:": "Если в указаниях не сказано иное:",
    "Keep the unit in this condition for a minimum of six hours.": (
        "Выдержите агрегат в этом состоянии не менее шести часов."
    ),
    "The Location Frame 460007235 (for right configuration units).": (
        "Установочная рама 460007235 (для агрегатов правого исполнения)."
    ),
    "The Location Frame 460007234 (for left configuration units)": (
        "Установочная рама 460007234 (для агрегатов левого исполнения)."
    ),
    "Make a record of the ambient temperature, (T1).": (
        "Запишите температуру окружающей среды (T1)."
    ),
    "Measure the ambient temperature, (T2).": (
        "Измерьте температуру окружающей среды (T2)."
    ),
    "The inflation equipment must be to MIL-G-8348.": (
        "Заправочное оборудование должно соответствовать MIL-G-8348."
    ),
    "Remove the wire thread insert. Make sure that broken pieces do not stay in the hole.": (
        "Удалите резьбовую вставку. Убедитесь, что в отверстии не осталось обломков."
    ),
    "Bend the outer coil of the wire thread insert to the centre of the hole.": (
        "Отогните наружный виток резьбовой вставки к центру отверстия."
    ),
    "If necessary, remove the wire thread inserts:": (
        "При необходимости удалите резьбовые вставки:"
    ),
    "Use the Milliohmmeter Megger, Type BT51, to measure the electrical bonding resistance.": (
        "Используйте миллиомметр Megger, тип BT51, для измерения сопротивления электрического соединения."
    ),
    "Use the Crowfoot Wrench T14500 to remove the charging valve (13-60). Remove the O-ring seal (13-67) from the charging valve (13-60).": (
        "Используйте рожковый ключ типа Crowfoot T14500, чтобы снять заправочный клапан (13-60). Снимите с заправочного клапана (13-60) уплотнительное кольцо круглого сечения (13-67)."
    ),
    "Use the Crowfoot Wrench T14500 to remove the charging valve (17-20). Remove the O-ring seal (17-27) from the charging valve (17-20).": (
        "Используйте рожковый ключ типа Crowfoot T14500, чтобы снять заправочный клапан (17-20). Снимите с заправочного клапана (17-20) уплотнительное кольцо круглого сечения (17-27)."
    ),
    "Use the Charging Adapter 460002502 to connect the hydraulic test rig to the charging valve (13-60).": (
        "Используйте заправочный адаптер 460002502, чтобы подключить гидравлический испытательный стенд к заправочному клапану (13-60)."
    ),
    "Make sure that all of the nitrogen pressure has been released: remove the charging valve (17-20).": (
        "Убедитесь, что давление азота полностью сброшено: снимите заправочный клапан (17-20)."
    ),
    "Release the lock indentations of the locking washer (19-54).": (
        "Отогните участки стопорения стопорной шайбы (19-54)."
    ),
    "Release the pressure in the gauge.": "Сбросьте давление в манометре.",
    "Release the nitrogen pressure.": "Сбросьте давление азота.",
    "The Bottom Press Adapter 460007260.": "Нижний нажимной адаптер 460007260.",
    "The Jacking Dome Adapter 460006223": "Адаптер поддомкратного купола 460006223",
    "The Holding Fixture 460006231.": "Удерживающее приспособление 460006231.",
    "Trolley Support Arms Towing Frame": "Тележка, опорные рычаги, буксировочная рама",
    "Hydraulic fluid . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . Material Ref. Item 02-501": (
        "Гидравлическая жидкость . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . Код ссылки материала 02-501"
    ),
    "Static dis- charge con- nector": "Разъем отвода статического электричества",
    "Unserviceable screw threads.": "Резьба, непригодная к эксплуатации.",
    "P5A = P1A + (P2A - P4A) OR": "P5A = P1A + (P2A - P4A) ИЛИ",
    "10-123-11MD or": "10-123-11MD или",
    "BEARING (20-250)": "ПОДШИПНИК (20-250)",
    "Damper (9-160)": "ДЕМПФЕР (9-160)",
    "Parts 1": "Части 1",
    "- Chromium": "- Хромовое покрытие",
    "Cond H1025": "сост. H1025",
    "or": "или",
    "This manual contains Description, Operation, Maintenance procedures and an Illustrated Parts List (IPL). IPL Figure and Item numbers in parentheses follow the part name to identify them.": (
        "Настоящее руководство содержит разделы «Описание», «Работа», процедуры технического обслуживания и иллюстрированный перечень деталей (IPL). Номера рисунков и позиций IPL в скобках следуют за наименованием детали для ее идентификации."
    ),
    "A Unit Identification Chart is included to show the modification status of the unit. The modification status is related to the unit part number by the dash number: the dash number is marked on the unit name plate adjacent to the part number.": (
        "В руководство включена таблица идентификации агрегата, показывающая статус модификации агрегата. Статус модификации связан с номером детали агрегата через номер исполнения: номер исполнения нанесен на табличку агрегата рядом с номером детали."
    ),
    "All dimensions and quantities in this manual are in SI units with Imperial units in parentheses. A comma shows a decimal part of an SI unit. A full point shows a decimal part of an Imperial unit.": (
        "Все размеры и количества, приведенные в настоящем руководстве, указаны в единицах СИ, а британские единицы приведены в скобках. Запятая обозначает десятичную часть единицы СИ. Точка обозначает десятичную часть британской единицы."
    ),
    "This manual refers to Process Specifications (M-DLPS and PCS) and Non-destructive Tests (M-DLNDT). These are available within the Safran Landing Systems Technical Publications on-line service.": (
        "В настоящем руководстве приведены ссылки на технологические спецификации (M-DLPS и PCS) и документы по неразрушающему контролю (M-DLNDT). Они доступны в онлайн-службе технических публикаций Safran Landing Systems."
    ),
    "Use approved persons and good aircraft engineering practice for all procedures in this manual.": (
        "Для всех процедур, приведенных в настоящем руководстве, используйте уполномоченный персонал и соблюдайте надлежащую авиационно-техническую практику."
    ),
    "The repairs in this CMM have been approved under Airbus’ EASA Design Organisation Approval No. EASA.21J.031.": (
        "Ремонты, приведенные в данном CMM, одобрены в рамках одобрения конструкторской организации Airbus EASA No. EASA.21J.031."
    ),
    "On occasion a REF. CODE can be identified in the NOMENCLATURE column in the DETAILED PARTS LIST. This is a Safran Landing Systems reference code and is used for cross-reference purposes only.": (
        "Иногда в графе NOMENCLATURE подробного списка деталей может указываться REF. CODE. Это код ссылки Safran Landing Systems, используемый только для перекрестных ссылок."
    ),
    "The accuracy and the adequacy of the instructions in this CMM have been technically verified by shop verification (performed or simulated) or by similarity with manufacturing instructions or with component maintenance manuals instructions from other programs that have been verified in shop.": (
        "Точность и достаточность указаний, приведенных в данном CMM, технически подтверждены производственной проверкой (выполненной или смоделированной) либо по аналогии с производственными инструкциями или с инструкциями руководств по техническому обслуживанию компонентов из других программ, прошедших производственную проверку."
    ),
    "Safran Landing Systems UK Ltd Component Maintenance Manual, Main Landing Gear Leg and Dressings, 32-12-21.": (
        "Safran Landing Systems UK Ltd. Руководство по техническому обслуживанию компонента. Стойка основного шасси и навесные элементы. 32-12-21."
    ),
    "Safran Landing Systems UK Ltd Component Maintenance Manual, Axle Harness 1M and 2M, 32-12-29.": (
        "Safran Landing Systems UK Ltd. Руководство по техническому обслуживанию компонента. Жгут оси 1M и 2M. 32-12-29."
    ),
    "The main landing gear leg is a two stage, telescopic shock absorber.": (
        "Стойка основного шасси представляет собой двухступенчатый телескопический амортизатор."
    ),
    "Description (Refer to Figures 1 and 2)": "Описание (См. рисунки 1 и 2)",
    "The main landing gear leg has a sliding tube subassembly that operates in a main fitting subassembly. The sliding tube subassembly operates through a lower bearing subassembly. The lower bearing subassembly also seals the sliding tube subassembly in the main fitting subassembly.": (
        "Стойка основного шасси имеет подсборку скользящей трубы, работающую в подсборке корпуса стойки. Подсборка скользящей трубы проходит через подсборку нижнего подшипника. Подсборка нижнего подшипника также герметизирует подсборку скользящей трубы в подсборке корпуса стойки."
    ),
    "An upper torque link subassembly attaches to the main fitting subassembly. A lower torque link subassembly attaches to the sliding tube subassembly. A damper attaches to the upper torque link subassembly. A pin installs through the damper and connects the upper and lower torque link subassemblies.": (
        "К подсборке корпуса стойки крепится подсборка верхнего шлиц-шарнира. К подсборке скользящей трубы крепится подсборка нижнего шлиц-шарнира. К подсборке верхнего шлиц-шарнира крепится демпфер. Через демпфер устанавливается штифт, соединяющий подсборки верхнего и нижнего шлиц-шарнира."
    ),
    "A slave link subassembly and a lower slave link subassembly attach opposite the upper and lower torque link subassemblies.": (
        "Подсборка ведомого звена и подсборка нижнего ведомого звена крепятся напротив подсборок верхнего и нижнего шлиц-шарнира."
    ),
    "A rod and a cylinder install in the sliding tube subassembly. A piston installs in the cylinder. An upper diaphragm tube subassembly installs in the main fitting subassembly. A baffle, a compression orifice plate and a diaphragm subassembly install in the upper diaphragm tube subassembly. The rod goes through the baffle.": (
        "В подсборку скользящей трубы устанавливаются шток и цилиндр. В цилиндр устанавливается поршень. В подсборку корпуса стойки устанавливается подсборка верхней диафрагменной трубы. В подсборку верхней диафрагменной трубы устанавливаются перегородка, пластина дроссельного отверстия сжатия и подсборка диафрагмы. Шток проходит через перегородку."
    ),
    "An upper bearing housing installs between the top of the sliding tube subassembly and the main fitting subassembly. A recoil orifice plate operates in the upper bearing housing.": (
        "Корпус верхнего подшипника устанавливается между верхней частью подсборки скользящей трубы и подсборкой корпуса стойки. В корпусе верхнего подшипника работает пластина дроссельного отверстия отбоя."
    ),
    "Operation (Refer to Figure 2)": "Работа (См. рисунок 2)",
    "Compression": "Сжатие",
    "The sliding tube subassembly moves into the main fitting subassembly. The subsequent decrease in volume causes hydraulic fluid to flow through the upper bearing housing: the recoil orifice plate moves and slows the flow of hydraulic fluid. The decrease in volume also causes hydraulic fluid to move through the diaphragm and lift the compression orifice plate: the hydraulic fluid flows through the baffle and into the upper diaphragm tube subassembly. This slows the speed of the compression.": (
        "Подсборка скользящей трубы входит в подсборку корпуса стойки. Последующее уменьшение объема вызывает перетекание гидравлической жидкости через корпус верхнего подшипника: пластина дроссельного отверстия отбоя перемещается и замедляет поток гидравлической жидкости. Уменьшение объема также вызывает прохождение гидравлической жидкости через диафрагму и подъем пластины дроссельного отверстия сжатия: гидравлическая жидкость проходит через перегородку и поступает в подсборку верхней диафрагменной трубы. Это замедляет скорость сжатия."
    ),
    "Hydraulic fluid that moves into the upper diaphragm tube compresses the nitrogen in the main fitting subassembly and the upper diaphragm tube subassembly. As the pressure of the nitrogen increases, the hydraulic fluid in the rod moves against the piston. The piston is pushed into the cylinder and compresses the nitrogen in it. This slows the speed of the compression more.": (
        "Гидравлическая жидкость, поступающая в верхнюю диафрагменную трубу, сжимает азот в подсборке корпуса стойки и в подсборке верхней диафрагменной трубы. По мере увеличения давления азота гидравлическая жидкость в штоке перемещается к поршню. Поршень вдавливается в цилиндр и сжимает находящийся в нем азот. Это дополнительно замедляет скорость сжатия."
    ),
    "WARNING: DO NOT GET HYDRAULIC FLUID ON YOUR SKIN OR IN YOUR EYES. DO NOT BREATHE THE FUMES. ONLY USE IN A LOCATION THAT HAS A CONTINUOUS FLOW OF CLEAN AIR. HYDRAULIC FLUID IS POISONOUS AND DANGEROUS.": (
        "ПРЕДУПРЕЖДЕНИЕ: НЕ ДОПУСКАЙТЕ ПОПАДАНИЯ ГИДРАВЛИЧЕСКОЙ ЖИДКОСТИ НА КОЖУ ИЛИ В ГЛАЗА. НЕ ВДЫХАЙТЕ ПАРЫ. ИСПОЛЬЗУЙТЕ ТОЛЬКО В МЕСТЕ С ПОСТОЯННЫМ ПРИТОКОМ ЧИСТОГО ВОЗДУХА. ГИДРАВЛИЧЕСКАЯ ЖИДКОСТЬ ЯДОВИТА И ОПАСНА."
    ),
    "WARNING: DO NOT GET CLEANING AGENTS ON YOUR SKIN, IN YOUR EYES OR NEAR A FLAME. DO NOT BREATHE THE FUMES. ONLY USE IN A LOCATION THAT HAS A CONTINUOUS FLOW OF CLEAN AIR. CLEANING AGENTS ARE POISONOUS AND FLAMMABLE.": (
        "ПРЕДУПРЕЖДЕНИЕ: НЕ ДОПУСКАЙТЕ ПОПАДАНИЯ ОЧИСТИТЕЛЕЙ НА КОЖУ, В ГЛАЗА ИЛИ В ЗОНУ ОТКРЫТОГО ПЛАМЕНИ. НЕ ВДЫХАЙТЕ ПАРЫ. ИСПОЛЬЗУЙТЕ ТОЛЬКО В МЕСТЕ С ПОСТОЯННЫМ ПРИТОКОМ ЧИСТОГО ВОЗДУХА. ОЧИСТИТЕЛИ ЯДОВИТЫ И ОГНЕОПАСНЫ."
    ),
    "WARNING: DO NOT GET PAINT STRIPPER ON YOUR SKIN, IN YOUR EYES OR NEAR A FLAME. DO NOT BREATHE THE FUMES. ONLY USE IN A LOCATION THAT HAS A CONTINUOUS FLOW OF CLEAN AIR. PAINT STRIPPER IS POISONOUS AND FLAMMABLE.": (
        "ПРЕДУПРЕЖДЕНИЕ: НЕ ДОПУСКАЙТЕ ПОПАДАНИЯ СМЫВКИ КРАСКИ НА КОЖУ, В ГЛАЗА ИЛИ В ЗОНУ ОТКРЫТОГО ПЛАМЕНИ. НЕ ВДЫХАЙТЕ ПАРЫ. ИСПОЛЬЗУЙТЕ ТОЛЬКО В МЕСТЕ С ПОСТОЯННЫМ ПРИТОКОМ ЧИСТОГО ВОЗДУХА. СМЫВКА КРАСКИ ЯДОВИТА И ОГНЕОПАСНА."
    ),
    "WARNING: RELEASE ALL NITROGEN PRESSURE BEFORE YOU REMOVE THE CHARGING VALVES (13-60, 17-20).": (
        "ПРЕДУПРЕЖДЕНИЕ: ПЕРЕД СНЯТИЕМ ЗАПРАВОЧНЫХ КЛАПАНОВ (13-60, 17-20) ПОЛНОСТЬЮ СБРОСЬТЕ ДАВЛЕНИЕ АЗОТА."
    ),
    "CAUTION: DO NOT PUT AN END LOAD OF MORE THAN 5,08 TONNES (5 TONS) ON THE MAIN LANDING GEAR LEG (1-1).": (
        "ОСТОРОЖНО: НЕ ПРИКЛАДЫВАЙТЕ К СТОЙКЕ ОСНОВНОГО ШАССИ (1-1) ОСЕВУЮ НАГРУЗКУ БОЛЕЕ 5,08 ТОННЫ (5 ТОНН)."
    ),
    "CAUTION: DISCARD THE SCREWS (15-90) AND THE LOCKING PLATES (15-80) WHEN REMOVED.": (
        "ОСТОРОЖНО: ПОСЛЕ СНЯТИЯ ВЫБРАСЫВАЙТЕ ВИНТЫ (15-90) И СТОПОРНЫЕ ПЛАСТИНЫ (15-80)."
    ),
    "Recoil": "Отбой",
    "After compression, the nitrogen pressure in the cylinder pushes the piston to the end of the cylinder: hydraulic fluid moves out of the cylinder and into the rod. The nitrogen pressure in the main fitting subassembly and the upper diaphragm subassembly pushes the hydraulic fluid through the baffle: the compression orifice plate is pushed against the diaphragm subassembly and limits the flow of hydraulic fluid through it. This slows the speed of the recoil. The sliding tube subassembly moves out of the main fitting subassembly.": (
        "После сжатия давление азота в цилиндре перемещает поршень к концу цилиндра: гидравлическая жидкость выходит из цилиндра и поступает в шток. Давление азота в подсборке корпуса стойки и в подсборке верхней диафрагмы проталкивает гидравлическую жидкость через перегородку: пластина дроссельного отверстия сжатия прижимается к подсборке диафрагмы и ограничивает поток гидравлической жидкости через нее. Это замедляет скорость отбоя. Подсборка скользящей трубы выходит из подсборки корпуса стойки."
    ),
    "The Upper and Lower Torque Link Subassemblies": "Подсборки верхнего и нижнего шлиц-шарнира",
    "The upper and lower torque link subassemblies prevent the sliding tube subassembly from turning in the main fitting subassembly.": (
        "Подсборки верхнего и нижнего шлиц-шарнира предотвращают поворот подсборки скользящей трубы в подсборке корпуса стойки."
    ),
    "The damper controls the movement of the upper and lower torque link subassemblies.": (
        "Демпфер управляет перемещением подсборок верхнего и нижнего шлиц-шарнира."
    ),
    "The hydraulic test rig must have a hand pump and a power pump. The power pump must have a controlled flow of not less than 4,5 l/min (4.62 in3/sec).": (
        "Гидравлический испытательный стенд должен иметь ручной насос и силовой насос. Силовой насос должен обеспечивать регулируемый расход не менее 4,5 л/мин (4.62 in3/sec)."
    ),
    "The temperature of the test fluid must be between 20 and 40 C (68 and 104 F).": (
        "Температура испытательной жидкости должна быть в пределах от 20 до 40 °C (68-104 °F)."
    ),
    "During all hydraulic tests, the unit and the test circuit must be hydraulically full.": (
        "Во время всех гидравлических испытаний агрегат и испытательный контур должны быть полностью заполнены гидравлической жидкостью."
    ),
    "During the proximity switch tests the ambient temperature must be between 15 and 25 C (59 and 77 F).": (
        "Во время испытаний датчиков приближения температура окружающей среды должна быть в пределах от 15 до 25 °C (59-77 °F)."
    ),
    "Piston (17-200) Leakage Tests": "Испытания поршня (17-200) на герметичность",
    "Use the Charging Adapter 460002502 and the Turner Inflation Equipment T14218: connect the charging valve (17-20) to the nitrogen supply. Open the charging valve (17-20).": (
        "Используйте заправочный адаптер 460002502 и заправочное оборудование Turner T14218: подключите заправочный клапан (17-20) к источнику азота. Откройте заправочный клапан (17-20)."
    ),
    "Slowly increase the nitrogen pressure to between 9,32 and 10,68 bar (135 and 155 lbf/in2). Make a record of the pressure. Close the charging valve (17-20) and hold the nitrogen pressure for 15 minutes.": (
        "Медленно увеличьте давление азота до 9,32-10,68 бар (135-155 lbf/in2). Запишите давление. Закройте заправочный клапан (17-20) и выдержите давление азота в течение 15 минут."
    ),
    "Open the charging valve (17-20) and measure the nitrogen pressure: it must be the same as the record in para (2). Leakage must not occur.": (
        "Откройте заправочный клапан (17-20) и измерьте давление азота: оно должно совпадать со значением, записанным в п. (2). Утечка не допускается."
    ),
    "Disconnect the nitrogen supply and remove the Turner Inflation Equipment T14218 and the Charging Adapter 460002502.": (
        "Отсоедините источник азота и снимите заправочное оборудование Turner T14218 и заправочный адаптер 460002502."
    ),
    "Refer to ASSEMBLY: install the charging valve (17-20) and complete the assembly procedure.": (
        "См. раздел «Сборка»: установите заправочный клапан (17-20) и завершите процедуру сборки."
    ),
    "Main Landing Gear Leg (1-1) Tests": "Испытания стойки основного шасси (1-1)",
    "Initial Operations": "Подготовительные операции",
    "Use these special tools to install the main landing gear leg (1-1) vertically in the loading press:": (
        "Используйте следующие специальные инструменты, чтобы установить стойку основного шасси (1-1) вертикально в нагрузочный пресс:"
    ),
    "Assemble the Load Cell and Adapter 460006232 and the Offset Adapter 460006234 to the main landing gear leg (1-1).": (
        "Установите на стойку основного шасси (1-1) тензодатчик Load Cell с адаптером 460006232 и смещенный адаптер 460006234."
    ),
    "Procedure to Fill and Pressurize the Main Landing Gear Leg (1-1)": (
        "Процедура заполнения и наддува стойки основного шасси (1-1)"
    ),
    "Make sure that there is no pressure in the main landing gear leg (1-1): open the charging valves (13-60 and 17-20).": (
        "Убедитесь в отсутствии давления в стойке основного шасси (1-1): откройте заправочные клапаны (13-60 и 17-20)."
    ),
    "Slowly increase the hydraulic pressure to between 13,11 and 14,48 bar (190 and 210 lbf/in2) and let the unit extend fully.": (
        "Медленно увеличьте гидравлическое давление до 13,11-14,48 бар (190-210 lbf/in2) и дайте агрегату полностью выдвинуться."
    ),
    "Release the hydraulic pressure and fully close the unit.": (
        "Сбросьте гидравлическое давление и полностью закройте агрегат."
    ),
    "Do para (c) and (d) until the hydraulic fluid that comes out of the unit does not have air in it.": (
        "Выполняйте пп. (c) и (d), пока выходящая из агрегата гидравлическая жидкость не перестанет содержать воздух."
    ),
    "Fully close the unit and disconnect the hydraulic test rig.": (
        "Полностью закройте агрегат и отсоедините гидравлический испытательный стенд."
    ),
    "Use the Charging Adapter 460002502 and the Turner Inflation Equipment T14218 to connect the nitrogen supply to the charging valve (13-60).": (
        "Используйте заправочный адаптер 460002502 и заправочное оборудование Turner T14218, чтобы подключить источник азота к заправочному клапану (13-60)."
    ),
    "Slowly increase the nitrogen pressure until the unit starts to extend. Hold the pressure and fully extend the unit. The pressure must not be more than 7,58 bar (110 lbf/in2).": (
        "Медленно увеличивайте давление азота, пока агрегат не начнет выдвигаться. Удерживайте давление и полностью выдвиньте агрегат. Давление не должно превышать 7,58 бар (110 lbf/in2)."
    ),
    "Refer to Figure 101 and measure the dimension X: it must be between 483,05 and 487,85 mm (19.017 and 19.207 in).": (
        "См. рисунок 101 и измерьте размер X: он должен быть в пределах от 483,05 до 487,85 мм (19.017-19.207 in)."
    ),
    "Use the Charging Adapter 460002502 and the Turner Inflation Equipment T14218 to connect the charging valve (17-20) to the nitrogen supply.": (
        "Используйте заправочный адаптер 460002502 и заправочное оборудование Turner T14218, чтобы подключить заправочный клапан (17-20) к источнику азота."
    ),
    "Slowly increase the nitrogen pressure to between 13,11 and 14,48 bar (190 and 210 lbf/in2).": (
        "Медленно увеличьте давление азота до 13,11-14,48 бар (190-210 lbf/in2)."
    ),
    "Slowly increase the nitrogen pressure to between 67,59 and 70,34 bar (980 and1020 lbf/in2).": (
        "Медленно увеличьте давление азота до 67,59-70,34 бар (980-1020 lbf/in2)."
    ),
    "Close the charging valve (17-20); use the Crowfoot Wrench T14500 to torque it to between 5,7 and 7,9 N m (50 and 70 lbf in).": (
        "Закройте заправочный клапан (17-20); используйте рожковый ключ типа Crowfoot T14500, чтобы затянуть его моментом 5,7-7,9 Н·м (50-70 lbf in)."
    ),
    "Use the Turner Inflation Equipment T14218 and the Charging Adapter 460002502 to connect the nitrogen supply to the charging valve (13-60).": (
        "Используйте заправочное оборудование Turner T14218 и заправочный адаптер 460002502, чтобы подключить источник азота к заправочному клапану (13-60)."
    ),
    "Slowly increase the nitrogen pressure to between 6,90 and 8,27 bar (100 and 120 lbf/in2).": (
        "Медленно увеличьте давление азота до 6,90-8,27 бар (100-120 lbf/in2)."
    ),
    "Close the charging valve (13-60); use the Crowfoot Wrench T14500 to torque it to between 5,7 and 7,9 N m (50 and 70 lbf in).": (
        "Закройте заправочный клапан (13-60); используйте рожковый ключ типа Crowfoot T14500, чтобы затянуть его моментом 5,7-7,9 Н·м (50-70 lbf in)."
    ),
    "Slowly increase the nitrogen pressure to between 9,32 and 10,68 bar (135 and": (
        "Медленно увеличьте давление азота до 9,32-10,68 бар (135 и"
    ),
    "155 lbf/in2). Make a record of the pressure. Close the charging valve (17-20) and hold the nitrogen pressure for 15 minutes.": (
        "155 lbf/in2). Запишите давление. Закройте заправочный клапан (17-20) и выдержите давление азота в течение 15 минут."
    ),
    "Slowly increase the nitrogen pressure until the unit starts to extend. Hold the pressure and fully extend the unit. The pressure must not be more than 7,58 bar": (
        "Медленно увеличивайте давление азота, пока агрегат не начнет выдвигаться. Удерживайте давление и полностью выдвиньте агрегат. Давление не должно превышать 7,58 бар"
    ),
    "Main Landing Gear Leg (1-1) Figure 101": "Стойка основного шасси (1-1), рисунок 101",
    "Make a record of the nitrogen pressure at the charging valve (13-60), (P1A) and the charging valve (17-20), (P1B).": (
        "Запишите давление азота на заправочном клапане (13-60), (P1A), и на заправочном клапане (17-20), (P1B)."
    ),
    "Compare the pressures P1A and P2A and compare the pressures P1B and P2B. The pressures P1A and P2A must be the same and the pressures P1B and P2B must be the same, unless:": (
        "Сравните давления P1A и P2A, а также давления P1B и P2B. Давления P1A и P2A должны совпадать, и давления P1B и P2B также должны совпадать, если только:"
    ),
    "There is a difference between the temperatures T1 and T2": (
        "Имеется разница между температурами T1 и T2"
    ),
    "If there is a difference between the temperatures T1 and T2, calculate the correct value for the nitrogen pressures (these will be P3A and P3B) and adjust the pressures to the corrected values. Use the formula:": (
        "Если имеется разница между температурами T1 и T2, рассчитайте скорректированные значения давления азота (это будут P3A и P3B) и доведите давление до скорректированных значений. Используйте формулу:"
    ),
    "OR": "ИЛИ",
    "Z =  1 for pressures in bar": "Z = 1 для давлений в барах",
    "Main Landing Gear Leg (1-1) Figure 101": "Стойка основного шасси (1-1), рисунок 101",
    "Open the charging valve (13-60) and measure the nitrogen pressure, (P4A).": (
        "Откройте заправочный клапан (13-60) и измерьте давление азота, (P4A)."
    ),
    "Open the charging valve (17-20) and measure the nitrogen pressure, (P4B).": (
        "Откройте заправочный клапан (17-20) и измерьте давление азота, (P4B)."
    ),
    "Calculate the correct values for the nitrogen pressures (these will be P5A and P5B) and adjust the pressures to the corrected values. Use the formula:": (
        "Рассчитайте скорректированные значения давления азота (это будут P5A и P5B) и доведите давление до скорректированных значений. Используйте формулу:"
    ),
    "P5A = P1A + (P2A - P4A) OR": "P5A = P1A + (P2A - P4A) ИЛИ",
    "Prepare for Transport and Storage": "Подготовка к транспортированию и хранению",
    "Open the charging valve (13-60) and reduce the nitrogen pressure to between 3,45 and 4,82 bar (50 and 70 lbf/in2).": (
        "Откройте заправочный клапан (13-60) и уменьшите давление азота до 3,45-4,82 бар (50-70 lbf/in2)."
    ),
    "Open the charging valve (17-20) and reduce the nitrogen pressure to between 3,45 and 4,82 bar (50 and 70 lbf/in2).": (
        "Откройте заправочный клапан (17-20) и уменьшите давление азота до 3,45-4,82 бар (50-70 lbf/in2)."
    ),
    "Write this data on a label and attach it to the unit: THE GEAR MUST BE INFLATED TO THE APPROPRIATE PRESSURES BEFORE BEING PLACED IN SERVICE.": (
        "Нанесите эти данные на табличку и прикрепите ее к агрегату: СТОЙКА ДОЛЖНА БЫТЬ ЗАПРАВЛЕНА ДО СООТВЕТСТВУЮЩИХ ДАВЛЕНИЙ ПЕРЕД ВВОДОМ В ЭКСПЛУАТАЦИЮ."
    ),
    "Complete the torque procedure for the retaining pins (13-10): refer to ASSEMBLY.": (
        "Выполните процедуру затяжки стопорных штифтов (13-10): см. раздел «Сборка»."
    ),
    "Proximity Switches (7-40 and 7-230) Adjustments and Tests": (
        "Регулировка и испытания датчиков приближения (7-40 и 7-230)"
    ),
    "Use the loading press: set the dimension between the pins (10-80 and 11-130) to between 632,80 and 636,95 mm (24.9134 and 25.0767 in).": (
        "Используйте нагрузочный пресс: установите размер между штифтами (10-80 и 11-130) в пределах 632,80-636,95 мм (24.9134-25.0767 in)."
    ),
    "Adjust the spacers (6-140, 7-50, 7-190 and 7-240) or laminated shims (6-140A, 7-50A, 7-90A and 7-240A): refer to ASSEMBLY.": (
        "Отрегулируйте проставки (6-140, 7-50, 7-190 и 7-240) или наборные регулировочные прокладки (6-140A, 7-50A, 7-90A и 7-240A): см. раздел «Сборка»."
    ),
    "NOTE: If the calculated gap is in the tolerance, the spacers (6-140, 7-50, 7-190 and 7-240) or laminated shims (6-140A, 7-50A, 7-90A and 7-240A) are not necessary.": (
        "ПРИМЕЧАНИЕ. Если рассчитанный зазор находится в пределах допуска, проставки (6-140, 7-50, 7-190 и 7-240) или наборные регулировочные прокладки (6-140A, 7-50A, 7-90A и 7-240A) не требуются."
    ),
    "Connect the 28 VDC power supply, the Lampbox 460005842 and the main landing gear leg (1-1).": (
        "Подключите источник питания 28 В пост. тока, блок Lampbox 460005842 и стойку основного шасси (1-1)."
    ),
    "Use the loading press to fully extend the main landing gear leg (1-1).": (
        "Используйте нагрузочный пресс, чтобы полностью выдвинуть стойку основного шасси (1-1)."
    ),
    "Use the loading press to slowly close the main landing gear leg (1-1):": (
        "Используйте нагрузочный пресс, чтобы медленно закрыть стойку основного шасси (1-1):"
    ),
    "The proximity switch (7-230) must operate before the main landing gear leg (1-1) has closed by 26,00 mm (1.0236 in).": (
        "Датчик приближения (7-230) должен сработать до того, как стойка основного шасси (1-1) закроется на 26,00 мм (1.0236 in)."
    ),
    "The proximity switch (7-40) must operate before the main landing gear leg (1-1) has closed by 29,30 mm (1.1535 in).": (
        "Датчик приближения (7-40) должен сработать до того, как стойка основного шасси (1-1) закроется на 29,30 мм (1.1535 in)."
    ),
    "Do para (4) and (5) again.": "Повторите пп. (4) и (5).",
    "Disconnect the 28 VDC supply and the Lampbox 460005842.": (
        "Отсоедините питание 28 В пост. тока и блок Lampbox 460005842."
    ),
    "Remove the main landing gear leg (1-1) from the loading press.": (
        "Снимите стойку основного шасси (1-1) с нагрузочного пресса."
    ),
    "NOTE: Make sure that the main landing gear leg (1-1) is electrically isolated from the equipment that is used to hold it.": (
        "ПРИМЕЧАНИЕ. Убедитесь, что стойка основного шасси (1-1) электрически изолирована от оборудования, используемого для ее удержания."
    ),
    "Measure between the bearing (20-250) and the test points given in Table 101. The electrical bonding resistance must not be more than the limit given in Table 101.": (
        "Измерьте между подшипником (20-250) и точками проверки, указанными в таблице 101. Сопротивление электрического соединения не должно превышать предельное значение, указанное в таблице 101."
    ),
    "Measure between the axle of the sliding tube subassembly (17-240) and the test points given in Table 102. The electrical bonding resistance must not be more than the limit given in Table 102.": (
        "Измерьте между осью подсборки скользящей трубы (17-240) и точками проверки, указанными в таблице 102. Сопротивление электрического соединения не должно превышать предельное значение, указанное в таблице 102."
    ),
    "NOTE: Refer to TESTING AND FAULT ISOLATION to find the necessary level of disassembly. This will give the condition of the component or the possible cause of its malfunction.": (
        "ПРИМЕЧАНИЕ. См. раздел «Испытания и поиск неисправностей», чтобы определить необходимый уровень разборки. Это позволит установить состояние компонента или возможную причину его неисправности."
    ),
    "Make sure that the work area, the tools and the equipment are clean.": (
        "Убедитесь, что рабочая зона, инструменты и оборудование чистые."
    ),
    "Procedure (Refer to IPL Figures 1 to 20)": "Процедура (См. рисунки IPL 1-20)",
    "Use these special tools as necessary during the procedure to lift and to hold the unit:": (
        "Используйте следующие специальные инструменты по мере необходимости в ходе процедуры для подъема и удержания агрегата:"
    ),
    "The Lifting Bar Assembly 460006208": "Подъемная штанга в сборе 460006208",
    "The Spherical Bearing Locator 460007282": "Установочный шаблон сферического подшипника 460007282",
    "The Pintle Location Assembly 460007281": "Установочное приспособление штифта навеса стойки 460007281",
    "Assemble the Load Cell and Adapter 460006232 and the Offset Adapter 460006234 to the main landing gear leg (1-1).": (
        "Установите на стойку основного шасси (1-1) тензодатчик с адаптером 460006232 и смещенный адаптер 460006234."
    ),
    "Open the charging valve (13-60) and release the nitrogen pressure. Do not close the charging valve (13-60).": (
        "Откройте заправочный клапан (13-60) и стравите давление азота. Не закрывайте заправочный клапан (13-60)."
    ),
    "and": "и",
    "and 460006261": "и 460006261",
    "Type 1 or 17-4PH to": "тип 1 или 17-4PH по",
    "Measure the nitrogen pressure at the charging valve (13-60), (P2A) and the charging valve (17-20), (P2B).": (
        "Измерьте давление азота у заправочного клапана (13-60), (P2A), и у заправочного клапана (17-20), (P2B)."
    ),
    "Z = 1 for pressures in bar": "Z = 1 для давлений в бар",
    "(15 for pressures in lbf/in2)": "(15 для давлений в lbf/in2)",
    "Repair Sleeves - Machining and Installation\tRepair No. 12-2": (
        "Ремонтные втулки - механическая обработка и установка\tРемонт № 12-2"
    ),
    "Repair Sleeves - Machining and Installation\tRepair No. 12-5": (
        "Ремонтные втулки - механическая обработка и установка\tРемонт № 12-5"
    ),
    "Oversize Thread Insert - Installation\tRepair No. 18-2": (
        "Резьбовая вставка увеличенного размера - установка\tРемонт № 18-2"
    ),
    "Labels (20-10, 20-30, 20-40, 20-60 and 20-80) and wiring diagram plate (1-110) . . . .\n- Installation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .": (
        "Таблички (20-10, 20-30, 20-40, 20-60 и 20-80) и табличка электрической схемы (1-110) . . . .\n- Установка . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ."
    ),
    "Proximity Switch (7-230) and Target (7-180) - Adjustment. . . . . . . . . . . . . . . . . . . . . .": (
        "Датчик приближения (7-230) и мишень (7-180) - Регулировка. . . . . . . . . . . . . . . . . . . . . ."
    ),
    "Proximity Switch (7-40) and Target (6-130) - Adjustment. . . . . . . . . . . . . . . . . . . . . .": (
        "Датчик приближения (7-40) и мишень (6-130) - Регулировка. . . . . . . . . . . . . . . . . . . . . ."
    ),
    "Load Cell and Adapter": "Тензодатчик и адаптер",
    "Remove the charging valves (13-60 and 17-20)": "Снимите заправочные клапаны (13-60 и 17-20)",
    "Remove the forward pintle bush (20-250A)": "Снимите втулку переднего штифта навеса (20-250A)",
    "Remove the bearings (5-280 and 5-290) and the bushes (20-380)": (
        "Снимите подшипники (5-280 и 5-290) и втулки (20-380)"
    ),
    "Remove the bearings (20-230, 20-240 and 20-290)": (
        "Снимите подшипники (20-230, 20-240 и 20-290)"
    ),
    "Lifting Bar Assembly": "Подъемная штанга в сборе",
    "Lift the sliding tube subassembly (17-240) and related parts": (
        "Поднимать подсборку скользящей трубы (17-240) и связанные с ней детали"
    ),
    "Transport and Build\nTrolley Support Arms Towing Frame\nJacking Dome Adapter\n\nAdapter": (
        "Транспортировочная и сборочная\nтележка, опорные рычаги, буксировочная рама\nАдаптер поддомкратного купола\n\nАдаптер"
    ),
    "Use with 460006232, 460006263\nand 460006261": (
        "Использовать с 460006232, 460006263\nи 460006261"
    ),
    "Use with 460006232, 460006263 and 460006261": "Использовать с 460006232, 460006263 и 460006261",
    "Assembly/Extraction Tool": "Сборочно-демонтажный инструмент",
    "Extractor Pad and Drawbolt": "Съемная опора и вытяжной болт",
    "\nTo remove the forward pintle bush (2-250A)": "\nДля снятия втулки переднего штифта навеса (2-250A)",
    "Hold the main landing gear leg (1-1) (left configuration)\n\nHold the main landing gear leg (1-2) (right configuration)": (
        "Удерживать стойку основного шасси (1-1) (левая конфигурация)\n\nУдерживать стойку основного шасси (1-2) (правая конфигурация)"
    ),
    "Hold the sliding tube subassembly (17-240) and related parts": (
        "Удерживать подсборку скользящей трубы (17-240) и связанные с ней детали"
    ),
    "M-D\nSpec": "Спецификация\nM-D",
    "M-D Spec": "Спецификация\nM-D",
    "PCS-3100\nand\n\nM-DLNDT3\nParts 1\nand 2": "PCS-3100\nи\n\nM-DLNDT3\nЧасти 1\nи 2",
    "PCS-3100 and M-DLNDT3 Parts 1 and 2": "PCS-3100\nи\n\nM-DLNDT3\nЧасти 1\nи 2",
    "and\n\nM-DLNDT3\nParts 1": "и\n\nM-DLNDT3\nЧасти 1",
    "and M-DLNDT3 Parts 1": "и\n\nM-DLNDT3\nЧасти 1",
    "and 2": "и 2",
    "and\n\nPCS-3002": "и\n\nPCS-3002",
    "and PCS-3002": "и\n\nPCS-3002",
    "Spec": "Спецификация",
    "Steel, 35CD4 or\n4340 to AMS6414 or 35NCD16 to NCT 10-123-11 MD": (
        "Сталь, 35CD4 или\n4340 по AMS6414 или 35NCD16 по NCT 10-123-11 MD"
    ),
    "Stainless Steel, Z15CN17-03\nType 1 or 17-4PH to\nAMS 5604/5643": (
        "Коррозионностойкая сталь, Z15CN17-03\nтип 1 или 17-4PH по\nAMS 5604/5643"
    ),
    "Stainless Steel, Z15CN17-03 Type 1 or 17-4PH to AMS 5604/5643": (
        "Коррозионностойкая сталь, Z15CN17-03\nтип 1 или 17-4PH по\nAMS 5604/5643"
    ),
    "Aluminium Alloy, L113 or\nL168-T6511": "Алюминиевый сплав, L113 или\nL168-T6511",
    "Aluminium Alloy, BS L168 or\nBS 2L93 or 7075-T73, T7351, T73510\nor T73511 to MTL-2701": (
        "Алюминиевый сплав, BS L168 или\nBS 2L93 или 7075-T73, T7351, T73510\nили T73511 по MTL-2701"
    ),
    "Post SB 201-32-22: cut the Bowden cable (1-45) and remove the cross bolts (1-47 and 1-49).": (
        "ПОСЛЕ SB 201-32-22: разрежьте трос Боудена (1-45) и снимите поперечные болты (1-47 и 1-49)."
    ),
    "Remove the bolt (2-10), the washer (2-20), the pin (2-30), the spacer (2-40) and the": (
        "Снимите болт (2-10), шайбу (2-20), штифт (2-30), проставку (2-40) и"
    ),
    "Release the tab washer (2-150). Remove the bolt (2-140), the tab washer (2-150) and the bonding cable (2-160).": (
        "Отогните усик отгибной шайбы (2-150). Снимите болт (2-140), отгибную шайбу (2-150) и кабель заземления (2-160)."
    ),
    "Remove the bolts (5-20 and 5-40), the washers (5-30 and 5-50), the nut (5-60) and the": (
        "Снимите болты (5-20 и 5-40), шайбы (5-30 и 5-50), гайку (5-60) и"
    ),
    "Use the Drift 460004331/7 and the Extractor 460006151/24 to remove the bearings (5-280 and 5-290) from the bracket (5-300).": (
        "Используйте выколотку 460004331/7 и съемник 460006151/24 для снятия подшипников (5-280 и 5-290) с кронштейна (5-300)."
    ),
    "Remove the split pin (6-60), the slotted nut (6-70), the washer (6-80) and the pivot pin (6-90).": (
        "Снимите шплинт (6-60), прорезную гайку (6-70), шайбу (6-80) и шарнирный штифт (6-90)."
    ),
    "Remove the slave link subassembly (6-190) and its attached parts.": (
        "Снимите подсборку ведомого звена (6-190) и присоединенные к ней детали."
    ),
    "NOTE: If the calculated gap is in the tolerance, the spacers (6-140) or the laminated shim (6-140A) is not installed.": (
        "ПРИМЕЧАНИЕ. Если рассчитанный зазор находится в пределах допуска, проставки (6-140) или наборная регулировочная прокладка (6-140A) не устанавливаются."
    ),
    "Remove the grooved spherical bearing (6-300) or the self lubricating bearing (6-300A) from the lower slave link (6-310).": (
        "Снимите сферический подшипник с канавкой (6-300) или самосмазывающийся подшипник (6-300A) с нижнего ведомого звена (6-310)."
    ),
    "NOTE: If the calculated gap is in the tolerance, the spacer (7-50) or the laminated shim (7-50A) is not installed.": (
        "ПРИМЕЧАНИЕ. Если рассчитанный зазор находится в пределах допуска, проставка (7-50) или наборная регулировочная прокладка (7-50A) не устанавливается."
    ),
    "NOTE: If the calculated gap is in the tolerance, the spacer (7-190) or the laminated shim (7-190A) is not installed.": (
        "ПРИМЕЧАНИЕ. Если рассчитанный зазор находится в пределах допуска, проставка (7-190) или наборная регулировочная прокладка (7-190A) не устанавливается."
    ),
    "Remove the housing (12-170) and its related parts.": "Снимите корпус (12-170) и связанные с ним детали.",
    "Remove the plate (13-100) from the inflation valve (13-110).": (
        "Снимите пластину (13-100) с заправочного клапана (13-110)."
    ),
    "Remove the split pin (13-140), the nut (13-150), the washers (13-160), the bolt (13-170) and the stop ring (13-180).": (
        "Снимите шплинт (13-140), гайку (13-150), шайбы (13-160), болт (13-170) и стопорное кольцо (13-180)."
    ),
    "Remove the shock absorber subassembly (13-50) and its related parts from the main fitting subassembly (20-90).": (
        "Снимите подсборку амортизатора (13-50) и связанные с ней детали со сборки корпуса стойки (20-90)."
    ),
    "Remove the upper diaphragm tube subassembly (15-360) and its related parts.": (
        "Снимите подсборку верхней диафрагменной трубы (15-360) и связанные с ней детали."
    ),
    "Remove the O-ring seal (16-20A or 16A-20A), the backing rings (16-30 or 16A-30), the O-ring seal (16-40A or 16A-40A) and the backing": (
        "Снимите уплотнительное кольцо круглого сечения (16-20A или 16A-20A), опорные кольца (16-30 или 16A-30), уплотнительное кольцо круглого сечения (16-40A или 16A-40A) и опорные"
    ),
    "rings (16-50 or 16A-50).": "кольца (16-50 или 16A-50).",
    "Remove the common lower bearing bushes (16-130A or 16A-130B) from the lower bearing housing (16-140B or 16A-140C).": (
        "Снимите общие втулки нижнего подшипника (16-130A или 16A-130B) с корпуса нижнего подшипника (16-140B или 16A-140C)."
    ),
    "Release the tab washer (17-110) and remove the bolts (17-100). Remove the tab washers (17-110).": (
        "Отогните усик отгибной шайбы (17-110) и снимите болты (17-100). Снимите отгибные шайбы (17-110)."
    ),
    "Remove the lock plate (17-120) and use the Pin Spanner 460007284 to remove the nut subassembly (17-130). Remove the rod (17-160) and the washer (17-170).": (
        "Снимите стопорную пластину (17-120) и используйте штифтовой ключ 460007284, чтобы снять подсборку гайки (17-130). Снимите шток (17-160) и шайбу (17-170)."
    ),
    "Use the Extractor 460006151/47 and the Drift 460004331/1 to remove the bearings (20-230 and 20-240).": (
        "Используйте съемник 460006151/47 и выколотку 460004331/1 для снятия подшипников (20-230 и 20-240)."
    ),
    "Paint the part: refer to REPAIR.": "Окрасьте деталь: см. раздел РЕМОНТ.",
    "Spring Data": "Данные пружины",
    "Slowly open the charging valve (17-20) and release all of the second stage nitrogen pressure.": (
        "Медленно откройте заправочный клапан (17-20) и стравите все давление азота второй ступени."
    ),
    "Slowly open the charging valve (13-60) and release all of the first stage nitrogen pressure.": (
        "Медленно откройте заправочный клапан (13-60) и стравите все давление азота первой ступени."
    ),
    "Use the Lifting Tackle 460006211 and install the sliding tube subassembly (17-240) in the Build Trolley 460007240.": (
        "Используйте подъемную оснастку 460006211 и установите подсборку скользящей трубы (17-240) в сборочную тележку 460007240."
    ),
    "Remove the upper bearing housing (15-40) and related parts as follows:": (
        "Снимите корпус верхнего подшипника (15-40) и связанные с ним детали следующим образом:"
    ),
    "Use Pin Spanner 460007279 to remove the upper bearing housing (15-40).": (
        "Используйте штифтовой ключ 460007279, чтобы снять корпус верхнего подшипника (15-40)."
    ),
    "Remove and discard the pins (15-120).": "Снимите и выбросьте штифты (15-120).",
    "Use the Torque Adapter 460007283, the Torque Reactor 460007278, the Holding Blocks 460006406 and the Bench Clamp MT1025 to remove the diaphragm subassembly (15-190), the compression orifice plate (15-220), the clapper seat (15-230) and the baffle (15-240).": (
        "Используйте моментный адаптер 460007283, упор реактивного момента 460007278, удерживающие блоки 460006406 и слесарные тиски MT1025, чтобы снять подсборку диафрагмы (15-190), пластину дроссельного отверстия сжатия (15-220), седло клапана (15-230) и перегородку (15-240)."
    ),
    "Use the Assembly/Extraction Tool 460006410 to remove the level tube (15-300) and remove the O-ring seal (15-310).": (
        "Используйте сборочно-демонтажный инструмент 460006410, чтобы снять уровневую трубку (15-300) и уплотнительное кольцо круглого сечения (15-310)."
    ),
    "Remove the lower bearing subassembly (16-110) and its related parts from the sliding tube subassembly (17-240).": (
        "Снимите сборку нижнего подшипника (16-110) и связанные с ней детали с подсборки скользящей трубы (17-240)."
    ),
    "Remove the lower bearing subassembly (16A-110D) and its related parts from the sliding tube subassembly (17-240).": (
        "Снимите сборку нижнего подшипника (16A-110D) и связанные с ней детали с подсборки скользящей трубы (17-240)."
    ),
    "Remove the inner liner (16A-117) from the lower bearing subassembly (16A-110D) and discard it.": (
        "Снимите внутренний вкладыш (16A-117) со сборки нижнего подшипника (16A-110D) и выбросьте его."
    ),
    "Remove the lower bearing (16A-150A) from the lower bearing housing subassembly (16A-120B). Discard the machined lower bearing (16A-150A).": (
        "Снимите нижний подшипник (16A-150A) с подсборки корпуса нижнего подшипника (16A-120B). Обработанный нижний подшипник (16A-150A) выбросьте."
    ),
    "Lower Bearing Subassembly (16-110D or 16A-110E) Post Ref. Code: 2253": (
        "Сборка нижнего подшипника (16-110D или 16A-110E), код ссылки ПОСЛЕ: 2253"
    ),
    "Remove the lower bearing subassembly (16-110D or 16A-110E) and its related parts from the sliding tube subassembly (17-240).": (
        "Снимите сборку нижнего подшипника (16-110D или 16A-110E) и связанные с ней детали с подсборки скользящей трубы (17-240)."
    ),
    "Remove the cap screws (17-30), the washers (17-40) and the valve support (17-50).": (
        "Снимите винты с цилиндрической головкой (17-30), шайбы (17-40) и опору клапана (17-50)."
    ),
    "Release the lock washer (17-90) and use the Torque Adapter 460006404 to remove the jacking dome (17-80). Remove the lock washer (17-90).": (
        "Отогните стопорную шайбу (17-90) и используйте моментный адаптер 460006404, чтобы снять поддомкратный купол (17-80). Снимите стопорную шайбу (17-90)."
    ),
    "Remove the cylinder (17-230) and its related parts from the sliding tube subassembly (17-240).": (
        "Снимите цилиндр (17-230) и связанные с ним детали с подсборки скользящей трубы (17-240)."
    ),
    "Hold the cylinder (17-230) in the Bench Clamp MT1025 and Holding Blocks MT1026/63.": (
        "Закрепите цилиндр (17-230) в слесарных тисках MT1025 и удерживающих блоках MT1026/63."
    ),
    "Remove the lubrication fittings (17-270) and the identification washers (17-280) from the sliding tube subassembly (17-240).": (
        "Снимите смазочные штуцеры (17-270) и идентификационные шайбы (17-280) с подсборки скользящей трубы (17-240)."
    ),
    "Remove the lubrication fittings (18-52) and identification washers (18-54).": (
        "Снимите смазочные штуцеры (18-52) и идентификационные шайбы (18-54)."
    ),
    "Remove the label (18-70) from the sliding tube (18-80).": (
        "Снимите табличку (18-70) со скользящей трубы (18-80)."
    ),
    "Use the Torque Adapter 460007232 to remove the locking nut (19-52). Remove the locking washer (19-54) and the outer race and the ball of the spherical bearing (19-50).": (
        "Используйте моментный адаптер 460007232, чтобы снять стопорную гайку (19-52). Снимите стопорную шайбу (19-54), наружную обойму и шарик сферического подшипника (19-50)."
    ),
    "NOTE: The outer race and the ball are parts of the spherical bearing (19-550).": (
        "ПРИМЕЧАНИЕ. Наружная обойма и шарик являются частями сферического подшипника (19-50)."
    ),
    "Remove the lubrication fitting (20-110) and the identification washer (20-120). Remove the lubrication adapter (20-130).": (
        "Снимите смазочный штуцер (20-110) и идентификационную шайбу (20-120). Снимите смазочный адаптер (20-130)."
    ),
    "Remove the lubrication fitting (20-140) and the identification washer (20-150). Remove the lubrication adapter (20-160).": (
        "Снимите смазочный штуцер (20-140) и идентификационную шайбу (20-150). Снимите смазочный адаптер (20-160)."
    ),
    "Remove the lubrication fitting (20-170) and the identification washer (20-180). Remove the lubrication adapter (20-190).": (
        "Снимите смазочный штуцер (20-170) и идентификационную шайбу (20-180). Снимите смазочный адаптер (20-190)."
    ),
    "Remove the lubrication fittings (20-200) and identification washers (20-210). Remove the lubrication adapters (20-220).": (
        "Снимите смазочные штуцеры (20-200) и идентификационные шайбы (20-210). Снимите смазочные адаптеры (20-220)."
    ),
    "Use the Hydraulic-Pneumatic Pump Set 460006497, the Bolt 460006498/7, the Press Pad 460006499/25 and the Extraction Tube 460004680 and remove the forward pintle bush (20-250A).": (
        "Используйте гидропневматическую насосную установку 460006497, болт 460006498/7, нажимную опору 460006499/25 и выпрессовочную трубку 460004680, чтобы снять втулку переднего штифта навеса (20-250A)."
    ),
    "Use the Extractor Plate 460007259/460006151/9 and the Drift 460004331/21 to remove the bushes (20-340 and 20-350).": (
        "Используйте плиту съемника 460007259/460006151/9 и выколотку 460004331/21, чтобы снять втулки (20-340 и 20-350)."
    ),
    "Use the Press Pad Assembly 460006267 and remove the drag arm sleeve (20-370A only).": (
        "Используйте сборку нажимной опоры 460006267 и снимите втулку тяги складывания (только 20-370A)."
    ),
    "Clean all the metal parts with white spirit, Material Ref. Item 11-524. Make sure that you fully remove all sealants, adhesives and jointing compounds.": (
        "Очистите все металлические детали уайт-спиритом, код ссылки материала 11-524. Убедитесь, что полностью удалены все герметики, клеевые составы и герметизирующие составы."
    ),
    "Use clean PVC or polythene gloves to prevent corrosion of metal parts.": (
        "Используйте чистые перчатки из ПВХ или полиэтилена, чтобы предотвратить коррозию металлических деталей."
    ),
    "Prevent corrosion of the metal parts that you do not immediately use for assembly procedures: refer to PCS-2800.": (
        "Предотвратите коррозию металлических деталей, которые не будут немедленно использованы при сборке: см. PCS-2800."
    ),
    "The procedure to examine the parts is in two levels:": (
        "Процедура проверки деталей состоит из двух уровней:"
    ),
    "Visually examine each part. Carefully examine changes of section and areas which contact sealing rings.": (
        "Визуально осмотрите каждую деталь. Особенно внимательно осмотрите переходы сечений и участки, контактирующие с уплотнительными кольцами."
    ),
    "Deterioration of protective treatment.": "Повреждение защитной обработки.",
    "Distortion and/or cracks.": "Деформация и/или трещины.",
    "Wear or fretting.": "Износ или фреттинг-коррозия.",
    "Scores, dents or burrs.": "Риски, вмятины или заусенцы.",
    "Measure all the parts that are in FITS AND CLEARANCES and compare with the dimensions in the table.": (
        "Измерьте все детали, указанные в разделе ПОСАДКИ И ЗАЗОРЫ, и сравните результаты с размерами, приведенными в таблице."
    ),
    "Special Dimension Check:": "Специальная проверка размеров:",
    "Examine the rod (17-160) for the diameter of radial damping holes. The diameter of each hole must be between 5,40 and 5,60 mm (0.213 and 0.220 in).": (
        "Проверьте шток (17-160) по диаметру радиальных демпфирующих отверстий. Диаметр каждого отверстия должен быть в пределах 5,40-5,60 мм (0.213-0.220 in)."
    ),
    "Examine the thread form of the diaphragm subassembly (15-190) and diaphragm (15-210A) with shadow graph projection.": (
        "Проверьте профиль резьбы подсборки диафрагмы (15-190) и диафрагмы (15-210A) с помощью теневой проекции."
    ),
    "Examine the 4 holes in the sliding tube (18-80) where the bracket (8-170) installs, for burrs. If you find burrs contact Safran Landing Systems who will supply an applicable repair.": (
        "Проверьте 4 отверстия в скользящей трубе (18-80), в которые устанавливается кронштейн (8-170), на наличие заусенцев. При обнаружении заусенцев свяжитесь с Safran Landing Systems, которая предоставит соответствующий ремонт."
    ),
    "NOTE: Use a good light source and 10x magnification to view the area, to look for burrs.": (
        "ПРИМЕЧАНИЕ. Используйте хороший источник света и 10-кратное увеличение для осмотра участка на наличие заусенцев."
    ),
    "Examine all parts shown in Tables 501 and 502 to the applicable NDT and information given.": (
        "Проверьте все детали, указанные в таблицах 501 и 502, соответствующим методом НК и с учетом приведенных указаний."
    ),
    "Parts that are included in Tables 501 and 502 must be fully disassembled to the lowest detail level for NDT inspection. This includes the removal of all of the bushes.": (
        "Детали, указанные в таблицах 501 и 502, должны быть полностью разобраны до наименьшего уровня детализации для контроля НК. Это включает снятие всех втулок."
    ),
    "Examination of Magnetic Steel Parts by Non-destructive Testing Table 501": (
        "Контроль магнитных стальных деталей методами неразрушающего контроля. Таблица 501"
    ),
    "Examination of Non-Magnetic Parts by Non-destructive Testing Table 502": (
        "Контроль немагнитных деталей методами неразрушающего контроля. Таблица 502"
    ),
}


PHRASE_RULES: list[tuple[str, str]] = [
    ("(Continued)", "(Продолжение)"),
    ("(Withdrawn)", "(Аннулирован)"),
    ("(Superseded)", "(Заменен)"),
    ("How To Use This Illustrated Parts List", "Как пользоваться данным иллюстрированным перечнем деталей"),
    ("Vendor Codes, Names and Addresses", "Коды поставщиков, наименования и адреса"),
    ("Detailed Parts List", "Подробный список деталей"),
    ("Numerical Index", "Числовой указатель"),
    ("Equipment and Materials", "Оборудование и материалы"),
    ("Description and Operation", "Описание и работа"),
    ("Testing and Fault Isolation", "Испытания и поиск неисправностей"),
    ("Special Detailed Inspection", "Специальный детальный осмотр"),
    ("Detailed Inspection", "Детальный осмотр"),
    ("Test Conditions", "Условия испытаний"),
    ("Fault Isolation", "Поиск неисправностей"),
    ("Special Tools, Fixtures and Equipment", "Специальные инструменты, приспособления и оборудование"),
    ("Fits and Clearances Definitions", "Определения посадок и зазоров"),
    ("Fits and Clearances", "Посадки и зазоры"),
    ("Electrical Bonding Resistance Test Points", "Точки проверки сопротивления электрического соединения"),
    ("Diagram of Operation", "Схема работы"),
    ("Illustrated Parts List", "Иллюстрированный перечень деталей"),
    ("Component Maintenance Manual", "Руководство по техническому обслуживанию компонента"),
    ("Main Landing Gear Leg", "Стойка основного шасси"),
    ("List of Effective Pages", "Перечень действующих страниц"),
    ("List of Service Bulletins", "Список сервисных бюллетеней"),
    ("Record of Temporary Revisions", "Журнал временных ревизий"),
    ("Record of Revisions", "Журнал ревизий"),
    ("Revision Record", "Журнал ревизий"),
    ("Unit Identification Chart", "Таблица идентификации агрегата"),
    ("Letter of Transmittal", "Сопроводительное письмо"),
    ("Title Page", "Титульный лист"),
    ("Table of Contents", "Содержание"),
    ("Illustrations", "Иллюстрации"),
    ("Repair Procedure Conditions", "Условия выполнения процедуры ремонта"),
    ("Approved Repairs", "Утвержденные ремонты"),
    ("Protective Treatment", "Защитная обработка"),
    ("Key Diagram", "Схема расположения"),
    ("Airbus Maintenance Planning Document", "документ Airbus по планированию технического обслуживания"),
    ("Maintenance Planning Document", "документ по планированию технического обслуживания"),
    ("Torque Link Repairs", "Ремонты шлиц-шарнира"),
    ("Torque Link Ремонтs", "Ремонты шлиц-шарнира"),
    ("Корпус стойки Ремонтs", "Ремонты корпуса стойки"),
    ("Скользящая труба Ремонтs", "Ремонты скользящей трубы"),
    ("Верхняя диафрагменная труба Ремонтs", "Ремонты верхней диафрагменной трубы"),
    ("Цилиндр Ремонтs", "Ремонты цилиндра"),
    ("Переходной блок Ремонтs", "Ремонты переходного блока"),
    ("Кронштейн крепления жгута Ремонтs", "Ремонты кронштейна крепления жгута"),
    ("Верхний кронштейн шарнира Ремонтs", "Ремонты верхнего кронштейна шарнира"),
    ("Torque Data", "Данные по моментам затяжки"),
    ("Remarks", "Примечания"),
    ("Introduction", "Введение"),
    ("General", "Общие сведения"),
    ("Procedure", "Процедура"),
    ("Storage", "Хранение"),
    ("Including", "включая"),
    ("Description", "Описание"),
    ("Operation", "Работа"),
    ("Data", "Данные"),
    ("Cleaning", "Очистка"),
    ("Check", "Проверка"),
    ("Assembly", "Сборка"),
    ("Disassembly", "Разборка"),
    ("Repair", "Ремонт"),
    ("Tables", "таблицы"),
    ("Table", "таблица"),
    ("and/or", "и/или"),
    (" and ", " и "),
    (" or ", " или "),
    ("Repair No.", "Ремонт №"),
    ("Repair to", "Ремонт"),
    ("Repair Bushes", "Ремонтные втулки"),
    ("Repair Bush(es)", "Ремонтная втулка(и)"),
    ("Repair Bush", "Ремонтная втулка"),
    ("Repair Bearing", "Подшипник"),
    ("Oversize Bushes", "Ремонтные втулки увеличенного размера"),
    ("Oversize Lubrication adapter", "Увеличенный смазочный адаптер"),
    ("Oversize Transfer Dowel", "Увеличенный переходной штифт"),
    ("Machining and Installation", "механическая обработка и установка"),
    ("Machining and installation", "механическая обработка и установка"),
    ("Machining", "механическая обработка"),
    ("Installation", "установка"),
    ("Inner Liner", "внутренний вкладыш"),
    ("Liner", "вкладыш"),
    ("Chromium Plate Termination", "завершение хромового покрытия"),
    ("Sheet", "лист"),
    ("Only", "только"),
    ("Vendor", "Поставщик"),
    ("Subject", "Тема"),
    ("Chart", "Таблица"),
    ("Fig.", "Рис."),
    ("ain Fitting", "корпуса стойки"),
    ("Main Fitting Subassembly", "Сборка корпуса стойки"),
    ("Main Fitting", "Корпус стойки"),
    ("Sliding Tube", "Скользящая труба"),
    ("Lower Bearing Subassembly", "Сборка нижнего подшипника"),
    ("Lower Torque Link", "Нижний шлиц-шарнир"),
    ("Upper Torque Link", "Верхний шлиц-шарнир"),
    ("Upper Slave Link", "Верхнее ведомое звено"),
    ("Lower Slave Link", "Нижнее ведомое звено"),
    ("Upper Diaphragm Tube Subassembly", "Сборка верхней диафрагменной трубы"),
    ("Upper Diaphragm Tube", "Верхняя диафрагменная труба"),
    ("Upper Pivot Bracket", "Верхний кронштейн шарнира"),
    ("Pivot Bracket", "Кронштейн шарнира"),
    ("Harness Support Bracket", "Кронштейн крепления жгута"),
    ("Transfer Block", "Переходной блок"),
    ("transfer block", "переходной блок"),
    ("Lock Stay Cardan", "Кардан фиксатора"),
    ("Spherical Bearing", "Сферический подшипник"),
    ("Pintle Pin", "Штифт навеса стойки"),
    ("Forward Pintle Pin", "Передний штифт навеса стойки"),
    ("Retaining Pin", "Стопорный штифт"),
    ("Valve Stem", "Шток клапана"),
    ("Inflation Valve", "Заправочный клапан"),
    ("Uplock Pin", "Штифт замка убранного положения"),
    ("Pivot Pin", "Шарнирный штифт"),
    ("Pin", "Штифт"),
    ("Bracket", "Кронштейн"),
    ("Slave Link", "Ведомое звено"),
    ("Cylinder", "Цилиндр"),
    ("Spacer", "Проставка"),
    ("Drag-arm Spacer", "Проставка тяги складывания"),
    ("Special Tools, Fixtures", "Специальные инструменты, приспособления"),
    ("and Equipment", "и оборудование"),
    ("Reference Publications", "Справочные публикации"),
    ("Application of Jointing Compound", "Нанесение герметизирующего состава"),
    ("Electrical Bonding Resistance Tests", "Проверка сопротивления электрического соединения"),
    ("Leakage Tests", "испытания на герметичность"),
    ("Maintenance procedures", "процедуры технического обслуживания"),
    ("Process Specifications", "технологические спецификации"),
    ("Non-destructive Tests", "неразрушающий контроль"),
    ("Technical Publications", "технические публикации"),
    ("on-line service", "онлайн-служба"),
    ("Imperial units", "британские единицы"),
    ("SI units", "единицы СИ"),
    ("Figure and Item numbers", "номера рисунков и позиций"),
    ("Refer to Figures", "См. рисунки"),
    ("Refer to Figure", "См. рисунок"),
    ("Subassemblies", "Подсборки"),
    ("Subassembly", "Подсборка"),
    ("subassemblies", "подсборки"),
    ("subassembly", "подсборка"),
    ("Sliding Tube Subassembly", "Подсборка скользящей трубы"),
    ("Bracket Subassembly", "Подсборка кронштейна"),
    ("Bolt Subassembly", "Подсборка болта"),
    ("Installation of Bushes", "Установка втулок"),
    ("Installation of Labels", "Установка табличек"),
    ("Assembly of Lower Bearing Subassembly", "Сборка нижнего подшипника"),
    ("Seal Configuration", "Конфигурация уплотнения"),
    ("Configuration", "Конфигурация"),
    ("Crimping of the Pin", "Расклепка штифта"),
    ("Application of Ardrox AV100D to the", "Нанесение Ardrox AV100D на"),
    ("to the Upper Diaphragm Tube", "на верхнюю диафрагменную трубу"),
    ("to the Pin", "на штифт"),
    ("Ardrox Application", "Нанесение Ardrox"),
    ("Dimensions After Installation in the Gland Housing", "Размеры после установки в корпус сальника"),
    ("Grease Groove", "Смазочная канавка"),
    ("Gland Housing", "Корпус сальника"),
    ("Lower Bearing", "Нижний подшипник"),
    ("Turner Inflation Equipment", "Заправочное оборудование Turner"),
    ("Axle Harness", "Жгут оси"),
    ("Dressings", "Навесные элементы"),
    ("Hydraulic-Pneumatic Pump Set", "Гидропневматическая насосная установка"),
    ("Hydraulic Test Rig", "Гидравлический испытательный стенд"),
    ("Nitrogen Supply", "Источник азота"),
    ("28 VDC Power Supply", "Источник питания 28 В пост. тока"),
    ("Crowfoot Wrench", "Рожковый ключ типа Crowfoot"),
    ("Charging Adapter", "Заправочный адаптер"),
    ("Holding Fixture", "Удерживающее приспособление"),
    ("Holding Blocks", "Удерживающие блоки"),
    ("Location Frame", "Установочная рама"),
    ("Torque Adapter", "Моментный адаптер"),
    ("Bottom Press Adapter", "Нижний нажимной адаптер"),
    ("Jacking Dome Adapter", "Адаптер поддомкратного купола"),
    ("Press Adapter", "Нажимной адаптер"),
    ("Extractor Plate", "Плита съемника"),
    ("Offset Adapter", "Смещенный адаптер"),
    ("Extraction Tube", "Выпрессовочная трубка"),
    ("Extraction Pad", "Выпрессовочная опора"),
    ("Extraction Bar", "Выпрессовочная штанга"),
    ("Lifting Tackle", "Подъемная оснастка"),
    ("Transport and Build Trolley", "Тележка для транспортировки и сборки"),
    ("Build Trolley", "Сборочная тележка"),
    ("Support Arms", "Опорные рычаги"),
    ("Towing Frame", "Буксировочная рама"),
    ("Bench Clamp", "Слесарные тиски"),
    ("Milliohmmeter Megger, Type BT51", "Миллиомметр Megger, тип BT51"),
    ("Proximity switch connector shell", "Корпус разъема датчика приближения"),
    ("Static discharge connector", "Разъем отвода статического электричества"),
    ("Heat shrink sleeve", "Термоусадочная трубка"),
    ("Ferrule", "Обжимная втулка"),
    ("Lampbox", "Ламповый блок"),
    ("Rod end", "Шарнирный наконечник"),
    ("Bowden cable", "Трос Боудена"),
    ("Labels", "Таблички"),
    ("Label", "Табличка"),
    ("Bushes", "Втулки"),
    ("Bush", "Втулка"),
    ("cap screws", "винты с цилиндрической головкой"),
    ("cap screw", "винт с цилиндрической головкой"),
    ("shims", "регулировочные прокладки"),
    ("shim", "регулировочная прокладка"),
    ("laminated shims", "наборные регулировочные прокладки"),
    ("laminated shim", "наборная регулировочная прокладка"),
    ("Bolts", "Болты"),
    ("Nuts", "Гайки"),
    ("Washers", "Шайбы"),
    ("Spacers", "Проставки"),
    ("lubrication fittings", "смазочные штуцеры"),
    ("lubrication fitting", "смазочный штуцер"),
    ("ground stud", "заземляющий штырь"),
    ("outer race", "наружная обойма"),
    ("retainers", "фиксаторы"),
    ("wedge", "клин"),
    ("target", "мишень"),
    ("diaphragm", "диафрагма"),
    ("baffle", "перегородка"),
    ("rod", "шток"),
    ("piston", "поршень"),
    ("shock absorber", "амортизатор"),
    ("unit", "агрегат"),
    ("Sealing ring", "Уплотнительное кольцо"),
    ("Seal", "Уплотнение"),
    ("Sleeve", "Втулка"),
    ("Washer", "Шайба"),
    ("Screws", "Винты"),
    ("Bolt", "Болт"),
    ("Nut", "Гайка"),
    ("Bung", "Заглушка"),
    ("Adapter", "Адаптер"),
    ("Extractor", "Съемник"),
    ("Drift", "Выколотка"),
    ("Press Pad", "Нажимная опора"),
    ("Joint seal", "Герметизирующее уплотнение"),
    ("Wiper ring", "Грязесъемное кольцо"),
    ("washer(s)", "шайба(ы)"),
    ("Backing rings", "Опорные кольца"),
    ("Backing ring", "Опорное кольцо"),
    ("Split pins", "Шплинты"),
    ("Split pin", "Шплинт"),
    ("Retaining pins", "Стопорные штифты"),
    ("Retaining pin", "Стопорный штифт"),
    ("Locking pins", "Стопорные штифты"),
    ("Locking pin", "Стопорный штифт"),
    ("Locking plate", "Стопорная пластина"),
    ("Lock plate", "Стопорная пластина"),
    ("Locking washer", "Стопорная шайба"),
    ("Lock washer", "Стопорная шайба"),
    ("Tab washers", "Отгибные шайбы"),
    ("Tab washer", "Отгибная шайба"),
    ("Cup washers", "Чашечные шайбы"),
    ("Cup washer", "Чашечная шайба"),
    ("O-ring seals", "Уплотнительные кольца круглого сечения"),
    ("O-ring seal", "Уплотнительное кольцо круглого сечения"),
    ("Wire Thread Inserts", "Резьбовые вставки"),
    ("Wire Thread Insert", "Резьбовая вставка"),
    ("wire thread inserts", "резьбовые вставки"),
    ("wire thread insert", "резьбовая вставка"),
    ("Oversize Thread Insert", "Резьбовая вставка увеличенного размера"),
    ("Threaded insert", "Резьбовая вставка"),
    ("Retaining ring", "Стопорное кольцо"),
    ("Level tube", "Уровневая трубка"),
    ("Clapper seat", "Седло клапана"),
    ("Compression orifice plate", "Пластина дроссельного отверстия сжатия"),
    ("Recoil orifice plate", "Пластина дроссельного отверстия отбоя"),
    ("Special Bolt", "Специальный болт"),
    ("Cross Bolt", "Поперечный болт"),
    ("Hydraulic fluid", "Гидравлическая жидкость"),
    ("Weight without hydraulic fluid", "Масса без гидравлической жидкости"),
    ("Weight with hydraulic fluid", "Масса с гидравлической жидкостью"),
    ("approximately", "приблизительно"),
    ("Nitrogen", "Азот"),
    ("Paint Removal", "Удаление краски"),
    ("Examine Parts Visually", "Визуально осмотрите детали"),
    ("Examine Dimensions", "Проверьте размеры"),
    ("Application of", "Нанесение"),
    ("Jointing Compound", "герметизирующего состава"),
    ("Hole Locations", "Расположение отверстий"),
    ("Materials", "Материалы"),
    ("Material Type", "Тип материала"),
    ("Material", "Материал"),
    ("Part Name", "Наименование детали"),
    ("Function", "Назначение"),
    ("Equipment", "Оборудование"),
    ("Special Tool", "Специальный инструмент"),
    ("Special Tools", "Специальные инструменты"),
    ("Part No.", "Номер детали"),
    ("EASA No.", "EASA №"),
    ("Name", "Наименование"),
    ("Repair Sleeves", "Ремонтные втулки"),
    ("Bearings", "Подшипники"),
    ("Bearing", "Подшипник"),
    ("Damper", "Демпфер"),
    ("Parts", "Части"),
    ("Ref. Item identification", "код ссылки"),
    ("Material Ref. Item", "Код ссылки материала"),
    ("Ref. Item", "Код ссылки"),
    ("LIMIT VALUE MILLIOHMS", "ПРЕДЕЛЬНОЕ ЗНАЧЕНИЕ, мОм"),
    ("TEST POINT", "ТОЧКА ПРОВЕРКИ"),
    ("IPL FIGURE AND ITEM No.", "РИС./ПОЗ. IPL"),
    ("IPL Fig/Item No.", "Рис./поз. IPL"),
    ("IPL Fig/Item", "Рис./поз. IPL"),
    ("Fig Item No.", "Рис./поз. №"),
    ("Fig Item", "Рис./поз."),
    ("Use with", "Использовать с"),
    ("Use the", "Используйте"),
    ("Hold the main landing gear leg", "Удерживать стойку основного шасси"),
    ("Hold the upper diaphragm tube", "Удерживать верхнюю диафрагменную трубу"),
    ("Hold the", "Удерживать"),
    ("Lift the", "Поднимать"),
    ("Main landing gear leg (1-1) tests", "Испытания стойки основного шасси (1-1)"),
    ("Proximity switch and target tests", "Испытания датчика приближения и мишени"),
    ("Use the Extractor", "Используйте съемник"),
    ("Use the Torque Adapter", "Используйте моментный адаптер"),
    ("Use the Charging Adapter", "Используйте заправочный адаптер"),
    ("Use the Crowfoot Wrench", "Используйте рожковый ключ типа Crowfoot"),
    ("to remove the bushes", "для снятия втулок"),
    ("to remove the bush", "для снятия втулки"),
    ("to remove the bearings", "для снятия подшипников"),
    ("to remove the bearing", "для снятия подшипника"),
    ("to remove the nuts", "для снятия гаек"),
    ("to remove the nut", "для снятия гайки"),
    ("to remove the forward pintle bush", "для снятия втулки переднего штифта навеса"),
    ("to hold the pin", "для удержания штифта"),
    ("to remove the charging valve", "для снятия заправочного клапана"),
    ("to remove the lubrication adapters", "для снятия смазочных адаптеров"),
    ("Remove the bushes", "Снимите втулки"),
    ("Remove the bush", "Снимите втулку"),
    ("Remove the", "Снимите"),
    ("Remove the bearings", "Снимите подшипники"),
    ("Remove the bearing", "Снимите подшипник"),
    ("Remove the nuts", "Снимите гайки"),
    ("Remove the nut", "Снимите гайку"),
    ("Remove and discard", "Снимите и выбросьте"),
    ("Open the charging valve", "Откройте заправочный клапан"),
    ("Close the charging valve", "Закройте заправочный клапан"),
    ("Close the charging valves", "Закрыть заправочные клапаны"),
    ("Remove the locking plate", "Снимите стопорную пластину"),
    ("Remove the locking nut", "Снимите стопорную гайку"),
    ("Remove the stop rings", "Снимите стопорные кольца"),
    ("Remove the jacking dome", "Снимите поддомкратный купол"),
    ("Remove the level tube", "Снимите уровневую трубку"),
    ("Remove the upper bearing housing", "Снимите корпус верхнего подшипника"),
    ("Remove the transfer dowels", "Снимите переходные штифты"),
    ("Remove the wiring diagram plate", "Снимите табличку электрической схемы"),
    ("Remove the lubrication adapters", "Снимите смазочные адаптеры"),
    ("Remove drag arm sleeve", "Снимите втулку тяги складывания"),
    ("Remove the 2M electrical axle harness", "Снимите электрический жгут оси 2M"),
    ("Remove the 1M electrical axle harness", "Снимите электрический жгут оси 1M"),
    ("Remove the charging valves", "Снимите заправочные клапаны"),
    ("Remove the lubrication fittings", "Снимите смазочные штуцеры"),
    ("Remove the lubrication fitting", "Снимите смазочный штуцер"),
    ("Remove the identification washers", "Снимите идентификационные шайбы"),
    ("Remove the identification washer", "Снимите идентификационную шайбу"),
    ("Remove the bonding cable", "Снимите кабель заземления"),
    ("Remove the related parts", "Снимите связанные детали"),
    ("wiring diagram plate", "табличка электрической схемы"),
    ("identification washers", "идентификационные шайбы"),
    ("identification washer", "идентификационная шайба"),
    ("bonding cable", "кабель заземления"),
    ("related parts", "связанные детали"),
    ("attached parts", "присоединенные детали"),
    ("its related parts", "связанные с ней детали"),
    ("its attached parts", "присоединенные к ней детали"),
    ("charging valves", "заправочные клапаны"),
    ("forward pintle bush", "втулка переднего штифта навеса"),
    ("cardan assembly", "кардан в сборе"),
    ("Cardan Assembly", "Кардан в сборе"),
    ("Lock Stay Cardan Subassembly", "Подсборка кардана фиксатора"),
    ("lock stay cardan subassembly", "подсборка кардана фиксатора"),
    ("Proximity Switches", "Датчики приближения"),
    ("Proximity Switch", "Датчик приближения"),
    ("proximity switches", "датчики приближения"),
    ("proximity switch", "датчик приближения"),
    ("Adjustment.", "Регулировка."),
    ("harness support bracket", "кронштейн крепления жгута"),
    ("harness support", "опора жгута"),
    ("pivot bracket subassembly", "подсборка кронштейна шарнира"),
    ("lubrication shaft subassembly", "подсборка смазочного вала"),
    ("dust cap", "пылезащитный колпачок"),
    ("clamp", "хомут"),
    ("slotted nut", "прорезная гайка"),
    ("grooved spherical bearing", "сферический подшипник с канавкой"),
    ("self lubricating bearing", "самосмазывающийся подшипник"),
    ("Torque Reaction Adapter", "Адаптер реактивного момента"),
    ("Press Pad Assembly", "Сборка нажимной опоры"),
    ("Extraction Tube", "Выпрессовочная трубка"),
    ("Extractor Plate", "Плита съемника"),
    ("Assembly/Extraction Tool", "Сборочно-демонтажный инструмент"),
    ("Extractor Pad and Drawbolt", "Съемная опора и вытяжной болт"),
    ("Lifting Bar Assembly", "Подъемная штанга в сборе"),
    ("Pintle Location Assembly", "Установочное приспособление штифта навеса"),
    ("Spherical Bearing Locator", "Установочный шаблон сферического подшипника"),
    ("Pin Spanner", "Штифтовой ключ"),
    ("left configuration", "левая конфигурация"),
    ("right configuration", "правая конфигурация"),
    ("Build Trolley", "Сборочная тележка"),
    ("Transport and Build", "Транспортировочная и сборочная"),
    ("Release the tab washer", "Отогните усик отгибной шайбы"),
    ("Release the tab washers", "Отогните усики отгибных шайб"),
    ("Release the cup washers", "Освободите чашечные шайбы"),
    ("Remove the two piece stop with inserts", "Снимите двухсоставной упор с вставками"),
    ("Remove the damaged paint", "Удалите поврежденное лакокрасочное покрытие"),
    ("refer to", "см."),
    ("Remove the bung", "Снимите заглушку"),
    ("Dry all the metal parts.", "Высушите все металлические детали."),
    ("Upper dia- phragm tube", "Верхняя диафрагменная труба"),
    ("The Adapter", "Адаптер"),
    ("POST SB", "ПОСЛЕ SB"),
    ("PRE SB", "ДО SB"),
    ("Housing", "Корпус"),
    ("Torque Reaction", "Реактивный момент"),
    ("Torque Reactor", "Упор реактивного момента"),
    ("Loading Press", "Нагрузочный пресс"),
    ("White spirit", "Уайт-спирит"),
    ("Jacking dome", "Поддомкратный купол"),
    ("Compression", "Сжатие"),
    ("Recoil", "Отбой"),
    (" or ", " или "),
    ("Inclusion class", "Класс включений"),
    ("on areas without chromium plate", "на участках без хромового покрытия"),
    ("Chromium plated areas", "Участки с хромовым покрытием"),
    ("plated areas", "участки с покрытием"),
    ("Stainless Steel", "Коррозионностойкая сталь"),
    ("Aluminium Alloy", "Алюминиевый сплав"),
    ("Steel", "Сталь"),
    ("Not applicable.", "Не применяется."),
    ("To be given subsequently.", "Будет указано позднее."),
    ("Corrosion.", "Коррозия."),
]


REGEX_RULES: list[tuple[re.Pattern[str], str | callable]] = [
    (re.compile(r"^Page\s+(\d+)\s+of\s+(\d+)$", re.IGNORECASE), r"Страница \1 из \2"),
    (re.compile(r"^Page\s+(\d+)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}/\d{4})$", re.IGNORECASE), r"Страница \1 \2 \3"),
    (re.compile(r"^Page\s+(\d+)$", re.IGNORECASE), r"Страница \1"),
    (re.compile(r"^Added repair no\.\s+([0-9-]+)$", re.IGNORECASE), r"Добавлен ремонт № \1"),
    (re.compile(r"\(Sheet\s+(\d+)\s+of\s+(\d+)\)", re.IGNORECASE), r"(Лист \1 из \2)"),
    (re.compile(r"\(Лист\s+(\d+)\s+of\s+(\d+)\)", re.IGNORECASE), r"(Лист \1 из \2)"),
]


def _letters_only(text: str) -> str:
    return "".join(ch for ch in text if ch.isalpha())


def _apply_case_style(source: str, target: str) -> str:
    letters = _letters_only(source)
    if not letters:
        return target
    if letters.isupper():
        return target.upper()
    if source[:1].isupper() and source[1:2].islower():
        return target[:1].upper() + target[1:]
    return target


def _replace_phrase(text: str, source: str, target: str) -> str:
    pattern = re.compile(re.escape(source), re.IGNORECASE)

    def repl(match: re.Match[str]) -> str:
        return _apply_case_style(match.group(0), target)

    return pattern.sub(repl, text)


def _cleanup_mixed_translation(text: str) -> str:
    cleaned = re.sub(r"\b(?:the|a|an)\s+(?=[А-Яа-яЁё])", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<![A-Za-z])or(?=(?:\s|$))", "или", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<![A-Za-z])and(?=(?:\s|$))", "и", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<![A-Za-z])from(?=(?:\s|$))", "из", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("Установка of ", "Установка ")
    cleaned = cleaned.replace("Сборка of ", "Сборка ")
    cleaned = cleaned.replace("Уплотнение Configuration", "Конфигурация уплотнения")
    cleaned = cleaned.replace("Уплотнение Конфигурация", "Конфигурация уплотнения")
    cleaned = cleaned.replace("M-D\nSpec", "Спецификация\nM-D")
    cleaned = cleaned.replace("Electrical Жгут оси", "Электрический жгут оси")
    cleaned = cleaned.replace(" Figure ", " рисунок ")
    cleaned = cleaned.replace(" to AMS", " по AMS")
    cleaned = cleaned.replace(" to MTL", " по MTL")
    cleaned = cleaned.replace(" to NCT", " по NCT")
    cleaned = cleaned.replace(" to BS", " по BS")
    cleaned = cleaned.replace(" to MAT", " по MAT")
    cleaned = cleaned.replace("Шплинтs", "Шплинты")
    cleaned = cleaned.replace("подшипникs", "подшипники")
    cleaned = cleaned.replace("Подшипникs", "Подшипники")
    cleaned = cleaned.replace("Втулкаs", "Втулки")
    cleaned = cleaned.replace("Уплотнениеs", "Уплотнения")
    cleaned = cleaned.replace("Процедураs", "процедуры")
    return cleaned


def _word_find_literal(text: str) -> str:
    out = text.replace("^", "^^")
    out = out.replace("\t", "^t")
    out = out.replace("\n", "^l")
    return out


def _translate_month_date_fragments(text: str) -> str:
    pattern = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})/(\d{4})\b")

    def repl(match: re.Match[str]) -> str:
        month = MONTH_NUMBER[match.group(1)]
        day = int(match.group(2))
        year = match.group(3)
        return f"{day:02d}.{month}.{year}"

    return pattern.sub(repl, text)


def _normalize_numeric_ranges(text: str) -> str:
    translated = re.sub(r"(?<=\d)\s+to\s+(?=\d)", "-", text, flags=re.IGNORECASE)
    translated = re.sub(r"(?<=\d)\s+and\s+(?=\d)", ", ", translated, flags=re.IGNORECASE)
    translated = re.sub(r"(?<=\d)\s+to\s*$", "-", translated, flags=re.IGNORECASE)
    translated = re.sub(r"(?<=\d)\s+and\s*$", ",", translated, flags=re.IGNORECASE)
    return translated


def _looks_code_like(text: str) -> bool:
    stripped = " ".join(text.split())
    if not stripped:
        return True
    if stripped in RESERVED_UNCHANGED:
        return True
    tokens = stripped.split()
    month_tokens = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
    if tokens and all(
        token in {"and", "to", "Blank"} or token in month_tokens or bool(re.fullmatch(r"[0-9./,-]+", token))
        for token in tokens
    ):
        return True
    return bool(CODE_LIKE_RE.fullmatch(stripped))


def translate_text(text: str) -> str:
    original = text
    if not LATIN_RE.search(text):
        return text
    stripped = " ".join(text.split())
    if stripped in EXACT_MAP:
        return EXACT_MAP[stripped]
    month_translated = _translate_month_date_fragments(original)
    range_normalized = _normalize_numeric_ranges(month_translated)
    if range_normalized != original and not LATIN_RE.search(range_normalized):
        return range_normalized
    if _looks_code_like(stripped):
        if range_normalized != original:
            return range_normalized
        if month_translated != original:
            return month_translated
        return original

    translated = range_normalized
    for pattern, replacement in REGEX_RULES:
        translated = pattern.sub(replacement, translated)
    stripped_after_regex = " ".join(translated.split())
    if stripped_after_regex in EXACT_MAP:
        return EXACT_MAP[stripped_after_regex]
    for source, target in sorted(PHRASE_RULES, key=lambda item: len(item[0]), reverse=True):
        translated = _replace_phrase(translated, source, target)
    translated = _cleanup_mixed_translation(translated)
    translated = translated.replace("  ", " ")
    return translated


def collect_unique_english_strings(docx_path: Path) -> list[str]:
    doc = Document(str(docx_path))
    segments = collect_segments(doc, include_headers=True, include_footers=True)
    unique: list[str] = []
    seen: set[str] = set()
    for seg in segments:
        text = seg.source_plain
        if not LATIN_RE.search(text):
            continue
        normalized = " ".join(text.split())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(text)
    unique.sort(key=len, reverse=True)
    return unique


def build_translation_map(unique_strings: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for source in unique_strings:
        target = translate_text(source)
        if target != source:
            mapping[source] = target
    return dict(sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True))


def find_unresolved(unique_strings: list[str], mapping: dict[str, str]) -> list[str]:
    unresolved: list[str] = []
    for source in unique_strings:
        if source in mapping:
            continue
        normalized = " ".join(source.split())
        if normalized in RESERVED_UNCHANGED:
            continue
        if _looks_code_like(normalized):
            continue
        unresolved.append(source)
    return unresolved


def _set_paragraph_text_minimal(paragraph, text: str) -> None:
    _, spans, _inline = paragraph_to_tagged(paragraph)
    span = spans[0] if spans else None
    _clear_paragraph_runs(paragraph)
    run = paragraph.add_run(text)
    if span is not None:
        _apply_style(run, span)


def replace_segments_direct(docx_path: Path, mapping: dict[str, str], *, min_len: int = 0) -> Counter:
    doc = Document(str(docx_path))
    segments = collect_segments(doc, include_headers=True, include_footers=True)
    counts: Counter[str] = Counter()
    changed = False
    for seg in segments:
        source = seg.source_plain
        target = mapping.get(source)
        if not target or len(source) < min_len:
            continue
        _set_paragraph_text_minimal(seg.paragraph_ref, target)
        counts[source] += 1
        changed = True
    if changed:
        doc.save(str(docx_path))
    return counts


def _iter_table_paragraphs(tables):
    for table in tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    yield paragraph
                yield from _iter_table_paragraphs(cell.tables)


def _iter_nested_tables(tables):
    for table in tables:
        yield table
        for row in table.rows:
            for cell in row.cells:
                yield from _iter_nested_tables(cell.tables)


def _iter_all_paragraphs(doc):
    yield from doc.paragraphs
    yield from _iter_table_paragraphs(doc.tables)
    for section in doc.sections:
        for part in (section.header, section.first_page_header, section.even_page_header):
            yield from part.paragraphs
            yield from _iter_table_paragraphs(part.tables)
        for part in (section.footer, section.first_page_footer, section.even_page_footer):
            yield from part.paragraphs
            yield from _iter_table_paragraphs(part.tables)


def _iter_all_tables(doc):
    yield from _iter_nested_tables(doc.tables)
    for section in doc.sections:
        for part in (section.header, section.first_page_header, section.even_page_header):
            yield from _iter_nested_tables(part.tables)
        for part in (section.footer, section.first_page_footer, section.even_page_footer):
            yield from _iter_nested_tables(part.tables)


def cleanup_remaining_paragraphs(docx_path: Path) -> Counter:
    doc = Document(str(docx_path))
    counts: Counter[str] = Counter()
    changed = False
    for paragraph in _iter_all_paragraphs(doc):
        source = paragraph.text
        if not source or not LATIN_RE.search(source):
            continue
        target = translate_text(source)
        if target == source:
            continue
        _set_paragraph_text_minimal(paragraph, target)
        counts[source] += 1
        changed = True
    if changed:
        doc.save(str(docx_path))
    return counts


def patch_header_footer_textboxes(docx_path: Path) -> int:
    replacements = {
        "<w:t>Page</w:t>": "<w:t>Стр.</w:t>",
        "Page ": "Стр. ",
        "No.": "№",
        "Jan": "янв.",
        "Feb": "февр.",
        "Mar": "мар.",
        "Apr": "апр.",
        "May": "мая",
        "Jun": "июн.",
        "Jul": "июл.",
        "Aug": "авг.",
        "Sep": "сент.",
        "Oct": "окт.",
        "Nov": "нояб.",
        "Dec": "дек.",
    }
    changed_files = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp_path = Path(tmp.name)
    try:
        with ZipFile(docx_path, "r") as zin, ZipFile(tmp_path, "w", compression=ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if re.fullmatch(r"word/(header|footer)\d+\.xml", info.filename):
                    text = data.decode("utf-8")
                    updated = text
                    updated = _translate_month_date_fragments(updated)
                    for source, target in replacements.items():
                        updated = updated.replace(source, target)
                    updated = _normalize_numeric_ranges(updated)
                    if updated != text:
                        data = updated.encode("utf-8")
                        changed_files += 1
                zout.writestr(info, data)
        shutil.move(str(tmp_path), str(docx_path))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return changed_files


def enforce_table_readability(docx_path: Path, *, min_font_size_pt: float = 9.0) -> int:
    doc = Document(str(docx_path))
    changed_runs = 0
    for table in _iter_all_tables(doc):
        table.autofit = True
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    fmt = paragraph.paragraph_format
                    fmt.space_before = Pt(0)
                    fmt.space_after = Pt(0)
                    fmt.line_spacing = 0.9
                    for run in paragraph.runs:
                        size = run.font.size.pt if run.font.size else None
                        if size is None or size < min_font_size_pt:
                            run.font.size = Pt(min_font_size_pt)
                            changed_runs += 1
    if changed_runs:
        doc.save(str(docx_path))
    return changed_runs


def auto_fix_layout(source_docx: Path, translated_docx: Path) -> dict[str, int]:
    source_doc = Document(str(source_docx))
    translated_doc = Document(str(translated_docx))
    source_segments = collect_segments(source_doc, include_headers=True, include_footers=True)
    translated_segments = collect_segments(translated_doc, include_headers=True, include_footers=True)

    source_by_loc = {seg.location: seg for seg in source_segments}
    aligned: list = []
    for seg in translated_segments:
        source_seg = source_by_loc.get(seg.location)
        if source_seg is None:
            continue
        seg.source_plain = source_seg.source_plain
        seg.target_tagged = seg.paragraph_ref.text
        aligned.append(seg)

    cfg = PipelineConfig(
        include_headers=True,
        include_footers=True,
        layout_check=True,
        layout_auto_fix=True,
        layout_auto_fix_passes=3,
        layout_font_reduction_pt=0.6,
        layout_spacing_factor=0.83,
    )
    total_issues = 0
    total_fixes = 0
    for _ in range(cfg.layout_auto_fix_passes):
        issues = validate_layout(translated_doc, aligned, cfg)
        total_issues += len(issues)
        fixed = fix_expansion_issues(aligned, issues, cfg)
        total_fixes += fixed
        if fixed == 0:
            break
    translated_doc.save(str(translated_docx))
    try:
        com_stats = update_fields_via_com(
            translated_docx,
            autofit_textboxes=True,
            min_font_size_pt=7.5,
            max_shrink_steps=6,
            expand_overflowing=False,
            max_height_growth=1.3,
        )
    except Exception as exc:
        com_stats = {
            "fields_updated": 0,
            "tocs_updated": 0,
            "textboxes_seen": 0,
            "textboxes_autofit": 0,
            "textboxes_shrunk": 0,
            "textboxes_expanded": 0,
            "com_autofit_error": str(exc),
        }
    return {
        "issues": total_issues,
        "auto_fixed_segments": total_fixes,
        **com_stats,
    }


def run_translation(source: Path, output: Path, report: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    unique_strings = collect_unique_english_strings(source)
    mapping = build_translation_map(unique_strings)
    unresolved = find_unresolved(unique_strings, mapping)

    shutil.copyfile(source, output)
    direct_counts = replace_segments_direct(output, mapping)
    cleanup_counts = cleanup_remaining_paragraphs(output)
    layout_stats = auto_fix_layout(source, output)
    table_font_floor_runs = enforce_table_readability(output)
    header_footer_patch_files = patch_header_footer_textboxes(output)

    report_payload = {
        "source": str(source),
        "output": str(output),
        "translation_context": {
            "glossary": str(ROOT / "for_test" / "new_formating" / "glossary.md"),
            "general_prompt": str(ROOT / "for_test" / "new_formating" / "general_prompt.md"),
        },
        "unique_english_strings": len(unique_strings),
        "mapped_strings": len(mapping),
        "unresolved_strings": unresolved,
        "replace_hits": int(sum(direct_counts.values())),
        "replaced_keys": int(len(direct_counts)),
        "cleanup_hits": int(sum(cleanup_counts.values())),
        "table_font_floor_runs": int(table_font_floor_runs),
        "header_footer_patch_files": int(header_footer_patch_files),
        "layout": layout_stats,
    }
    report.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate original_new_part1.docx into Russian with formatting preserved.")
    parser.add_argument(
        "--source",
        default=str(ROOT / "for_test" / "new_formating" / "section" / "original_new_part1.docx"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "for_test" / "new_formating" / "section_translate" / "original_new_part1_ru.docx"),
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "tmp" / "docs" / "part1_manual_translation_report.json"),
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    report = Path(args.report).resolve()
    report_payload = run_translation(source, output, report)
    print(json.dumps(report_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
