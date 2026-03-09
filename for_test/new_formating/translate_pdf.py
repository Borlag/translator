"""
Перевод PDF-файла с иллюстрациями CMM с английского на русский.
Использует pymupdf: редактирование текстовых слоёв, изображения не затрагиваются.
"""

import fitz  # pymupdf
import re
import os

INPUT_PDF = r"C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\picture.pdf"
OUTPUT_PDF = r"C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\picture_ru.pdf"
FONT_FILE = r"C:\Windows\Fonts\arial.ttf"
FONT_NAME = "arial"

# =============================================================================
# СЛОВАРЬ ПЕРЕВОДА — от самых длинных/конкретных к коротким/общим
# =============================================================================

# 1) ПОЛНЫЕ ФРАЗЫ / ТОЧНЫЕ СОВПАДЕНИЯ СТРОК (с учётом регистра)
FIXED = {
    # --- ШАПКА СТРАНИЦЫ ---
    "PART No. 201587001 AND 201587002 COMPONENT MAINTENANCE MANUAL":
        "ДЕТ. № 201587001 И 201587002 РУК. ПО ТО КОМПОНЕНТОВ",
    "MAIN LANDING GEAR LEG": "СТОЙКА ОСНОВНОГО ШАССИ",

    # --- НАЗВАНИЯ РИСУНКОВ (заглавная буква = из аннотации) ---
    "Bracket (2-120 and 2-130) - Protective Treatment":
        "Кронштейн (2-120 и 2-130) – Защитная обработка",
    "Bracket (4-140 and 4-150) - Protective Treatment":
        "Кронштейн (4-140 и 4-150) – Защитная обработка",
    "Bracket (5-160) - Protective Treatment":
        "Кронштейн (5-160) – Защитная обработка",
    "Bracket (6-170 and 6-180) - Protective Treatment":
        "Кронштейн (6-170 и 6-180) – Защитная обработка",
    "Bracket (2-80 and 2-90) - Protective Treatment":
        "Кронштейн (2-80 и 2-90) – Защитная обработка",
    "Bracket (4-210) - Protective Treatment":
        "Кронштейн (4-210) – Защитная обработка",
    "Bracket (8-170) - Protective Treatment":
        "Кронштейн (8-170) – Защитная обработка",
    "Bracket (9-150) Only - Protective Treatment":
        "Кронштейн (9-150) – Защитная обработка",
    "Bracket (9-150A) - Protective Treatment":
        "Кронштейн (9-150A) – Защитная обработка",
    "Transfer Block (2-340, 2-340A, 2-350 and 2-350A) - Protective Treatment":
        "Блок передачи нагрузки (2-340, 2-340A, 2-350 и 2-350A) – Защитная обработка",
    "Transfer Block (2-340B and 2-350B) - Protective Treatment":
        "Блок передачи нагрузки (2-340B и 2-350B) – Защитная обработка",
    "Slave Link (6-230) Only - Protective Treatment":
        "Ведомая тяга (6-230) – Защитная обработка",
    "Slave Link (6-230A) - Protective Treatment":
        "Ведомая тяга (6-230A) – Защитная обработка",
    "Pin (10-80) - Protective Treatment": "Штифт (10-80) – Защитная обработка",
    "Pin (11-130) - Protective Treatment": "Штифт (11-130) – Защитная обработка",
    "Pin (13-190 and 13-190A) - Protective Treatment":
        "Штифт (13-190 и 13-190A) – Защитная обработка",
    "Valve Stem (12-90) - Protective Treatment":
        "Шток клапана (12-90) – Защитная обработка",
    "Valve Stem (12-90A) - Protective Treatment":
        "Шток клапана (12-90A) – Защитная обработка",
    "Retaining Pin (13-10) - Protective Treatment":
        "Фиксирующий штифт (13-10) – Защитная обработка",
    "Inflation Valve (13-110 and 13-110A) - Protective Treatment":
        "Клапан накачки (13-110 и 13-110A) – Защитная обработка",
    "Upper Diaphragm Tube (15-390) - Protective Treatment":
        "Верхняя диафрагменная труба (15-390) – Защитная обработка",
    "Upper Diaphragm Tube (15-390A) - Protective Treatment":
        "Верхняя диафрагменная труба (15-390A) – Защитная обработка",
    "Upper Diaphragm Tube Subassembly (15-360) - Protective Treatment":
        "Подсборка верхней диафрагменной трубы (15-360) – Защитная обработка",
    "Upper Diaphragm Tube Subassembly (15-360A) - Protective Treatment":
        "Подсборка верхней диафрагменной трубы (15-360A) – Защитная обработка",
    "Cylinder (17-230) - Protective Treatment":
        "Цилиндр (17-230) – Защитная обработка",
    "Cylinder (17-230A) - Protective Treatment":
        "Цилиндр (17-230A) – Защитная обработка",
    "Sliding Tube (18-80) or (18-80A) - Protective Treatment":
        "Скользящая труба (18-80) или (18-80A) – Защитная обработка",
    "Sliding Tube (18-80B) - Protective Treatment":
        "Скользящая труба (18-80B) – Защитная обработка",
    "Sliding Tube (18-80B)- Protective Treatment":
        "Скользящая труба (18-80B) – Защитная обработка",
    "Sliding tube (18-80D, 18-80E, 18-80F and 18-80G) - Protective Treatment":
        "Скользящая труба (18-80D, 18-80E, 18-80F и 18-80G) – Защитная обработка",
    "Main Fitting Subassembly (20-90, 20-90A, 20-100 and 20-100A) - Protective Treatment":
        "Подсборка корпуса стойки (20-90, 20-90A, 20-100 и 20-100A) – Защитная обработка",
    "Main Fitting Subassembly (20-90B and 20-100B) - Protective Treatment":
        "Подсборка корпуса стойки (20-90B и 20-100B) – Защитная обработка",
    "Main Fitting (20-410) and (20-420) - Protective Treatment":
        "Корпус стойки (20-410) и (20-420) – Защитная обработка",
    "Main Fitting (20-410) and (20-420)- Protective Treatment":
        "Корпус стойки (20-410) и (20-420) – Защитная обработка",
    "Main Fitting (20-410A) and (20-420A) - Protective Treatment":
        "Корпус стойки (20-410A) и (20-420A) – Защитная обработка",
    "Main Fitting (20-410B, 20-420B, 20-410C, 20-420C, 20-410D and 20-420D) - Protective Treatment":
        "Корпус стойки (20-410B, 20-420B, 20-410C, 20-420C, 20-410D и 20-420D) – Защитная обработка",
    "Main Fitting (20-410B, 20-420B, 20-410D and 20-420D) - Protective Treatment":
        "Корпус стойки (20-410B, 20-420B, 20-410D и 20-420D) – Защитная обработка",
    "Upper Pivot Bracket (10-160) Only - Protective Treatment":
        "Верхний поворотный кронштейн (10-160) – Защитная обработка",
    "Upper Pivot Bracket (10-160A) - Protective Treatment":
        "Верхний поворотный кронштейн (10-160A) – Защитная обработка",
    "Harness Support Bracket (7-100, 7-100A, 7-110 and 7-110A) - Protective Treatment":
        "Кронштейн крепления жгута (7-100, 7-100A, 7-110 и 7-110A) – Защитная обработка",
    "Harness Support Bracket (11-140) - Protective Treatment":
        "Кронштейн крепления жгута (11-140) – Защитная обработка",
    "Forward Pintle Pin (1-60A) - Protective Treatment":
        "Передний штифт навески (1-60A) – Защитная обработка",
    "Spacer (4-180) Only - Protective Treatment":
        "Проставка (4-180) – Защитная обработка",
    "Drag-arm Spacer (4-180A) - Protective Treatment":
        "Проставка подкоса (4-180A) – Защитная обработка",
    "Spacer (9-190) Only - Protective Treatment":
        "Проставка (9-190) – Защитная обработка",
    "Spacer (9-190A) - Protective Treatment":
        "Проставка (9-190A) – Защитная обработка",
    "Spacer (11-30) - Protective Treatment":
        "Проставка (11-30) – Защитная обработка",
    "Uplock Pin (5-400A) - Protective Treatment":
        "Штифт замка убранного положения (5-400A) – Защитная обработка",
    "Figure Deleted": "Рисунок аннулирован",

    # --- ДИАГРАММЫ РЕМОНТОВ ---
    "Approved Repairs - Key Diagram": "Допустимые ремонты – Ключевая схема",
    "Main Fitting Repairs - Key Diagram": "Ремонты корпуса стойки – Ключевая схема",
    "T orque Link Repairs - Key Diagram": "Ремонты шлиц-шарнира – Ключевая схема",
    "Torque Link Repairs - Key Diagram": "Ремонты шлиц-шарнира – Ключевая схема",
    "Sliding Tube Repairs - Key Diagram": "Ремонты скользящей трубы – Ключевая схема",
    "Upper Diaphragm Tube Repairs - Key Diagram":
        "Ремонты верхней диафрагменной трубы – Ключевая схема",
    "Cylinder Repairs - Key Diagram": "Ремонты цилиндра – Ключевая схема",
    "Transfer Block Repairs - Key Diagram":
        "Ремонты блока передачи нагрузки – Ключевая схема",
    "Harness Support Bracket Repairs - Key Diagram":
        "Ремонты кронштейна крепления жгута – Ключевая схема",
    "Upper Pivot Bracket Repairs - Key Diagram":
        "Ремонты верхнего поворотного кронштейна – Ключевая схема",
    "Repair to Lower Bearing Subassembly": "Ремонт подсборки нижнего подшипника",
    "Repair to Pivot Pin": "Ремонт шарнирного штифта",
    "Repair to Uplock Pin": "Ремонт штифта замка убранного положения",
    "Repair to Sliding Tube": "Ремонт скользящей трубы",
    "Repair to Sliding Tube - Machining": "Ремонт скользящей трубы – мехобработка",
    "Repair to Pin": "Ремонт штифта",
    "Repair to Bracket": "Ремонт кронштейна",
    "Repair Bush - Machining and Installation":
        "Ремонтная втулка – мехобработка и установка",
    "Repair Bushes - Machining and Installation":
        "Ремонтные втулки – мехобработка и установка",

    # --- ДОПОЛНИТЕЛЬНЫЕ РЕМОНТНЫЕ РИСУНКИ (из скана) ---
    "Repair to Main Fitting": "Ремонт корпуса стойки",
    "Repair to Main Fitting - Machining": "Ремонт корпуса стойки – мехобработка",
    "Repair to Main Fitting - Repair Bush Installation":
        "Ремонт корпуса стойки – установка ремонтной втулки",
    "Repair to Main Fitting - Repair Bush Machining and Installation":
        "Ремонт корпуса стойки – мехобработка и установка ремонтной втулки",
    "Repair to Main Fitting - Chromium Plate Termination":
        "Ремонт корпуса стойки – завершение хромирования",
    "Repair to Main Fitting - Bush": "Ремонт корпуса стойки – втулка",
    "Oversize Bushes - Machining and Installation":
        "Ремонтные втулки – мехобработка и установка",
    "Oversize Bush(es) - Machining and Installation":
        "Ремонтные втулки – мехобработка и установка",
    "Repair Bearing - Machining and Installation":
        "Ремонтный подшипник – мехобработка и установка",
    "Oversize Bearings - Machining and Installation":
        "Ремонтные подшипники – мехобработка и установка",
    "Oversize Lubrication adapter - Installation":
        "Ремонтный смазочный адаптер – установка",
    "Lower Bearing Subassembly Machining and Liner Installation":
        "Мехобработка подсборки нижнего подшипника и установка вкладыша",
    "Lower Bearing Subassembly - Machining and Inner Liner Installation":
        "Подсборка нижнего подшипника – мехобработка и установка внутреннего вкладыша",
    "Oversize Transfer Dowel - Installation": "Ремонтный штифт передачи – установка",

    # --- ПОЛНЫЕ СТРОКИ-АННОТАЦИИ (из ремонтных рисунков) ---
    "DIAMETER A MUST BE FOLLOW THE LINE OF EXISTING BORE.":
        "ДИАМЕТР A ДОЛЖЕН СЛЕДОВАТЬ ОСИ СУЩЕСТВУЮЩЕГО ОТВЕРСТИЯ.",
    "DIAMETER(S) A AND/OR B MUST FOLLOW THE LINE OF EXISTING BORES.":
        "ДИАМЕТРЫ A И/ИЛИ B ДОЛЖНЫ СЛЕДОВАТЬ ОСИ СУЩЕСТВУЮЩИХ ОТВЕРСТИЙ.",
    "DIAMETER A MUST FOLLOW THE AXIS OF EXISTING BORE.":
        "ДИАМЕТР A ДОЛЖЕН СЛЕДОВАТЬ ОСИ СУЩЕСТВУЮЩЕГО ОТВЕРСТИЯ.",
    "DIAMETERS A AND B MUST FOLLOW THE AXIS OF EXISTING BORE.":
        "ДИАМЕТРЫ A И B ДОЛЖНЫ СЛЕДОВАТЬ ОСИ СУЩЕСТВУЮЩЕГО ОТВЕРСТИЯ.",
    "DIAMETERS A AND B MUST FOLLOW THE AXIS OF EXISTING BORES.":
        "ДИАМЕТРЫ A И B ДОЛЖНЫ СЛЕДОВАТЬ ОСИ СУЩЕСТВУЮЩИХ ОТВЕРСТИЙ.",
    "DIAMETERS A, B, C AND D MUST FOLLOW THE AXIS OF EXISTING BORES.":
        "ДИАМЕТРЫ A, B, C И D ДОЛЖНЫ СЛЕДОВАТЬ ОСИ СУЩЕСТВУЮЩИХ ОТВЕРСТИЙ.",
    "APPLY LOCTITE GRADE 270 TO ADAPTOR INTERFACE WITH MAIN FITTING: REFER TO PCS-5303.":
        "НАНЕСТИ LOCTITE GRADE 270 НА СТЫКОВУЮ ПОВЕРХНОСТЬ АДАПТЕРА С КОРПУСОМ СТОЙКИ: СМ. PCS-5303.",
    "INSTALL THE APPLICABLE LUBRICATION ADAPTOR: REFER TO TABLE 601.":
        "УСТАНОВИТЬ СООТВЕТСТВУЮЩИЙ СМАЗОЧНЫЙ АДАПТЕР: СМ. ТАБЛИЦУ 601.",
    "ALL CAVITIES AND VOIDS MUST BE FILLED TO PREVENT MOISTURE INGRESS.":
        "ВСЕ ПОЛОСТИ И ПУСТОТЫ ДОЛЖНЫ БЫТЬ ЗАПОЛНЕНЫ ДЛЯ ПРЕДОТВРАЩЕНИЯ ПОПАДАНИЯ ВЛАГИ.",
    "APPLY MOLYKOTE 111 TO THE BOLT SHANKS, THREADS, UNDERCUTS AND ALL INTERFACES":
        "НАНЕСТИ MOLYKOTE 111 НА СТЕРЖНИ БОЛТОВ, РЕЗЬБУ, ПРОТОЧКИ И ВСЕ СТЫКОВЫЕ ПОВЕРХНОСТИ",
    "BETWEEN MATING PARTS MUST BE COATED BEFORE ASSEMBLY: REFER TO PCS-7303.":
        "СОПРЯГАЕМЫХ ДЕТАЛЕЙ ДО СБОРКИ: СМ. PCS-7303.",
    "APPLY A FULL BEAD OF SEALANT, PR340-2 WITH A MAXIMUM HEIGHT OF":
        "НАНЕСТИ ПОЛНЫЙ ВАЛИК ГЕРМЕТИКА PR340-2 С МАКСИМАЛЬНОЙ ВЫСОТОЙ",
    "1,000mm (0.0394in) ABOVE ADJOINING SURFACES: REFER TO PCS-7200.":
        "1,000 мм (0,0394 дюйм.) НАД ПРИЛЕГАЮЩИМИ ПОВЕРХНОСТЯМИ: СМ. PCS-7200.",
    "APPLY SEALANT: REFER TO PCS-7200 TYPE 2.": "НАНЕСТИ ГЕРМЕТИК: СМ. PCS-7200 ТИП 2.",
    "APPLY FILLET SEALANT: REFER TO PCS-7200. MAKE SURE THAT THE SEALANT":
        "НАНЕСТИ ГЕРМЕТИК УГЛОВЫМ ШВОМ: СМ. PCS-7200. УБЕДИТЬСЯ, ЧТО ГЕРМЕТИК",
    "COMPLETELY COVERS EXPOSED PRIMER PAINT.":
        "ПОЛНОСТЬЮ ЗАКРЫВАЕТ ОТКРЫТУЮ ГРУНТОВОЧНУЮ КРАСКУ.",
    "APPLY FILLET SEALANT: REFER TO PCS-7200. MAKE SURE THAT THE SEALANT COMPLETELY":
        "НАНЕСТИ ГЕРМЕТИК УГЛОВЫМ ШВОМ: СМ. PCS-7200. УБЕДИТЬСЯ, ЧТО ГЕРМЕТИК ПОЛНОСТЬЮ",
    "COVERS EXPOSED PRIMER PAINT.": "ЗАКРЫВАЕТ ОТКРЫТУЮ ГРУНТОВОЧНУЮ КРАСКУ.",
    "DEBURR THE SHARP EDGES WITH 0,500 to 1,000mm (0.0197 to 0.0394in) RAD.":
        "СНЯТЬ ЗАУСЕНЦЫ С ОСТРЫХ КРОМОК РАДИУСОМ 0,500–1,000 мм (0,0197–0,0394 дюйм.).",
    "DEBURR THE SHARP EDGES WITH 0,130 to 0,380mm (0.0051 to 0.0150in) RAD.":
        "СНЯТЬ ЗАУСЕНЦЫ С ОСТРЫХ КРОМОК РАДИУСОМ 0,130–0,380 мм (0,0051–0,0150 дюйм.).",
    "UNLESS GIVEN DIFFERENTLY. ": "ЕСЛИ НЕ ЗАДАНО ИНАЧЕ.",
    "UNLESS GIVEN DIFFERENTLY.": "ЕСЛИ НЕ ЗАДАНО ИНАЧЕ.",
    "PRIMER PAINT ONLY: REFER TO PCS-2500. NO WITNESS OF TOP COAT PAINT PERMITTED":
        "ТОЛЬКО ГРУНТОВОЧНАЯ КРАСКА: СМ. PCS-2500. СЛЕДЫ ФИНИШНОГО ПОКРЫТИЯ НЕДОПУСТИМЫ",
    "ON THESE SURFACES.": "НА ЭТИХ ПОВЕРХНОСТЯХ.",
    "NITRIDING DEPTH 0,18 to 0,23mm (0.007 to 0.009in), 0,02 to 0,04mm (0.0008 TO":
        "ГЛУБИНА АЗОТИРОВАНИЯ 0,18–0,23 мм (0,007–0,009 дюйм.), 0,02–0,04 мм (0,0008–",
    "0.0016in) REMOVAL OVER AREA SHOWN 750HV MIN.":
        "0,0016 дюйм.) СЪЁМ В УКАЗАННОЙ ЗОНЕ 750HV МИН.",
    "DIAMETER H AND CORNER RADIUS ARE TO BE MACHINED ON RECEIPT":
        "ДИАМЕТР H И УГЛОВОЙ РАДИУС ДОЛЖНЫ БЫТЬ ОБРАБОТАНЫ ПО ПОЛУЧЕНИИ",
    "BY OVERHAUL AGENCY TO SUIT INDIVIDUAL BORE OF MAIN FITTING.":
        "РЕМОНТНЫМ ПРЕДПРИЯТИЕМ ПОД ИНДИВИДУАЛЬНОЕ ОТВЕРСТИЕ КОРПУСА СТОЙКИ.",
    "LOCALLY APPLY CADMIUM PLATE TO THE REWORKED OUTER DIAMETER":
        "МЕСТНО НАНЕСТИ КАДМИРОВАНИЕ НА ДОРАБОТАННЫЙ НАРУЖНЫЙ ДИАМЕТР",
    "OF THE BUSHES: REFER TO PCS-2141.": "ВТУЛОК: СМ. PCS-2141.",
    "APPLY CADMIUM PLATE ALL OVER: REFER TO PCS-2101. THE CADMIUM PLATE":
        "НАНЕСТИ КАДМИРОВАНИЕ ПО ВСЕЙ ПОВЕРХНОСТИ: СМ. PCS-2101. КАДМИРОВАНИЕ",
    "THICKNESS MUST BE BETWEEN 0,010 to 0,015mm (0.0004 to 0.0006in).":
        "ДОЛЖНО БЫТЬ ТОЛЩИНОЙ 0,010–0,015 мм (0,0004–0,0006 дюйм.).",
    "APPLY CADMIUM PLATE ALL OVER: REFER TO PCS-2101. THE PLATING THICKNESS ":
        "НАНЕСТИ КАДМИРОВАНИЕ ПО ВСЕЙ ПОВЕРХНОСТИ: СМ. PCS-2101. ТОЛЩИНА ПОКРЫТИЯ",
    "APPLY CADMIUM PLATE ALL OVER: REFER TO PCS-2101. THE PLATING THICKNESS":
        "НАНЕСТИ КАДМИРОВАНИЕ ПО ВСЕЙ ПОВЕРХНОСТИ: СМ. PCS-2101. ТОЛЩИНА ПОКРЫТИЯ",
    "MUST BE BETWEEN 0,010 to 0,015mm (0.0004 to 0.0006in).":
        "ДОЛЖНА БЫТЬ В ДИАПАЗОНЕ 0,010–0,015 мм (0,0004–0,0006 дюйм.).",
    "REMOVE THE BREAK EDGES WITHIN 0,500 to 2,000mm (0.0197 to 0.0787in) RAD.":
        "СНЯТЬ ОСТРЫЕ КРОМКИ В ПРЕДЕЛАХ РАДИУСА 0,500–2,000 мм (0,0197–0,0787 дюйм.).",
    "REMOVE THE SHARP EDGES OF 0,250 to 1,000mm (0.0098 to 0.0394in) RAD.":
        "СНЯТЬ ОСТРЫЕ КРОМКИ РАДИУСОМ 0,250–1,000 мм (0,0098–0,0394 дюйм.).",
    "NO PRIMER PAINT TO BE": "ГРУНТОВОЧНАЯ КРАСКА НЕ ДОЛЖНА БЫТЬ",
    "VISIBLE AFTER SEALANT": "ВИДНА ПОСЛЕ НАНЕСЕНИЯ ГЕРМЕТИКА",
    "NO PRIMER PAINT TO BE VISIBLE": "ГРУНТОВОЧНАЯ КРАСКА НЕ ДОЛЖНА БЫТЬ ВИДНА",
    "AFTER SEALANT APPLICATION": "ПОСЛЕ НАНЕСЕНИЯ ГЕРМЕТИКА",
    "CADMIUM PLATE OPTIONAL AND NO PAINT.":
        "КАДМИРОВАНИЕ ОПЦИОНАЛЬНО, КРАСКА НЕ НАНОСИТСЯ.",
    "FOR THE MAIN FITTING (20-410B), (20-410C), (20-420B) and (20-420C):":
        "ДЛЯ КОРПУСА СТОЙКИ (20-410B), (20-410C), (20-420B) и (20-420C):",
    "THE MINIMUM WALL THICKNESS IS 15,382mm (0.6056in).":
        "МИНИМАЛЬНАЯ ТОЛЩИНА СТЕНКИ 15,382 мм (0,6056 дюйм.).",
    "FOR THE MAIN FITTING (20-410D) and (20-420D):":
        "ДЛЯ КОРПУСА СТОЙКИ (20-410D) и (20-420D):",
    "THE MINIMUM WALL THICKNESS IS 15,582mm (0.6134in).":
        "МИНИМАЛЬНАЯ ТОЛЩИНА СТЕНКИ 15,582 мм (0,6134 дюйм.).",
    "CADMIUM PLATE MUST OVERLAP": "КАДМИРОВАНИЕ ДОЛЖНО ПЕРЕКРЫВАТЬ",
    # --- Split-line patterns from PDF (individual lines) ---
    "DIAMETERS A MUST FOLLOW": "ДИАМЕТРЫ A ДОЛЖНЫ СЛЕДОВАТЬ",
    "THE LINE OF EXISTING BORES": "ЛИНИИ СУЩЕСТВУЮЩИХ ОТВЕРСТИЙ",
    "THE LINE OF EXISTING BORE": "ЛИНИИ СУЩЕСТВУЮЩЕГО ОТВЕРСТИЯ",
    "DIAMETER A MUST FOLLOW": "ДИАМЕТР A ДОЛЖЕН СЛЕДОВАТЬ",
    "DIAMETER A MUST BE FOLLOW THE LINE OF EXISTING BORE.":
        "ДИАМЕТР A ДОЛЖЕН СЛЕДОВАТЬ ЛИНИИ СУЩЕСТВУЮЩЕГО ОТВЕРСТИЯ.",
    "DIAMETER A MUST FOLLOW THE AXIS OF EXISTING BORE.":
        "ДИАМЕТР A ДОЛЖЕН СЛЕДОВАТЬ ОСИ СУЩЕСТВУЮЩЕГО ОТВЕРСТИЯ.",
    "ORIENTATION IS IMPORTANT": "ОРИЕНТАЦИЯ ВАЖНА",
    "(HOLE TO DEPTH OF": "(ОТВЕРСТИЕ НА ГЛУБИНУ",
    "FROM THIS SURFACE)": "ОТ ЭТОЙ ПОВЕРХНОСТИ)",
    "FROM THIS SURFACE": "ОТ ЭТОЙ ПОВЕРХНОСТИ",
    "PRIMER PAINT ONLY: REFER TO PCS-2500. NO WITNESS OF TOP COAT PAINT PERMITTED":
        "ТОЛЬКО ГРУНТОВОЧНАЯ КРАСКА: СМ. PCS-2500. СЛЕДЫ ФИНИШНОЙ КРАСКИ НЕДОПУСТИМЫ",
    "ON THESE SURFACES.": "НА ЭТИХ ПОВЕРХНОСТЯХ.",
    "IF THE BASE METAL": "ЕСЛИ ОСНОВНОЙ МЕТАЛЛ",
    "IS NOT DAMAGED": "НЕ ПОВРЕЖДЁН",
    "IS DAMAGED": "ПОВРЕЖДЁН",
    "OVERSIZE BUSH 450237810": "РЕМОНТНАЯ ВТУЛКА 450237810",
    "(WITH LUBRICATION ADAPTOR)": "(С СМАЗОЧНЫМ АДАПТЕРОМ)",
    "(WITHOUT LUBRICATION ADAPTOR)": "(БЕЗ СМАЗОЧНОГО АДАПТЕРА)",
    "(WITHOUT OVERSIZE REAR SPHERICAL BEARING)": "(БЕЗ РЕМОНТНОГО ЗАДНЕГО СФЕРИЧЕСКОГО ПОДШИПНИКА)",
    "(WITH OVERSIZE REAR SPHERICAL BEARING)": "(С РЕМОНТНЫМ ЗАДНИМ СФЕРИЧЕСКИМ ПОДШИПНИКОМ)",
    "NITRIDING DEPTH 0,18 to 0,23mm (0.007 to 0.009in), 0,02 to 0,04mm (0.0008 TO":
        "ГЛУБИНА АЗОТИРОВАНИЯ 0,18–0,23 мм (0,007–0,009 дюйм.), 0,02–0,04 мм (0,0008–",
    "0.0016in) REMOVAL OVER AREA SHOWN 750HV MIN.":
        "0,0016 дюйм.) УДАЛЕНИЕ ПО УКАЗАННОЙ ЗОНЕ. МИН. 750HV.",
    "ARE TO BE NITRIDED BEFORE DESPATCH TO OVERHAUL AGENCY.":
        "ПОДЛЕЖАТ АЗОТИРОВАНИЮ ПЕРЕД ОТПРАВКОЙ В РЕМОНТНУЮ ОРГАНИЗАЦИЮ.",
    "DIAMETER H AND CORNER RADIUS ARE TO BE MACHINED ON RECEIPT":
        "ДИАМЕТР H И РАДИУС СКРУГЛЕНИЯ ПОДЛЕЖАТ МЕХОБРАБОТКЕ ПРИ ПОЛУЧЕНИИ",
    "BY OVERHAUL AGENCY TO SUIT INDIVIDUAL BORE OF MAIN FITTING.":
        "РЕМОНТНОЙ ОРГАНИЗАЦИЕЙ ПОД КОНКРЕТНОЕ ОТВЕРСТИЕ КОРПУСА СТОЙКИ.",
    "LOCALLY APPLY CADMIUM PLATE TO THE REWORKED OUTER DIAMETER":
        "МЕСТНО НАНЕСТИ КАДМИРОВАНИЕ НА ПЕРЕРАБОТАННЫЙ НАРУЖНЫЙ ДИАМЕТР",
    "OF THE BUSHES: REFER TO PCS-2141.":
        "ВТУЛОК: СМ. PCS-2141.",
    "APPLY MOLYKOTE 111 TO THE BOLT SHANKS, THREADS, UNDERCUTS AND ALL INTERFACES":
        "НАНЕСТИ MOLYKOTE 111 НА СТЕРЖНИ БОЛТОВ, РЕЗЬБУ, ПОДРЕЗЫ И ВСЕ СОПРЯЖЁННЫЕ",
    "BETWEEN MATING PARTS MUST BE COATED BEFORE ASSEMBLY: REFER TO PCS-7303.":
        "ПОВЕРХНОСТИ СОПРЯГАЕМЫХ ДЕТАЛЕЙ ДО СБОРКИ: СМ. PCS-7303.",
    "ALL CAVITIES AND VOIDS MUST BE FILLED TO PREVENT MOISTURE INGRESS.":
        "ВСЕ ПОЛОСТИ ДОЛЖНЫ БЫТЬ ЗАПОЛНЕНЫ ДЛЯ ПРЕДОТВРАЩЕНИЯ ПОПАДАНИЯ ВЛАГИ.",
    "APPLY A FULL BEAD OF SEALANT, PR340-2 WITH A MAXIMUM HEIGHT OF":
        "НАНЕСТИ ПОЛНЫЙ ВАЛИК ГЕРМЕТИКА PR340-2 С МАКСИМАЛЬНОЙ ВЫСОТОЙ",
    "1,000mm (0.0394in) ABOVE ADJOINING SURFACES: REFER TO PCS-7200.":
        "1,000 мм (0,0394 дюйм.) НАД ПРИЛЕГАЮЩИМИ ПОВЕРХНОСТЯМИ: СМ. PCS-7200.",
    "APPLY SEALANT: REFER TO PCS-7200 TYPE 2.":
        "НАНЕСТИ ГЕРМЕТИК: СМ. PCS-7200, ТИП 2.",
    "NOTE: REPAIR BUSHES 450237351, 450237352, 450237353, 450237354 AND 450237355":
        "ПРИМЕЧАНИЕ: РЕМОНТНЫЕ ВТУЛКИ 450237351, 450237352, 450237353, 450237354 И 450237355",
    "OVERSIZE SPHERICAL": "РЕМОНТНЫЙ СФЕРИЧЕСКИЙ",
    "BEARING ASSEMBLY": "ПОДСБОРКА ПОДШИПНИКА",
    "REFER TO FIGURE 6001": "СМ. РИСУНОК 6001",
    # --- p18: dimension lines with lowercase "for" ---
    "305,00mm (12.000in) for (18-80)": "305,00 мм (12.000 дюйм.) для (18-80)",
    "220,00mm (8.661in) for (18-80A)": "220,00 мм (8.661 дюйм.) для (18-80A)",
    # --- p18: CENTERLINE ---
    "CENTERLINE OF": "ОСЬ",
    "CENTERLINE OF SLIDING TUBE": "ОСЬ СКОЛЬЗЯЩЕЙ ТРУБЫ",
    # --- p18: LIMIT OF SERMETEL W ---
    "LIMIT OF SERMETEL W": "ГРАНИЦА SERMETEL W",
    # --- p67: TO REMAIN ON WORKING DIA. ---
    "TO REMAIN ON WORKING DIA.": "НЕ ДОЛЖНО ОСТАВАТЬСЯ НА РАБОЧЕМ ДИА.",
    # --- p69: REPAIR No. patterns without space ---
    "REPAIR No.11-6 and 11-20": "РЕМОНТ №11-6 и 11-20",
    "REPAIR No.11-10, 11-16, 11-26,": "РЕМОНТ №11-10, 11-16, 11-26,",
    "REPAIR No.11-17": "РЕМОНТ №11-17",
    "REPAIR No.9-7": "РЕМОНТ №9-7",
    # --- p79: GRINDING standalone and TEMINATION (typo in source) ---
    "IT CAN BE FINISHED BY GRINDING": "КРАЯ МОГУТ БЫТЬ ОТШЛИФОВАНЫ",
    "TEMINATION TO M-DLPS1031-6": "ЗАВЕРШЕНИЕ ПО M-DLPS1031-6",
    "TEMINATION TO M-DLPS1031-1": "ЗАВЕРШЕНИЕ ПО M-DLPS1031-1",
    "TEMINATION TO M-DLPS1031-3": "ЗАВЕРШЕНИЕ ПО M-DLPS1031-3",
    # --- p91: THICKNESS OF / NOT EXTEND BEYOND ---
    "THICKNESS OF": "ТОЛЩИНА",
    "ABOVE OUTER": "НАД НАРУЖНЫМ",
    "NOT EXTEND BEYOND THE": "НЕ ДОЛЖНО ВЫХОДИТЬ ЗА",
    # --- p105: SERMETAL (typo in source) ---
    "SERMETAL COATING AND": "ПОКРЫТИЕ SERMETAL И",
    "OVER THIS LENGTH": "НА ЭТОЙ ДЛИНЕ",
    # --- p108: DIMENSION / REFERENCE / SECONDS ---
    "DIMENSION B": "РАЗМЕР B",
    "DIMENSION D": "РАЗМЕР D",
    "DIMENSION E": "РАЗМЕР E",
    "(REFERENCE)": "(СПРАВОЧНО)",
    "REFERENCE BOTTOM": "СПРАВОЧНО НИЖНИЙ",
    # --- p113: ENTRY OF LUBRICATION ---
    "ENTRY OF LUBRICATION": "ВХОД СМАЗКИ",
    "OVER RADIUS.": "ПО РАДИУСУ.",
    # --- p126/148: FIGURE 602 standalone ---
    "FIGURE 602)": "РИСУНОК 602)",
    "FIGURE 602": "РИСУНОК 602",
    # --- p128: DRAIN HOLE ---
    "DRAIN HOLE J": "ДРЕНАЖНОЕ ОТВЕРСТИЕ J",
    # --- p132: CHECK HONE ---
    "CHECK HONE": "КОНТРОЛЬНАЯ ХОНИНГОВА",
    # --- p133: TO EXISTING RADIUS ---
    "TO EXISTING RADIUS": "ДО СУЩЕСТВУЮЩЕГО РАДИУСА",
    # --- p134: MACHINE standalone ---
    "TABLE 1": "ТАБЛИЦА 1",
    # --- p156: AS LISTED IN TABLE 1 ---
    "AS LISTED IN TABLE 1": "СОГЛАСНО ТАБЛИЦЕ 1",
    "AS IN TABLE 602": "СОГЛАСНО ТАБЛИЦЕ 602",
    # --- p157: TO INTERSECTION ---
    "TO INTERSECTION": "ДО ПЕРЕСЕЧЕНИЯ",
    # --- p161: BACKING RING / O RING standalone ---
    "BACKING RING": "ОПОРНОЕ КОЛЬЦО",
    "O RING": "УПЛОТНИТЕЛЬНОЕ КОЛЬЦО",
    # --- p177/180: standalone words ---
    "APPLICATION": "НАНЕСЕНИЕ",
    "ORIENTATION": "ОРИЕНТАЦИЯ",
    "VIEW ON W": "ВИД ПО СТРЕЛКЕ W",
    "VIEW ON V": "ВИД ПО СТРЕЛКЕ V",
    "VIEW ON": "ВИД ПО СТРЕЛКЕ",
    "LARGER VIEW AT": "УВЕЛИЧЕННЫЙ ВИД",
    "90 DEGREES ROTATED": "ПОВЁРНУТО НА 90 ГРАДУСОВ",
}

# 2) УПОРЯДОЧЕННЫЙ СПИСОК ЗАМЕН (длинные → короткие, применяются через str.replace)
PHRASES = [
    # --- СЕЧЕНИЯ / ВИДЫ / ДЕТАЛИ (с буквой/парой букв) ---
    # обрабатываются через regex ниже, но на случай точных совпадений:
    ("FOR ALL SECTION VIEWS SEE SHEET 2", "ВСЕ СЕЧЕНИЯ – СМ. ЛИСТ 2"),
    ("VIEW ON ARROW Z", "ВИД ПО СТРЕЛКЕ Z"),
    ("VIEW ON ARROW", "ВИД ПО СТРЕЛКЕ"),
    ("LARGER VIEW AT A", "УВЕЛИЧЕННЫЙ ВИД A"),
    ("LARGER VIEW AT Y", "УВЕЛИЧЕННЫЙ ВИД Y"),
    ("LARGER VIEW AT Z", "УВЕЛИЧЕННЫЙ ВИД Z"),
    ("ENLARGED DETAIL Z", "УВЕЛИЧЕННАЯ ДЕТАЛЬ Z"),
    ("ENLARGED DETAIL", "УВЕЛИЧЕННАЯ ДЕТАЛЬ"),
    ("NOT TO SCALE", "БЕЗ МАСШТАБА"),
    ("LARGER DETAIL Z NOT TO SCALE", "УВЕЛИЧЕННАЯ ДЕТАЛЬ Z БЕЗ МАСШТАБА"),
    ("LARGER DETAIL Z", "УВЕЛИЧЕННАЯ ДЕТАЛЬ Z"),

    # --- ХРОМОВОЕ ПОКРЫТИЕ ---
    ("NO ZINC-NICKEL OR PAINT DEPOSIT TO REMAIN ON OR PROUD OF WORKING DIA. AFTER GRINDING CHROME",
     "ЦИНК-НИКЕЛЕВОЕ ИЛИ КРАСОЧНОЕ ПОКРЫТИЕ НА РАБОЧЕМ ДИАМЕТРЕ НЕДОПУСТИМО ПОСЛЕ ШЛИФОВАНИЯ ХРОМА"),
    ("NO ZINC-NICKEL OR PAINT DEPOSIT TO REMAIN ON WORKING DIA. AFTER CHROME PLATING",
     "ЦИНК-НИКЕЛЕВОЕ ИЛИ КРАСОЧНОЕ ПОКРЫТИЕ НА РАБОЧЕМ ДИАМЕТРЕ ПОСЛЕ ХРОМИРОВАНИЯ НЕДОПУСТИМО"),
    ("NO ZINC-NICKEL OR PAINT DEPOSIT TO REMAIN ON WORKING DIA. AFTER CHROME PLATING",
     "ЦИНК-НИКЕЛЕВОЕ ИЛИ КРАСОЧНОЕ ПОКРЫТИЕ НА РАБОЧЕМ ДИАМЕТРЕ ПОСЛЕ ХРОМИРОВАНИЯ НЕДОПУСТИМО"),
    ("NO ZINC-NICKEL OR PAINT ON THIS SURFACE", "БЕЗ ЦИНК-НИКЕЛЯ ИЛИ КРАСКИ НА ЭТОЙ ПОВЕРХНОСТИ"),
    ("NO ZINC-NICKEL OR PAINT DEPOSIT", "БЕЗ ЦИНК-НИКЕЛЕВОГО ИЛИ КРАСОЧНОГО ПОКРЫТИЯ"),
    ("NO ZINC-NICKEL OR", "БЕЗ ЦИНК-НИКЕЛЯ ИЛИ"),
    ("PAINT DEPOSIT TO REMAIN ON OR PROUD OF WORKING DIA.", "КРАСОЧНОЕ ПОКРЫТИЕ НА РАБОЧЕМ ДИАМЕТРЕ НЕДОПУСТИМО"),
    ("PAINT DEPOSIT TO REMAIN ON", "КРАСКА НЕ ДОЛЖНА ОСТАВАТЬСЯ НА"),
    ("PAINT DEPOSIT TO REMAIN", "КРАСКА НЕ ДОЛЖНА ОСТАВАТЬСЯ"),
    ("DEPOSIT TO REMAIN ON OR", "НЕ ДОЛЖНО ОСТАВАТЬСЯ ИЛИ"),
    ("DEPOSIT TO REMAIN ON", "НЕ ДОЛЖНО ОСТАВАТЬСЯ НА"),
    ("DEPOSIT TO REMAIN", "НЕ ДОЛЖНО ОСТАВАТЬСЯ"),
    ("ON OR PROUD OF WORKING DIA.", "ИЛИ ВЫСТУПАТЬ НАД РАБОЧИМ ДИА."),
    ("OR PROUD OF WORKING DIA.", "ИЛИ ВЫСТУПАТЬ НАД РАБОЧИМ ДИА."),
    ("PROUD OF WORKING DIA.", "ВЫСТУПАТЬ НАД РАБОЧИМ ДИА."),
    ("WORKING DIA.", "РАБОЧИЙ ДИА."),
    ("ZINC-NICKEL DEPOSIT OVERLAP AND RUN OUT BAND", "ЗОНА ПЕРЕКРЫТИЯ И ВЫБЕГА ЦИНК-НИКЕЛЕВОГО ПОКРЫТИЯ"),
    ("ZINC-NICKEL DEPOSIT OVERLAP", "ПЕРЕКРЫТИЕ ЦИНК-НИКЕЛЕВОГО ПОКРЫТИЯ"),
    ("ZINC-NICKEL DEPOSIT", "СЛОЙ ЦИНК-НИКЕЛЕВОГО ПОКРЫТИЯ"),
    ("EXTERNAL THICK ZINC-NICKEL PLATING LIMIT", "ГРАНИЦА ТОЛСТОГО ЦИНК-НИКЕЛЕВОГО ПОКРЫТИЯ (СНАРУЖИ)"),
    ("INTERNAL THICK ZINC-NICKEL PLATING LIMIT", "ГРАНИЦА ТОЛСТОГО ЦИНК-НИКЕЛЕВОГО ПОКРЫТИЯ (ИЗНУТРИ)"),
    ("EXTERNAL THICK ZINC-NICKEL", "НАРУЖНЫЙ ТОЛСТЫЙ СЛОЙ ЦИНК-НИКЕЛЯ"),
    ("INTERNAL THICK ZINC-NICKEL", "ВНУТРЕННИЙ ТОЛСТЫЙ СЛОЙ ЦИНК-НИКЕЛЯ"),
    ("PLATING LIMIT", "ГРАНИЦА ПОКРЫТИЯ"),
    ("ZINC NICKEL PLATE", "ЦИНК-НИКЕЛЕВОЕ ПОКРЫТИЕ"),
    ("ZINC-NICKEL", "ЦИНК-НИКЕЛЬ"),

    ("CHROME PLATING WILL TERMINATE ANYWHERE ON THE CHAMFER DIA. AFTER CHROME PLATING",
     "ХРОМОВОЕ ПОКРЫТИЕ МОЖЕТ ОКАНЧИВАТЬСЯ В ЛЮБОМ МЕСТЕ ДИАМЕТРА ФАСКИ ПОСЛЕ ХРОМИРОВАНИЯ"),
    ("CHROMIUM PLATE CAN STOP ANYWHERE ON THE CHAMFER. IT CAN BE FINISHED BY GRINDING",
     "ХРОМИРОВАНИЕ МОЖЕТ ЗАКАНЧИВАТЬСЯ В ЛЮБОМ МЕСТЕ ФАСКИ. КРАЯ МОГУТ БЫТЬ ОТШЛИФОВАНЫ"),
    ("CHROMIUM PLATE MAY TERMINATE", "ХРОМИРОВАНИЕ МОЖЕТ ОКАНЧИВАТЬСЯ"),
    ("MAY TERMINATE ANYWHERE ON THIS RADIUS", "МОЖЕТ ОКАНЧИВАТЬСЯ В ЛЮБОМ МЕСТЕ НА ЭТОМ РАДИУСЕ"),
    ("ANYWHERE ON THIS RADIUS", "В ЛЮБОМ МЕСТЕ НА ЭТОМ РАДИУСЕ"),
    ("ON THIS RADIUS", "НА ЭТОМ РАДИУСЕ"),
    ("MAY TERMINATE ANYWHERE", "МОЖЕТ ОКАНЧИВАТЬСЯ В ЛЮБОМ МЕСТЕ"),
    ("MAY TERMINATE", "МОЖЕТ ОКАНЧИВАТЬСЯ"),
    ("CAN STOP", "МОЖЕТ ЗАКАНЧИВАТЬСЯ"),
    ("CHROMIUM PLATE AND PAINT TO OVERLAP ON CHROMIUM RADIUS",
     "ХРОМИРОВАНИЕ И КРАСКА С ПЕРЕКРЫТИЕМ НА ХРОМИРОВАННОМ РАДИУСЕ"),
    ("PAINT MUST OVERLAP CADMIUM PLATE", "КРАСКА ДОЛЖНА ПЕРЕКРЫВАТЬ КАДМИРОВАНИЕ"),
    ("CADMIUM PLATE AND PAINT TO OVERLAP", "КАДМИРОВАНИЕ И КРАСКА С ПЕРЕКРЫТИЕМ"),
    ("CADMIUM PLATE AND PRIMER PAINT", "КАДМИРОВАНИЕ И ГРУНТОВОЧНАЯ КРАСКА"),
    ("CADMIUM PLATE AND PAINT", "КАДМИРОВАНИЕ И КРАСКА"),
    ("NO BARE METAL PERMISSIBLE", "НЕЗАЩИЩЁННЫЙ МЕТАЛЛ НЕДОПУСТИМ"),
    ("NO CADMIUM OR PAINT DEPOSIT TO", "КАДМИРОВАНИЕ И КРАСКА НЕ ДОЛЖНЫ"),
    ("NO CADMIUM PLATE OR PAINT BEYOND THIS LINE", "БЕЗ КАДМИРОВАНИЯ ИЛИ КРАСКИ ЗА ЭТОЙ ЛИНИЕЙ"),
    ("NO CADMIUM PLATE", "БЕЗ КАДМИРОВАНИЯ"),
    ("NO CADMIUM", "БЕЗ КАДМИРОВАНИЯ"),
    ("DO NOT CADMIUM PLATE", "НЕ НАНОСИТЬ КАДМИРОВАНИЕ"),
    ("PLATE OR PAINT BEYOND THIS LINE", "ПОКРЫТИЯ ИЛИ КРАСКИ ЗА ЭТОЙ ЛИНИЕЙ"),
    ("PLATE OR PAINT", "ПОКРЫТИЯ ИЛИ КРАСКИ"),
    ("BEYOND THIS LINE", "ЗА ЭТОЙ ЛИНИЕЙ"),
    ("LENGTH OF CADMIUM PLATE", "ДЛИНА КАДМИРОВАНИЯ"),
    ("CADMIUM PLATE", "КАДМИРОВАНИЕ"),
    ("LENGTH OF CHROMIUM PLATE", "ДЛИНА ХРОМИРОВАНИЯ"),
    ("CHROMIUM PLATED LENGTH", "ДЛИНА ХРОМИРОВАНИЯ"),
    ("NOT CHROMIUM PLATED", "БЕЗ ХРОМИРОВАНИЯ"),
    ("CHROMIUM PLATE MUST STOP IN THIS LENGTH. AN IRREGULAR LINE IS PERMITTED.",
     "ХРОМИРОВАНИЕ ДОЛЖНО ОКАНЧИВАТЬСЯ В ЭТОЙ ДЛИНЕ. ДОПУСТИМА НЕРОВНАЯ ЛИНИЯ."),
    ("THE CHROMIUM PLATE MUST STOP IN THIS LENGTH. AN IRREGULAR LINE IS PERMITTED",
     "ХРОМИРОВАНИЕ ДОЛЖНО ОКАНЧИВАТЬСЯ В ЭТОЙ ДЛИНЕ. ДОПУСТИМА НЕРОВНАЯ ЛИНИЯ"),
    ("CHROMIUM PLATE MUST STOP IN THIS LENGTH. WAVY OR IRREGULAR LINE IS PERMITTED",
     "ХРОМИРОВАНИЕ ДОЛЖНО ОКАНЧИВАТЬСЯ В ЭТОЙ ДЛИНЕ. ДОПУСТИМА ВОЛНИСТАЯ ИЛИ НЕРОВНАЯ ЛИНИЯ"),
    ("CHROMIUM PLATE MUST TERMINATE IN THIS LENGTH. WAVY OR IRREGULAR LINE IS PERMITTED",
     "ХРОМИРОВАНИЕ ДОЛЖНО ОКАНЧИВАТЬСЯ В ЭТОЙ ДЛИНЕ. ДОПУСТИМА ВОЛНИСТАЯ ИЛИ НЕРОВНАЯ ЛИНИЯ"),
    ("CHROMIUM PLATE MUST TERMINATE WITHIN", "ХРОМИРОВАНИЕ ДОЛЖНО ОКАНЧИВАТЬСЯ В ПРЕДЕЛАХ"),
    ("MUST TERMINATE WITHIN", "ДОЛЖНО ОКАНЧИВАТЬСЯ В ПРЕДЕЛАХ"),
    ("TERMINATE WITHIN", "ОКАНЧИВАТЬСЯ В ПРЕДЕЛАХ"),
    ("THE CHROMIUM PLATE MUST STOP", "ХРОМИРОВАНИЕ ДОЛЖНО ОКАНЧИВАТЬСЯ"),
    ("CHROMIUM PLATE MUST STOP", "ХРОМИРОВАНИЕ ДОЛЖНО ОКАНЧИВАТЬСЯ"),
    ("CHROMIUM PLATE MUST", "ХРОМИРОВАНИЕ ДОЛЖНО"),
    ("MUST STOP", "ДОЛЖНО ОКАНЧИВАТЬСЯ"),
    ("TERMINATE IN THIS LENGTH.", "ОКАНЧИВАТЬСЯ В ЭТОЙ ДЛИНЕ."),
    ("IN THIS LENGTH. AN IRREGULAR", "В ЭТОЙ ДЛИНЕ. ДОПУСТИМА НЕРОВНАЯ"),
    ("IN THIS LENGTH", "В ЭТОЙ ДЛИНЕ"),
    ("AN IRREGULAR", "ДОПУСТИМА НЕРОВНАЯ"),
    ("WAVY OR IRREGULAR LINE IS PERMITTED", "ДОПУСТИМА ВОЛНИСТАЯ ИЛИ НЕРОВНАЯ ЛИНИЯ"),
    ("WAVY OR IRREGULAR LINE IS PERMISSIBLE.", "ВОЛНИСТАЯ ИЛИ НЕРОВНАЯ ЛИНИЯ ДОПУСТИМА."),
    ("WAVY OR IRREGULAR LINE PERMISSIBLE", "ВОЛНИСТАЯ ИЛИ НЕРОВНАЯ ЛИНИЯ ДОПУСТИМА"),
    ("WAVY IRREGULAR LINE IS PERMISSIBLE.", "ВОЛНИСТАЯ ИЛИ НЕРОВНАЯ ЛИНИЯ ДОПУСТИМА."),
    ("WAVY IRREGULAR LINE IS PERMISSIBLE", "ВОЛНИСТАЯ ИЛИ НЕРОВНАЯ ЛИНИЯ ДОПУСТИМА"),
    ("AN IRREGULAR LINE IS PERMITTED.", "ДОПУСТИМА НЕРОВНАЯ ЛИНИЯ."),
    ("AN IRREGULAR LINE IS PERMITTED", "ДОПУСТИМА НЕРОВНАЯ ЛИНИЯ"),
    ("IRREGULAR LINE IS PERMITTED", "ДОПУСТИМА НЕРОВНАЯ ЛИНИЯ"),
    ("LINE IS PERMITTED", "ЛИНИЯ ДОПУСТИМА"),
    ("LINE IS PERMISSIBLE.", "ЛИНИЯ ДОПУСТИМА."),
    ("LINE IS PERMISSIBLE", "ЛИНИЯ ДОПУСТИМА"),
    ("WAVY OR IRREGULAR", "ВОЛНИСТАЯ ИЛИ НЕРОВНАЯ"),
    ("LINE PERMISSIBLE", "ЛИНИЯ ДОПУСТИМА"),
    ("CHROMIUM PLATE MUST NOT EXTENDED ONTO", "ХРОМИРОВАНИЕ НЕ ДОЛЖНО РАСПРОСТРАНЯТЬСЯ НА"),
    ("CHROMIUM PLATE TERMINATION TO", "ЗАВЕРШЕНИЕ ХРОМИРОВАНИЯ ПО"),
    ("CHROMIUM PLATE TEMINATION TO", "ЗАВЕРШЕНИЕ ХРОМИРОВАНИЯ ПО"),
    ("CHROMIUM PLATE TERMINATION", "ЗАВЕРШЕНИЕ ХРОМИРОВАНИЯ"),
    ("CHROMIUM PLATE DEPOSIT", "СЛОЙ ХРОМОВОГО ПОКРЫТИЯ"),
    ("ENTER OR STAND PROUD OF CHROMIUM", "ВХОДИТЬ ИЛИ ВЫСТУПАТЬ НАД ХРОМИРОВАННЫМ"),
    ("PLATED BORE", "ПОКРЫТЫМ ОТВЕРСТИЕМ"),
    ("PLATED DIAMETER", "ПОКРЫТЫМ ДИАМЕТРОМ"),
    ("CHROMIUM PLATED SURFACE", "ХРОМИРОВАННАЯ ПОВЕРХНОСТЬ"),
    ("TWO PLACESCHROMIUM PLATED", "ДВА МЕСТА — ХРОМИРОВАННАЯ ПОВЕРХНОСТЬ"),
    ("CHROMIUM PLATED", "ХРОМИРОВАННЫЙ"),
    ("FULL CHROME PLATING THICKNESS", "ПОЛНАЯ ТОЛЩИНА ХРОМОВОГО ПОКРЫТИЯ"),
    ("FULL CHROME", "ПОЛНАЯ ТОЛЩИНА ХРОМА"),
    ("CHROME PLATING DEPOSIT", "СЛОЙ ХРОМОВОГО ПОКРЫТИЯ"),
    ("CHROME PLATING WILL", "ХРОМОВОЕ ПОКРЫТИЕ"),
    ("CHROME PLATING", "ХРОМОВОЕ ПОКРЫТИЕ"),
    # Specific CHROME TERMINATION entries must come BEFORE general "CHROME TERMINATION"
    ("BARREL OUTER DIA. LOWER CHROME TERMINATION", "ЗАВЕРШЕНИЕ ХРОМИРОВАНИЯ НИЖНЕГО НАРУЛ. ДИА. СТВОЛА"),
    ("BARREL OUTER DIA. UPPER CHROME TERMINATION", "ЗАВЕРШЕНИЕ ХРОМИРОВАНИЯ ВЕРХНЕГО НАРУЛ. ДИА. СТВОЛА"),
    ("JOURNAL A,B,C CHROME TERMINATION", "ЗАВЕРШЕНИЕ ХРОМИРОВАНИЯ ПОЯСКОВ A,B,C"),
    ("JOURNAL A OUTER CHROME TERMINATION", "ЗАВЕРШЕНИЕ ХРОМИРОВАНИЯ НАРУЖНОГО ДИА. ПОЯСКА A"),
    ("JOURNAL C INNER CHROME TERMINATION", "ЗАВЕРШЕНИЕ ХРОМИРОВАНИЯ ВНУТР. ДИА. ПОЯСКА C"),
    ("HPC SEAL ABUTMENT LOWER CHROME TERMINATION", "ЗАВЕРШЕНИЕ ХРОМИРОВАНИЯ НА НИЖНЕЙ ОПОРЕ УПЛОТНЕНИЯ HPC"),
    ("HPC SEAL ABUTMENT UPPER CHROME TERMINATION", "ЗАВЕРШЕНИЕ ХРОМИРОВАНИЯ НА ВЕРХНЕЙ ОПОРЕ УПЛОТНЕНИЯ HPC"),
    ("HPC SEAL ABUTMENT LOWER", "НИЖНЯЯ ОПОРА УПЛОТНЕНИЯ HPC"),
    ("HPC SEAL ABUTMENT UPPER", "ВЕРХНЯЯ ОПОРА УПЛОТНЕНИЯ HPC"),
    ("HPC SEAL ABUTMENT", "ОПОРА УПЛОТНЕНИЯ HPC"),
    ("SEAL ABUTMENT", "ОПОРА УПЛОТНЕНИЯ"),
    ("BREAK FLANGE FACE CHROME TERMINATION", "ЗАВЕРШЕНИЕ ХРОМИРОВАНИЯ ТОРЦА ТОРМОЗНОГО ФЛАНЦА"),
    ("CHROME TERMINATION", "ЗАВЕРШЕНИЕ ХРОМИРОВАНИЯ"),
    ("CHROME RUN OUT BAND (TO RADIUS INTERSECTION POINT)", "ЗОНА ВЫБЕГА ХРОМА (ДО ТОЧКИ ПЕРЕСЕЧЕНИЯ РАДИУСА)"),
    ("(TO RADIUS INTERSECTION POINT)", "(ДО ТОЧКИ ПЕРЕСЕЧЕНИЯ РАДИУСА)"),
    ("CHROME RUN OUT BAND", "ЗОНА ВЫБЕГА ХРОМА"),
    ("CHROMIUM DEPOSIT", "СЛОЙ ХРОМОВОГО ПОКРЫТИЯ"),
    ("CHROMIUM RUNOUT", "БИЕНИЕ ХРОМОВОГО ПОКРЫТИЯ"),
    ("AREA OF CHROMIUM PLATE", "ЗОНА ХРОМИРОВАНИЯ"),
    ("LENGTH OF CHROMIUM", "ДЛИНА ХРОМИРОВАНИЯ"),
    ("DIA. AFTER GRINDING CHROME", "ДИА. ПОСЛЕ ШЛИФОВАНИЯ ХРОМА"),
    ("DIA. AFTER GRINDING", "ДИА. ПОСЛЕ ШЛИФОВАНИЯ"),
    ("AFTER GRINDING CHROME", "ПОСЛЕ ШЛИФОВАНИЯ ХРОМА"),
    ("DIA. AFTER CHROME PLATING", "ДИА. ПОСЛЕ ХРОМИРОВАНИЯ"),
    ("DIAMETER AFTER GRINDING OF CHROMIUM PLATE", "ДИАМЕТР ПОСЛЕ ШЛИФОВАНИЯ ХРОМОВОГО ПОКРЫТИЯ"),
    ("DIAMETER AFTER GRINDING CHROMIUM PLATE", "ДИАМЕТР ПОСЛЕ ШЛИФОВАНИЯ ХРОМОВОГО ПОКРЫТИЯ"),
    ("DIAMETER BEFORE CHROMIUM PLATE", "ДИАМЕТР ДО ХРОМИРОВАНИЯ"),
    ("DIAMETER AFTER CHROMIUM PLATE", "ДИАМЕТР ПОСЛЕ ХРОМИРОВАНИЯ"),
    ("DIAMETER AFTER MACHINING NICKEL PLATE", "ДИАМЕТР ПОСЛЕ МЕХОБРАБОТКИ НИКЕЛЕВОГО ПОКРЫТИЯ"),
    ("DIAMETER BEFORE NICKEL AND CHROMIUM PLATE", "ДИАМЕТР ДО НИКЕЛИРОВАНИЯ И ХРОМИРОВАНИЯ"),
    ("DIAMETER BEFORE NICKEL", "ДИАМЕТР ДО НИКЕЛИРОВАНИЯ"),
    ("DIA. BEFORE SULPHAMATE", "ДИА. ДО СУЛЬФАМАТНОГО"),
    ("DIAMETER AFTER CHROME PLATING", "ДИАМЕТР ПОСЛЕ ХРОМИРОВАНИЯ"),
    ("SECTION Y-Y BEFORE CHROMIUM PLATE", "СЕЧЕНИЕ Y-Y ДО ХРОМИРОВАНИЯ"),
    ("SECTION Y-Y AFTER CHROMIUM PLATE", "СЕЧЕНИЕ Y-Y ПОСЛЕ ХРОМИРОВАНИЯ"),
    ("DIAMETER A BEFORE CHROMIUM PLATE", "ДИАМЕТР A ДО ХРОМИРОВАНИЯ"),
    ("DIAMETER A AFTER GRINDING", "ДИАМЕТР A ПОСЛЕ ШЛИФОВАНИЯ"),
    ("AFTER CHROMIUM PLATE IS APPLIED AND BEFORE GRINDING", "ПОСЛЕ НАНЕСЕНИЯ ХРОМИРОВАНИЯ И ДО ШЛИФОВАНИЯ"),
    ("BEFORE CHROMIUM PLATE IS APPLIED", "ДО НАНЕСЕНИЯ ХРОМИРОВАНИЯ"),
    ("AFTER CHROMIUM PLATE AND GRINDING", "ПОСЛЕ ХРОМИРОВАНИЯ И ШЛИФОВАНИЯ"),
    ("BEFORE CHROMIUM PLATE", "ДО ХРОМИРОВАНИЯ"),
    ("AFTER CHROMIUM PLATE", "ПОСЛЕ ХРОМИРОВАНИЯ"),
    ("THIN CHROMIUM PLATE", "ТОНКОЕ ХРОМИРОВАНИЕ"),
    ("CHROMIUM PLATE", "ХРОМИРОВАНИЕ"),
    ("AFTER GRINDING CHROMIUM PLATE", "ПОСЛЕ ШЛИФОВАНИЯ ХРОМОВОГО ПОКРЫТИЯ"),
    ("THICKNESS OF CHROMIUM PLATE ABOVE OUTER SURFACE OF FLANGE",
     "ТОЛЩИНА ХРОМОВОГО ПОКРЫТИЯ НАД НАРУЖНОЙ ПОВЕРХНОСТЬЮ ФЛАНЦА"),
    ("THE CHROMIUM PLATE MUST NOT EXTEND BEYOND THE DIMENSIONS SHOWN",
     "ХРОМИРОВАНИЕ НЕ ДОЛЖНО ВЫХОДИТЬ ЗА УКАЗАННЫЕ РАЗМЕРЫ"),
    ("MAKE EDGES SMOOTH AFTER CHROMIUM PLATE IS APPLIED AND BEFORE GRINDING",
     "ЗАЧИСТИТЬ КРАЯ ПОСЛЕ НАНЕСЕНИЯ ХРОМИРОВАНИЯ И ДО ШЛИФОВАНИЯ"),
    ("EDGES SMOOTHED OUT.", "КРАЯ ЗАЧИЩЕНЫ."),
    ("EDGE SMOOTHED", "КРОМКА СГЛАЖЕНА"),
    ("MAKE EDGES SMOOTH", "ЗАЧИСТИТЬ КРАЯ ДО ГЛАДКОСТИ"),
    ("SMOOTH BLEND TO ADJACENT SURFACES", "ПЛАВНОЕ СОПРЯЖЕНИЕ С ПРИЛЕГАЮЩИМИ ПОВЕРХНОСТЯМИ"),
    ("BLEND SMOOTHLY TO", "ПЛАВНО СОПРЯЧЬ С"),
    ("BLEND SMOOTHLY", "ПЛАВНО СОПРЯЧЬ"),
    ("TO ADJACENT SURFACES", "С ПРИЛЕГАЮЩИМИ ПОВЕРХНОСТЯМИ"),
    ("ADJACENT SURFACES", "ПРИЛЕГАЮЩИЕ ПОВЕРХНОСТИ"),
    ("SMOOTH EDGE", "ГЛАДКИЙ КРАЙ"),
    ("SMOOTH BLEND", "ПЛАВНОЕ СОПРЯЖЕНИЕ"),
    ("EDGE BLENDED WITH A SMOOTH TRANSITION", "КРАЯ ПЛАВНО СОПРЯЖЕНЫ"),
    ("AFTER GRINDING", "ПОСЛЕ ШЛИФОВАНИЯ"),
    ("BEFORE CHROMIUM", "ДО ХРОМИРОВАНИЯ"),
    ("AFTER CHROMIUM", "ПОСЛЕ ХРОМИРОВАНИЯ"),
    ("CHROMIUM", "ХРОМИРОВАНИЕ"),

    # --- НИКЕЛЕВОЕ ПОКРЫТИЕ ---
    ("SULPHAMATE NICKEL PLATE DEPOSIT", "СЛОЙ СУЛЬФАМАТНОГО НИКЕЛЕВОГО ПОКРЫТИЯ"),
    ("SULPHAMATE NICKEL PLATE", "СУЛЬФАМАТНОЕ НИКЕЛЕВОЕ ПОКРЫТИЕ"),
    ("NICKEL PLATE MUST NOT ENTER LUBRICATION HOLES OR CROSS HOLE",
     "НИКЕЛЕВОЕ ПОКРЫТИЕ НЕ ДОЛЖНО ВХОДИТЬ В СМАЗОЧНЫЕ ИЛИ ПОПЕРЕЧНЫЕ ОТВЕРСТИЯ"),
    ("LUBRICATION HOLES OR CROSS HOLE", "СМАЗОЧНЫЕ ИЛИ ПОПЕРЕЧНЫЕ ОТВЕРСТИЯ"),
    ("WITH LUBRICATION ADAPTOR", "С СМАЗОЧНЫМ АДАПТЕРОМ"),
    ("WITHOUT LUBRICATION ADAPTOR", "БЕЗ СМАЗОЧНОГО АДАПТЕРА"),
    ("LENGTH OF NICKEL PLATE", "ДЛИНА НИКЕЛЕВОГО ПОКРЫТИЯ"),
    ("NICKEL PLATE", "НИКЕЛЕВОЕ ПОКРЫТИЕ"),
    ("DIAMETER MINIMUM", "МИНИМАЛЬНЫЙ ДИАМЕТР"),

    # --- КРАСКА / ПОКРЫТИЕ ---
    ("PAINT TO PCS-2500 OVER SERMETEL W TO LENGTH", "КРАСКА ПО PCS-2500 ПОВЕРХ SERMETEL W НА ДЛИНУ"),
    ("PRIMER PAINT TO PCS-2500", "ГРУНТ ПО PCS-2500"),
    ("PRIMER PAINT WHERE BUSH FLANGE WILL TOUCH", "ГРУНТ В МЕСТЕ КОНТАКТА ФЛАНЦА ВТУЛКИ"),
    ("PRIMER PAINT", "ГРУНТОВОЧНАЯ КРАСКА"),
    ("PAINT TO PCS-2500", "КРАСКА ПО PCS-2500"),
    ("FINISH PAINT", "ОКОНЧАТЕЛЬНАЯ КРАСКА"),
    ("PAINT DEPOSIT OVERLAP", "ПЕРЕКРЫТИЕ СЛОЯ КРАСКИ"),
    ("PAINT DEPOSIT", "СЛОЙ КРАСКИ"),
    ("NO PAINT", "БЕЗ ОКРАСКИ"),
    ("DO NOT PAINT", "НЕ КРАСИТЬ"),
    ("UP TO BUSH FLANGES", "ДО БУРТИКОВ ВТУЛКИ"),
    ("IN HOLES", "В ОТВЕРСТИЯХ"),
    ("DEPOSIT OVERLAP", "ПЕРЕКРЫТИЕ ПОКРЫТИЯ"),
    ("SERMETAL COATING", "ПОКРЫТИЕ SERMETEL"),
    ("SERMETEL COATING", "ПОКРЫТИЕ SERMETEL"),
    ("EXTERNAL SERMETEL LIMIT", "ГРАНИЦА SERMETEL (СНАРУЖИ)"),
    ("INTERNAL SERMETEL LIMIT", "ГРАНИЦА SERMETEL (ИЗНУТРИ)"),
    ("LIMIT OF SERMETEL W TERMINATION FROM CENTER", "ГРАНИЦА SERMETEL W ОТ ЦЕНТРА"),
    ("ALUMINIUM COAT OPTIONAL ON THESE SURFACES", "АЛЮМИНИЕВОЕ ПОКРЫТИЕ НА ЭТИХ ПОВЕРХНОСТЯХ — ДОПУСТИМО"),
    ("ALUMINIUM COAT", "АЛЮМИНИЕВОЕ ПОКРЫТИЕ"),
    ("THESE SURFACES", "ЭТИ ПОВЕРХНОСТИ"),
    ("DO NOT PRIMER PAINT", "НЕ НАНОСИТЬ ГРУНТОВОЧНУЮ КРАСКУ"),
    ("DO NOT PRIMER", "НЕ НАНОСИТЬ ГРУНТ"),
    ("TRACES OF", "СЛЕДЫ"),
    ("OVER SERMETEL W TO LENGTH", "ПОВЕРХ SERMETEL W НА ДЛИНУ"),
    ("SEALANT TO PCS-7200", "ГЕРМЕТИК ПО PCS-7200"),
    ("APPLY SEALANT TO PCS-7200", "НАНЕСТИ ГЕРМЕТИК ПО PCS-7200"),
    ("APPLY SEALANT TO", "НАНЕСТИ ГЕРМЕТИК ПО"),
    ("APPLY SEALANT", "НАНЕСТИ ГЕРМЕТИК"),
    ("SEALANT TO", "ГЕРМЕТИК ПО"),
    ("ADHESIVE TO", "АДГЕЗИВ ПО"),
    ("RUN OUT BAND", "ЗОНА ВЫБЕГА"),
    ("DEPOSIT OVERLAP", "ПЕРЕКРЫТИЕ ПОКРЫТИЯ"),
    ("WORKING DIA.", "РАБОЧИЙ ДИАМЕТР"),
    ("TO HPC ABUTMENT FACE", "ДО ОПОРНОГО ТОРЦА HPC"),

    # --- ОБРАБОТКА ПОВЕРХНОСТЕЙ ---
    ("DO NOT MACHINE THIS FACE", "НЕ ОБРАБАТЫВАТЬ ЭТУ ПОВЕРХНОСТЬ"),
    ("MACHINE THIS FACE", "МЕХОБРАБОТКА ЭТОЙ ПОВЕРХНОСТИ"),
    ("INTENTIONALLY BLANK", "НАМЕРЕННО ОСТАВЛЕНО ПУСТЫМ"),
    ("APPLY COAT OF ALUMINIUM (IVD)", "НАНЕСТИ ПОКРЫТИЕ АЛЮМИНИЕМ (IVD)"),
    ("IVD COATING OPTIONAL IN END FACE", "ПОКРЫТИЕ IVD НА ТОРЦЕ ОПЦИОНАЛЬНО"),
    ("IVD COATING OPTIONAL", "ПОКРЫТИЕ IVD ОПЦИОНАЛЬНО"),
    ("APPLY ELECTRICALLY", "НАНЕСТИ ЭЛЕКТРОПРОВОДЯЩИЙ"),
    ("CONDUCTING MOLYKOTE 111", "ПРОВОДЯЩИЙ MOLYKOTE 111"),
    ("OR RUBBERISED SEALANT", "ИЛИ РЕЗИНОВЫЙ ГЕРМЕТИК"),
    ("IN THE BORES TO", "В ОТВЕРСТИЯ ПО"),
    ("ON BOTH BUSHES TO", "НА ОБЕ ВТУЛКИ ПО"),
    ("WITHOUT OVERFLOW ON", "БЕЗ ПЕРЕТЕКАНИЯ НА"),
    ("FACES OF THE BUSHES", "ТОРЦЕВЫЕ ПОВЕРХНОСТИ ВТУЛОК"),
    ("SPOTFACE TYPICAL 12 PLACES INCLUDING CHAMFER", "ПОДРЕЗКА ТИПИЧНО 12 МЕСТ С ФАСКОЙ"),
    ("SPOTFACE TYPICAL 4 PLACES INCLUDING CHAMFER", "ПОДРЕЗКА ТИПИЧНО 4 МЕСТА С ФАСКОЙ"),
    ("SPOTFACE TYPICAL 2 PLACES INCLUDING CHAMFER", "ПОДРЕЗКА ТИПИЧНО 2 МЕСТА С ФАСКОЙ"),
    ("SPOTFACE TYPICAL 12 PLACES", "ПОДРЕЗКА ТИПИЧНО 12 МЕСТ"),
    ("SPOTFACE TYPICAL 4 PLACES", "ПОДРЕЗКА ТИПИЧНО 4 МЕСТА"),
    ("SPOTFACE TYPICAL 2 PLACES", "ПОДРЕЗКА ТИПИЧНО 2 МЕСТА"),
    ("SPOTFACE TYPICAL", "ПОДРЕЗКА ТИПИЧНО"),
    ("SPOTFACE CORNER RADIUS (TYP)", "РАДИУС СКРУГЛЕНИЯ ПОДРЕЗКИ (ТИП.)"),
    ("SPOTFACE CORNER RADIUS", "РАДИУС СКРУГЛЕНИЯ ПОДРЕЗКИ"),
    ("BETWEEN SPOTFACES", "МЕЖДУ ПОДРЕЗКАМИ"),
    ("ACROSS SPOTFACES", "МЕЖДУ ПОДРЕЗКАМИ"),
    ("SPOTFACE TO DIAMETER", "ПОДРЕЗКА ДО ДИАМЕТРА"),
    ("SPOTFACE TO", "ПОДРЕЗКА ДО"),
    ("SPOTFACE RAD.", "РАДИУС ПОДРЕЗКИ"),
    ("SPOTFACE RADIUS", "РАДИУС ПОДРЕЗКИ"),
    ("SPOTFACE", "ПОДРЕЗКА"),
    ("SHOT PEEN", "ДРОБЕСТРУЙНАЯ ОБРАБОТКА"),
    ("DEBURR", "СНЯТЬ ЗАУСЕНЦЫ"),
    ("MACHINE TO", "МЕХОБРАБОТКА ДО"),
    ("GRIND TO", "ШЛИФОВАТЬ ДО"),
    ("MACHINING THE REPAIR BUSH", "МЕХОБРАБОТКА РЕМОНТНОЙ ВТУЛКИ"),
    ("MACHINING REPAIR SLEEVE", "МЕХОБРАБОТКА РЕМОНТНОЙ ВТУЛКИ"),
    ("OVERSIZE BEARING - MACHINING", "РЕМОНТНЫЙ ПОДШИПНИК – МЕХОБРАБОТКА"),
    ("SMOOTH TO RADIUS.", "СГЛАЖЕНО ДО РАДИУСА."),
    ("MACHINING", "МЕХАНИЧЕСКАЯ ОБРАБОТКА"),
    ("SERMETEL W ON INTERNAL DIAMETER OVER LENGTH", "SERMETEL W НА ВНУТРЕННЕМ ДИАМЕТРЕ НА ДЛИНУ"),
    ("NOTE:-", "ПРИМЕЧАНИЕ:"),
    ("NOTE:", "ПРИМЕЧАНИЕ:"),
    ("REF. EXTENT OF SHOT PEENING", "СПРАВОЧНО: ЗОНА ДРОБЕСТРУЙНОЙ ОБРАБОТКИ"),

    # --- АННОТАЦИИ РАЗМЕРОВ ---
    ("PLATING THICKNESS", "ТОЛЩИНА ПОКРЫТИЯ"),
    ("GRINDING CHROME", "ШЛИФОВАНИЯ ХРОМА"),
    ("CHROME", "ХРОМ"),
    ("A SMOOTH TRANSITION", "ПЛАВНЫЙ ПЕРЕХОД"),
    ("DIMENSIONS SHOWN", "УКАЗАННЫХ РАЗМЕРАХ"),
    ("TERMINATION FROM CENTER", "ОТ ЦЕНТРА"),
    ("MINIMUM EXTENT OF FINE LIMIT  DIA.", "МИНИМАЛЬНАЯ ДЛИНА УЧАСТКА С ТОЧНЫМ ДОПУСКОМ"),
    ("MINIMUM EXTENT OF FINE LIMIT DIA.", "МИНИМАЛЬНАЯ ДЛИНА УЧАСТКА С ТОЧНЫМ ДОПУСКОМ"),
    ("MIN.EXTENT OF FINE LIMIT", "МИН. ДЛИНА УЧАСТКА С ТОЧНЫМ ДОПУСКОМ"),
    ("MINIMUM EXTENT", "МИНИМАЛЬНАЯ ДЛИНА"),
    ("EXTENT OF FINE LIMIT  DIA.", "ДЛИНА УЧАСТКА С ТОЧНЫМ ДОПУСКОМ"),
    ("EXTENT OF FINE LIMIT DIA.", "ДЛИНА УЧАСТКА С ТОЧНЫМ ДОПУСКОМ"),
    ("OF FINE LIMIT  DIA.", "ТОЧНОГО ДОПУСКА ДИА."),
    ("OF FINE LIMIT DIA.", "ТОЧНОГО ДОПУСКА ДИА."),
    ("OF FINE LIMIT", "ТОЧНОГО ДОПУСКА"),
    ("FINE LIMIT", "ТОЧНЫЙ ДОПУСК"),
    ("REMAINDER MAY BE", "ОСТАЛЬНОЕ МОЖЕТ БЫТЬ"),
    ("EXTENT OF CADMIUM FADE OUT", "ПРОТЯЖЁННОСТЬ ВЫХОДА КАДМИРОВАНИЯ"),
    ("EXTENT OF CHROME FADE OUT", "ПРОТЯЖЁННОСТЬ ВЫХОДА ХРОМА"),
    ("EXTENT OF CHROMIUM PLATE FADE OUT", "ПРОТЯЖЁННОСТЬ ВЫХОДА ХРОМИРОВАНИЯ"),
    ("EXTENT OF CHROMIUM FADE OUT", "ПРОТЯЖЁННОСТЬ ВЫХОДА ХРОМИРОВАНИЯ"),
    ("PERMISSIBLE FADE OUT", "ДОПУСТИМЫЙ ВЫХОД ПОКРЫТИЯ"),
    ("FADE OUT", "ВЫХОД ПОКРЫТИЯ"),
    ("WITHIN THIS BAND", "В ПРЕДЕЛАХ ЭТОЙ ЗОНЫ"),
    ("MAXIMUM RUNOUT", "МАКСИМАЛЬНОЕ БИЕНИЕ"),
    ("MINIMUM NO PLATING", "МИНИМУМ БЕЗ ПОКРЫТИЯ"),
    ("NO PLATING", "БЕЗ ПОКРЫТИЯ"),

    # --- ГЕОМЕТРИЯ / РАЗМЕРЫ ---
    ("CENTERLINE OF SLIDING TUBE", "ОСЬ СКОЛЬЗЯЩЕЙ ТРУБЫ"),
    ("DIAMETER THRU BORE INCLUDING CHAMFERS", "ДИАМЕТР СКВОЗНОГО ОТВЕРСТИЯ С ФАСКАМИ"),
    ("DIAMETERS THRU BORES INCLUDING CHAMFERS", "ДИАМЕТРЫ СКВОЗНЫХ ОТВЕРСТИЙ С ФАСКАМИ"),
    ("DIAMETER THRU BORE", "ДИАМЕТР СКВОЗНОГО ОТВЕРСТИЯ"),
    ("THRU BORE", "СКВОЗНОЕ ОТВЕРСТИЕ"),
    ("DIAMETER REF.", "ДИАМЕТР СПРАВОЧНО"),
    ("INNER DIAMETER", "ВНУТРЕННИЙ ДИАМЕТР"),
    ("THROUGH DIA.", "СКВОЗНОЙ ДИА."),
    ("EXTENT OF FINE LIMIT DIA.", "ДЛИНА УЧАСТКА С ТОЧНЫМ ДОПУСКОМ"),
    ("MINIMUM WALL THICKNESS", "МИНИМАЛЬНАЯ ТОЛЩИНА СТЕНКИ"),
    ("MINIMUM WALL", "МИНИМАЛЬНАЯ ТОЛЩИНА СТЕНКИ"),
    ("MINIMUM LUG WIDTH", "МИНИМАЛЬНАЯ ШИРИНА УШКА"),
    ("MIN. WALL THICKNESS", "МИН. ТОЛЩИНА СТЕНКИ"),
    ("WALL THICKNESS", "ТОЛЩИНА СТЕНКИ"),
    ("THE SURFACE FINISH MUST BE", "ШЕРОХОВАТОСТЬ ПОВЕРХНОСТИ ДОЛЖНА БЫТЬ"),
    ("OR BETTER UNLESS GIVEN DIFFERENTLY.", "ИЛИ ЛУЧШЕ, ЕСЛИ НЕ УКАЗАНО ИНОЕ."),
    ("OR BETTER UNLESS GIVEN DIFFERENTLY", "ИЛИ ЛУЧШЕ, ЕСЛИ НЕ УКАЗАНО ИНОЕ"),
    ("SURFACE FINISH", "ШЕРОХОВАТОСТЬ ПОВЕРХНОСТИ"),
    ("SURFACES", "ПОВЕРХНОСТИ"),
    ("SURFACE", "ПОВЕРХНОСТЬ"),
    ("THIS FACE ONLY", "ТОЛЬКО ЭТА ПОВЕРХНОСТЬ"),
    ("IN THIS FACE ONLY", "ТОЛЬКО В ЭТОЙ ПОВЕРХНОСТИ"),
    ("ON FACE", "НА ТОРЦЕ"),
    ("FACE C", "ПОВЕРХНОСТЬ C"),
    ("FACE D", "ПОВЕРХНОСТЬ D"),
    ("FACE E", "ПОВЕРХНОСТЬ E"),
    ("FACE B", "ПОВЕРХНОСТЬ B"),
    ("FACE Q", "ПОВЕРХНОСТЬ Q"),
    ("FACE", "ПОВЕРХНОСТЬ"),
    ("AROUND DOWEL HOLES", "ВОКРУГ ОТВЕРСТИЙ ПОД ШТИФТ"),
    ("AROUND CHANGEOVER VALVE HOLES", "ВОКРУГ ОТВЕРСТИЙ ПЕРЕПУСКНОГО КЛАПАНА"),
    ("1 POSITION ONLY", "ТОЛЬКО 1 ПОЗИЦИЯ"),
    ("3 POSITIONS", "3 ПОЗИЦИИ"),
    ("2 POSITIONS", "2 ПОЗИЦИИ"),
    ("IN THIS FACE", "В ЭТОЙ ПОВЕРХНОСТИ"),
    ("TWO PLACES", "ДВА МЕСТА"),
    ("TYPICAL BOTH BORES", "ТИПИЧНО ОБА ОТВЕРСТИЯ"),
    ("TYPICAL BOTH", "ТИПИЧНО ОБА"),
    ("(BOTH FLANGES)", "(ОБА ФЛАНЦА)"),
    ("BOTH FLANGES", "ОБА ФЛАНЦА"),
    ("BOTH BORES", "ОБА ОТВЕРСТИЯ"),
    ("BOTH HOLES", "ОБА ОТВЕРСТИЯ"),
    ("CHAMFER BOTH ENDS", "ФАСКА С ОБОИХ КОНЦОВ"),
    ("BOTH ENDS", "ОБА КОНЦА"),
    ("BOTH SIDES", "ОБА КОНЦА"),
    ("BOTH BUSHES", "ОБЕ ВТУЛКИ"),
    ("ALL AROUND AT", "ПО ВСЕМУ ПЕРИМЕТРУ НА"),
    ("ALL AROUND", "ПО ВСЕМУ ПЕРИМЕТРУ"),
    ("TYPICAL BOTH REPAIR SLEEVES", "ТИПИЧНО ДЛЯ ОБЕИХ РЕМОНТНЫХ ВТУЛОК"),
    ("TYPICAL AROUND 4 SLOTS.", "ТИПИЧНО ПО 4 ПАЗАМ."),
    ("TYPICAL AROUND 4 SLOTS", "ТИПИЧНО ПО 4 ПАЗАМ"),
    ("TYPICAL AROUND LUG", "ТИПИЧНО ПО УШКУ"),
    ("TYPICAL 2 LUGS", "ТИПИЧНО 2 УШКА"),
    ("TYPICAL 2 PLACES", "ТИПИЧНО 2 МЕСТА"),
    ("TYPICAL 4 PLACES", "ТИПИЧНО 4 МЕСТА"),
    ("TYPICAL 6 PLACES", "ТИПИЧНО 6 МЕСТ"),
    ("TYPICAL 12 PLACES", "ТИПИЧНО 12 МЕСТ"),
    ("TYPICAL 2", "ТИПИЧНО 2"),
    ("TYPICAL 3", "ТИПИЧНО 3"),
    ("TYPICAL 4", "ТИПИЧНО 4"),
    ("TYP. 2 PLACES", "ТИПИЧНО 2 МЕСТА"),
    ("TYP. 2", "ТИПИЧНО 2"),
    ("CENTERS TYPICAL", "ТИПОВЫЕ ЦЕНТРЫ"),
    ("TYPICAL", "ТИПИЧНО"),
    ("TYPICAL 3 POSITIONS", "ТИПИЧНО 3 ПОЗИЦИИ"),
    ("14 PLACES", "14 МЕСТ"),
    ("12 PLACES", "12 МЕСТ"),
    ("10 PLACES", "10 МЕСТ"),
    ("9 PLACES", "9 МЕСТ"),
    ("8 PLACES", "8 МЕСТ"),
    ("6 PLACES", "6 МЕСТ"),
    ("4 PLACES", "4 МЕСТА"),
    ("3 PLACES", "3 МЕСТА"),
    ("2 PLACES", "2 МЕСТА"),
    ("2 LUGS", "2 УШКА"),
    ("4 HOLES", "4 ОТВЕРСТИЯ"),
    ("3 HOLES", "3 ОТВЕРСТИЯ"),
    ("2 HOLES", "2 ОТВЕРСТИЯ"),
    ("3 BORES", "3 ОТВЕРСТИЯ"),
    ("2 BORES", "2 ОТВЕРСТИЯ"),
    ("WITHOUT REPAIR BUSHES", "БЕЗ РЕМОНТНЫХ ВТУЛОК"),
    ("WITH REPAIR BUSHES INSTALLED", "С УСТАНОВЛЕННЫМИ РЕМОНТНЫМИ ВТУЛКАМИ"),
    ("WITH REPAIR BUSHES", "С РЕМОНТНЫМИ ВТУЛКАМИ"),
    ("WITHOUT REPAIR BUSH", "БЕЗ РЕМОНТНОЙ ВТУЛКИ"),
    ("WITH REPAIR BUSH", "С РЕМОНТНОЙ ВТУЛКОЙ"),
    ("WITHOUT BEARING", "БЕЗ ПОДШИПНИКА"),
    ("WITH BEARING", "С ПОДШИПНИКОМ"),
    ("WITHOUT BUSHES", "БЕЗ ВТУЛОК"),
    ("WITH BUSHES", "С ВТУЛКАМИ"),
    ("BORE AND CHAMFER INCLUDED", "ОТВЕРСТИЕ И ФАСКА ВКЛЮЧЕНЫ"),
    ("CHAMFER INCLUDED)", "ФАСКА ВКЛЮЧЕНА)"),
    ("CHAMFER INCLUDED", "ФАСКА ВКЛЮЧЕНА"),
    ("REARSIDE ONLY", "ТОЛЬКО ТЫЛЬНАЯ СТОРОНА"),
    ("INCLUDING CHAMFERS", "С ФАСКАМИ"),
    ("INCLUDING CHAMFER", "С ФАСКОЙ"),
    ("INCLUDING RADIUS", "С РАДИУСОМ"),
    ("CHAMFERS ONLY", "ТОЛЬКО ФАСКИ"),
    ("CHAMFER", "ФАСКА"),
    ("RADIUS BEFORE CHROMIUM PLATE", "РАДИУС ДО ХРОМИРОВАНИЯ"),
    ("UNPLATED LENGTH", "ДЛИНА БЕЗ ПОКРЫТИЯ"),
    ("BARREL OUTER DIA. LOWER", "НИЖНИЙ НАРУЖНЫЙ ДИА. СТВОЛА"),
    ("BARREL OUTER DIA. UPPER", "ВЕРХНИЙ НАРУЖНЫЙ ДИА. СТВОЛА"),
    ("BARREL OUTER DIA.", "НАРУЖНЫЙ ДИА. СТВОЛА"),
    ("BARREL", "СТВОЛ"),
    ("RADIUS", "РАДИУС"),
    ("RUNOUT", "БИЕНИЕ"),
    ("RAD.", "РАД."),
    ("REF.", "СПРАВОЧНО"),
    ("(REF)", "(СПРАВОЧНО)"),
    ("MIN.", "МИН."),
    ("MAX.", "МАКС."),
    ("MINIMUM", "МИНИМУМ"),
    ("MAXIMUM", "МАКСИМУМ"),
    ("DIAMETER", "ДИАМЕТР"),
    ("EXTERNALLY", "СНАРУЖИ"),
    ("INTERNALLY", "ИЗНУТРИ"),
    ("EXTERNAL", "НАРУЖНЫЙ"),
    ("INTERNAL", "ВНУТРЕННИЙ"),
    ("LIMIT OF D", "ГРАНИЦА D"),
    ("LIMIT OF A", "ГРАНИЦА A"),
    ("DEGREES", "ГРАДУСОВ"),

    # --- ОТВЕРСТИЯ И КОНСТРУКТИВНЫЕ ЭЛЕМЕНТЫ ---
    ("AXLE NUT CROSS BOLT HOLES", "ОТВЕРСТИЯ ПОД БОЛТЫ ГАЙКИ ОСИ"),
    ("AXLE NUT CROSS", "ПОПЕРЕЧНЫЙ БОЛТ ГАЙКИ ОСИ"),
    ("AXLE BORE", "ОТВЕРСТИЕ ОСИ"),
    ("DRAIN HOLE", "ДРЕНАЖНОЕ ОТВЕРСТИЕ"),
    ("HOLES AND LUGS", "ОТВЕРСТИЯ И УШКИ"),
    ("KNUCKLE TOOLING LUG", "ТЕХНОЛОГИЧЕСКОЕ УШО ТРАВЕРСЫ"),
    ("GREASE HOLES", "ОТВЕРСТИЯ ДЛЯ СМАЗКИ"),
    ("KNUCKLE BORES", "ОТВЕРСТИЯ ТРАВЕРСЫ"),
    ("BRAKE FLANGE", "ТОРМОЗНОЙ ФЛАНЕЦ"),
    ("LUG BORES", "ОТВЕРСТИЯ УШКОВ"),
    ("PINTLE CROSS BORES", "ПОПЕРЕЧНЫЕ ОТВЕРСТИЯ ПОД ШТИФТ НАВЕСКИ"),
    ("PINTLE BORES", "ОТВЕРСТИЯ ПОД ШТИФТ НАВЕСКИ"),
    ("LOCK LINK BORE", "ОТВЕРСТИЕ ПОД ЗАМКОВУЮ ТЯГУ"),
    ("LOWER CARDAN BORE", "НИЖНЕЕ ОТВЕРСТИЕ КАРДАНА"),
    ("UPPER DIAPHRAGM TUBE CROSS BORE", "ПОПЕРЕЧНОЕ ОТВЕРСТИЕ ВЕРХНЕЙ ДИАФРАГМЕННОЙ ТРУБЫ"),
    ("CROSS BORE", "ПОПЕРЕЧНОЕ ОТВЕРСТИЕ"),
    ("TORQUE LINK AND RETAINING PIN BORES", "ОТВЕРСТИЯ ПОД ШЛИЦ-ШАРНИР И ФИКСИРУЮЩИЙ ШТИФТ"),
    ("RETRACTION BORES", "ОТВЕРСТИЯ МЕХАНИЗМА УБОРКИ"),
    ("DRAG ARM HOLES", "ОТВЕРСТИЯ ПОДКОСА"),
    ("TOOLING LUG", "ТЕХНОЛОГИЧЕСКОЕ УШО"),
    ("CHANGE OVER VALVE HOLES AND LUGS", "ОТВЕРСТИЯ И УШКИ ПЕРЕПУСКНОГО КЛАПАНА"),
    ("CHANGE OVER VALVE", "ПЕРЕПУСКНОЙ КЛАПАН"),
    ("UPLOCK LUGS", "УШКИ ЗАМКА УБРАННОГО ПОЛОЖЕНИЯ"),
    ("LOWER DOOR LUGS", "УШКИ НИЖНЕЙ СТВОРКИ"),
    ("UPPER DOOR LUGS", "УШКИ ВЕРХНЕЙ СТВОРКИ"),
    ("TYPICAL 2 BRAKE MANIFOLD LUGS", "ТИПИЧНО 2 УШКА ТОРМОЗНОГО КОЛЛЕКТОРА"),
    ("TYPICAL 2 TRANSFER BLOCK LUGS", "ТИПИЧНО 2 УШКА БЛОКА ПЕРЕДАЧИ"),
    ("BRAKE MANIFOLD LUGS", "УШКИ ТОРМОЗНОГО КОЛЛЕКТОРА"),
    ("TRANSFER BLOCK LUGS", "УШКИ БЛОКА ПЕРЕДАЧИ"),
    ("FOR MAIN FITTING (20-410C, 20-420C) ONLY",
     "ТОЛЬКО ДЛЯ КОРПУСА СТОЙКИ (20-410C, 20-420C)"),
    ("FOR MAIN FITTING (20-410C\nAND 20-420C) ONLY",
     "ТОЛЬКО ДЛЯ КОРПУСА СТОЙКИ (20-410C\nИ 20-420C)"),
    ("PINTLE BORES FOR MAIN FITTING (20-410C AND 20-420C) ONLY",
     "ОТВЕРСТИЯ ПОД ШТИФТ НАВЕСКИ — ТОЛЬКО ДЛЯ КОРПУСА СТОЙКИ (20-410C И 20-420C)"),
    ("SECTION C-C LOCK LINK BORE", "СЕЧЕНИЕ C-C — ОТВЕРСТИЕ ПОД ЗАМКОВУЮ ТЯГУ"),
    ("SECTION E-E TYPICAL 2 TRANSFER BLOCK LUGS", "СЕЧЕНИЕ E-E ТИПИЧНО 2 УШКА БЛОКА ПЕРЕДАЧИ"),
    ("SECTION F-F TYPICAL 2 BRAKE MANIFOLD LUGS", "СЕЧЕНИЕ F-F ТИПИЧНО 2 УШКА ТОРМОЗНОГО КОЛЛЕКТОРА"),
    ("SECTION H-H TOOLING LUG", "СЕЧЕНИЕ H-H ТЕХНОЛОГИЧЕСКОЕ УШО"),
    ("HOLE TO DEPTH OF", "ОТВЕРСТИЕ НА ГЛУБИНУ"),
    ("FROM THIS SURFACE", "ОТ ЭТОЙ ПОВЕРХНОСТИ"),

    # --- РЕМОНТ ---
    ("INSTALL BEARING", "УСТАНОВИТЬ ПОДШИПНИК"),
    ("OVERSIZE TRANSFER DOWEL", "РЕМОНТНЫЙ ШТИФТ ПЕРЕДАЧИ"),
    ("OVERSIZE BACKING RING", "РЕМОНТНОЕ ОПОРНОЕ КОЛЬЦО"),
    ("OVERSIZE O RING", "РЕМОНТНОЕ УПЛОТНИТЕЛЬНОЕ КОЛЬЦО"),
    ("OVERSIZE SPHERICAL", "РЕМОНТНЫЙ СФЕРИЧЕСКИЙ"),
    ("OVERSIZE COMPONENTS", "РЕМОНТНЫЕ КОМПОНЕНТЫ"),
    ("TYPICAL INSTALLATION OF", "ТИПОВАЯ УСТАНОВКА"),
    ("INSTALLATION", "УСТАНОВКА"),
    ("CORRECT OVERSIZE BEARING", "ПРАВИЛЬНЫЙ РЕМОНТНЫЙ ПОДШИПНИК"),
    ("CORRECT OVERSIZE", "ПРАВИЛЬНЫЙ РАЗМЕР"),
    ("MODIFIED REPAIR LOWER BEARING", "МОДИФИЦИРОВАННЫЙ РЕМОНТНЫЙ НИЖНИЙ ПОДШИПНИК"),
    ("MODIFIED REPAIR LOWER", "МОДИФИЦИРОВАННЫЙ РЕМОНТ НИЖНЕГО"),
    ("BEARING SUBASSEMBLY", "ПОДСБОРКА ПОДШИПНИКА"),
    ("NEW INNER LINER", "НОВЫЙ ВНУТРЕННИЙ ВКЛАДЫШ"),
    ("REPAIR BEARING", "РЕМОНТНЫЙ ПОДШИПНИК"),
    ("REPAIR LUBRICATION", "РЕМОНТНЫЙ СМАЗОЧНЫЙ"),
    ("ADAPTOR", "АДАПТЕР"),
    ("PLANE PASSES", "ПЛОСКОСТЬ ПРОХОДИТ"),
    ("FLANGES OF REPAIR", "ФЛАНЦЫ РЕМОНТНЫХ"),
    ("BUSHES MAY BE", "ВТУЛОК МОГУТ БЫТЬ"),
    ("LOCALLY REMOVED", "УДАЛЕНЫ МЕСТНО"),
    ("IF NECESSARY", "ПРИ НЕОБХОДИМОСТИ"),
    ("REFER TO TABLE 601", "СМ. ТАБЛИЦУ 601"),
    ("(REFER TO TABLE 1)", "(СМ. ТАБЛИЦУ 1)"),
    ("COMMON ZONE", "ОБЩАЯ ЗОНА"),
    ("REPAIR BUSH", "РЕМОНТНАЯ ВТУЛКА"),
    ("REPAIR SLEEVE", "РЕМОНТНАЯ ВТУЛКА"),
    ("LINER DIMENSIONS", "РАЗМЕРЫ ВКЛАДЫША"),
    ("OVERSIZE BEARING", "РЕМОНТНЫЙ ПОДШИПНИК"),
    ("SECTION Z-Z WITHOUT BEARING", "СЕЧЕНИЕ Z-Z БЕЗ ПОДШИПНИКА"),
    ("SECTION Z-Z WITH BEARING", "СЕЧЕНИЕ Z-Z С ПОДШИПНИКОМ"),
    ("DIM. C", "РАЗМЕР C"),
    ("DIM. D", "РАЗМЕР D"),
    ("DIM. B", "РАЗМЕР B"),
    ("DIM.", "РАЗМЕР"),
    ("SURFACE", "ПОВЕРХНОСТЬ"),
    ("HOLE", "ОТВЕРСТИЕ"),

    # --- КЛЮЧЕВЫЕ СХЕМЫ ---
    ("REFER TO FIGURE", "СМ. РИСУНОК"),
    ("MAIN FITTING", "КОРПУС СТОЙКИ"),
    ("SLIDING TUBE", "СКОЛЬЗЯЩАЯ ТРУБА"),
    ("LOWER TORQUE LINK", "НИЖНИЙ ШЛИЦ-ШАРНИР"),
    ("UPPER DIAPHRAGM TUBE", "ВЕРХНЯЯ ДИАФРАГМЕННАЯ ТРУБА"),
    ("UPPER TORQUE LINK", "ВЕРХНИЙ ШЛИЦ-ШАРНИР"),
    ("TRANSFER BLOCK", "БЛОК ПЕРЕДАЧИ НАГРУЗКИ"),
    ("HARNESS SUPPORT BRACKET", "КРОНШТЕЙН КРЕПЛЕНИЯ ЖГУТА"),
    ("UPPER PIVOT BRACKET", "ВЕРХНИЙ ПОВОРОТНЫЙ КРОНШТЕЙН"),
    ("CYLINDER", "ЦИЛИНДР"),

    # --- РЕМОНТ К ШТИФТАМ / ШАРНИРАМ ---
    ("BEFORE CHROMIUM PLATE", "ДО ХРОМИРОВАНИЯ"),
    ("M-DLPS1031-2 CHROMIUM PLATE TERMINATION", "ЗАВЕРШЕНИЕ ХРОМИРОВАНИЯ ПО M-DLPS1031-2"),
    ("PART SECTION Z-Z", "ЧАСТИЧНЫЙ СЕЧЕНИЕ Z-Z"),

    # --- ПРОЧЕЕ ---
    ("LARGER VIEW AT", "УВЕЛИЧЕННЫЙ ВИД"),
    ("SECTION Z-Z", "СЕЧЕНИЕ Z-Z"),   # catch-all если не попало по regex
    ("DETAIL Z", "ДЕТАЛЬ Z"),
    ("VIEW A", "ВИД A"),

    # --- ДОПОЛНИТЕЛЬНЫЕ МНОГОСЛОВНЫЕ ---
    ("OVERSIZE BUSH", "РЕМОНТНАЯ ВТУЛКА"),
    ("OVERSIZE BORE", "РЕМОНТНОЕ ОТВЕРСТИЕ"),
    ("SEE TABLE", "СМ. ТАБЛИЦУ"),
    ("REMOVE SHARP EDGES", "СНЯТЬ ОСТРЫЕ КРОМКИ"),
    ("REMOVE SHARP", "СНЯТЬ ОСТРЫЕ"),
    ("SHARP EDGES", "ОСТРЫЕ КРОМКИ"),
    ("REMOVE EDGE", "СНЯТЬ КРОМКУ"),
    ("LUG WIDTH", "ШИРИНА ПРОУШИНЫ"),
    ("OVER LENGTH", "ПО ДЛИНЕ"),
    ("CORNER RADIUS", "РАДИУС СКРУГЛЕНИЯ"),
    ("OUTSIDE DIAMETER", "НАРУЖНЫЙ ДИАМЕТР"),
    ("INSIDE DIAMETER", "ВНУТРЕННИЙ ДИАМЕТР"),
    ("FULL LENGTH", "НА ВСЮ ДЛИНУ"),
    ("THIS LENGTH", "ЭТА ДЛИНА"),
    ("THIS BAND", "ЭТА ЗОНА"),
    ("THIS SURFACE", "ЭТА ПОВЕРХНОСТЬ"),
    ("THIS AREA", "ЭТА ЗОНА"),
    ("THIS LINE", "ЭТА ЛИНИЯ"),
    ("THIS DIMENSION", "ЭТОТ РАЗМЕР"),
    ("POINT C", "ТОЧКА C"),
    ("POINT B", "ТОЧКА B"),
    ("POINT A", "ТОЧКА A"),
    ("POINT D", "ТОЧКА D"),
    ("INCLUSIVE CHAMFER", "ВКЛЮЧАЯ ФАСКУ"),
    ("NOT EXTENDED ONTO", "НЕ РАСПРОСТРАНЯТЬСЯ НА"),
    ("MUST NOT ENTER", "НЕ ДОЛЖНО ВХОДИТЬ В"),
    ("MUST FOLLOW", "ДОЛЖНО СЛЕДОВАТЬ"),
    ("MUST BE", "ДОЛЖНО БЫТЬ"),
    ("WILL TOUCH", "БУДЕТ КАСАТЬСЯ"),
    ("BELOW THIS SURFACE", "НИЖЕ ЭТОЙ ПОВЕРХНОСТИ"),
    ("ABOVE THIS SURFACE", "ВЫШЕ ЭТОЙ ПОВЕРХНОСТИ"),
    ("ON THIS SURFACE", "НА ЭТОЙ ПОВЕРХНОСТИ"),
    ("ON THIS DIA.", "НА ЭТОМ ДИАМЕТРЕ"),
    ("ON THIS DIAMETER", "НА ЭТОМ ДИАМЕТРЕ"),
    ("ON THIS FACE", "НА ЭТОЙ ПОВЕРХНОСТИ"),
    ("ON THE CHAMFER", "НА ФАСКЕ"),
    ("NOT PRIMER", "БЕЗ ГРУНТА"),
    ("DEEP INCL.", "ГЛУБИНОЙ ВКЛ."),
    ("DEEP INCLUDING", "ГЛУБИНОЙ ВКЛЮЧАЯ"),
    ("DEEP", "ГЛУБИНОЙ"),
    ("INCL.", "ВКЛ."),
    ("NO FURTHER WORK REQUIRED", "ДОПОЛНИТЕЛЬНАЯ ОБРАБОТКА НЕ ТРЕБУЕТСЯ"),
    ("FURTHER WORK", "ДОПОЛНИТЕЛЬНАЯ ОБРАБОТКА"),
    ("AS REQUIRED", "ПО НЕОБХОДИМОСТИ"),
    ("AS NECESSARY", "ПО НЕОБХОДИМОСТИ"),
    ("WHERE APPLICABLE", "ГДЕ ПРИМЕНИМО"),
    ("WHERE FITTED", "ПРИ УСТАНОВКЕ"),
    ("WHERE BUSH FLANGE", "В МЕСТЕ ФЛАНЦА ВТУЛКИ"),
    ("ENSURE THAT", "УБЕДИТЬСЯ, ЧТО"),
    ("OPTIONAL ON", "ДОПУСТИМО НА"),
    ("OPTIONAL", "ДОПУСТИМО"),
    ("SPLIT PIN", "ШПЛИНТ"),
    ("BOLT", "БОЛТ"),
    ("WASHER", "ШАЙБА"),
    ("NUT", "ГАЙКА"),
    ("RADIUS (TYP)", "РАДИУС (ТИП.)"),
    ("(TYP)", "(ТИП.)"),
    ("REFER TO TABLE 1", "СМ. ТАБЛИЦУ 1"),
    ("REFER TO TABLE", "СМ. ТАБЛИЦУ"),
    ("REFER TO", "СМ."),
    ("DIA A", "ДИА. A"),
    ("DIA B", "ДИА. B"),
    ("DIA C", "ДИА. C"),
    ("DIA D", "ДИА. D"),
    ("DIA", "ДИА."),
    ("APPLY", "НАНЕСТИ"),

    # --- ДОПОЛНИТЕЛЬНЫЕ ФРАЗЫ ---
    ("FROM OUTSIDE", "СНАРУЖИ"),
    ("OVER RADIUS.", "ПО РАДИУСУ."),
    ("OVER RADIUS", "ПО РАДИУСУ"),
    ("SPHERICAL RAD.", "СФЕРИЧЕСКИЙ РАД."),
    ("SPHERICAL RADIUS", "СФЕРИЧЕСКИЙ РАДИУС"),
    ("SPHERICAL", "СФЕРИЧЕСКИЙ"),
    ("CENTERLINE", "ОСЕВАЯ ЛИНИЯ"),
    ("TO EXISTING RADIUS", "ДО СУЩЕСТВУЮЩЕГО РАДИУСА"),
    ("TO EXISTING", "ДО СУЩЕСТВУЮЩЕГО"),
    ("EXISTING", "СУЩЕСТВУЮЩИЙ"),
    ("REFERENCE", "СПРАВОЧНО"),
    ("DIMENSION", "РАЗМЕР"),
    ("THROUGH", "СКВОЗНОЙ"),
    ("CHECK", "КОНТРОЛЬ"),
    ("OVERSIZE", "РЕМОНТНЫЙ"),

    # --- ДОПОЛНИТЕЛЬНЫЕ КОНТЕКСТНЫЕ ФРАЗЫ ---
    ("SEE FIG.", "СМ. РИС."),
    ("(3 OFF)", "(3 ШТ.)"),
    ("(2 OFF)", "(2 ШТ.)"),
    ("DIM P", "РАЗМЕР P"),
    ("ON THE", "НА"),
    ("DO NOT", "НЕ"),
    ("FOR THE", "ДЛЯ"),
    ("IN THE", "В"),

    # --- ОДИНОЧНЫЕ СЛОВА (FALLBACK) — ДЛИННЫЕ формы ДО коротких! ---
    # Суффиксные формы ПЕРЕД базовыми (предотвращаем garbling)
    ("OVERLAPPING", "ПЕРЕКРЫТИЕ"),
    ("OVERLAP", "ПЕРЕКРЫТИЕ"),
    ("COATINGS", "ПОКРЫТИЯ"),
    ("COATING", "ПОКРЫТИЕ"),
    ("PLATING", "ПОКРЫТИЕ"),
    ("SEALANT", "ГЕРМЕТИК"),
    ("FLANGES", "ФЛАНЦЫ"),
    ("FLANGE", "ФЛАНЕЦ"),
    ("TERMINATED", "ЗАВЕРШЁННОЕ"),
    ("TERMINATION", "ЗАВЕРШЕНИЕ"),
    ("TERMINATE", "ОКАНЧИВАТЬСЯ"),
    ("ANYWHERE", "В ЛЮБОМ МЕСТЕ"),
    ("REMAINDER", "ОСТАЛЬНОЕ"),
    ("PERMITTED", "ДОПУСТИМО"),
    ("PERMISSIBLE", "ДОПУСТИМО"),
    ("IRREGULAR", "НЕРОВНАЯ"),
    ("INCLUSIVELY", "ВКЛЮЧИТЕЛЬНО"),
    ("INCLUSIVE", "ВКЛЮЧИТЕЛЬНО"),
    ("INCLUDING", "ВКЛЮЧАЯ"),
    ("INCLUDED", "ВКЛЮЧЁН"),
    ("SMOOTHLY", "ПЛАВНО"),
    ("SMOOTHED", "СГЛАЖЕНО"),
    ("SMOOTH", "ГЛАДКИЙ"),
    ("DAMAGED", "ПОВРЕЖДЁН"),
    ("REQUIRED", "ТРЕБУЕТСЯ"),
    ("APPLIED", "НАНЕСЕНО"),
    ("LARGER", "УВЕЛИЧЕННЫЙ"),
    ("BEFORE", "ДО"),
    ("AFTER", "ПОСЛЕ"),
    ("PAINT", "КРАСКА"),
    ("PLATED", "ПОКРЫТЫЙ"),
    ("PLATES", "ПОКРЫТИЯ"),
    ("PLATE", "ПОКРЫТИЕ"),
    ("POINTS", "ТОЧКИ"),
    ("POINT", "ТОЧКА"),
    ("LENGTHS", "ДЛИНЫ"),
    ("LENGTH", "ДЛИНА"),
    ("WIDTH", "ШИРИНА"),
    ("DEPTH", "ГЛУБИНА"),
    ("BLENDED", "СОПРЯЖЁННЫЙ"),
    ("BLENDING", "СОПРЯЖЕНИЕ"),
    ("BLEND", "СОПРЯЖЕНИЕ"),
    ("BORES", "ОТВЕРСТИЯ"),
    ("BORE", "ОТВЕРСТИЕ"),
    ("BUSHES", "ВТУЛКИ"),
    ("BUSH", "ВТУЛКА"),
    ("EDGES", "КРОМКИ"),
    ("EDGE", "КРОМКА"),
    ("LINER", "ВКЛАДЫШ"),
    ("LINES", "ЛИНИИ"),
    ("LINE", "ЛИНИЯ"),
    ("BAND", "ЗОНА"),
    ("AREAS", "ЗОНЫ"),
    ("AREA", "ЗОНА"),
    ("MUST", "ДОЛЖНО"),
    ("ONLY", "ТОЛЬКО"),
    ("THIS", "ЭТА"),
    ("THAT", "ЭТО"),
    ("EXTENT OF", "ПРОТЯЖЁННОСТЬ"),
    ("EXTENT", "ПРОТЯЖЁННОСТЬ"),
    ("DEPOSITS", "СЛОИ"),
    ("DEPOSIT", "СЛОЙ"),
    ("LOWER", "НИЖНИЙ"),
    ("UPPER", "ВЕРХНИЙ"),
    ("OUTER", "НАРУЖНЫЙ"),
    ("INNER", "ВНУТРЕННИЙ"),
    ("WALLS", "СТЕНКИ"),
    ("WALL", "СТЕНКА"),
    ("THICKNESS", "ТОЛЩИНА"),
    ("PLACES", "МЕСТ"),
    ("POSITIONS", "ПОЗИЦИИ"),
    ("CORNERS", "УГЛЫ"),
    ("CORNER", "УГЛОВОЙ"),
    ("REPAIRS", "РЕМОНТ"),
    ("REPAIR", "РЕМОНТ"),
    ("INSTALLED", "УСТАНОВЛЕН"),
    ("INSTALLATION", "УСТАНОВКА"),
    ("MINUTES", "МИНУТ"),
    ("MINUTE", "МИНУТА"),
    ("MACHINED", "ОБРАБОТАН"),
    ("MACHINE", "МЕХОБРАБОТКА"),
    ("GRINDING", "ШЛИФОВАНИЕ"),
    ("DRAIN", "ДРЕНАЖ"),
    ("THREAD", "РЕЗЬБА"),
    ("CROSS", "ПОПЕРЕЧНЫЙ"),
    ("FULL", "ПОЛНЫЙ"),
    ("THIN", "ТОНКИЙ"),
    ("THICK", "ТОЛСТЫЙ"),
    ("ABOVE", "ВЫШЕ"),
    ("BELOW", "НИЖЕ"),
    ("OVER", "ПО"),
    ("BETWEEN", "МЕЖДУ"),
    ("OUTSIDE", "НАРУЖНЫЙ"),
    ("INSIDE", "ВНУТРЕННИЙ"),
    ("REMAINING", "ОСТАВШИЙСЯ"),
    ("REMAIN", "ОСТАВАТЬСЯ"),
    ("FINISHED", "ОБРАБОТАННЫЙ"),
    ("SURFACES", "ПОВЕРХНОСТИ"),
    ("LUGS", "УШКИ"),
    ("RING", "КОЛЬЦО"),
    ("SUBASSEMBLY", "ПОДСБОРКА"),
    ("PIN", "ШТИФТ"),
    ("LIMIT", "ГРАНИЦА"),
    ("WITHOUT", "БЕЗ"),
    ("ONTO", "НА"),
    ("INTO", "В"),
    ("FROM", "ОТ"),
    ("WITH", "С"),
]

# Regex-паттерны: применяются до PHRASES
REGEX_PATTERNS = [
    # Figure NNN - Sheet N → Рисунок NNN – Лист N
    (re.compile(r'^Figure\s+(\d+)\s*-\s*Sheet\s+(\d+)\s*$'), r'Рисунок \1 – Лист \2'),
    # Figure NNN → Рисунок NNN (standalone)
    (re.compile(r'^Figure\s+(\d+)\s*$'), r'Рисунок \1'),
    # FIGURE NNN (uppercase standalone or in parentheses)
    (re.compile(r'\bFIGURE\s+(\d+)'), r'РИСУНОК \1'),
    # PART SECTION X-X → ЧАСТИЧНОЕ СЕЧЕНИЕ X-X (must come before general SECTION)
    (re.compile(r'\bPART\s+SECTION\s+([A-Z]{1,2})-([A-Z]{1,2})\b'), r'ЧАСТИЧНОЕ СЕЧЕНИЕ \1-\2'),
    (re.compile(r'\bPART\s+SECTION\s+([A-Z]{1,2})\b'), r'ЧАСТИЧНОЕ СЕЧЕНИЕ \1'),
    # PART VIEW ON ARROW X → ЧАСТИЧНЫЙ ВИД ПО СТРЕЛКЕ X
    (re.compile(r'\bPART\s+VIEW\s+ON\s+ARROW\s+([A-Z]{1,2})\b'), r'ЧАСТИЧНЫЙ ВИД ПО СТРЕЛКЕ \1'),
    (re.compile(r'\bPART\s+VIEW\s+ON\s+ARROW\b'), r'ЧАСТИЧНЫЙ ВИД ПО СТРЕЛКЕ'),
    # SECTION X-X (где X — одна или две буквы, например A-A, Z-Z, L-L, U-U)
    (re.compile(r'\bSECTION\s+([A-Z]{1,2})-([A-Z]{1,2})\b'), r'СЕЧЕНИЕ \1-\2'),
    # SECTION X (без тире)
    (re.compile(r'\bSECTION\s+([A-Z]{1,2})\b'), r'СЕЧЕНИЕ \1'),
    # DETAIL X
    (re.compile(r'\bDETAIL\s+([A-Z]{1,2})\b'), r'ДЕТАЛЬ \1'),
    # VIEW ON ARROW [X] — must come before generic VIEW X
    (re.compile(r'\bVIEW\s+ON\s+ARROW\s+([A-Z]{1,2})\b'), r'ВИД ПО СТРЕЛКЕ \1'),
    (re.compile(r'\bVIEW\s+ON\s+ARROW\b'), r'ВИД ПО СТРЕЛКЕ'),
    # VIEW X (одна буква или W, V и т.д.) — exclude ON, AT
    (re.compile(r'\bVIEW\s+(?!ON\b|AT\b)([A-Z]{1,2})\b'), r'ВИД \1'),
    # REPAIR No. X-Y (с буквами или цифрами)
    (re.compile(r'\bRepair No\.\s*(\S+)'), r'Ремонт № \1'),
    (re.compile(r'\bREPAIR No\.\s*(\S+)'), r'РЕМОНТ № \1'),
    # "REPAIR" + space + number pattern (standalone REPAIR without "No.")
    (re.compile(r'\bREPAIR\s+(\d+-\d+)'), r'РЕМОНТ \1'),
    # Page NNN  (в шапке)
    (re.compile(r'^Page\s+(\S+)$'), r'Стр. \1'),
    # DEGREES -> ГРАДУСОВ (число + DEGREES)
    (re.compile(r'(\d+)\s+DEGREES'), r'\1 ГРАДУСОВ'),
    (re.compile(r'(\d+)\s+DEGREE'), r'\1 ГРАДУС'),
    (re.compile(r'(\d+)\s+SECONDS'), r'\1 СЕКУНД'),
    (re.compile(r'(\d+)\s+MINUTES'), r'\1 МИНУТ'),
    # Замена "and"/"AND" на "и"/"И"
    (re.compile(r'\bAND\b'), 'И'),
    (re.compile(r'\band\b'), 'и'),
    # Замена "or"/"OR" на "или"/"ИЛИ"
    (re.compile(r'\bOR\b'), 'ИЛИ'),
    (re.compile(r'\bor\b'), 'или'),
    # "Only" -> "только", "for" -> "для"
    (re.compile(r'\bfor\b'), 'для'),
    (re.compile(r'\bOnly\b'), 'только'),
    # FOR MAIN FITTING (...) ONLY
    (re.compile(r'FOR MAIN FITTING\s+\(([^)]+)\)\s+ONLY'),
        lambda m: f'ТОЛЬКО ДЛЯ КОРПУСА СТОЙКИ ({m.group(1)})'),
    # FOR MAIN FITTING (...\n...) ONLY — обрабатывается отдельно
    # UNPLATED LENGTH -> handled above
    # (Qty N) -> (Кол-во N)
    (re.compile(r'\(Qty\.?\s*(\d+)\)'), r'(Кол-во \1)'),
    # Qty N -> Кол-во N
    (re.compile(r'\bQty\.?\s*(\d+)'), r'Кол-во \1'),
]


def should_skip(text: str) -> bool:
    """True если строку НЕ надо переводить."""
    s = text.strip()
    if not s:
        return True
    # Только цифры, пробелы, запятые, точки, скобки, дефисы (размерные значения)
    if re.match(r'^[\d\s,\.\(\)\-\+/]+$', s):
        return True
    # Одна–три буквы (метка измерения: A, B, AA, Z)
    if re.match(r'^[A-Za-z]{1,3}$', s):
        return True
    # ATA-код 32-12-22
    if re.match(r'^\d{2}-\d{2}-\d{2}$', s):
        return True
    # Дата (Dec 6/2019, Mar 18/2025)
    if re.match(r'^[A-Z][a-z]{2}\s+\d+/\d{4}$', s):
        return True
    # Номер детали A321xxx
    if re.match(r'^A321[A-Z0-9\-\.]+$', s):
        return True
    # Коды спецификаций
    if re.match(r'^(?:PCS|IFC|AMS|M-DLPS|MIL|DEF|NCT|PSC|SB|CMM|IPC|DPL|NDT|AMM|ATA|SRM|AECMA|FED|STAN)[-\d]', s):
        return True
    # Материалы (SERMETEL W, ALOCROM, MASTINOX, MOLYKOTE)
    if re.match(r'^(?:SERMETEL|ALOCROM|MASTINOX|MOLYKOTE|AVIOX)\b', s):
        return True
    # Компания / CAGE
    if 'SAFRAN LANDING SYSTEMS' in s or s.startswith('CAGE:'):
        return True
    # Уже кириллица
    if re.search(r'[А-Яа-яЁё]', s) and not re.search(r'[A-Za-z]', s):
        return True
    # Нет латинских букв вообще
    if not re.search(r'[A-Za-z]', s):
        return True
    return False


def translate_line(text: str) -> str:
    """Переводит строку текста."""
    s = text.strip()

    # 1) Точное совпадение в FIXED
    if s in FIXED:
        return FIXED[s]

    # 2) Пропустить?
    if should_skip(s):
        return text

    # 3) Применить regex-паттерны
    result = s
    for pat, repl in REGEX_PATTERNS:
        if callable(repl):
            result = pat.sub(repl, result)
        else:
            result = pat.sub(repl, result)

    # 4) Применить фразовые замены
    for en, ru in PHRASES:
        result = result.replace(en, ru)

    # 5) Если совсем ничего не изменилось и только латиница — вернуть оригинал
    # (не переводить неизвестные строки — лучше оставить на английском)
    if result == s:
        return text

    return result


# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ PDF
# =============================================================================

def translate_pdf(input_path: str, output_path: str) -> None:
    doc = fitz.open(input_path)
    total = len(doc)
    print(f"Страниц: {total}")

    for page_idx, page in enumerate(doc):
        if page_idx % 20 == 0:
            print(f"  Обрабатываю стр. {page_idx+1}/{total}...")

        # Собираем данные о строках, которые нужно заменить
        replacements = []  # list of (bbox, translated_text, font_size, is_bold)

        d = page.get_text("dict")
        for block in d["blocks"]:
            if block["type"] != 0:
                continue  # пропускаем изображения
            for line in block["lines"]:
                # Собираем текст и bbox линии
                spans = line["spans"]
                if not spans:
                    continue

                line_text = "".join(sp["text"] for sp in spans)
                if not line_text.strip():
                    continue

                translated = translate_line(line_text)
                if translated == line_text:
                    continue  # ничего не изменилось

                # Берём параметры шрифта из первого span
                sp0 = spans[0]
                font_size = sp0["size"]
                is_bold = bool(sp0["flags"] & 2**4)  # bold flag
                line_bbox = fitz.Rect(line["bbox"])

                replacements.append((line_bbox, translated, font_size, is_bold))

        if not replacements:
            continue

        # Применяем замены: сначала маскируем оригинал, потом вставляем перевод
        for bbox, trans_text, font_size, is_bold in replacements:
            # 1) Добавляем аннотацию редактирования (белый прямоугольник)
            # Немного расширяем bbox чтобы полностью перекрыть оригинал
            expanded = bbox + (-1.5, -1.5, 1.5, 1.5)
            page.add_redact_annot(expanded, fill=(1, 1, 1))

        # Применяем редактирование — удаляем текст, НЕ трогаем изображения
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE
        )

        # 2) Вставляем переведённый текст
        for bbox, trans_text, font_size, is_bold in replacements:
            # Пробуем вписать в bbox, уменьшаем размер если не влезает
            # Для автоматического выравнивания используем insert_textbox
            fitted = False
            for size_factor in [1.0, 0.9, 0.82, 0.75, 0.68, 0.60, 0.55, 0.50, 0.45]:
                cur_size = font_size * size_factor
                if cur_size < 5.5:
                    cur_size = 5.5
                overflow = page.insert_textbox(
                    bbox,
                    trans_text,
                    fontfile=FONT_FILE,
                    fontname=FONT_NAME,
                    fontsize=cur_size,
                    color=(0, 0, 0),
                    align=fitz.TEXT_ALIGN_LEFT,
                )
                if overflow >= 0:
                    fitted = True
                    break
            if not fitted:
                # Как минимум вставляем с минимальным размером (5.5pt абсолютный мин.)
                fallback_size = max(font_size * 0.40, 5.5)
                page.insert_textbox(
                    bbox,
                    trans_text,
                    fontfile=FONT_FILE,
                    fontname=FONT_NAME,
                    fontsize=fallback_size,
                    color=(0, 0, 0),
                    align=fitz.TEXT_ALIGN_LEFT,
                )

    doc.save(output_path, garbage=4, deflate=True)
    print(f"\nСохранено: {output_path}")


if __name__ == "__main__":
    if not os.path.exists(FONT_FILE):
        raise FileNotFoundError(f"Шрифт не найден: {FONT_FILE}")
    translate_pdf(INPUT_PDF, OUTPUT_PDF)
    print("Готово!")
