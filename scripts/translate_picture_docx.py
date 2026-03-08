from __future__ import annotations

import argparse
import re
import zipfile
from collections import Counter
from pathlib import Path

from lxml import etree

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

MONTHS = {
    "jan": "янв",
    "feb": "фев",
    "mar": "мар",
    "apr": "апр",
    "may": "мая",
    "jun": "июн",
    "jul": "июл",
    "aug": "авг",
    "sep": "сен",
    "oct": "окт",
    "nov": "ноя",
    "dec": "дек",
}

COMPONENT_FORMS: dict[str, tuple[str, str]] = {
    "bracket": ("Кронштейн", "кронштейна"),
    "cylinder": ("Цилиндр", "цилиндра"),
    "drag-arm spacer": ("Проставка рычага подкоса", "проставки рычага подкоса"),
    "forward pintle pin": ("Передний штифт навеса стойки", "переднего штифта навеса стойки"),
    "harness support bracket": ("Кронштейн крепления жгута", "кронштейна крепления жгута"),
    "inflation valve": ("Клапан накачивания", "клапана накачивания"),
    "lower bearing subassembly": ("Сборка нижнего подшипника", "сборки нижнего подшипника"),
    "main fitting": ("Корпус стойки", "корпуса стойки"),
    "main fitting subassembly": ("Сборка корпуса стойки", "сборки корпуса стойки"),
    "oversize bearings": ("Ремонтные подшипники", "ремонтных подшипников"),
    "oversize bush(es)": ("Ремонтные втулки", "ремонтных втулок"),
    "oversize bushes": ("Ремонтные втулки", "ремонтных втулок"),
    "oversize lubrication adapter": ("Ремонтный смазочный адаптер", "ремонтного смазочного адаптера"),
    "oversize transfer dowel": ("Ремонтный передаточный штифт", "ремонтного передаточного штифта"),
    "pin": ("Штифт", "штифта"),
    "pivot pin": ("Штифт шарнира", "штифта шарнира"),
    "repair bearing": ("Ремонтный подшипник", "ремонтного подшипника"),
    "repair bush": ("Ремонтная втулка", "ремонтной втулки"),
    "repair bushes": ("Ремонтные втулки", "ремонтных втулок"),
    "retaining pin": ("Фиксирующий штифт", "фиксирующего штифта"),
    "slave link": ("Ведомая тяга", "ведомой тяги"),
    "sliding tube": ("Скользящая труба", "скользящей трубы"),
    "spacer": ("Проставка", "проставки"),
    "torque link": ("Шлиц-шарнир", "шлиц-шарнира"),
    "transfer block": ("Передаточный блок", "передаточного блока"),
    "uplock pin": ("Штифт замка убранного положения", "штифта замка убранного положения"),
    "upper diaphragm tube": ("Верхняя диафрагменная трубка", "верхней диафрагменной трубки"),
    "upper diaphragm tube subassembly": (
        "Сборка верхней диафрагменной трубки",
        "сборки верхней диафрагменной трубки",
    ),
    "upper pivot bracket": ("Верхний кронштейн шарнира", "верхнего кронштейна шарнира"),
    "valve stem": ("Шток клапана", "штока клапана"),
}

COMPONENT_KEYS = sorted(COMPONENT_FORMS, key=len, reverse=True)

CAPTION_SUBLABELS = {
    "bush": "Втулка",
    "chromium plate termination": "Граница хромового покрытия",
    "installation": "Установка",
    "key diagram": "Ключевая схема",
    "machining": "Механическая обработка",
    "machining and inner liner installation": "Механическая обработка и установка внутренней втулки",
    "machining and installation": "Механическая обработка и установка",
    "machining and liner installation": "Механическая обработка и установка втулки",
    "protective treatment": "Защитная обработка",
    "repair bush installation": "Установка ремонтной втулки",
    "repair bush machining and installation": "Механическая обработка и установка ремонтной втулки",
}

CUSTOM_PHRASES: tuple[tuple[str, str], ...] = (
    ("BLEND SMOOTHLY TO ADJACENT SURFACES", "ПЛАВНО СОПРЯЧЬ С ПРИЛЕГАЮЩИМИ ПОВЕРХНОСТЯМИ"),
    ("SPHERICAL RAD", "СФЕРИЧ. РАД."),
    ("REMOVE EDGES", "СНЯТЬ КРОМКИ"),
    ("INSTALL BUSH FLUSH TO", "УСТАНОВИТЬ ВТУЛКУ ЗАПОДЛИЦО"),
    ("BELOW SURFACE", "НИЖЕ ПОВЕРХНОСТИ"),
    ("APPLY SEALANT", "НАНЕСТИ ГЕРМЕТИК"),
    ("TO PCS-7200", "ПО PCS-7200"),
    ("DO NOT APPLY CADMIUM PLATE", "КАДМИЕВОЕ ПОКРЫТИЕ НЕ НАНОСИТЬ"),
    ("DO NOT APPLY", "НЕ НАНОСИТЬ"),
    ("BUSHES", "ВТУЛКИ"),
    ("BUSH", "ВТУЛКА"),
    ("PLANE PASSES THROUGH", "ПЛОСКОСТЬ ПРОХОДИТ ЧЕРЕЗ"),
    ("OVERSIZE BUSHES", "РЕМОНТНЫЕ ВТУЛКИ"),
    ("OVERSIZE BUSH(ES)", "РЕМОНТНЫЕ ВТУЛКИ"),
    ("OVERSIZE BUSH", "РЕМОНТНАЯ ВТУЛКА"),
    ("OVERSIZE BEARING", "РЕМОНТНЫЙ ПОДШИПНИК"),
    ("INSTALL BUSH FLUSH TO BELOW SURFACE", "УСТАНОВИТЬ ВТУЛКУ ЗАПОДЛИЦО НИЖЕ ПОВЕРХНОСТИ"),
    ("CHECK HONE", "КОНТР. ХОНИНГ."),
    ("ELECTRICALLY CONDUCTING", "ЭЛЕКТРОПРОВОДЯЩИЙ"),
    ("RUBBERISED SEALANT", "РЕЗИНОПОДОБНЫЙ ГЕРМЕТИК"),
    ("IN THE BORES", "В ОТВЕРСТИЯ"),
    ("ON BOTH BUSHES", "НА ОБЕ ВТУЛКИ"),
    ("FACES OF THE BUSHES", "ТОРЦЫ ВТУЛОК"),
    ("WITHOUT OVERFLOW", "БЕЗ ВЫТЕКАНИЯ"),
    ("NITRIDING DEPTH", "ГЛУБИНА НИТРИРОВАНИЯ"),
    ("REMOVAL OVER AREA SHOWN", "СНЯТИЕ НА ПОКАЗАННОМ УЧАСТКЕ"),
    ("ARE TO BE NITRIDED BEFORE DESPATCH TO OVERHAUL AGENCY", "ДОЛЖНЫ БЫТЬ НИТРИРОВАНЫ ПЕРЕД ОТПРАВКОЙ В РЕМОНТНОЕ ПОДРАЗДЕЛЕНИЕ"),
    ("CORNER RADIUS", "УГЛОВОЙ РАДИУС"),
    ("ARE TO BE MACHINED ON RECEIPT", "ОБРАБОТАТЬ ПРИ ПОЛУЧЕНИИ"),
    ("BORES", "ОТВЕРСТИЯ"),
    ("SPHERICAL RAD.", "СФЕРИЧ. РАД."),
    ("TYP.", "ТИП."),
    ("REPAIR TO", "РЕМОНТ"),
    ("REPAIR NO.", "РЕМОНТ №"),
    ("PART NO.", "№ ДЕТАЛЕЙ"),
    ("COMPONENT MAINTENANCE MANUAL", "РУКОВОДСТВО ПО ТЕХНИЧЕСКОМУ ОБСЛУЖИВАНИЮ КОМПОНЕНТОВ"),
    ("MAIN LANDING GEAR LEG", "СТОЙКА ОСНОВНОГО ШАССИ"),
    ("REPAIR BUSHES", "РЕМОНТНЫЕ ВТУЛКИ"),
    ("REPAIR BUSH", "РЕМОНТНАЯ ВТУЛКА"),
    ("REPAIR BEARING", "РЕМОНТНЫЙ ПОДШИПНИК"),
    ("DEPOSIT", "СЛОЙ"),
    ("COMMON ZONE", "ОБЩАЯ ЗОНА"),
    ("WORKING DIA.", "РАБОЧИЙ ДИАМ."),
    ("WORKING DIAMETER", "РАБОЧИЙ ДИАМЕТР"),
    ("EXTENT OF SHOT PEENING", "ПРОТЯЖЁННОСТЬ ДРОБЕСТРУЙНОЙ ОБРАБОТКИ"),
    ("SHOT PEENING", "ДРОБЕСТРУЙНАЯ ОБРАБОТКА"),
    ("UNPLATED LENGTH", "ДЛИНА БЕЗ ПОКРЫТИЯ"),
    ("PLATING THICKNESS", "ТОЛЩИНА ПОКРЫТИЯ"),
    ("FULL CHROME", "ПОЛНОЕ ХРОМИРОВАНИЕ"),
    ("IT CAN BE FINISHED BY GRINDING", "МОЖЕТ БЫТЬ ДОВЕДЕНО ШЛИФОВАНИЕМ"),
    ("WILL TERMINATE ANYWHERE ON THE", "МОЖЕТ ЗАКАНЧИВАТЬСЯ В ЛЮБОЙ ТОЧКЕ"),
    ("MUST NOT EXTEND ONTO", "НЕ ДОЛЖНО ЗАХОДИТЬ НА"),
    ("MUST NOT ENTER", "НЕ ДОЛЖНО ЗАХОДИТЬ В"),
    ("MAKE EDGES SMOOTH", "СГЛАДИТЬ КРОМКИ"),
    ("SHOT PEEN", "ДРОБЕСТРУЙНАЯ ОБРАБОТКА"),
    ("MACHINE TO", "ОБРАБОТАТЬ ДО"),
    ("GRIND TO", "ШЛИФОВАТЬ ДО"),
    ("AFTER THREAD", "ПОСЛЕ НАРЕЗАНИЯ РЕЗЬБЫ"),
    ("CENTERS", "МЕЖОСЕВОЕ РАССТОЯНИЕ"),
    ("NOT TO SCALE", "НЕ В МАСШТАБЕ"),
    ("SLEEVE", "ВТУЛКА"),
    ("LARGER VIEW AT", "УВЕЛИЧЕННЫЙ ВИД В ТОЧКЕ"),
    ("LARGER VIEW", "УВЕЛИЧЕННЫЙ ВИД"),
    ("DIMENSION", "РАЗМЕР"),
    ("MACHINE THIS FACE ONLY", "ОБРАБОТАТЬ ТОЛЬКО ЭТУ ПОВЕРХНОСТЬ"),
    ("MACHINE THIS FACE", "ОБРАБОТАТЬ ЭТУ ПОВЕРХНОСТЬ"),
    ("IRREGULAR LINE IS PERMITTED", "НЕРОВНАЯ ЛИНИЯ ДОПУСКАЕТСЯ"),
    ("DEC", "ДЕК"),
    ("MAR", "МАР"),
    ("LUGS", "ПРОУШИНЫ"),
    ("LUG", "ПРОУШИНА"),
    ("CROSS BORES", "ПОПЕРЕЧНЫЕ ОТВЕРСТИЯ"),
    ("CROSS BORE", "ПОПЕРЕЧНОЕ ОТВЕРСТИЕ"),
    ("NO ZINC-NICKEL OR PAINT", "БЕЗ ЦИНК-НИКЕЛЕВОГО ПОКРЫТИЯ ИЛИ КРАСКИ"),
    ("NO CADMIUM", "БЕЗ КАДМИЕВОГО ПОКРЫТИЯ"),
    ("UP TO BUSH FLANGES", "ДО ФЛАНЦЕВ ВТУЛОК"),
    ("ON THIS SURFACE", "НА ЭТОЙ ПОВЕРХНОСТИ"),
    ("OVER LENGTH", "НА ДЛИНЕ"),
    ("FROM CENTER", "ОТ ЦЕНТРА"),
    ("PAGE", "СТРАНИЦА"),
    ("FOR", "ДЛЯ"),
    ("ON", "НА"),
    ("BEFORE", "ДО"),
    ("AFTER", "ПОСЛЕ"),
    ("LIMIT OF", "ПРЕДЕЛ"),
    ("EXTENT OF FINE LIMIT", "ПРОТЯЖЁННОСТЬ УЧАСТКА ТОЧНОГО ДОПУСКА"),
    ("FINE LIMIT", "УЧАСТОК ТОЧНОГО ДОПУСКА"),
    ("REMAINDER MAY BE", "ОСТАЛЬНАЯ ЧАСТЬ МОЖЕТ БЫТЬ"),
    ("TO REMAIN ON OR PROUD OF", "ОСТАВАТЬСЯ НА ИЛИ ВЫСТУПАТЬ НАД"),
    ("TO ENTER OR STAND PROUD OF", "ЗАХОДИТЬ В ИЛИ ВЫСТУПАТЬ НАД"),
    ("PROUD OF", "ВЫСТУПАТЬ НАД"),
    ("AROUND", "ВОКРУГ"),
    ("CHANGE OVER VALVE", "ПЕРЕКЛЮЧАЮЩИЙ КЛАПАН"),
    ("CHANGEOVER VALVE", "ПЕРЕКЛЮЧАЮЩИЙ КЛАПАН"),
    ("DOWEL", "УСТАНОВОЧНЫЙ ШТИФТ"),
    ("DEGREE", "ГРАДУСА"),
    ("MUST BE", "ДОЛЖНА БЫТЬ"),
    ("MUST FOLLOW THE LINE OF EXISTING BORE", "ДОЛЖЕН СЛЕДОВАТЬ ЛИНИИ СУЩЕСТВУЮЩЕГО ОТВЕРСТИЯ"),
    ("MUST FOLLOW", "ДОЛЖЕН СЛЕДОВАТЬ"),
    ("EXISTING BORE", "СУЩЕСТВУЮЩЕГО ОТВЕРСТИЯ"),
    ("PAINT TO PCS-2500", "КРАСКА ПО PCS-2500"),
    ("PRIMER PAINT TO PCS-2500", "ГРУНТОВОЧНАЯ КРАСКА ПО PCS-2500"),
    ("PRIMER PAINT", "ГРУНТОВОЧНАЯ КРАСКА"),
    ("DO NOT PAINT", "НЕ ОКРАШИВАТЬ"),
    ("INTERNALLY", "ИЗНУТРИ"),
    ("INTERNAL", "ВНУТРЕННИЙ"),
    ("EXTERNAL", "НАРУЖНЫЙ"),
    ("REARSIDE ONLY", "ТОЛЬКО С ОБРАТНОЙ СТОРОНЫ"),
    ("ENLARGED DETAIL", "УВЕЛИЧЕННАЯ ДЕТАЛЬ"),
    ("DIM.", "РАЗМ."),
    ("QTY.", "КОЛ-ВО"),
    ("QTY", "КОЛ-ВО"),
    ("LENGTH OF CADMIUM PLATE", "ДЛИНА КАДМИЕВОГО ПОКРЫТИЯ"),
    ("VIEW ON ARROW", "ВИД ПО СТРЕЛКЕ"),
    ("FROM OUTSIDE SURFACE", "ОТ НАРУЖНОЙ ПОВЕРХНОСТИ"),
    ("FROM THIS SURFACE", "ОТ ЭТОЙ ПОВЕРХНОСТИ"),
    ("TO DEPTH OF", "НА ГЛУБИНУ"),
    ("CENTERLINE OF SLIDING TUBE", "ОСЬ СКОЛЬЗЯЩЕЙ ТРУБЫ"),
    ("CENTERLINE", "ОСЬ"),
    ("FOR ALL SECTION VIEWS SEE SHEET 2", "ДЛЯ ВСЕХ ВИДОВ В СЕЧЕНИИ СМ. ЛИСТ 2"),
    ("AXLE BORE", "ОТВЕРСТИЕ ОСИ"),
    ("AXLE BORE", "ОТВЕРСТИЯ ОСИ"),
    ("AXLE NUT CROSS BOLT HOLES", "ОТВЕРСТИЯ ПОПЕРЕЧНОГО БОЛТА ГАЙКИ ОСИ"),
    ("KNUCKLE BORES", "ОТВЕРСТИЯ ВИЛКИ"),
    ("TOOLING LUG", "ТЕХНОЛОГИЧЕСКАЯ ПРОУШИНА"),
    ("TOOLING LUGS", "ТЕХНОЛОГИЧЕСКИЕ ПРОУШИНЫ"),
    ("GREASE HOLES", "СМАЗОЧНЫЕ ОТВЕРСТИЯ"),
    ("BRAKE FLANGE", "ТОРМОЗНОЙ ФЛАНЕЦ"),
    ("BARREL OUTER DIA.", "НАРУЖНЫЙ ДИАМЕТР ЦИЛИНДРА"),
    ("BARREL", "ЦИЛИНДР"),
    ("JOURNAL", "ШЕЙКА"),
    ("PINTLE CROSS BORES", "ПОПЕРЕЧНЫЕ ОТВЕРСТИЯ ШТИФТА НАВЕСА"),
    ("PINTLE BORES", "ОТВЕРСТИЯ ШТИФТА НАВЕСА"),
    ("DRAG ARM HOLES", "ОТВЕРСТИЯ РЫЧАГА ПОДКОСА"),
    ("RETRACTION BORES", "ОТВЕРСТИЯ УЗЛА УБОРКИ"),
    ("UPPER DOOR LUGS", "ВЕРХНИЕ ПРОУШИНЫ СТВОРКИ"),
    ("LOWER DOOR LUGS", "НИЖНИЕ ПРОУШИНЫ СТВОРКИ"),
    ("UPLOCK LUGS", "ПРОУШИНЫ ЗАМКА УБРАННОГО ПОЛОЖЕНИЯ"),
    ("UPPER DIAPHRAGM TUBE CROSS BORE", "ПОПЕРЕЧНОЕ ОТВЕРСТИЕ ВЕРХНЕЙ ДИАФРАГМЕННОЙ ТРУБКИ"),
    ("CLOCK LINK BORE", "ОТВЕРСТИЕ ТЯГИ ЗАМКА"),
    ("LOCK LINK BORE", "ОТВЕРСТИЕ ТЯГИ ЗАМКА"),
    ("TRANSFER BLOCK LUGS", "ПРОУШИНЫ ПЕРЕДАТОЧНОГО БЛОКА"),
    ("BRAKE MANIFOLD LUGS", "ПРОУШИНЫ ТОРМОЗНОГО КОЛЛЕКТОРА"),
    ("LOWER CARDAN BORE", "ОТВЕРСТИЕ НИЖНЕГО КАРДАНА"),
    ("APPLY SEALANT TO PCS-7200", "НАНЕСТИ ГЕРМЕТИК ПО PCS-7200"),
    ("SEALANT TO PCS-7200", "ГЕРМЕТИК ПО PCS-7200"),
    ("SEALANT TO PCS 7200", "ГЕРМЕТИК ПО PCS-7200"),
    ("APPLY FILLET SEALANT", "НАНЕСТИ ВАЛИКОВЫЙ ГЕРМЕТИК"),
    ("APPLY LOCTITE GRADE 270", "НАНЕСТИ LOCTITE GRADE 270"),
    ("APPLY MOLYKOTE 111", "НАНЕСТИ MOLYKOTE 111"),
    ("APPLY CADMIUM PLATE ALL OVER", "НАНЕСТИ КАДМИЕВОЕ ПОКРЫТИЕ ПО ВСЕЙ ПОВЕРХНОСТИ"),
    ("APPLY COAT OF ALUMINIUM (IVD)", "НАНЕСТИ ПОКРЫТИЕ ИЗ АЛЮМИНИЯ (IVD)"),
    ("APPLY ELECTRICALLY CONDUCTING MOLYKOTE 111", "НАНЕСТИ ЭЛЕКТРОПРОВОДЯЩИЙ MOLYKOTE 111"),
    ("APPLY ELECTRICALLY CONDUCTING", "НАНЕСТИ ЭЛЕКТРОПРОВОДЯЩИЙ"),
    ("RUBBERISED SEALANT", "РЕЗИНОПОДОБНЫЙ ГЕРМЕТИК"),
    ("ADHESIVE TO PCS-5303", "КЛЕЙ ПО PCS-5303"),
    ("ADHESIVE", "КЛЕЙ"),
    ("SEALANT", "ГЕРМЕТИК"),
    ("INSTALL THE APPLICABLE LUBRICATION ADAPTOR", "УСТАНОВИТЬ СООТВЕТСТВУЮЩИЙ СМАЗОЧНЫЙ АДАПТЕР"),
    ("REPAIR LUBRICATION ADAPTOR", "СМАЗОЧНЫЙ АДАПТЕР РЕМОНТНОГО РАЗМЕРА"),
    ("LUBRICATION ADAPTOR", "СМАЗОЧНЫЙ АДАПТЕР"),
    ("LUBRICATION BORE", "СМАЗОЧНОЕ ОТВЕРСТИЕ"),
    ("REFER TO TABLE", "СМ. ТАБЛИЦУ"),
    ("REFER TO FIGURE", "СМ. РИСУНОК"),
    ("PART SECTION", "ЧАСТИЧНОЕ СЕЧЕНИЕ"),
    ("SECTION", "СЕЧЕНИЕ"),
    ("DETAIL", "ДЕТАЛЬ"),
    ("VIEW", "ВИД"),
    ("FACE", "ПОВЕРХНОСТЬ"),
    ("POINT", "ТОЧКА"),
    ("SURFACE FINISH", "ШЕРОХОВАТОСТЬ ПОВЕРХНОСТИ"),
    ("SURFACE", "ПОВЕРХНОСТЬ"),
    ("BOTH SIDES", "ОБЕ СТОРОНЫ"),
    ("BOTH BORES", "ОБА ОТВЕРСТИЯ"),
    ("BOTH HOLES", "ОБА ОТВЕРСТИЯ"),
    ("2 HOLES", "2 ОТВЕРСТИЯ"),
    ("3 HOLES", "3 ОТВЕРСТИЯ"),
    ("4 HOLES", "4 ОТВЕРСТИЯ"),
    ("BOTH FLANGES", "ОБА ФЛАНЦА"),
    ("BOTH ENDS", "ОБА КОНЦА"),
    ("THIS FACE ONLY", "ТОЛЬКО ЭТА ПОВЕРХНОСТЬ"),
    ("THIS FACE", "ЭТА ПОВЕРХНОСТЬ"),
    ("ONLY", "ТОЛЬКО"),
    ("INNER DIAMETER", "ВНУТРЕННИЙ ДИАМЕТР"),
    ("DIAMETER", "ДИАМЕТР"),
    ("DIA.", "ДИАМ."),
    ("DIA", "ДИАМ."),
    ("RADIUS", "РАДИУС"),
    ("RAD.", "РАД."),
    ("RAD", "РАД."),
    ("REF.", "СПРАВ."),
    ("REF", "СПРАВ."),
    ("MIN.", "МИН."),
    ("MAX.", "МАКС."),
    ("MINIMUM", "МИНИМУМ"),
    ("MAXIMUM", "МАКСИМУМ"),
    ("DEEP", "ГЛУБИНА"),
    ("DEPTH", "ГЛУБИНА"),
    ("THROUGH", "СКВОЗНОЙ"),
    ("THRU", "СКВОЗНОЙ"),
    ("BORE", "ОТВЕРСТИЕ"),
    ("HOLES", "ОТВЕРСТИЯ"),
    ("HOLE", "ОТВЕРСТИЕ"),
    ("PLACES", "МЕСТА"),
    ("POSITIONS", "ПОЗИЦИИ"),
    ("POSITION", "ПОЗИЦИЯ"),
    ("TYPICAL", "ТИПОВОЕ"),
    ("INCLUDING CHAMFERS", "ВКЛЮЧАЯ ФАСКИ"),
    ("INCLUDING CHAMFER", "ВКЛЮЧАЯ ФАСКУ"),
    ("CHAMFERS ONLY", "ТОЛЬКО ФАСКИ"),
    ("CHAMFER", "ФАСКА"),
    ("CHAMFERS", "ФАСКИ"),
    ("INCLUSIVE", "СУММАРНЫЙ"),
    ("SPOTFACE", "ПОДРЕЗКА ПЛОЩАДКИ"),
    ("CHROMIUM PLATED SURFACE", "ПОВЕРХНОСТЬ С ХРОМОВЫМ ПОКРЫТИЕМ"),
    ("CHROMIUM PLATED LENGTH", "ДЛИНА ХРОМИРОВАННОГО УЧАСТКА"),
    ("CHROMIUM PLATE DEPOSIT", "СЛОЙ ХРОМОВОГО ПОКРЫТИЯ"),
    ("CHROMIUM PLATE", "ХРОМОВОЕ ПОКРЫТИЕ"),
    ("CHROME PLATING DEPOSIT", "СЛОЙ ХРОМОВОГО ПОКРЫТИЯ"),
    ("CHROME PLATING", "ХРОМОВОЕ ПОКРЫТИЕ"),
    ("CHROME", "ХРОМ"),
    ("FULL CHROME PLATING THICKNESS", "ПОЛНАЯ ТОЛЩИНА ХРОМОВОГО ПОКРЫТИЯ"),
    ("CADMIUM PLATE", "КАДМИЕВОЕ ПОКРЫТИЕ"),
    ("SULPHAMATE NICKEL PLATE", "СУЛЬФАМАТНОЕ НИКЕЛЕВОЕ ПОКРЫТИЕ"),
    ("ZINC-NICKEL DEPOSIT OVERLAP", "ПЕРЕКРЫТИЕ ЦИНК-НИКЕЛЕВОГО ПОКРЫТИЯ"),
    ("PAINT DEPOSIT OVERLAP", "ПЕРЕКРЫТИЕ ЛАКОКРАСОЧНОГО ПОКРЫТИЯ"),
    ("NO PAINT", "БЕЗ КРАСКИ"),
    ("NO PLATING", "БЕЗ ПОКРЫТИЯ"),
    ("NO BARE METAL PERMISSIBLE", "ОГОЛЁННЫЙ МЕТАЛЛ НЕ ДОПУСКАЕТСЯ"),
    ("DO NOT CADMIUM PLATE", "КАДМИЕВОЕ ПОКРЫТИЕ НЕ НАНОСИТЬ"),
    ("NO PRIMER PAINT TO BE VISIBLE AFTER SEALANT APPLICATION", "ПОСЛЕ НАНЕСЕНИЯ ГЕРМЕТИКА ГРУНТОВОЧНАЯ КРАСКА НЕ ДОЛЖНА БЫТЬ ВИДНА"),
    ("PAINT MUST OVERLAP CADMIUM PLATE", "КРАСКА ДОЛЖНА ПЕРЕКРЫВАТЬ КАДМИЕВОЕ ПОКРЫТИЕ"),
    ("CADMIUM PLATE MUST OVERLAP CHROMIUM PLATE", "КАДМИЕВОЕ ПОКРЫТИЕ ДОЛЖНО ПЕРЕКРЫВАТЬ ХРОМОВОЕ ПОКРЫТИЕ"),
    ("CADMIUM PLATE AND PAINT TO OVERLAP", "КАДМИЕВОЕ ПОКРЫТИЕ И КРАСКА ДОЛЖНЫ ПЕРЕКРЫВАТЬСЯ"),
    ("CADMIUM PLATE AND PRIMER PAINT", "КАДМИЕВОЕ ПОКРЫТИЕ И ГРУНТОВОЧНАЯ КРАСКА"),
    ("CHROMIUM PLATE MAY TERMINATE ANYWHERE ON THIS RADIUS", "ХРОМОВОЕ ПОКРЫТИЕ МОЖЕТ ЗАКАНЧИВАТЬСЯ В ЛЮБОЙ ТОЧКЕ ЭТОГО РАДИУСА"),
    ("CHROMIUM PLATE TERMINATION", "ГРАНИЦА ХРОМОВОГО ПОКРЫТИЯ"),
    ("CHROMIUM PLATE TEMINATION", "ГРАНИЦА ХРОМОВОГО ПОКРЫТИЯ"),
    ("CHROMIUM PLATE MUST TERMINATE WITHIN THIS BAND", "ХРОМОВОЕ ПОКРЫТИЕ ДОЛЖНО ЗАКАНЧИВАТЬСЯ ВНУТРИ ЭТОЙ ПОЛОСЫ"),
    ("CHROMIUM PLATE MUST TERMINATE WITHIN THIS LENGTH", "ХРОМОВОЕ ПОКРЫТИЕ ДОЛЖНО ЗАКАНЧИВАТЬСЯ ВНУТРИ ЭТОЙ ДЛИНЫ"),
    ("CHROMIUM PLATE MUST TERMINATE IN THIS LENGTH", "ХРОМОВОЕ ПОКРЫТИЕ ДОЛЖНО ЗАКАНЧИВАТЬСЯ В ЭТОЙ ДЛИНЕ"),
    ("CHROMIUM PLATE MUST STOP IN THIS LENGTH", "ХРОМОВОЕ ПОКРЫТИЕ ДОЛЖНО ЗАКАНЧИВАТЬСЯ В ЭТОЙ ДЛИНЕ"),
    ("CHROMIUM PLATE CAN STOP ANYWHERE ON THE CHAMFER", "ХРОМОВОЕ ПОКРЫТИЕ МОЖЕТ ЗАКАНЧИВАТЬСЯ В ЛЮБОЙ ТОЧКЕ ФАСКИ"),
    ("CHROMIUM PLATE RUNOUT", "СХОД ХРОМОВОГО ПОКРЫТИЯ"),
    ("RUN OUT BAND", "ПОЛОСА СХОДА"),
    ("RUNOUT", "СХОД"),
    ("FADE OUT", "ПЛАВНЫЙ СХОД"),
    ("AREA OF CHROMIUM PLATE", "ЗОНА ХРОМОВОГО ПОКРЫТИЯ"),
    ("LENGTH OF CHROMIUM PLATE", "ДЛИНА ХРОМОВОГО ПОКРЫТИЯ"),
    ("WAVY OR IRREGULAR LINE PERMISSIBLE", "ВОЛНИСТАЯ ИЛИ НЕРОВНАЯ ЛИНИЯ ДОПУСКАЕТСЯ"),
    ("WAVY OR IRREGULAR LINE IS PERMISSIBLE", "ВОЛНИСТАЯ ИЛИ НЕРОВНАЯ ЛИНИЯ ДОПУСКАЕТСЯ"),
    ("WAVY OR IRREGULAR LINE IS PERMITTED", "ВОЛНИСТАЯ ИЛИ НЕРОВНАЯ ЛИНИЯ ДОПУСКАЕТСЯ"),
    ("IRREGULAR LINE IS PERMITTED", "НЕРОВНАЯ ЛИНИЯ ДОПУСКАЕТСЯ"),
    ("AN IRREGULAR LINE IS PERMITTED", "НЕРОВНАЯ ЛИНИЯ ДОПУСКАЕТСЯ"),
    ("THICKNESS OF CHROMIUM PLATE ABOVE OUTER SURFACE OF FLANGE", "ТОЛЩИНА ХРОМОВОГО ПОКРЫТИЯ НАД НАРУЖНОЙ ПОВЕРХНОСТЬЮ ФЛАНЦА"),
    ("AFTER CHROMIUM PLATE IS APPLIED AND BEFORE GRINDING", "ПОСЛЕ НАНЕСЕНИЯ ХРОМОВОГО ПОКРЫТИЯ И ДО ШЛИФОВАНИЯ"),
    ("AFTER CHROMIUM PLATE IS APPLIED", "ПОСЛЕ НАНЕСЕНИЯ ХРОМОВОГО ПОКРЫТИЯ"),
    ("AFTER GRINDING CHROMIUM PLATE", "ПОСЛЕ ШЛИФОВАНИЯ ХРОМОВОГО ПОКРЫТИЯ"),
    ("AFTER GRINDING", "ПОСЛЕ ШЛИФОВАНИЯ"),
    ("BEFORE CHROMIUM PLATE IS APPLIED", "ДО НАНЕСЕНИЯ ХРОМОВОГО ПОКРЫТИЯ"),
    ("BEFORE CHROMIUM PLATE", "ДО НАНЕСЕНИЯ ХРОМОВОГО ПОКРЫТИЯ"),
    ("DIAMETER BEFORE CHROMIUM PLATE", "ДИАМЕТР ДО НАНЕСЕНИЯ ХРОМОВОГО ПОКРЫТИЯ"),
    ("DIAMETER A BEFORE CHROMIUM PLATE", "ДИАМЕТР A ДО НАНЕСЕНИЯ ХРОМОВОГО ПОКРЫТИЯ"),
    ("DIA. AFTER GRINDING CHROME", "ДИАМ. ПОСЛЕ ШЛИФОВАНИЯ ХРОМА"),
    ("DIA. AFTER CHROME PLATING", "ДИАМ. ПОСЛЕ ХРОМИРОВАНИЯ"),
    ("DIAMETER AFTER CHROMIUM PLATE", "ДИАМЕТР ПОСЛЕ ХРОМОВОГО ПОКРЫТИЯ"),
    ("DIAMETER A AFTER GRINDING", "ДИАМЕТР A ПОСЛЕ ШЛИФОВАНИЯ"),
    ("MACHINE THIS FACE", "ОБРАБОТАТЬ ЭТУ ПОВЕРХНОСТЬ"),
    ("DEBURR THE SHARP EDGES", "СНЯТЬ ЗАУСЕНЦЫ С ОСТРЫХ КРОМОК"),
    ("REMOVE THE SHARP EDGES OF", "СНЯТЬ ОСТРЫЕ КРОМКИ"),
    ("REMOVE EDGE", "СНЯТЬ КРОМКУ"),
    ("EDGE BLENDED WITH A SMOOTH TRANSITION", "КРОМКУ ПЛАВНО СОПРЯЧЬ"),
    ("EDGE SMOOTHED OUT", "КРОМКУ СГЛАДИТЬ"),
    ("EDGES SMOOTHED OUT", "КРОМКИ СГЛАДИТЬ"),
    ("BLEND SMOOTHLY TO EXISTING RADIUS", "ПЛАВНО СОПРЯЧЬ С СУЩЕСТВУЮЩИМ РАДИУСОМ"),
    ("SMOOTH BLEND TO LARGER RADIUS", "ПЛАВНО СОПРЯЧЬ С БОЛЬШИМ РАДИУСОМ"),
    ("TO RADIUS INTERSECTION POINT", "ДО ТОЧКИ ПЕРЕСЕЧЕНИЯ РАДИУСОВ"),
    ("LUG WIDTH", "ШИРИНА ПРОУШИНЫ"),
    ("MINIMUM WALL THICKNESS", "МИНИМАЛЬНАЯ ТОЛЩИНА СТЕНКИ"),
    ("WALL THICKNESS", "ТОЛЩИНА СТЕНКИ"),
    ("BETWEEN SPOTFACES", "МЕЖДУ ПОДРЕЗКАМИ ПЛОЩАДКИ"),
    ("BEYOND THIS LINE", "ЗА ЭТОЙ ЛИНИЕЙ"),
    ("INTENTIONALLY BLANK", "УМЫШЛЕННО ОСТАВЛЕНО ПУСТЫМ"),
    ("OR BETTER UNLESS GIVEN DIFFERENTLY.", "ИЛИ ЛУЧШЕ, ЕСЛИ НЕ УКАЗАНО ИНОЕ."),
    ("OR BETTER UNLESS GIVEN DIFFERENTLY", "ИЛИ ЛУЧШЕ, ЕСЛИ НЕ УКАЗАНО ИНОЕ"),
    ("NOTE:", "ПРИМЕЧАНИЕ:"),
    ("NOTE:-", "ПРИМЕЧАНИЕ: "),
    ("IF THE BASE METAL IS DAMAGED", "ЕСЛИ ОСНОВНОЙ МЕТАЛЛ ПОВРЕЖДЁН"),
    ("IF THE BASE METAL IS NOT DAMAGED", "ЕСЛИ ОСНОВНОЙ МЕТАЛЛ НЕ ПОВРЕЖДЁН"),
    ("WITH LUBRICATION ADAPTOR", "СО СМАЗОЧНЫМ АДАПТЕРОМ"),
    ("WITHOUT LUBRICATION ADAPTOR", "БЕЗ СМАЗОЧНОГО АДАПТЕРА"),
    ("WITH BEARING", "С ПОДШИПНИКОМ"),
    ("WITHOUT BEARING", "БЕЗ ПОДШИПНИКА"),
    ("WITH BUSHES", "С ВТУЛКАМИ"),
    ("WITHOUT BUSHES", "БЕЗ ВТУЛОК"),
    ("WITH BUSH", "С ВТУЛКОЙ"),
    ("WITHOUT BUSH", "БЕЗ ВТУЛКИ"),
    ("WITHOUT REPAIR BUSHES", "БЕЗ РЕМОНТНЫХ ВТУЛОК"),
    ("WITHOUT REPAIR BEARING", "БЕЗ РЕМОНТНОГО ПОДШИПНИКА"),
    ("INTERNAL SERMETEL LIMIT", "ВНУТРЕННИЙ ПРЕДЕЛ SERMETEL"),
    ("EXTERNAL SERMETEL LIMIT", "НАРУЖНЫЙ ПРЕДЕЛ SERMETEL"),
    ("INTERNAL THICK ZINC-NICKEL PLATING LIMIT", "ВНУТРЕННИЙ ПРЕДЕЛ ТОЛСТОГО ЦИНК-НИКЕЛЕВОГО ПОКРЫТИЯ"),
    ("EXTERNAL THICK ZINC-NICKEL PLATING LIMIT", "НАРУЖНЫЙ ПРЕДЕЛ ТОЛСТОГО ЦИНК-НИКЕЛЕВОГО ПОКРЫТИЯ"),
    ("SERMETEL W TO IFC 40-860-03MD", "SERMETEL W ПО IFC 40-860-03MD"),
    ("SERMETEL W TOIFC 40-860-03MD", "SERMETEL W ПО IFC 40-860-03MD"),
    ("SERMETEL WTO IFC 40-860-03MD", "SERMETEL W ПО IFC 40-860-03MD"),
)

EXACT_TEXT_MAP = {
    "SAFRAN LANDING SYSTEMS UK LTD CAGE: K0654": "SAFRAN LANDING SYSTEMS UK Ltd КОД CAGE: K0654",
    "PART NO. 201587001 AND 201587002 COMPONENT MAINTENANCE MANUAL MAIN LANDING GEAR LEG": (
        "№ ДЕТАЛЕЙ 201587001 И 201587002 РУКОВОДСТВО ПО ТЕХНИЧЕСКОМУ ОБСЛУЖИВАНИЮ КОМПОНЕНТОВ "
        "СТОЙКА ОСНОВНОГО ШАССИ"
    ),
    "FIGURE DELETED FIGURE 609": "Рисунок 609 удалён",
    "FIGURE DELETED FIGURE 611": "Рисунок 611 удалён",
    "APPROVED REPAIRS - KEY DIAGRAM FIGURE 649": "Утверждённые ремонты - Ключевая схема Рисунок 649",
    (
        "INSTALL THE APPLICABLE LUBRICATION ADAPTOR: REFER TO TABLE 601. "
        "CADMIUM PLATE OPTIONAL AND NO PAINT. REPAIR TO MAIN FITTING FIGURE 601"
    ): (
        "УСТАНОВИТЬ СООТВЕТСТВУЮЩИЙ СМАЗОЧНЫЙ АДАПТЕР: СМ. ТАБЛИЦУ 601. "
        "КАДМИЕВОЕ ПОКРЫТИЕ ДОПУСКАЕТСЯ, КРАСКУ НЕ НАНОСИТЬ. "
        "Ремонт корпуса стойки Рисунок 601"
    ),
    "NOTE: THE SURFACE FINISH MUST BE": "ПРИМЕЧАНИЕ: ШЕРОХОВАТОСТЬ ПОВЕРХНОСТИ ДОЛЖНА БЫТЬ",
    (
        "APPLY FILLET SEALANT: REFER TO PCS-7200. MAKE SURE THAT THE SEALANT COMPLETELY COVERS EXPOSED "
        "PRIMER PAINT."
    ): "НАНЕСТИ ВАЛИКОВЫЙ ГЕРМЕТИК: СМ. PCS-7200. УБЕДИТЬСЯ, ЧТО ГЕРМЕТИК ПОЛНОСТЬЮ ПОКРЫВАЕТ ОГОЛЁННУЮ ГРУНТОВОЧНУЮ КРАСКУ.",
    (
        "APPLY LOCTITE GRADE 270 TO ADAPTOR INTERFACE WITH MAIN FITTING: REFER TO PCS-5303."
    ): "НАНЕСТИ LOCTITE GRADE 270 НА ПОВЕРХНОСТЬ СОПРЯЖЕНИЯ АДАПТЕРА С КОРПУСОМ СТОЙКИ: СМ. PCS-5303.",
    (
        "APPLY LOCTITE GRADE 270 TO ADAPTOR INTERFACE WITH MAIN FITTING: REFER TO PCS-5303. "
        "INSTALL THE APPLICABLE LUBRICATION ADAPTOR: REFER TO TABLE 601."
    ): (
        "НАНЕСТИ LOCTITE GRADE 270 НА ПОВЕРХНОСТЬ СОПРЯЖЕНИЯ АДАПТЕРА С КОРПУСОМ СТОЙКИ: СМ. PCS-5303. "
        "УСТАНОВИТЬ СООТВЕТСТВУЮЩИЙ СМАЗОЧНЫЙ АДАПТЕР: СМ. ТАБЛИЦУ 601."
    ),
    (
        "APPLY CADMIUM PLATE ALL OVER: REFER TO PCS-2101. THE CADMIUM PLATE THICKNESS MUST BE BETWEEN "
        "0,010 TO 0,015MM (0.0004 TO 0.0006IN)."
    ): (
        "НАНЕСТИ КАДМИЕВОЕ ПОКРЫТИЕ ПО ВСЕЙ ПОВЕРХНОСТИ: СМ. PCS-2101. "
        "ТОЛЩИНА КАДМИЕВОГО ПОКРЫТИЯ ДОЛЖНА БЫТЬ 0,010 ДО 0,015 ММ (0,0004 ДО 0,0006 ДЮЙМА)."
    ),
    (
        "APPLY CADMIUM PLATE ALL OVER: REFER TO PCS-2101. THE PLATING THICKNESS MUST BE BETWEEN "
        "0,010 TO 0,015MM (0.0004 TO 0.0006IN). DEBURR THE SHARP EDGES WITH 0,130 TO 0,380MM "
        "(0.0051 TO 0.0150IN) RAD. UNLESS GIVEN DIFFERENTLY."
    ): (
        "НАНЕСТИ КАДМИЕВОЕ ПОКРЫТИЕ ПО ВСЕЙ ПОВЕРХНОСТИ: СМ. PCS-2101. ТОЛЩИНА ПОКРЫТИЯ ДОЛЖНА БЫТЬ "
        "0,010 ДО 0,015 ММ (0,0004 ДО 0,0006 ДЮЙМА). СНЯТЬ ЗАУСЕНЦЫ С ОСТРЫХ КРОМОК РАДИУСОМ "
        "0,130 ДО 0,380 ММ (0,0051 ДО 0,0150 ДЮЙМА), ЕСЛИ НЕ УКАЗАНО ИНОЕ."
    ),
    "PART NO. 201587001 AND 201587002 COMPONENT MAINTENANCE MANUAL MAIN LANDING GEAR LEG15 DEGREES": (
        "№ ДЕТАЛЕЙ 201587001 И 201587002 РУКОВОДСТВО ПО ТЕХНИЧЕСКОМУ ОБСЛУЖИВАНИЮ КОМПОНЕНТОВ "
        "СТОЙКА ОСНОВНОГО ШАССИ"
    ),
}

RAW_CLEANUPS: tuple[tuple[str, str], ...] = (
    ("LtdCAGE", "Ltd CAGE"),
    ("Protective TreatmentFigure", "Protective Treatment Figure"),
    ("NOTE:THE", "NOTE: THE"),
    ("MUST BE", " MUST BE"),
    ("PAINT.Repair", "PAINT. Repair"),
    ("NO PAINTIN", "NO PAINT IN"),
    ("THISSURFACE", "THIS SURFACE"),
    ("TOPCS", "TO PCS"),
    ("ONLYA321", "ONLY A321"),
    ("GEARLEG", "GEAR LEG"),
    ("LEG15", "LEG 15"),
    ("DIMENSIONDMACHINE", "DIMENSION D MACHINE"),
    ("FITTINGREFER", "FITTING REFER"),
    ("TUBEREFER", "TUBE REFER"),
    ("BLOCKREFER", "BLOCK REFER"),
    ("BRACKET REFER", "BRACKET REFER"),
    ("BORESFOR", "BORES FOR"),
    ("HOLESFOR", "HOLES FOR"),
    ("LUGFOR", "LUG FOR"),
    ("TYPICAL2", "TYPICAL 2"),
    ("TYPICAL12", "TYPICAL 12"),
    ("WTYPICAL", "W TYPICAL"),
    ("CTYPICAL", "C TYPICAL"),
    ("JTYPICAL", "J TYPICAL"),
    ("FTYPICAL", "F TYPICAL"),
    ("HTYPICAL", "H TYPICAL"),
    ("ATWO", "A TWO"),
    ("A SPOTFACEZ", "A SPOTFACE Z"),
    ("A SPOTFACECSPOTFACE", "A SPOTFACE C SPOTFACE"),
    ("DIAMETERA", "DIAMETER A"),
    ("DIAMETERB", "DIAMETER B"),
    ("FACEQ", "FACE Q"),
    ("IFC30", "IFC 30"),
    ("TOIFC", "TO IFC"),
    ("TO M-DLPS", "TO M-DLPS"),
    ("PCS 7200", "PCS-7200"),
    ("PCS 7304", "PCS-7304"),
    ("PCS 5303", "PCS-5303"),
    ("PCS-72004", "PCS-7200 4"),
    ("PCS-7200SECTION", "PCS-7200 SECTION"),
    ("ELECTRICALLYCONDUCTING", "ELECTRICALLY CONDUCTING"),
    ("SEALANTIN", "SEALANT IN"),
    ("PLACESAPPLY", "PLACES APPLY"),
    ("HONENO", "HONE NO"),
    ("GRINDINGPART", "GRINDING PART"),
    ("PAINTDETAIL", "PAINT DETAIL"),
    ("BLENDTO", "BLEND TO"),
    ("TEMINATION", "TERMINATION"),
    ("PLATETERMINATION", "PLATE TERMINATION"),
    ("PLATEDEPOSIT", "PLATE DEPOSIT"),
    ("PLATEDIA", "PLATE DIA."),
    ("RUNOUTSECTION", "RUNOUT SECTION"),
    ("LIMITSECTION", "LIMIT SECTION"),
    ("EXTENTOF", "EXTENT OF"),
    ("DIADETAIL", "DIA. DETAIL"),
    ("RADIUSREF.", "RADIUS REF."),
    ("WITHOUT LUBRICATION ADAPTOR)A321", "WITHOUT LUBRICATION ADAPTOR) A321"),
    ("IS DAMAGEDA321", "IS DAMAGED A321"),
    ("NOT DAMAGEDA321", "NOT DAMAGED A321"),
    ("VIEW ABracket", "VIEW A Bracket"),
    ("SECTION Z-ZWITH", "SECTION Z-Z WITH"),
    ("SECTION Y-YWITH", "SECTION Y-Y WITH"),
    ("SECTION U-UCHROMIUM", "SECTION U-U CHROMIUM"),
    ("SECTION V-VCHROMIUM", "SECTION V-V CHROMIUM"),
    ("A0,05", "A 0,05"),
)

PAGE_RE = re.compile(
    r"^Page\s+(?P<page>\d+)(?:\s*(?P<month>[A-Za-z]{3})\s*(?P<day>\d{1,2})/(?P<year>\d{4}))?$",
    re.IGNORECASE,
)
SECTION_DATE_RE = re.compile(
    r"^(?P<section>\d{2}-\d{2}-\d{2})\s*(?P<month>[A-Za-z]{3})\s*(?P<day>\d{1,2})/(?P<year>\d{4})$",
    re.IGNORECASE,
)
REPAIR_PAGE_RE = re.compile(
    r"^Repair No\.\s*(?P<num>[\d\-]+)\s*Page\s*(?P<page>\d+)\s*(?P<month>[A-Za-z]{3})\s*(?P<day>\d{1,2})/(?P<year>\d{4})$",
    re.IGNORECASE,
)


def parse_glossary_pairs(glossary_text: str | None) -> list[tuple[str, str]]:
    if not glossary_text:
        return []

    pairs: list[tuple[str, str]] = []
    for raw_line in glossary_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cols = [col.strip() for col in line.strip("|").split("|")]
        if len(cols) < 2:
            continue
        source, target = cols[0], cols[1]
        if not source or not target or source.lower() in {"english", "standard"} or set(source) == {"-"}:
            continue
        pairs.append((source, target))

    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return pairs


def build_phrase_patterns(glossary_text: str | None) -> list[tuple[re.Pattern[str], str]]:
    merged: dict[str, str] = {}
    for source, target in list(CUSTOM_PHRASES) + parse_glossary_pairs(glossary_text):
        key = source.strip().lower()
        if key and key not in merged:
            merged[key] = target.strip()

    patterns: list[tuple[re.Pattern[str], str]] = []
    for source, target in sorted(merged.items(), key=lambda item: len(item[0]), reverse=True):
        escaped = re.escape(source)
        pattern = re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
        patterns.append((pattern, target))
    return patterns


def translate_component_expr(text: str, *, genitive: bool = False) -> str:
    out = text.strip()
    lowered = out.lower()
    for key in COMPONENT_KEYS:
        if lowered.startswith(key):
            form = COMPONENT_FORMS[key][1 if genitive else 0]
            out = form + out[len(key) :]
            break

    out = re.sub(r"\band\b", "и", out, flags=re.IGNORECASE)
    out = re.sub(r"\bor\b", "или", out, flags=re.IGNORECASE)
    out = re.sub(r"\bonly\b", "только", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def translate_sublabel(text: str) -> str:
    key = text.strip().lower()
    if key in CAPTION_SUBLABELS:
        return CAPTION_SUBLABELS[key]
    return text.strip()


def normalize_source(text: str) -> str:
    out = text.replace("\u00a0", " ").replace("\t", " ").replace("\r", " ").replace("\n", " ")
    out = out.replace("–", "-").replace("—", "-")
    out = re.sub(r"\s+", " ", out).strip()

    for source, target in RAW_CLEANUPS:
        out = out.replace(source, target)

    out = re.sub(r"(?<=\))(?=[A-Za-z0-9])", " ", out)
    out = re.sub(r"(?<=[A-Za-z])(?=\d{1,3}[,.])", " ", out)
    out = re.sub(r"(?<=\d)(?=[A-Z]{2,})", " ", out)
    out = re.sub(r"(?<=\d)(?=[A-Z][a-z])", " ", out)
    out = re.sub(r"(?<=THICKNESS)(?=\d)", " ", out)
    out = re.sub(r"(?<=PLACES)(?=\d)", " ", out)
    out = re.sub(r"(?<=RAD\.)(?=\d)", " ", out)
    out = re.sub(r"(?<=MAX\.)(?=\d)", " ", out)
    out = re.sub(r"(?<=DIA\.)(?=\d)", " ", out)
    out = re.sub(r"(?<=TYPICAL)(?=\d)", " ", out)
    out = re.sub(r"(?<=FIGURE)(?=\d)", " ", out)
    out = re.sub(r"(?<=[A-Za-z])(?=\()", " ", out)
    out = re.sub(r"(?<=\))(?=[A-Za-z])", " ", out)
    out = re.sub(r"(?<=[.])(?=[A-Za-z])", " ", out)
    out = re.sub(
        r"(?<=[A-Z])(?=FOR\b|ONLY\b|TYPICAL\b|INCLUDING\b|BARREL\b|KNUCKLE\b|LUG\b|LOWER\b|UPPER\b|PINTLE\b|DRAG\b|RETRACTION\b|BRAKE\b|CHANGE\b|TORQUE\b|TRANSFER\b|VIEW\b|SPOTFACE\b|SECTION\b|DETAIL\b)",
        " ",
        out,
    )
    out = re.sub(r"(?<=\b[ABCDEFGHIJKLMNOPQRSTUVWXYZ])(?=SPOTFACE|VIEW|SECTION|DETAIL|DIA|RAD)", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"(Figure\s+\d+)\s*32-12-22$", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"(\bRepair.*Figure\s+\d+)\s*32-12-22$", r"\1", out, flags=re.IGNORECASE)
    return out


def normalize_exact_key(text: str) -> str:
    out = normalize_source(text)
    out = re.sub(r"\s+", " ", out).strip().upper()
    return out


def format_page_date(page: str, month: str | None, day: str | None, year: str | None) -> str:
    if not month or not day or not year:
        return f"Страница {page}"
    month_ru = MONTHS.get(month.lower(), month.lower())
    return f"Страница {page} {int(day)} {month_ru}/{year}"


def translate_page_or_section(text: str) -> str | None:
    m = PAGE_RE.match(text)
    if m:
        return format_page_date(m.group("page"), m.group("month"), m.group("day"), m.group("year"))

    m = SECTION_DATE_RE.match(text)
    if m:
        month_ru = MONTHS.get(m.group("month").lower(), m.group("month").lower())
        return f"{m.group('section')} {int(m.group('day'))} {month_ru}/{m.group('year')}"

    m = REPAIR_PAGE_RE.match(text)
    if m:
        month_ru = MONTHS.get(m.group("month").lower(), m.group("month").lower())
        return f"РЕМОНТ № {m.group('num')}\nСтраница {m.group('page')}\n{int(m.group('day'))} {month_ru}/{m.group('year')}"
    return None


def translate_caption(text: str) -> str | None:
    protected = re.match(
        r"^(?P<label>.+?)\s+-\s+Protective Treatment\s*Figure\s*(?P<fig>\d+)(?:\s+-\s+Sheet\s*(?P<sheet>\d+))?$",
        text,
        flags=re.IGNORECASE,
    )
    if protected:
        label = translate_component_expr(protected.group("label"))
        out = f"{label} - Защитная обработка Рисунок {protected.group('fig')}"
        if protected.group("sheet"):
            out += f" - Лист {protected.group('sheet')}"
        return out

    repairs = re.match(
        r"^(?P<label>.+?)\s+Repairs\s*-\s*Key Diagram\s*Figure\s*(?P<fig>\d+)$",
        text,
        flags=re.IGNORECASE,
    )
    if repairs:
        label = translate_component_expr(repairs.group("label"), genitive=True)
        return f"Ремонты {label} - Ключевая схема Рисунок {repairs.group('fig')}"

    repair_to_sub = re.match(
        r"^Repair to (?P<label>.+?)\s+-\s+(?P<sub>.+?)\s*Figure\s*(?P<fig>\d+)(?:\s+-\s+Sheet\s*(?P<sheet>\d+))?$",
        text,
        flags=re.IGNORECASE,
    )
    if repair_to_sub:
        label = translate_component_expr(repair_to_sub.group("label"), genitive=True)
        sub = translate_sublabel(repair_to_sub.group("sub"))
        out = f"Ремонт {label} - {sub} Рисунок {repair_to_sub.group('fig')}"
        if repair_to_sub.group("sheet"):
            out += f" - Лист {repair_to_sub.group('sheet')}"
        return out

    generic = re.match(
        r"^(?P<label>.+?)\s+-\s+(?P<sub>.+?)\s*Figure\s*(?P<fig>\d+)(?:\s+-\s+Sheet\s*(?P<sheet>\d+))?$",
        text,
        flags=re.IGNORECASE,
    )
    if generic and "protective treatment" not in generic.group("sub").lower():
        label = translate_component_expr(generic.group("label"))
        sub = translate_sublabel(generic.group("sub"))
        out = f"{label} - {sub} Рисунок {generic.group('fig')}"
        if generic.group("sheet"):
            out += f" - Лист {generic.group('sheet')}"
        return out

    repair_to = re.match(
        r"^Repair to (?P<label>.+?)\s*Figure\s*(?P<fig>\d+)(?:\s*-\s*Sheet\s*(?P<sheet>\d+))?$",
        text,
        flags=re.IGNORECASE,
    )
    if repair_to:
        label = translate_component_expr(repair_to.group("label"), genitive=True)
        out = f"Ремонт {label} Рисунок {repair_to.group('fig')}"
        if repair_to.group("sheet"):
            out += f" - Лист {repair_to.group('sheet')}"
        return out

    liner = re.match(
        r"^(Lower Bearing Subassembly)\s+(Machining and Liner Installation)\s*Figure\s*(?P<fig>\d+)$",
        text,
        flags=re.IGNORECASE,
    )
    if liner:
        return f"Сборка нижнего подшипника - Механическая обработка и установка втулки Рисунок {liner.group('fig')}"

    return None


def apply_phrase_patterns(text: str, patterns: list[tuple[re.Pattern[str], str]]) -> str:
    out = text
    if not re.search(r"[A-Za-z]", out):
        return out
    for pattern, replacement in patterns:
        out = pattern.sub(replacement, out)
    return out


def translate_special_compact(text: str) -> str | None:
    normalized = text.strip()
    normalized_upper = normalized.upper()

    exact_compact = {
        "BINTERNALLYWW": "B\nВНУТР.",
        "CINTERNALLY": "C\nВНУТР.",
        "VIEW ON ARROWZ": "ВИД ПО\nСТРЕЛКЕ Z",
        "VIEW ON ARROW Z": "ВИД ПО\nСТРЕЛКЕ Z",
        "PART VIEW ON ARROW Z": "ЧАСТИЧНЫЙ ВИД\nПО СТРЕЛКЕ Z",
        "C (2 DIAMETERS)": "C\n(2 ДИАМ.)",
        "(INNER DIAMETER)": "(ВНУТР. ДИАМ.)",
        "B (2 HOLES)": "B (2 ОТВ.)",
        "B (4 HOLES)": "B (4 ОТВ.)",
        "(BOTH HOLES)": "(ОБА ОТВ.)",
        "NO CADMIUM PLATE NO PAINT(2 PLACES)": "БЕЗ КАДМ. ПОКР.\nБЕЗ КРАСКИ\n(2 МЕСТА)",
        "NO CADMIUM PLATE NO PAINT (2 PLACES)": "БЕЗ КАДМ. ПОКР.\nБЕЗ КРАСКИ\n(2 МЕСТА)",
        "NO CADMIUM PLATE NO PAINTIN HOLES (Qty 6)": "БЕЗ КАДМ. ПОКР.\nБЕЗ КРАСКИ\nВ ОТВ.\n(КОЛ-ВО 6)",
        "NO CADMIUM PLATE NO PAINT": "БЕЗ КАДМ. ПОКР.\nБЕЗ КРАСКИ",
        "NO CADMIUM PLATE OR PAINT BEYOND THIS LINE": "БЕЗ КАДМ. ПОКР.\nИЛИ КРАСКИ\nЗА ЭТОЙ ЛИНИЕЙ",
        "NO PAINT": "БЕЗ КРАСКИ",
        "VNO PAINT": "V\nБЕЗ КРАСКИ",
        "FOR ALL SECTION VIEWS SEE SHEET 2": "ДЛЯ ВСЕХ\nВИДОВ В СЕЧ.\nСМ. ЛИСТ 2",
        "CHOLEASPOTFACE": "C ОТВ.\nA ПОДРЕЗКА",
        "ASPOTFACE": "A ПОДРЕЗКА",
        "ASPOTFACEZ": "A ПОДРЕЗКА\nZ",
        "ASPOTFACECSPOTFACE": "A ПОДРЕЗКА\nC ПОДРЕЗКА",
        "CSPOTFACE": "C ПОДРЕЗКА",
        "CSPOTFACEB": "C ПОДРЕЗКА\nB",
        "DETAIL W14 PLACES": "ДЕТАЛЬ W\n14 МЕСТ",
        "4 PLACES A": "4 МЕСТА\nA",
        "A4 PLACES": "A\n4 МЕСТА",
        "A2 PLACES": "A\n2 МЕСТА",
        "A SPOTFACE": "A ПОДРЕЗКА",
        "A SPOTFACECSPOTFACE": "A ПОДРЕЗКА\nC ПОДРЕЗКА",
        "DETAIL Y": "ДЕТАЛЬ Y",
        "DETAIL Y (REFER TO FIGURE 601)": "ДЕТАЛЬ Y\n(СМ. РИС. 601)",
        "FINISH PAINT DETAIL Y": "ФИНИШНАЯ\nКРАСКА\nДЕТАЛЬ Y",
        "REPAIR BEARING 450237824": "РЕМОНТНЫЙ\nПОДШИПНИК\n450237824",
        "OVERSIZE BUSH 450237810": "РЕМОНТНАЯ\nВТУЛКА\n450237810",
        "INSTALL BUSH FLUSH TO BELOW SURFACE": "УСТАНОВИТЬ ВТУЛКУ\nЗАПОДЛИЦО НИЖЕ\nПОВЕРХНОСТИ",
        "SPHERICAL RAD. 4 PLACES": "СФЕРИЧ. РАД.\n4 МЕСТА",
        "MINIMUM WALL THICKNESS 4 PLACES": "МИН. ТОЛЩИНА\nСТЕНКИ\n4 МЕСТА",
        "4 PLACES": "4 МЕСТА",
        "(4 PLACES)": "(4 МЕСТА)",
        "A (4 PLACES)": "A\n(4 МЕСТА)",
        "DIA A (4 PLACES)": "ДИАМ. A\n(4 МЕСТА)",
        "B HOLE (4 PLACES)": "ОТВ. B\n(4 МЕСТА)",
        "SECTION Z-Z": "СЕЧЕНИЕ Z-Z",
        "PART SECTION Z-Z": "ЧАСТИЧНОЕ\nСЕЧЕНИЕ Z-Z",
        "SECTION Z-Z WITHOUT BEARING": "СЕЧЕНИЕ Z-Z\nБЕЗ ПОДШИПНИКА",
        "SECTION Z-Z WITH BEARING": "СЕЧЕНИЕ Z-Z\nС ПОДШИПНИКОМ",
        "SECTION Z-Z (WITHOUT BUSHES)": "СЕЧЕНИЕ Z-Z\n(БЕЗ ВТУЛОК)",
        "SECTION Z-Z (WITH BUSHES)": "СЕЧЕНИЕ Z-Z\n(С ВТУЛКАМИ)",
        "SECTION Z-Z (WITH BUSHES)": "СЕЧЕНИЕ Z-Z\n(С ВТУЛКАМИ)",
        "SECTION Z-Z (WITHOUT BUSHES)": "СЕЧЕНИЕ Z-Z\n(БЕЗ ВТУЛОК)",
        "SECTION Z-Z (WITHOUT BUSH)": "СЕЧЕНИЕ Z-Z\n(БЕЗ ВТУЛКИ)",
        "SECTION Z-Z (WITH BUSH)": "СЕЧЕНИЕ Z-Z\n(С ВТУЛКОЙ)",
        "SECTION Z-Z (WITH LUBRICATION ADAPTOR)": "СЕЧЕНИЕ Z-Z\n(СО СМАЗОЧНЫМ\nАДАПТЕРОМ)",
        "SECTION Z-Z (WITHOUT LUBRICATION ADAPTOR)": "СЕЧЕНИЕ Z-Z\n(БЕЗ СМАЗОЧНОГО\nАДАПТЕРА)",
        "SECTION Z-Z (WITH OVERSIZE REAR SPHERICAL BEARING) 90 DEGREES ROTATED": "СЕЧЕНИЕ Z-Z\n(С РЕМОНТНЫМ ЗАДНИМ\nСФЕРИЧЕСКИМ ПОДШИПНИКОМ)\nПОВЕРНУТО НА 90°",
        "SECTION Z-Z (WITHOUT OVERSIZE REAR SPHERICAL BEARING) 90 DEGREES ROTATED 1,6(63) OR BETTER UNLESS GIVEN DIFFERENTLY.": "СЕЧЕНИЕ Z-Z\n(БЕЗ РЕМОНТНОГО ЗАДНЕГО\nСФЕРИЧЕСКОГО ПОДШИПНИКА)\nПОВЕРНУТО НА 90°\n1,6(63) ИЛИ ЛУЧШЕ",
        "APPLY MOLYKOTE 111 TO THE BOLT SHANKS, THREADS, UNDERCUTS AND ALL INTERFACES BETWEEN MATING PARTS MUST BE COATED BEFORE ASSEMBLY: REFER TO PCS-7303. ALL CAVITIES AND VOIDS MUST BE FILLED TO PREVENT MOISTURE INGRESS. APPLY A FULL BEAD OF SEALANT, PR340-2 WITH A MAXIMUM HEIGHT OF 1,000MM (0.0394IN) ABOVE ADJOINING SURFACES: REFER TO PCS-7200. APPLY SEALANT: REFER TO PCS-7200 TYPE 2.": "НАНЕСТИ MOLYKOTE 111 НА СТЕРЖНИ,\nРЕЗЬБУ, ПОДРЕЗКИ И ВСЕ ПОВЕРХНОСТИ\nСОПРЯЖЕНИЯ ДЕТАЛЕЙ: СМ. PCS-7303.\nЗАПОЛНИТЬ ВСЕ ПОЛОСТИ И ПУСТОТЫ\nДЛЯ ПРЕДОТВРАЩЕНИЯ ПОПАДАНИЯ ВЛАГИ.\nНАНЕСТИ СПЛОШНОЙ ВАЛИК ГЕРМЕТИКА\nPR340-2 ВЫСОТОЙ НЕ БОЛЕЕ\n1,000 ММ (0,0394 ДЮЙМА)\nНАД ПРИЛЕГАЮЩИМИ ПОВЕРХНОСТЯМИ:\nСМ. PCS-7200.\nНАНЕСТИ ГЕРМЕТИК:\nСМ. PCS-7200 ТИП 2.",
        "REMOVE THE BREAK EDGES WITHIN 0,500 TO 2,000MM (0.0197 TO 0.0787IN) RAD. FOR THE MAIN FITTING (20-410B), (20-410C), (20-420B) AND (20-420C): THE MINIMUM WALL THICKNESS IS 15,382MM (0.6056IN). FOR THE MAIN FITTING (20-410D) AND (20-420D):THE MINIMUM WALL THICKNESS IS 15,582MM (0.6134IN). PRIMER PAINT ONLY: REFER TO PCS-2500. NO WITNESS OF TOP COAT PAINT PERMITTED ON THESE SURFACES.": "СКРУГЛИТЬ КРОМКИ РАДИУСОМ\n0,500 ДО 2,000 ММ\n(0,0197 ДО 0,0787 ДЮЙМА).\nДЛЯ КОРПУСА СТОЙКИ\n(20-410B), (20-410C), (20-420B)\nИ (20-420C) МИН. ТОЛЩИНА\nСТЕНКИ 15,382 ММ\n(0,6056 ДЮЙМА).\nДЛЯ КОРПУСА СТОЙКИ\n(20-410D) И (20-420D)\nМИН. ТОЛЩИНА СТЕНКИ\n15,582 ММ (0,6134 ДЮЙМА).\nТОЛЬКО ГРУНТОВОЧНАЯ КРАСКА:\nСМ. PCS-2500.\nСЛЕДЫ ФИНИШНОЙ КРАСКИ\nНА ЭТИХ ПОВЕРХНОСТЯХ\nНЕ ДОПУСКАЮТСЯ.",
        "18,80 TO 19,30MM (0.740 TO 0.760IN) RAD BLEND SMOOTHLY TO ADJACENT SURFACES": "18,80 ДО 19,30 ММ\n(0,740 ДО 0,760 ДЮЙМА)\nРАД.\nПЛАВНО СОПРЯЧЬ\nС ПРИЛЕГАЮЩИМИ\nПОВЕРХНОСТЯМИ",
        "14,00 TO 16,00MM (0.551 TO 0.630IN) APPLY ELECTRICALLY CONDUCTING MOLYKOTE 111 OR RUBBERISED SEALANT IN THE BORES TO IFC 30-145-03 MD OR PCS-7304 TYP. 2 PLACES 12,00 TO 14,00MM (0.472 TO 0.551IN) APPLY ELECTRICALLY CONDUCTING MOLYKOTE 111": "14,00 ДО 16,00 ММ\n(0,551 ДО 0,630 ДЮЙМА)\nНАНЕСТИ ЭЛЕКТРОПРОВОДЯЩИЙ\nMOLYKOTE 111 ИЛИ\nРЕЗИНОПОДОБНЫЙ ГЕРМЕТИК\nВ ОТВЕРСТИЯ ПО IFC 30-145-03MD\nИЛИ PCS-7304, ТИП. 2 МЕСТА\n12,00 ДО 14,00 ММ\n(0,472 ДО 0,551 ДЮЙМА)\nНАНЕСТИ ЭЛЕКТРОПРОВОДЯЩИЙ\nMOLYKOTE 111",
        "OR RUBBERISED SEALANT ON BOTH BUSHES TO IFC 30-145-03 MD OR PCS-7304 TYP. 2 PLACES APPLY SEALANT TO PCS-7200 WITHOUT OVERFLOW ON FACES OF THE BUSHES": "ИЛИ РЕЗИНОПОДОБНЫЙ\nГЕРМЕТИК НА ОБЕ ВТУЛКИ\nПО IFC 30-145-03MD\nИЛИ PCS-7304, ТИП. 2 МЕСТА\nНАНЕСТИ ГЕРМЕТИК\nПО PCS-7200\nБЕЗ ВЫТЕКАНИЯ\nНА ТОРЦЫ ВТУЛОК",
        "NITRIDING DEPTH 0,18 TO 0,23MM (0.007 TO 0.009IN), 0,02 TO 0,04MM (0.0008 TO 0.0016IN) REMOVAL OVER AREA SHOWN 750 HV MIN. NOTE: REPAIR BUSHES 450237351, 450237352, 450237353, 450237354 AND 450237355 ARE TO BE NITRIDED BEFORE DESPATCH TO OVERHAUL AGENCY. DIAMETER H AND CORNER RADIUS ARE TO BE MACHINED ON RECEIPT": "ГЛУБИНА НИТРИРОВАНИЯ\n0,18 ДО 0,23 ММ\n(0,007 ДО 0,009 ДЮЙМА)\nСНЯТИЕ 0,02 ДО 0,04 ММ\n(0,0008 ДО 0,0016 ДЮЙМА)\nНА ПОКАЗАННОМ УЧАСТКЕ\n750 HV МИН.\nПРИМЕЧАНИЕ:\nРЕМОНТНЫЕ ВТУЛКИ 450237351,\n450237352, 450237353,\n450237354 И 450237355\nДОЛЖНЫ БЫТЬ НИТРИРОВАНЫ\nПЕРЕД ОТПРАВКОЙ В РЕМОНТНОЕ\nПОДРАЗДЕЛЕНИЕ.\nДИАМЕТР H И УГЛОВОЙ\nРАДИУС ОБРАБОТАТЬ\nПРИ ПОЛУЧЕНИИ",
    }
    exact_compact_map = {key.upper(): value for key, value in exact_compact.items()}
    if normalized_upper in exact_compact_map:
        return exact_compact_map[normalized_upper]

    m = re.match(r"^(?P<label>[A-Z])\s*\(DIAMETER\)$", normalized, flags=re.IGNORECASE)
    if m:
        return f"{m.group('label').upper()} (ДИАМ.)"

    m = re.match(r"^(?P<label>[A-Z])\s*\(CHAMFER\)$", normalized, flags=re.IGNORECASE)
    if m:
        return f"{m.group('label').upper()} (ФАСКА)"

    m = re.match(
        r"^(?P<label>[A-Z])\(HOLE TO DEPTH OF\s*(?P<mm>[\d.,]+)mm\s*\((?P<inch>[\d.,]+)in\)\s*FROM THIS SURFACE\)$",
        normalized,
        flags=re.IGNORECASE,
    )
    if m:
        mm = m.group("mm").replace(".", ",")
        inch = m.group("inch").replace(".", ",")
        return f"{m.group('label').upper()} (ОТВ. ГЛУБ.\n{mm} мм ({inch} дюйма)\nОТ ЭТОЙ ПОВ.)"

    m = re.match(
        r"^DETAIL\s+(?P<label>[A-Z])(?P<mm>[\d.,]+)mm\s*\((?P<inch>[\d.,]+)in\)\s*DIA\.\s*THIS FACE ONLY$",
        normalized,
        flags=re.IGNORECASE,
    )
    if m:
        mm = m.group("mm").replace(".", ",")
        inch = m.group("inch").replace(".", ",")
        return f"ДЕТАЛЬ {m.group('label').upper()}\n{mm} мм ({inch} дюйма)\nДИАМ.\nТОЛЬКО ЭТА ПОВ."

    m = re.match(
        r"^(?P<mm>[\d.,]+)mm\s*\((?P<inch>[\d.,]+)in\)\s*DIA\.\s*THIS FACE ONLY$",
        normalized,
        flags=re.IGNORECASE,
    )
    if m:
        mm = m.group("mm").replace(".", ",")
        inch = m.group("inch").replace(".", ",")
        return f"{mm} мм ({inch} дюйма)\nДИАМ.\nТОЛЬКО ЭТА ПОВ."

    m = re.match(
        r"^NO PAINT\s*(?P<mm1>[\d.,]+)\s*to\s*(?P<mm2>[\d.,]+)mm\s*\((?P<in1>[\d.,]+)\s*to\s*(?P<in2>[\d.,]+)in\)\s*DIA\.$",
        normalized,
        flags=re.IGNORECASE,
    )
    if m:
        mm1 = m.group("mm1").replace(".", ",")
        mm2 = m.group("mm2").replace(".", ",")
        in1 = m.group("in1").replace(".", ",")
        in2 = m.group("in2").replace(".", ",")
        return f"БЕЗ КРАСКИ\n{mm1} до {mm2} мм\n({in1} до {in2} дюйма)\nДИАМ."

    m = re.match(
        r"^NO PAINT\s*(?P<mm1>[\d.,]+)\s*to\s*(?P<mm2>[\d.,]+)mm$",
        normalized,
        flags=re.IGNORECASE,
    )
    if m:
        mm1 = m.group("mm1").replace(".", ",")
        mm2 = m.group("mm2").replace(".", ",")
        return f"БЕЗ КРАСКИ\n{mm1} до {mm2} мм"

    m = re.match(r"^REPAIR BUSH\s+(?P<pn>\d+)\s+MACHINING$", normalized, flags=re.IGNORECASE)
    if m:
        return f"РЕМОНТНАЯ ВТУЛКА\n{m.group('pn')}\nМЕХ. ОБРАБОТКА"

    if normalized == "DIMENSION DMACHINE THIS FACE C ONLY":
        return "РАЗМЕР D,\nОБР. ТОЛЬКО\nПОВ. C"

    if normalized == "APPLY SEALANT TO PCS-7200SECTION Z-ZWITH BUSH":
        return "НАНЕСТИ ГЕРМЕТИК\nПО PCS-7200\nСЕЧЕНИЕ Z-Z\nС ВТУЛКОЙ"

    return None


def convert_units(text: str) -> str:
    def mm_repl(match: re.Match[str]) -> str:
        value = match.group(1).replace(".", ",")
        return f"{value} мм"

    def inch_repl(match: re.Match[str]) -> str:
        value = match.group(1).replace(".", ",")
        return f"{value} дюйма"

    out = re.sub(r"(\d+(?:[.,]\d+)?)\s*mm\b", mm_repl, text, flags=re.IGNORECASE)
    out = re.sub(r"(\d+(?:[.,]\d+)?)\s*in\b", inch_repl, out, flags=re.IGNORECASE)
    out = re.sub(r"(?<!\d)\.(?=\d)", "0,", out)
    out = re.sub(r"(?<=\d)\.(?=\d)", ",", out)
    out = re.sub(r"\bDEGREES\b", "ГРАДУСОВ", out)
    out = re.sub(r"\bMINUTES\b", "МИНУТ", out)
    out = re.sub(r"\bSECONDS\b", "СЕКУНД", out)
    return out


def tidy_translation(text: str) -> str:
    out = text
    out = re.sub(r"\bРИСУНОК\b", "Рисунок", out)
    out = re.sub(r"\bЛИСТ\b", "Лист", out)
    out = re.sub(r"\bСТРАНИЦА\b", "Страница", out)
    out = re.sub(r"\bДИАМ\.(?=\S)", "ДИАМ. ", out)
    out = re.sub(r"\bРАД\.(?=\S)", "РАД. ", out)
    out = re.sub(r"\bСПРАВ\.(?=\S)", "СПРАВ. ", out)
    out = re.sub(r"\s*-\s*", " - ", out)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\( ", "(", out)
    out = re.sub(r" \)", ")", out)
    out = re.sub(r"(?<=\d),(?=\d{3}\b)", ",", out)
    return out


def translate_text(text: str, patterns: list[tuple[re.Pattern[str], str]]) -> str:
    normalized = normalize_source(text)
    if not normalized:
        return text

    translated_page = translate_page_or_section(normalized)
    if translated_page:
        return translated_page

    special = translate_special_compact(normalized)
    if special:
        return special

    exact = EXACT_TEXT_MAP.get(normalize_exact_key(normalized))
    if exact:
        return exact

    caption = translate_caption(normalized)
    if caption:
        return caption

    out = apply_phrase_patterns(normalized, patterns)
    out = convert_units(out)
    out = re.sub(r"\band\b", "И", out, flags=re.IGNORECASE)
    out = re.sub(r"\bor\b", "ИЛИ", out, flags=re.IGNORECASE)
    out = tidy_translation(out)

    exact_after = EXACT_TEXT_MAP.get(normalize_exact_key(out))
    if exact_after:
        return exact_after
    return out


def text_nodes_for_textbox(txbx: etree._Element) -> list[etree._Element]:
    return txbx.xpath(".//w:t", namespaces=NS)


def textbox_text(txbx: etree._Element) -> str:
    return "".join(node.text or "" for node in text_nodes_for_textbox(txbx)).strip()


def set_text(nodes: list[etree._Element], value: str) -> None:
    if not nodes:
        return
    first = nodes[0]
    run = first.getparent()
    lines = value.split("\n")
    first.text = lines[0] if lines else ""
    if first.text[:1].isspace() or first.text[-1:].isspace():
        first.set(XML_SPACE, "preserve")
    elif XML_SPACE in first.attrib:
        del first.attrib[XML_SPACE]

    if len(lines) > 1 and run is not None:
        insert_at = list(run).index(first) + 1
        for line in lines[1:]:
            br = etree.Element(f"{{{NS['w']}}}br")
            new_t = etree.Element(f"{{{NS['w']}}}t")
            new_t.text = line
            if line[:1].isspace() or line[-1:].isspace():
                new_t.set(XML_SPACE, "preserve")
            run.insert(insert_at, br)
            insert_at += 1
            run.insert(insert_at, new_t)
            insert_at += 1

    for node in nodes[1:]:
        node.text = ""


def allowed_english(text: str) -> str:
    out = text
    out = re.sub(r"\b(?:PCS|IFC|PR|CAGE|SAFRAN|Ltd|IVD|TYPE)\b", "", out)
    out = re.sub(r"\b(?:A321[0-9A-Z-]*|M-DLPS[0-9-]+|[A-Z]{1,3}-\d{2}-\d{2}-\d{2}(?:-\d+)?)\b", "", out)
    out = re.sub(r"\b(?:[A-Z])\b", "", out)
    return out


def process_docx(
    input_path: Path,
    output_path: Path,
    *,
    glossary_text: str | None = None,
    audit_path: Path | None = None,
) -> tuple[int, int, list[tuple[str, int]]]:
    patterns = build_phrase_patterns(glossary_text)
    changed = 0
    total = 0
    untranslated: Counter[str] = Counter()

    with zipfile.ZipFile(input_path, "r") as zin:
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename.startswith("word/") and item.filename.endswith(".xml"):
                    try:
                        root = etree.fromstring(data)
                    except Exception:
                        zout.writestr(item, data)
                        continue

                    xml_changed = False
                    for txbx in root.xpath(".//w:txbxContent", namespaces=NS):
                        nodes = text_nodes_for_textbox(txbx)
                        if not nodes:
                            continue
                        source_text = textbox_text(txbx)
                        if not source_text:
                            continue

                        total += 1
                        translated = translate_text(source_text, patterns)
                        if translated != source_text:
                            set_text(nodes, translated)
                            changed += 1
                            xml_changed = True

                        remaining = allowed_english(translated)
                        if re.search(r"[A-Za-z]{3,}", remaining):
                            untranslated[translated] += 1

                    if xml_changed:
                        data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")

                zout.writestr(item, data)

    leftover = untranslated.most_common(200)
    if audit_path:
        lines = [f"{count}\t{text}" for text, count in leftover]
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text("\n".join(lines), encoding="utf-8")

    return total, changed, leftover


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Translate textbox OCR labels inside picture_1.docx locally.")
    parser.add_argument("--input", required=True, help="Input DOCX path")
    parser.add_argument("--output", required=True, help="Output DOCX path")
    parser.add_argument("--glossary", help="Glossary markdown path")
    parser.add_argument("--audit-file", help="Optional audit output for untranslated leftovers")
    args = parser.parse_args(argv)

    glossary_text = None
    if args.glossary:
        glossary_text = Path(args.glossary).read_text(encoding="utf-8")

    total, changed, leftover = process_docx(
        Path(args.input),
        Path(args.output),
        glossary_text=glossary_text,
        audit_path=Path(args.audit_file) if args.audit_file else None,
    )

    print(f"textboxes processed: {total}")
    print(f"textboxes changed: {changed}")
    print(f"top untranslated leftovers: {len(leftover)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
