from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.docxru.docx_reader import collect_segments
from src.docxru.tagging import paragraph_to_tagged, tagged_to_runs


HEADER_TITLE_EN = "PART No. 201587001 AND 201587002 COMPONENT MAINTENANCE MANUAL MAIN LANDING GEAR LEG"
HEADER_TITLE_RU = (
    "ДЕТАЛЬ № 201587001 И 201587002 РУКОВОДСТВО ПО ТЕХНИЧЕСКОМУ ОБСЛУЖИВАНИЮ "
    "КОМПОНЕНТА СТОЙКА ОСНОВНОГО ШАССИ"
)


EXACT_SOURCE_MAP: dict[str, str] = {
    "Machine diameter(s) A to remove the damage or wear or corrosion to": (
        "Обработайте диаметр(ы) A для удаления повреждений, износа или коррозии. "
        "После обработки диаметр должен быть"
    ),
    "Apply cadmium plate to the repaired areas. The plating thickness must be between 0,010 and 0,020 mm (0.0004 and 0.0008 in): refer to PCS-2100 or PCS-2141. No bare metal is permitted.": (
        "Нанесите кадмиевое покрытие на отремонтированные участки. Толщина покрытия должна "
        "быть от 0,010 до 0,020 mm (от 0.0004 до 0.0008 in): см. PCS-2100 или PCS-2141. "
        "Оголение основного металла не допускается."
    ),
    "Shot peen the reworked areas only: refer to PCS-2300.": (
        "Упрочните дробеструйной обработкой только доработанные участки: см. PCS-2300."
    ),
    "Prepare the repair sleeve(s) 62-4505252-00 with these dimensions for installation (qty 1 to 4 as necessary): refer to Figure 602.": (
        "Подготовьте ремонтную(ые) втулку(и) 62-4505252-00 до указанных размеров для "
        "установки (от 1 до 4 шт. по необходимости): см. рисунок 602."
    ),
    "Machine diameter Z, use the formula:": "Обработайте диаметр Z по формуле:",
    "Apply cadmium plate or zinc nickel plate over the reworked areas. The cadmium plate thickness must be between 0,010 and 0,015 mm": (
        "Нанесите кадмиевое или цинк-никелевое покрытие на доработанные участки. Толщина "
        "кадмиевого покрытия должна быть от 0,010 до 0,015 mm"
    ),
    "NOTE: Make sure that the sleeve chamfer is protruded out of the lug face during the installation and do not remain in the lug width.": (
        "ПРИМЕЧАНИЕ: Убедитесь, что во время установки фаска втулки выступает за торец ушка "
        "и не остается в пределах его ширины."
    ),
    "Machine the bore of sleeve(s) to the dimensions shown: refer to Figure 601.": (
        "Обработайте отверстия втулок до указанных размеров: см. рисунок 601."
    ),
    "NOTE: Make sure that the sleeve chamfer is machined and do not remain in the lug width.": (
        "ПРИМЕЧАНИЕ: Убедитесь, что фаска втулки обработана и не остается в пределах ширины ушка."
    ),
    "Apply cadmium plate or zinc nickel plate to the machined bores and faces. The cadmium plate thickness must be between 0,010 and 0,020 mm": (
        "Нанесите кадмиевое или цинк-никелевое покрытие на обработанные отверстия и "
        "поверхности. Толщина кадмиевого покрытия должна быть от 0,010 до 0,020 mm"
    ),
    "Apply applicable paint to the repaired areas: refer to PCS-2500.": (
        "Нанесите соответствующую краску на отремонтированные участки: см. PCS-2500."
    ),
    "Record the repair number onto the documentation which is attached to the part. Optionally, identify the part with the Safran Landing Systems repair number 64-4505126-00 adjacent to the existing part number: refer to PCS-6000-07.": (
        "Занесите номер ремонта в документацию, прикрепленную к детали. При необходимости "
        "нанесите на деталь ремонтный номер Safran Landing Systems 64-4505126-00 рядом с "
        "существующим номером детали: см. PCS-6000-07."
    ),
    "NOTE: Alternative equivalents are permitted.": "ПРИМЕЧАНИЕ: Допускаются альтернативные эквиваленты.",
    "NOTE:\tAlternative equivalents are permitted.": "ПРИМЕЧАНИЕ: Допускаются альтернативные эквиваленты.",
    "NOTE: This operation includes 23 hours de-embrittlement at 185 to 195 oC (366 to 384 oF).": (
        "ПРИМЕЧАНИЕ: Эта операция включает 23 часа устранения хрупкости при температуре "
        "от 185 до 195 oC (от 366 до 384 oF)."
    ),
    "NOTE: An electrical bonding test is not necessary.": (
        "ПРИМЕЧАНИЕ: Проверка электрического соединения не требуется."
    ),
    "NOTE: If this repair is applied to diameters A and B, identify the part with the Messier-Dowty Limited repair number 450266430-AB adjacent to the part number.": (
        "ПРИМЕЧАНИЕ: Если данный ремонт применяется к диаметрам A и B, нанесите на деталь "
        "ремонтный номер Messier-Dowty Limited 450266430-AB рядом с номером детали."
    ),
    "NOTE: If the repair is applied to the diameters A, B and C, apply sulphamate nickel plate: refer to Figure 601 and identify the part with the Safran Landing Systems repair number 64-4505141-00-ABC adjacent to the existing part number: refer to PCS-6000-07.": (
        "ПРИМЕЧАНИЕ: Если ремонт применяется к диаметрам A, B и C, нанесите "
        "сульфаматно-никелевое покрытие: см. рисунок 601. Затем нанесите на деталь "
        "ремонтный номер Safran Landing Systems 64-4505141-00-ABC рядом с существующим "
        "номером детали: см. PCS-6000-07."
    ),
}


CURRENT_TEXT_MAP: dict[str, str] = {
    "ПРИМЕЧАНИЕ: Допустимы альтернативные эквиваленты.": "ПРИМЕЧАНИЕ: Допускаются альтернативные эквиваленты.",
    "ПРИМЕЧАНИЕ:Убедитесь, что фаска втулки выступает за торец ушка во время установки и не остается в пределах ширины ушка.": (
        "ПРИМЕЧАНИЕ: Убедитесь, что во время установки фаска втулки выступает за торец ушка "
        "и не остается в пределах его ширины."
    ),
    "ПРИМЕЧАНИЕ:Убедитесь, что фаска втулки обработана и не остается в пределах ширины ушка.": (
        "ПРИМЕЧАНИЕ: Убедитесь, что фаска втулки обработана и не остается в пределах ширины ушка."
    ),
}


LOCATION_MAP: dict[str, str] = {
    "body/p9": "Диам. Z = Диам. A (по замеру) от + 0,008 до + 0,032 mm (от + 0.0004 до 0.0012 in).",
    "body/p333": "F = A (по измерению) от + 0,018 до 0,059 mm (от 0.0008 до 0.0023 in).",
    "body/p557": "Диаметр F = диаметр A (по измерению) от + 0,018 до 0,059 mm (от 0.0007 до 0.0023 in).",
    "body/p576": "Диам. H = диам. D (по замеру) от + 0,010 до 0,039 mm (от 0.0004 до 0.0015 in).",
    "body/p808": "G = A или B (по измерению) от + 0,069 до 0,138 mm (от 0.0027 до 0.0054 in).",
    "body/p809": "H = E или F (по измерению) от - 0,25 mm (0.010 in) до + 0,25 mm (0.010 in).",
    "body/p1017": "G = A или B (по измерению) от + 0,029 до 0,078 mm (от 0.0011 до 0.0031 in).",
    "body/p1018": "H = E или F (по измерению) от - 0,25 mm (0.010 in) до + 0,25 mm (0.010 in).",
    "body/p1376": "Диаметр G = A (по замеру) от - 0,006 до + 0,028 mm (от - 0.0002 до + 0.0011 in).",
    "body/p1379": "Диаметр H = B (по замеру) от - 0,006 до + 0,028 mm (от - 0.0002 до + 0.0011 in).",
    "body/p1769": "B = A (по измерению) от + 0,002 до 0,051 mm (от 0.00008 до 0.0020 in).",
    "body/p2764": "Диам. C = Диам. A (по замеру) от - 0,006 до + 0,023 mm (от - 0.0002 до + 0.0009 in).",
    "body/p4114": "Диаметр M = диаметр A и/или диаметр B (как измерено) от + 0,007 до",
    "body/p4115": "+ 0,041 mm (от + 0.0003 до + 0.0016 in).",
    "body/p4746": "Диаметр D (до кадмирования) = диаметр A (по замеру) от + 0,023 до - 0,006 mm (от + 0.0009 до - 0.0002 in).",
    "body/p4747": "Диаметр D (после кадмирования) = диаметр A (по замеру) от + 0,014 до + 0,053 mm (от + 0.0005 до + 0.0021 in).",
    "body/p4904": "B = A (по измерению) от - 0,006 до + 0,023 mm (от - 0.0002 до + 0.0009 in).",
    "body/p5206": "Диам. D (до покрытия) = Диам. A (по замеру) от - 0,006 до + 0,023 mm (от - 0.0002 до",
    "body/p5208": "Диам. D (после покрытия) = Диам. A (по замеру) от + 0,014 до + 0,053 mm (от + 0.0005 до",
    "body/p5380": "B (до кадмирования) = A (по измерению) от - 0,006 до + 0,023 mm (от - 0.0002 до",
    "body/p5382": "B (после кадмирования) = A (по измерению) от + 0,014 до + 0,053 mm (от + 0.0005 до",
    "body/p5926": "C (после кадмирования) = A или B (по замеру) от + 0,013 mm (+0.0005 in) до + 0,041 mm (0.0016 in).",
    "body/p6087": "B = A (по измерению) от + 0,010 до + 0,039 mm (от + 0.0004 до + 0.0015 in).",
    "body/p6529": (
        "Вычислите диаметр ремонтной втулки 450265251, используя формулу: Диам. B = Диам. A "
        "(по замеру) от + 0,023 до 0,072 mm (от 0.0009 до 0.0028 in)."
    ),
}


FUNCTION_SOURCE_MAP: dict[str, str] = {
    "Install the repair bushes": "Установка ремонтных втулок",
    "Install the repair bush": "Установка ремонтной втулки",
    "Install the repair bearing": "Установка ремонтного подшипника",
    "Align the repair bush": "Совмещение ремонтной втулки",
    "Main landing gear leg (1-1) tests": "Испытания стойки основного шасси (1-1)",
    "Proximity switch and target tests": "Испытания датчика приближения и мишени",
    "Electrical bonding resistance tests": "Проверка сопротивления электрического соединения",
    "To install the repair sleeve": "Для установки ремонтной втулки",
    "Install the forward pintle bush": "Установка передней втулки штифта навеса",
    "Torque the jacking dome (17-80)": "Затяжка поддомкратного купола (17-80)",
    "Get the correct dimension across the bushes (20-330)": (
        "Для получения правильного размера между втулками (20-330)"
    ),
    "To prevent damage to the mating surfaces of the lower bearing subassembly": (
        "Для предотвращения повреждения сопрягаемых поверхностей нижней подсборки подшипника"
    ),
}


GENITIVE_PREFIX_MAP: list[tuple[str, str]] = [
    ("ремонтные втулки", "ремонтных втулок"),
    ("ремонтную втулку", "ремонтной втулки"),
    ("ремонтный подшипник", "ремонтного подшипника"),
    ("ремонтные подшипники", "ремонтных подшипников"),
    ("втулки", "втулок"),
    ("втулку", "втулки"),
    ("подшипник", "подшипника"),
    ("подшипники", "подшипников"),
    ("штифт", "штифта"),
    ("штифты", "штифтов"),
    ("уплотнения", "уплотнений"),
    ("сборку корпуса стойки", "сборки корпуса стойки"),
    ("подсборку скользящей трубки", "подсборки скользящей трубки"),
    ("стойку основного шасси", "стойки основного шасси"),
    ("переднюю втулку штифта навеса", "передней втулки штифта навеса"),
    ("нижнюю сборку подшипника", "нижней сборки подшипника"),
]


_SIGNED_NUM = r"[+\-]?\s*\d[\d.,]*"
NUMERIC_RANGE_RE = re.compile(
    rf"(?P<a>{_SIGNED_NUM})\s+до\s+(?P<b>{_SIGNED_NUM})(?=(?:\s*(?:mm|in|oC|oF)\b|\)|\s|$))"
)
BETWEEN_RE = re.compile(rf"\bмежду\s+(?P<a>{_SIGNED_NUM})\s+и\s+(?P<b>{_SIGNED_NUM})", re.IGNORECASE)
AND_RANGE_RE = re.compile(
    rf"(?P<a>{_SIGNED_NUM})\s+и\s+(?P<b>{_SIGNED_NUM})(?=\s*(?:mm|in|oC|oF)\b)"
)
PAGE_RE = re.compile(r"^Page\s+(\d+)$", re.IGNORECASE)
PAGE_DATE_RE = re.compile(r"^Page\s+(\d+)\s+([A-Z][a-z]{2}\s+\d{1,2}/\d{4})$")
PAGE_ANY_RE = re.compile(r"^Page\s+([0-9.]+)(?:\s+([A-Z][a-z]{2}\s+\d{1,2}/\d{4}))?$")
REPAIR_TITLE_RE = re.compile(r"^Repair to ")


def _build_single_span_tagged(paragraph, text: str) -> tuple[str, list, dict[str, str]]:
    _, spans, inline_map = paragraph_to_tagged(paragraph)
    if not spans:
        return text, spans, inline_map
    first = spans[0]
    flag_part = "|" + "|".join(first.flags) if first.flags else ""
    tagged = f"⟦S_{first.span_id}{flag_part}⟧{text}⟦/S_{first.span_id}⟧"
    return tagged, spans, inline_map


def _write_text(paragraph, text: str) -> bool:
    text = text.replace("—", "-").replace("–", "-").replace("−", "-")
    if paragraph.text == text:
        return False
    tagged, spans, inline_map = _build_single_span_tagged(paragraph, text)
    if spans:
        tagged_to_runs(paragraph, tagged, spans, inline_run_map=inline_map)
    else:
        paragraph.text = text
    return True


def _ru_codes(text: str) -> str:
    return text.replace(" and ", " и ").replace(" or ", " или ")


def _to_genitive_phrase(text: str) -> str:
    stripped = text.strip()
    for prefix, repl in GENITIVE_PREFIX_MAP:
        if stripped.startswith(prefix):
            return repl + stripped[len(prefix) :]
    return stripped


def _looks_like_drawing_label(source: str | None, style_name: str) -> bool:
    style_key = (style_name or "").strip().lower()
    if style_key == "normal":
        return True
    if not source:
        return False
    if source.upper() == source and any(ch.isalpha() for ch in source):
        return True
    return False


def _normalize_range_language(text: str, *, drawing_label: bool) -> str:
    out = text
    out = BETWEEN_RE.sub(lambda m: f"от {m.group('a')} до {m.group('b')}", out)
    out = AND_RANGE_RE.sub(lambda m: f"от {m.group('a')} до {m.group('b')}", out)
    if drawing_label:
        out = NUMERIC_RANGE_RE.sub(lambda m: f"{m.group('a')}-{m.group('b')}", out)
        out = out.replace("(от ", "(").replace(" до ", "-")
    elif "=" in out:
        out = NUMERIC_RANGE_RE.sub(lambda m: f"от {m.group('a')} до {m.group('b')}", out)
    return out


def _cleanup_common_text(text: str) -> str:
    out = text
    out = out.replace("ПРИМЕЧАНИЕ:", "ПРИМЕЧАНИЕ: ")
    out = re.sub(r"\s{2,}", " ", out)
    out = out.replace("оC", "oC").replace("оF", "oF")
    out = out.replace("мм (дюйм)", "mm (in)")
    out = out.replace("Открытый металл не допускается.", "Оголение основного металла не допускается.")
    out = out.replace("применимую краску", "соответствующую краску")
    out = out.replace("переобработанные", "доработанные")
    out = out.replace("Обработайте дробью", "Упрочните дробеструйной обработкой")
    out = out.replace("купол домкрата", "поддомкратный купол")
    out = out.replace("от от", "от")
    out = out.replace("отот", "от")
    out = out.replace("+ от ", "от + ")
    out = out.replace("- от ", "от - ")
    out = out.replace("+ отот ", "от + ")
    out = out.replace("- отот ", "от - ")
    out = re.sub(r"\(\+\s*от\s*", "(от + ", out)
    out = re.sub(r"\(-\s*от\s*", "(от - ", out)
    out = re.sub(r"\((по [^)]+?) от ([+\-])", r"(\1) от \2", out)
    out = re.sub(r"\((как измерено) от ([+\-])", r"(\1) от \2", out)
    out = out.replace("ПРИМЕЧАНИЕ:  ", "ПРИМЕЧАНИЕ: ")
    return out.strip()


def _translate_header_footer(source: str, text: str) -> str | None:
    if source == HEADER_TITLE_EN:
        return HEADER_TITLE_RU
    if source == "Page 603":
        return "Стр. 603"
    m = PAGE_RE.match(source)
    if m:
        return f"Стр. {m.group(1)}"
    m = PAGE_DATE_RE.match(source)
    if m:
        return f"Стр. {m.group(1)} {m.group(2)}"
    m = PAGE_ANY_RE.match(source)
    if m:
        number = m.group(1)
        date = m.group(2)
        return f"Стр. {number}" + (f" {date}" if date else "")
    return None


def _translate_repair_heading(source: str, current: str) -> str | None:
    if not REPAIR_TITLE_RE.match(source):
        return None
    out = current.strip()
    out = re.sub(r"\s*-\s*", ". ", out)
    out = re.sub(r"\s+Рисунок\s+", ". Рисунок ", out)
    out = re.sub(r"\.\s+Лист\s+", ", лист ", out)
    out = re.sub(r"\s+,", ",", out)
    if not out.endswith("."):
        out += "."
    return out


def _translate_section_detail_view(source: str) -> str | None:
    if source.startswith("SECTION "):
        body = source.removeprefix("SECTION ").strip()
        body = body.replace("WITH BUSHES", "С ВТУЛКАМИ")
        body = body.replace("WITH SLEEVES", "С ВТУЛКАМИ")
        body = body.replace("WITHOUT SLEEVES", "БЕЗ ВТУЛОК")
        body = body.replace("WITHOUT REPAIR BUSHES", "БЕЗ РЕМОНТНЫХ ВТУЛОК")
        body = body.replace("WITH REPAIR BUSHES", "С РЕМОНТНЫМИ ВТУЛКАМИ")
        body = body.replace("REFER TO FIGURE", "см. рисунок")
        body = _ru_codes(body)
        return f"СЕЧЕНИЕ {body}"
    if source.startswith("DETAIL "):
        body = _ru_codes(source.removeprefix("DETAIL ").strip())
        return f"ФРАГМЕНТ {body}"
    if source.startswith("VIEW ON ARROW "):
        return f"ВИД ПО СТРЕЛКЕ {source.removeprefix('VIEW ON ARROW ').strip()}"
    if source.startswith("VIEW "):
        body = _ru_codes(source.removeprefix("VIEW ").strip())
        return f"ВИД {body}"
    return None


def _translate_function(source: str | None, current: str) -> str | None:
    if source and source in FUNCTION_SOURCE_MAP:
        return FUNCTION_SOURCE_MAP[source]

    if source:
        m = re.fullmatch(r"Install the repair bush(?:es)? (.+)", source)
        if m:
            object_suffix = _ru_codes(m.group(1).strip())
            base = "ремонтных втулок" if "bushes" in source else "ремонтной втулки"
            return f"Установка {base} {object_suffix}"
        m = re.fullmatch(r"Install the (bush|bearing|pin|seals?) \((.+)\)", source)
        if m:
            noun_map = {
                "bush": "втулки",
                "bearing": "подшипника",
                "pin": "штифта",
                "seal": "уплотнения",
                "seals": "уплотнений",
            }
            noun = noun_map[m.group(1)]
            return f"Установка {noun} ({_ru_codes(m.group(2))})"
        m = re.fullmatch(r"Install the bushes \((.+)\)", source)
        if m:
            return f"Установка втулок ({_ru_codes(m.group(1))})"
        m = re.fullmatch(r"Use with (.+)", source, flags=re.IGNORECASE)
        if m:
            suffix = _ru_codes(m.group(1))
            suffix = suffix.replace("Press Pad", "прижимной пластиной")
            suffix = suffix.replace("press pad", "прижимной пластиной")
            return f"Используется с {suffix}"
        m = re.fullmatch(r"To install the repair bush (.+)", source)
        if m:
            return f"Для установки ремонтной втулки {_ru_codes(m.group(1))}"
        m = re.fullmatch(r"Lift the (.+)", source)
        if m:
            obj = _ru_codes(m.group(1))
            return f"Подъем {obj}"
        m = re.fullmatch(r"Hold the (.+)", source)
        if m:
            obj = _ru_codes(m.group(1))
            return f"Удержание {obj}"
        m = re.fullmatch(r"Remove/torque the (.+)", source)
        if m:
            obj = _ru_codes(m.group(1))
            return f"Снятие/затяжка {obj}"

    stripped = current.strip()
    if stripped.startswith("Установите "):
        return "Установка " + _to_genitive_phrase(stripped.removeprefix("Установите "))
    if stripped.startswith("Используйте "):
        return "Используется " + stripped.removeprefix("Используйте ")
    if stripped.startswith("Использовать "):
        return "Используется " + stripped.removeprefix("Использовать ")
    if stripped.startswith("Поднимите "):
        return "Подъем " + _to_genitive_phrase(stripped.removeprefix("Поднимите "))
    if stripped.startswith("Удерживайте "):
        return "Удержание " + _to_genitive_phrase(stripped.removeprefix("Удерживайте "))
    if stripped.startswith("Затяните "):
        return "Затяжка " + _to_genitive_phrase(stripped.removeprefix("Затяните "))
    if stripped.startswith("Совместите "):
        return "Совмещение " + _to_genitive_phrase(stripped.removeprefix("Совместите "))
    if stripped.startswith("Получите "):
        return "Получение " + stripped.removeprefix("Получите ").lower()
    return None


def revise_document(source_path: Path, target_path: Path, backup_path: Path | None) -> dict[str, int]:
    if backup_path is not None and not backup_path.exists():
        shutil.copyfile(target_path, backup_path)

    source_doc = Document(str(source_path))
    target_doc = Document(str(target_path))

    source_segments = collect_segments(source_doc, include_headers=True, include_footers=True)
    target_segments = collect_segments(target_doc, include_headers=True, include_footers=True)
    source_by_location = {seg.location: seg for seg in source_segments}

    stats = {
        "updated": 0,
        "exact": 0,
        "function": 0,
        "labels": 0,
        "ranges": 0,
        "headers": 0,
        "repair_titles": 0,
    }

    for seg in target_segments:
        paragraph = seg.paragraph_ref
        current = seg.source_plain or ""
        source_seg = source_by_location.get(seg.location)
        source = source_seg.source_plain if source_seg is not None else None
        style_name = str(seg.context.get("paragraph_style") or "")
        new_text: str | None = None

        if source and (seg.location.startswith("header") or seg.location.startswith("footer")):
            new_text = _translate_header_footer(source, current)
            if new_text:
                stats["headers"] += 1

        if new_text is None and source and source in EXACT_SOURCE_MAP:
            new_text = EXACT_SOURCE_MAP[source]
            stats["exact"] += 1

        if new_text is None and current in CURRENT_TEXT_MAP:
            new_text = CURRENT_TEXT_MAP[current]
            stats["exact"] += 1

        if new_text is None and seg.location in LOCATION_MAP:
            new_text = LOCATION_MAP[seg.location]
            stats["exact"] += 1

        if new_text is None and source:
            label_text = _translate_section_detail_view(source)
            if label_text:
                new_text = label_text
                stats["labels"] += 1

        if new_text is None and source:
            repair_title = _translate_repair_heading(source, current)
            if repair_title:
                new_text = repair_title
                stats["repair_titles"] += 1

        column_header = str(seg.context.get("column_header") or "").strip()
        if new_text is None and source == "Function":
            new_text = "Назначение"
            stats["function"] += 1
        if new_text is None and column_header in {"Функция", "Назначение"}:
            translated_function = _translate_function(source, current)
            if translated_function:
                new_text = translated_function
                stats["function"] += 1

        candidate = new_text if new_text is not None else current
        drawing_label = _looks_like_drawing_label(source, style_name)
        ranged = _normalize_range_language(candidate, drawing_label=drawing_label)
        if ranged != candidate:
            candidate = ranged
            stats["ranges"] += 1

        cleaned = _cleanup_common_text(candidate)
        if cleaned != current:
            if _write_text(paragraph, cleaned):
                stats["updated"] += 1

    target_doc.save(str(target_path))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual revision pass for section 2 Russian DOCX.")
    parser.add_argument("--source", required=True, help="English source DOCX.")
    parser.add_argument("--target", required=True, help="Russian target DOCX.")
    parser.add_argument("--backup", help="Optional backup path for the original target DOCX.")
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    target_path = Path(args.target).resolve()
    backup_path = Path(args.backup).resolve() if args.backup else None

    stats = revise_document(source_path, target_path, backup_path)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
