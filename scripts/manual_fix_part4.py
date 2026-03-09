from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": NS_W}
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

FIGURE_CAPTION_RE = re.compile(
    r"^(?P<title>.+?) - Защитная обработка рисунок (?P<figure>\d+)(?: - Лист (?P<sheet>\d+))?$"
)
KEY_DIAGRAM_RE = re.compile(r"^(?P<title>.+?) - Схема расположения рисунок (?P<figure>\d+)$")
REPAIR_LABEL_RE = re.compile(r"^(?P<prefix>.*?Ремонт №)\s*(?P<body>.+)$", re.IGNORECASE)
CONTINUATION_AND_RE = re.compile(r"^и\s+\d+-\d+$", re.IGNORECASE)
CONTINUATION_LIST_RE = re.compile(r"^\d+-\d+(?:,\s*\d+-\d+)+$")

EXACT_REPLACEMENTS = {
    "ГРУНТОВОЧНАЯ КРАСКА, ПОВЕРХНОСТЬ C": "ГРУНТОВОЧНАЯ КРАСКА НА ПОВЕРХНОСТИ C",
    "C ПОСЛЕ НАРЕЗАНИЯ РЕЗЬБЫ": "ПОВЕРХНОСТЬ C ПОСЛЕ НАРЕЗАНИЯ РЕЗЬБЫ",
    "КАДМИЕВОЕ ПОКРЫТИЕ И КРАСКА ДОЛЖНЫ ПЕРЕКРЫВАТЬСЯ НА ХРОМОВОМ РАДИУСЕ": (
        "КАДМИРОВАНИЕ И ЛАКОКРАСОЧНОЕ ПОКРЫТИЕ ДОЛЖНЫ ПЕРЕКРЫВАТЬСЯ НА ХРОМИРОВАННОМ РАДИУСЕ"
    ),
    "КАДМИЕВОЕ ПОКРЫТИЕ И КРАСКА ДОЛЖНЫ ПЕРЕКРЫВАТЬСЯ": (
        "КАДМИРОВАНИЕ И ЛАКОКРАСОЧНОЕ ПОКРЫТИЕ ДОЛЖНЫ ПЕРЕКРЫВАТЬСЯ"
    ),
    "КАДМИЕВОЕ ПОКРЫТИЕ C": "КАДМИРОВАНИЕ",
    "и ГРУНТОВОЧНАЯ КРАСКА": "И ГРУНТОВОЧНАЯ КРАСКА",
    "ОТВЕРСТИЯ ПОПЕРЕЧНОГО БОЛТА ГАЙКИ ОСИ": "ОТВЕРСТИЯ ПОД ПОПЕРЕЧНЫЙ БОЛТ ГАЙКИ ОСИ",
    "ПРОТЯЖЕННОСТЬ УЧАСТКА МЕНЬШЕГО ПРЕД. ДИАМ.": "ДЛИНА УЧАСТКА МЕНЬШЕГО ПРЕДЕЛЬНОГО ДИАМЕТРА",
    "УЧАСТКА МЕНЬШЕГО ПРЕД. ДИАМ.": "УЧАСТКА МЕНЬШЕГО ПРЕДЕЛЬНОГО ДИАМЕТРА",
    "НИ КАДМИЕВОЕ ПОКРЫТИЕ, НИ КРАСКА НЕ ДОЛЖНЫ ВЫХОДИТЬ ЗА ЭТУ ЛИНИЮ": (
        "КАДМИРОВАНИЕ И ЛАКОКРАСОЧНОЕ ПОКРЫТИЕ НЕ ДОЛЖНЫ ВЫХОДИТЬ ЗА ЭТУ ЛИНИЮ"
    ),
    "(1 .2106-1 .2303 in) ????. ???????? ???????? ??????": (
        "(1.2106-1.2303 in) ДИАМ. ПОДРЕЗКА ПЛОЩАДКИ, ТИПОВО"
    ),
    "12 МЕСТА ВКЛЮЧАЯ": "12 МЕСТ, ВКЛЮЧАЯ",
    "ТИПОВО 12 МЕСТА": "ТИПОВО 12 МЕСТ",
    "НИЖНЯЯ ГРАНИЦА ХРОМОВОГО ПОКРЫТИЯ НАРУЖ. ДИАМ. ЦИЛИНДРИЧЕСКОЙ ЧАСТИ": (
        "НИЖНЯЯ ГРАНИЦА ХРОМОВОГО ПОКРЫТИЯ НА НАРУЖНОМ ДИАМЕТРЕ ЦИЛИНДРИЧЕСКОЙ ЧАСТИ"
    ),
    "ВЕРХНЯЯ ГРАНИЦА ХРОМОВОГО ПОКРЫТИЯ НАРУЖ. ДИАМ. ЦИЛИНДРИЧЕСКОЙ ЧАСТИ": (
        "ВЕРХНЯЯ ГРАНИЦА ХРОМОВОГО ПОКРЫТИЯ НА НАРУЖНОМ ДИАМЕТРЕ ЦИЛИНДРИЧЕСКОЙ ЧАСТИ"
    ),
    "Утвержденные ремонты Таблица 602": "Таблица 602. Утвержденные ремонты",
    "Утвержденные ремонты Таблица 602 (Продолжение)": "Таблица 602. Утвержденные ремонты (Продолжение)",
    "См. рисунки 649-657 и Таблица 602.": "См. рисунки 649-657 и таблицу 602.",
    "Landing Systems Ремонт №": "Landing Systems № ремонта",
}

SUBSTRING_REPLACEMENTS = (
    ("REARSIDE ONLY", "ТОЛЬКО С ЗАДНЕЙ СТОРОНЫ"),
    ("REARSIDE ТОЛЬКО", "ТОЛЬКО С ЗАДНЕЙ СТОРОНЫ"),
    ("(2 HOLES)", "(2 ОТВЕРСТИЯ)"),
    ("(4 HOLES)", "(4 ОТВЕРСТИЯ)"),
    ("2 HOLES", "2 ОТВЕРСТИЯ"),
    ("4 HOLES", "4 ОТВЕРСТИЯ"),
    ("A SPOT", "A ПОДРЕЗКА ПЛОЩАДКИ"),
    ("SPOTFACE", "ПОДРЕЗКА ПЛОЩАДКИ"),
    ("THRU BORE", "СКВОЗНОГО ОТВЕРСТИЯ"),
    ("DIAMETER", "ДИАМЕТР"),
    ("RADIUS", "РАДИУС"),
)

PAIR_REPLACEMENTS = (
    (
        "НАРУЖНОЕ ТОЛСТОЕ ЦИНК-НИКЕЛЕВОЕ ПОКРЫТИЕ",
        "ГРАНИЦА ПОКРЫТИЯ",
        "НАРУЖНАЯ ГРАНИЦА ТОЛСТОГО",
        "ЦИНК-НИКЕЛЕВОГО ПОКРЫТИЯ",
    ),
    (
        "ВНУТРЕННЕЕ ТОЛСТОЕ ЦИНК-НИКЕЛЕВОЕ ПОКРЫТИЕ",
        "ГРАНИЦА ПОКРЫТИЯ",
        "ВНУТРЕННЯЯ ГРАНИЦА ТОЛСТОГО",
        "ЦИНК-НИКЕЛЕВОГО ПОКРЫТИЯ",
    ),
)


def _text_nodes(paragraph: etree._Element) -> list[etree._Element]:
    return paragraph.xpath(".//w:t", namespaces=NS)


def _paragraph_text(paragraph: etree._Element) -> str:
    return "".join(node.text or "" for node in _text_nodes(paragraph)).strip()


def _set_text(nodes: list[etree._Element], value: str) -> None:
    if not nodes:
        return
    first = nodes[0]
    first.text = value
    if value[:1].isspace() or value[-1:].isspace():
        first.set(XML_SPACE, "preserve")
    elif XML_SPACE in first.attrib:
        del first.attrib[XML_SPACE]
    for node in nodes[1:]:
        node.text = ""
        if XML_SPACE in node.attrib:
            del node.attrib[XML_SPACE]


def _plural_form(value: int) -> str:
    rem100 = value % 100
    rem10 = value % 10
    if 11 <= rem100 <= 14:
        return "МЕСТ"
    if rem10 == 1:
        return "МЕСТО"
    if 2 <= rem10 <= 4:
        return "МЕСТА"
    return "МЕСТ"


def _normalize_places(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        number = int(match.group(1))
        return f"{number} {_plural_form(number)}"

    return re.sub(r"\b(\d+)\s+МЕСТА\b", repl, text)


def _fix_caption(text: str) -> str:
    match = FIGURE_CAPTION_RE.match(text)
    if not match:
        return text
    title = match.group("title")
    if title.endswith(" Только"):
        title = f"{title[:-7]}, только"
    result = f"{title} - Защитная обработка, рисунок {match.group('figure')}"
    if match.group("sheet"):
        result += f", лист {match.group('sheet')}"
    return result


def _fix_key_diagram(text: str) -> str:
    match = KEY_DIAGRAM_RE.match(text)
    if not match:
        return text
    return f"{match.group('title')} - схема расположения, рисунок {match.group('figure')}"


def _normalize_repair_list(text: str) -> str:
    match = REPAIR_LABEL_RE.match(text)
    if not match:
        return text

    prefix = match.group("prefix").rstrip()
    body = re.sub(r"\bРЕМОНТ\s*", "", match.group("body"), flags=re.IGNORECASE).strip()
    body = re.sub(r"\s{2,}", " ", body)
    if not body or body.endswith(","):
        return f"{prefix} {body}".strip()

    items = [item.strip() for item in body.split(",") if item.strip()]
    if not items or not all(re.fullmatch(r"\d+-\d+", item) for item in items):
        return f"{prefix} {body}".strip()

    if len(items) == 1:
        joined = items[0]
    elif len(items) == 2:
        joined = f"{items[0]} и {items[1]}"
    else:
        joined = f"{', '.join(items[:-1])} и {items[-1]}"
    return f"{prefix} {joined}"


def _normalize_tokens(text: str) -> str:
    text = re.sub(r"\bMIN\.", "МИН.", text)
    text = re.sub(r"\bMAX\.", "МАКС.", text)
    text = re.sub(r"\bREF\.", "СПРАВ.", text)
    text = re.sub(r"\bRAD\.", "РАД.", text)
    text = re.sub(r"\bМИН\.(?=[A-ZА-Я])", "МИН. ", text)
    text = re.sub(r"\bМАКС\.(?=[A-ZА-Я])", "МАКС. ", text)
    text = re.sub(r"\bСПРАВ\.(?=[A-ZА-Я])", "СПРАВ. ", text)
    text = re.sub(r"\bРАД\.(?=[A-ZА-Я])", "РАД. ", text)
    text = re.sub(r"ДИАМ\. MAX\.", "ДИАМ. МАКС.", text)
    return text


def _transform_text(text: str) -> tuple[str, Counter[str]]:
    stats: Counter[str] = Counter()
    original = text

    if text in EXACT_REPLACEMENTS:
        text = EXACT_REPLACEMENTS[text]
        stats["exact_replacements"] += 1

    updated = _fix_caption(text)
    if updated != text:
        text = updated
        stats["captions_fixed"] += 1

    updated = _fix_key_diagram(text)
    if updated != text:
        text = updated
        stats["key_diagrams_fixed"] += 1

    for source, target in SUBSTRING_REPLACEMENTS:
        if source in text:
            text = text.replace(source, target)
            stats["substring_replacements"] += 1

    updated = _normalize_tokens(text)
    if updated != text:
        text = updated
        stats["token_normalizations"] += 1

    if "ПОДРЕЗКА ПЛОЩАДКИ РАДИУС" in text:
        text = text.replace("ПОДРЕЗКА ПЛОЩАДКИ РАДИУС", "РАДИУС ПОДРЕЗКИ ПЛОЩАДКИ")
        stats["spotface_radius_fixed"] += 1
    if "ПОДРЕЗКА ПЛОЩАДКИ РАД." in text:
        text = text.replace("ПОДРЕЗКА ПЛОЩАДКИ РАД.", "РАДИУС ПОДРЕЗКИ ПЛОЩАДКИ")
        stats["spotface_radius_fixed"] += 1
    if "ВКЛЮЧАЯ ФАСКА" in text:
        text = text.replace("ВКЛЮЧАЯ ФАСКА", "ВКЛЮЧАЯ ФАСКИ")
        stats["grammar_fixes"] += 1
    if "ПОЛОСА СХОДА ПОКРЫТИЯ" in text:
        text = text.replace("ПОЛОСА СХОДА ПОКРЫТИЯ", "ПОЛОСА СХОДА ХРОМОВОГО ПОКРЫТИЯ")
        stats["run_out_band_fixed"] += 1

    updated = _normalize_repair_list(text)
    if updated != text:
        text = updated
        stats["repair_labels_fixed"] += 1

    updated = _normalize_places(text)
    if updated != text:
        text = updated
        stats["plural_fixes"] += 1

    text = re.sub(r"\s{2,}", " ", text).strip()
    if text != original:
        stats["paragraphs_changed"] += 1
    return text, stats


def _apply_sequence_fixes(paragraphs: list[etree._Element]) -> Counter[str]:
    stats: Counter[str] = Counter()
    texts = [_paragraph_text(paragraph) for paragraph in paragraphs]

    for first, second, first_target, second_target in PAIR_REPLACEMENTS:
        for idx in range(len(paragraphs) - 1):
            if texts[idx] == first and texts[idx + 1] == second:
                _set_text(_text_nodes(paragraphs[idx]), first_target)
                _set_text(_text_nodes(paragraphs[idx + 1]), second_target)
                texts[idx] = first_target
                texts[idx + 1] = second_target
                stats["sequence_pairs_fixed"] += 1

    for idx in range(len(paragraphs) - 1):
        current = texts[idx]
        nxt = texts[idx + 1]
        if current.startswith("Ремонт №") and CONTINUATION_AND_RE.fullmatch(nxt):
            merged = f"{current} {nxt}"
            _set_text(_text_nodes(paragraphs[idx]), merged)
            _set_text(_text_nodes(paragraphs[idx + 1]), "")
            texts[idx] = merged
            texts[idx + 1] = ""
            stats["repair_continuations_merged"] += 1

        if current.startswith("Ремонт №") and current.endswith(",") and CONTINUATION_LIST_RE.fullmatch(nxt):
            merged = f"{current} {nxt}"
            _set_text(_text_nodes(paragraphs[idx]), merged)
            _set_text(_text_nodes(paragraphs[idx + 1]), "")
            texts[idx] = merged
            texts[idx + 1] = ""
            stats["repair_continuations_merged"] += 1

    return stats


def _iter_word_xml_parts(docx_path: Path) -> list[str]:
    with ZipFile(docx_path) as zin:
        return [
            name
            for name in zin.namelist()
            if name.startswith("word/")
            and name.endswith(".xml")
            and (
                name == "word/document.xml"
                or Path(name).name.startswith("header")
                or Path(name).name.startswith("footer")
            )
        ]


def apply_fixes(docx_path: Path) -> Counter[str]:
    stats: Counter[str] = Counter()
    xml_parts = _iter_word_xml_parts(docx_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp_path = Path(tmp.name)

    try:
        with ZipFile(docx_path, "r") as zin, ZipFile(tmp_path, "w", compression=ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename not in xml_parts:
                    zout.writestr(info, data)
                    continue

                root = etree.fromstring(data)
                paragraphs = [
                    paragraph
                    for paragraph in root.iter(f"{{{NS_W}}}p")
                    if _paragraph_text(paragraph)
                ]

                if info.filename == "word/document.xml":
                    stats.update(_apply_sequence_fixes(paragraphs))

                for paragraph in paragraphs:
                    current = _paragraph_text(paragraph)
                    if not current:
                        continue
                    updated, paragraph_stats = _transform_text(current)
                    if updated != current:
                        _set_text(_text_nodes(paragraph), updated)
                    stats.update(paragraph_stats)

                data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")
                zout.writestr(info, data)

        shutil.move(str(tmp_path), str(docx_path))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply GPT review fixes to part4 Russian DOCX.")
    parser.add_argument("--docx", required=True, help="Translated DOCX to patch in place")
    parser.add_argument("--report", required=True, help="Where to write the JSON report")
    args = parser.parse_args()

    docx_path = Path(args.docx).resolve()
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    stats = apply_fixes(docx_path)
    payload = {
        "docx": str(docx_path),
        "stats": dict(sorted(stats.items())),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
