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

CANONICAL_COMPANY = "SAFRAN LANDING SYSTEMS UK Ltd КОД CAGE: K0654"
CANONICAL_PART = (
    "№ детали 201587001 и 201587002 "
    "РУКОВОДСТВО ПО ТЕХНИЧЕСКОМУ ОБСЛУЖИВАНИЮ КОМПОНЕНТОВ "
    "СТОЙКА ОСНОВНОГО ШАССИ"
)

CUSTOM_PHRASES: tuple[tuple[str, str], ...] = (
    ("LARGER VIEW AT", "УВЕЛИЧЕННЫЙ ВИД В ТОЧКЕ"),
    ("SPOT FACE", "ПОДРЕЗКА ПЛОЩАДКИ ПОД ГОЛОВКУ КРЕПЕЖА"),
    ("BOTH HOLES", "ОБА ОТВЕРСТИЯ"),
    ("BOTH SIDES", "ОБЕ СТОРОНЫ"),
    ("CHROMIUM PLATE DEPOSIT", "НАНЕСЕНИЕ ХРОМОВОГО ПОКРЫТИЯ"),
    ("CHROMIUM PLATE", "ХРОМОВОЕ ПОКРЫТИЕ"),
    ("NO PLATING", "БЕЗ ПОКРЫТИЯ"),
    ("CADMIUM LENGTH", "ДЛИНА УЧАСТКА КАДМИРОВАНИЯ"),
    ("LUBRICATION ADAPTOR", "АДАПТЕР ДЛЯ СМАЗКИ"),
    ("REFER TO FIGURE", "СМ. РИСУНОК"),
    ("INCLUDING CHAMFERS", "ВКЛЮЧАЯ ФАСКИ"),
    ("UPPER SLAVE LINK", "ВЕРХНЯЯ ВЕДОМАЯ ТЯГА"),
    ("LOWER SLAVE LINK", "НИЖНЯЯ ВЕДОМАЯ ТЯГА"),
    ("SLAVE LINK", "ВЕДОМАЯ ТЯГА"),
    ("TRANSFER BLOCK", "ПЕРЕДАТОЧНЫЙ БЛОК"),
    ("INFLATION VALVE", "КЛАПАН НАКАЧИВАНИЯ"),
    ("VALVE STEM", "ШТОК КЛАПАНА"),
    ("RETAINING PIN", "ФИКСИРУЮЩИЙ ШТИФТ"),
    ("UPPER DIAPHRAGM TUBE", "ВЕРХНЯЯ ДИАФРАГМЕННАЯ ТРУБКА"),
    ("MAIN FITTING", "КОРПУС СТОЙКИ"),
    ("SECTION", "СЕЧЕНИЕ"),
    ("DETAIL", "ДЕТАЛЬ"),
    ("VIEW", "ВИД"),
    ("SURFACE", "ПОВЕРХНОСТЬ"),
    ("HOLES", "ОТВЕРСТИЯ"),
    ("HOLE", "ОТВЕРСТИЕ"),
    ("PLACES", "МЕСТА"),
    ("TYPICAL", "ТИПОВОЕ"),
    ("MAXIMUM", "МАКСИМУМ"),
    ("MINIMUM", "МИНИМУМ"),
    ("DEPTH", "ГЛУБИНА"),
    ("DEEP", "ГЛУБИНА"),
    ("RADIUS", "РАДИУС"),
    ("DIA.", "ДИАМ."),
    ("DIA", "ДИАМ."),
    ("RAD.", "РАД."),
    ("RAD", "РАД."),
    ("REF.", "СПРАВ."),
    ("REF", "СПРАВ."),
)

EXACT_REPLACEMENTS = {
    "20,00mm87 дюйма) РАД.2 МЕСТА": "20,00 мм (0,787 дюйма) РАД. 2 МЕСТА",
    "20,00mm87 дюйма) РАД. 2 МЕСТА": "20,00 мм (0,787 дюйма) РАД. 2 МЕСТА",
    "0(0,000(0,000,50 до(0,020 до 0,029inТИПИЧНЫЙ РЕМОНТ SL": (
        "0,50 до 0,75 мм (0,020 до 0,029 дюйма) ТИПИЧНЫЙ РЕМОНТ SL"
    ),
    "10,25mmo 0,404 дюйма) METERECK": "10,25 мм (0,404 дюйма)",
    "51,00 до 51,03mm0,95 до 1,45 мм (2,008 до 2,009 дюйма ДИАМ.. (0,037 до 0,057 дюйма)": (
        "51,00 до 51,03 мм (2,008 до 2,009 дюйма) ДИАМ. "
        "0,95 до 1,45 мм (0,037 до 0,057 дюйма)"
    ),
    "7200 ACES": "PCS-7200",
}

CAPTION_RE = re.compile(r"^(?P<label>.+?)\s*Рисунок\s+(?P<fig>\d+)$", re.DOTALL)
FIGURE_DELETED_RE = re.compile(r"^Рисунок\s+удал[её]н\s+Рисунок\s+(\d+)$", re.IGNORECASE)
PAGE_RE = re.compile(r"^Страница\s+\d+\s+мар\s+18/2025$", re.IGNORECASE)
PART_RE = re.compile(r"201587001.*201587002", re.IGNORECASE | re.DOTALL)
COMPANY_RE = re.compile(r"SAFRAN\s+LANDING\s+SYSTEMS\s+UK\s+LTD", re.IGNORECASE)


def parse_glossary_pairs(glossary_text: str | None) -> list[tuple[str, str]]:
    if not glossary_text:
        return []

    pairs: list[tuple[str, str]] = []
    for raw_line in glossary_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("|") and line.endswith("|"):
            cols = [c.strip() for c in line.strip("|").split("|")]
            if len(cols) < 2:
                continue
            source, target = cols[0], cols[1]
            if source.lower() == "english" or set(source) == {"-"}:
                continue
            if not source or not target:
                continue
            pairs.append((source, target))
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return pairs


def compile_phrase_map(glossary_text: str | None) -> list[tuple[re.Pattern[str], str]]:
    phrase_pairs = list(CUSTOM_PHRASES)
    phrase_pairs.extend(parse_glossary_pairs(glossary_text))

    merged: dict[str, str] = {}
    for source, target in phrase_pairs:
        key = source.strip().lower()
        if key and key not in merged:
            merged[key] = target.strip()

    compiled: list[tuple[re.Pattern[str], str]] = []
    for source, target in sorted(merged.items(), key=lambda item: len(item[0]), reverse=True):
        escaped = re.escape(source)
        pattern = re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
        compiled.append((pattern, target))
    return compiled


def _fix_inches_number(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        number = match.group(1).replace(".", ",")
        return f"{number} дюйма"

    return re.sub(r"(\d+(?:[.,]\d+)?)\s*in\b", repl, text, flags=re.IGNORECASE)


def _fix_mm_number(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        number = match.group(1).replace(".", ",")
        return f"{number} мм"

    return re.sub(r"(\d+(?:[.,]\d+)?)\s*mm\b", repl, text, flags=re.IGNORECASE)


def _apply_phrase_replacements(text: str, replacements: list[tuple[re.Pattern[str], str]]) -> str:
    out = text
    if not any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in out):
        return out
    for pattern, replacement in replacements:
        out = pattern.sub(replacement, out)
    return out


def repair_text(text: str, phrase_replacements: list[tuple[re.Pattern[str], str]]) -> str:
    original = text
    out = text.replace("\t", " ")
    out = out.replace("\u00a0", " ")
    out = out.replace("—", "-")
    out = out.replace("–", "-")

    if out in EXACT_REPLACEMENTS:
        return EXACT_REPLACEMENTS[out]

    if COMPANY_RE.search(out) and "CAGE" in out.upper():
        return CANONICAL_COMPANY

    if PART_RE.search(out) and ("РУКОВОДСТВО" in out.upper() or "Руководство" in out):
        return CANONICAL_PART

    m_deleted = FIGURE_DELETED_RE.match(out.strip())
    if m_deleted:
        return f"Рисунок {m_deleted.group(1)} удалён"

    out = re.sub(r"\b№\s*детал[еий]+\b", "№ детали", out, flags=re.IGNORECASE)
    out = re.sub(r"\bНомер\s+детал[еий]+\b", "№ детали", out, flags=re.IGNORECASE)
    out = out.replace("No.", "№")
    out = re.sub(r"\bИ\b", "и", out)
    out = re.sub(r"\bКОД\s*CAGE\b", "КОД CAGE", out)
    out = re.sub(r"\bLtd\s*КОД\b", "Ltd КОД", out)

    out = _apply_phrase_replacements(out, phrase_replacements)
    out = _fix_mm_number(out)
    out = _fix_inches_number(out)

    out = re.sub(r"(?<![A-Za-z])0\.(\d)", r"0,\1", out)
    out = re.sub(r"(?<=\d)\.(?=\d)", ",", out)
    out = re.sub(r"\bДИАМ\b\.?", "ДИАМ.", out)
    out = re.sub(r"\bДИА\b\.?", "ДИА.", out)
    out = re.sub(r"\bРАД\b\.?", "РАД.", out)
    out = re.sub(r"\bСПРАВ\b\.?", "СПРАВ.", out)
    out = re.sub(r"\bТИПОВОЕ\b", "ТИПОВОЕ", out)
    out = re.sub(r"\bРАЗДЕЛ\b", "СЕЧЕНИЕ", out)
    out = re.sub(r"\bРАЗРЕЗ\b", "РАЗРЕЗ", out)

    out = re.sub(r"(?<=\b[A-ZА-Я])(?=ПОДРЕЗКА|РАДИУС|РАД\.|ОТВЕРСТИЕ|ПОВЕРХНОСТЬ|ВИД|СЕЧЕНИЕ|ДЕТАЛЬ)", " ", out)
    out = re.sub(r"(?<=\b[A-ZА-Я])(?=\d+[.,])", " ", out)
    out = re.sub(r"(?<=ДИАМЕТР)(?=[A-ZА-Я]\b)", " ", out)
    out = re.sub(r"(?<=ДИАМ\.)(?=[A-ZА-Я]\b)", " ", out)
    out = re.sub(r"(?<=ДИАМ\.)(?=\d)", " ", out)
    out = re.sub(r"(?<=РАД\.)(?=\d)", " ", out)
    out = re.sub(r"(?<=\d)(?=[A-ZА-Я][А-ЯA-Z])", " ", out)
    out = re.sub(r"(?<=\))(?=[A-ZА-Я])", " ", out)
    out = re.sub(r"(?<=[А-Яа-я])(?=[A-Z]\b)", " ", out)
    out = re.sub(r"(?<=[A-ZА-Я])\(", " (", out)
    out = re.sub(r"\)(?=\S)", ") ", out)
    out = re.sub(r"(?<=\d),(?=\d{3}\b)", ",", out)
    out = re.sub(r"\s*x\s*(?=\d)", " x ", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+-\s+", " - ", out)
    out = re.sub(r"\)\s*-\s*Защитная обработка", ") - Защитная обработка", out)
    out = re.sub(r"\s*Рисунок\s+", " Рисунок ", out)
    out = re.sub(r"\bТолько\b", "только", out)
    out = re.sub(r"\bОТВЕРСТИЯ\)\(", "ОТВЕРСТИЯ) (", out)
    out = re.sub(r"\bСЕЧЕНИЕ\s+СЕЧЕНИЕ\b", "СЕЧЕНИЕ", out)
    out = re.sub(r"\bДЕТАЛЬ\s+ДЕТАЛЬ\b", "ДЕТАЛЬ", out)
    out = out.replace("SERMETEL WПО", "SERMETEL W ПО")
    out = out.replace("ДОIFC", "ДО IFC")
    out = out.replace("ПОСЛЕ ШЛИФОВКИСЕЧЕНИЕ", "ПОСЛЕ ШЛИФОВКИ СЕЧЕНИЕ")
    out = out.replace("НАНЕСИТЕ ГЕРМЕТИК НА PCS - 72004 МЕСТА", "НАНЕСИТЕ ГЕРМЕТИК НА PCS-7200, 4 МЕСТА")
    out = out.replace("НАНЕСТИ ГЕРМЕТИК НА PCS - 7200", "НАНЕСТИ ГЕРМЕТИК НА PCS-7200")
    out = out.replace("ГЕРМЕТИК PCS 7200", "ГЕРМЕТИК PCS-7200")
    out = out.replace("Герметик по PCS - 7200", "Герметик по PCS-7200")
    out = out.replace("ДИАМ.4 МЕСТА", "ДИАМ. 4 МЕСТА")
    out = out.replace("ДИАМ.2 МЕСТА", "ДИАМ. 2 МЕСТА")
    out = out.replace("РАД.2 МЕСТА", "РАД. 2 МЕСТА")
    out = out.replace("ДИАМ.ПОДРЕЗКА", "ДИАМ. ПОДРЕЗКА")
    out = out.replace("C ОТВЕРСТИЕA", "C ОТВЕРСТИЕ A")
    out = out.replace("DНА ПОВЕРХНОСТИ", "D НА ПОВЕРХНОСТИ")
    out = out.replace("Ремонт № 11 - 2Страница", "Ремонт № 11-2 Страница")

    m_caption = CAPTION_RE.match(out.strip())
    if m_caption:
        label = re.sub(r"\s+", " ", m_caption.group("label")).strip()
        out = f"{label} Рисунок {m_caption.group('fig')}"

    out = re.sub(r"[ ]{2,}", " ", out).strip()
    if out in EXACT_REPLACEMENTS:
        return EXACT_REPLACEMENTS[out]

    if PAGE_RE.match(out):
        return original
    return out


def text_nodes_for_textbox(txbx: etree._Element) -> list[etree._Element]:
    return txbx.xpath(".//w:t", namespaces=NS)


def textbox_text(txbx: etree._Element) -> str:
    return "".join(node.text or "" for node in text_nodes_for_textbox(txbx)).strip()


def set_text(nodes: list[etree._Element], value: str) -> None:
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


def iter_word_xml_parts(docx_path: Path) -> list[str]:
    with zipfile.ZipFile(docx_path) as zf:
        names = [
            name
            for name in zf.namelist()
            if name.startswith("word/")
            and name.endswith(".xml")
            and (
                name == "word/document.xml"
                or Path(name).name.startswith("header")
                or Path(name).name.startswith("footer")
            )
        ]
    return names


def repair_docx(input_path: Path, output_path: Path, glossary_path: Path | None) -> Counter[str]:
    glossary_text = None
    if glossary_path and glossary_path.exists():
        glossary_text = glossary_path.read_text(encoding="utf-8-sig", errors="replace")
    phrase_replacements = compile_phrase_map(glossary_text)

    stats: Counter[str] = Counter()
    xml_parts = iter_word_xml_parts(input_path)

    with zipfile.ZipFile(input_path) as src, zipfile.ZipFile(output_path, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename not in xml_parts:
                dst.writestr(item, data)
                continue

            root = etree.fromstring(data)
            changed_in_part = 0
            for txbx in root.xpath(".//w:txbxContent", namespaces=NS):
                nodes = text_nodes_for_textbox(txbx)
                if not nodes:
                    continue
                current = textbox_text(txbx)
                if not current:
                    continue
                repaired = repair_text(current, phrase_replacements)
                if repaired != current:
                    set_text(nodes, repaired)
                    changed_in_part += 1
                    stats["textboxes_changed"] += 1
                    if repaired == CANONICAL_COMPANY:
                        stats["company_lines_fixed"] += 1
                    elif repaired == CANONICAL_PART:
                        stats["part_lines_fixed"] += 1
                    elif "Рисунок" in repaired:
                        stats["captions_fixed"] += 1
                    elif any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in current):
                        stats["latin_residue_fixed"] += 1
                    else:
                        stats["other_fixed"] += 1

            dst.writestr(item, etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes"))
            if changed_in_part:
                stats[f"part::{item.filename}"] = changed_in_part

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair OCR-like textbox text in a translated DOCX.")
    parser.add_argument("--input", required=True, help="Source DOCX")
    parser.add_argument("--output", required=True, help="Output DOCX")
    parser.add_argument("--glossary", default=None, help="Glossary markdown file")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    glossary_path = Path(args.glossary).resolve() if args.glossary else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = repair_docx(input_path, output_path, glossary_path)

    print(f"Repaired: {output_path}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
