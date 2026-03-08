from __future__ import annotations

import argparse
import csv
import io
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import translate_picture_docx as text_translator


SUSPICIOUS_RE = re.compile(
    r"\b(?:VIEW|DETAIL|HOLE|HOLES|DIAMETER|DIAMETERS|INTERNALLY|SPOTFACE|PAINT|NO|CADMIUM|PLATE|"
    r"CHAMFER|THIS|FACE|ONLY|PLACES|ARROW|BOTH|SHEET|REPAIR|BUSH|MACHINING|SECTION|FIG\.?|REFER|"
    r"APPLY|SEALANT|SURFACE|DEPTH|FROM|INNER|OUTER|BORE|BORES|THROUGH|TYP(?:ICAL)?|PERMITTED|"
    r"NITRIDING|HONE)\b",
    re.IGNORECASE,
)

ALLOWED_LATIN_RE = re.compile(r"\b(?:PCS|IFC|HV|PR\d+|PR|MOLYKOTE|A\d+[A-Z0-9-]*|[A-Z]{1,3})\b")

OCR_NORMALIZATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bPES\b", re.IGNORECASE), "PCS"),
    (re.compile(r"\b2-7\b", re.IGNORECASE), "Z-Z"),
    (re.compile(r"\b2-2\b", re.IGNORECASE), "Z-Z"),
    (re.compile(r"\bSECTION\s+2\s*-\s*7\b", re.IGNORECASE), "SECTION Z-Z"),
    (re.compile(r"\bSECTION\s+2\s*-\s*2\b", re.IGNORECASE), "SECTION Z-Z"),
    (re.compile(r"\bVIEW\s+[\\/|]\b", re.IGNORECASE), "VIEW Y"),
    (re.compile(r"\bDIA\.\s*B[-_]+\b", re.IGNORECASE), "DIA. B"),
    (re.compile(r"\s+"), " "),
)

MANUAL_PATCHES: dict[str, list[tuple[tuple[int, int, int, int], str]]] = {
    "image2634.png": [
        ((2640, 3928, 2985, 4018), "ВИД Y"),
    ],
    "image2636.png": [
        ((100, 1120, 590, 1215), "СЕЧЕНИЕ T"),
        ((1020, 1135, 1995, 1418), "СЕЧЕНИЕ U"),
        ((0, 1720, 325, 1778), "3 МЕСТА"),
        ((2090, 2870, 2610, 2965), "СЕЧЕНИЕ W"),
        ((250, 3205, 745, 3298), "СЕЧЕНИЕ V"),
        ((195, 4470, 690, 4565), "СЕЧЕНИЕ Y"),
        ((3120, 4440, 3610, 4532), "СЕЧЕНИЕ X"),
        ((1870, 4630, 2355, 4720), "СЕЧЕНИЕ Z"),
    ],
    "image6226.png": [
        ((2140, 1418, 2340, 1458), "4 ОТВ."),
        ((1590, 1510, 1948, 1608), "0,20 мм\n(0,008 дюйма)\nДИАМ."),
        ((1510, 1660, 1845, 1898), "1,915 мм\n(0,0754 дюйма)\nМИН. ТОЛЩИНА\nСТЕНКИ\n4 МЕСТА"),
        ((2110, 1680, 2472, 1918), "0,50 до 1,00 мм\n(0,020 до 0,040 дюйма)\nСНЯТЬ КРОМКИ\nРАДИУС\n4 МЕСТА"),
        ((1450, 2430, 1665, 2748), "ДИАМ. A\n16,17 мм\n(0,6366 дюйма)\nМАКСИМУМ\n4 ОТВ.\nБЕЗ КРАСКИ"),
        ((2100, 2405, 2462, 2555), "СФЕРИЧ. РАД.\n4 МЕСТА\nБЕЗ КРАСКИ"),
        ((1785, 2620, 2148, 2856), "1,00 до 2,00 мм\n(0,040 до 0,080 дюйма)\nСНЯТЬ КРОМКИ\nРАДИУС\n4 МЕСТА"),
    ],
}


def run_tesseract(image_path: Path, *, tsv: bool = False) -> str:
    cmd = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        str(image_path),
        "stdout",
        "-l",
        "eng+rus",
        "--psm",
        "11",
    ]
    if tsv:
        cmd.append("tsv")
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=True,
    ).stdout


def normalize_ocr_text(text: str) -> str:
    out = text.strip()
    out = out.replace("—", "-").replace("–", "-")
    out = out.replace("[", "").replace("]", "")
    for pattern, replacement in OCR_NORMALIZATIONS:
        out = pattern.sub(replacement, out)
    return out.strip(" -_,.;:")


def image_suspicion_score(text: str) -> int:
    return len(SUSPICIOUS_RE.findall(text))


def should_translate_line(text: str) -> bool:
    if not text or not re.search(r"[A-Za-z]", text):
        return False
    cleaned = ALLOWED_LATIN_RE.sub("", text)
    cleaned = re.sub(r"\b\d[\d,./()xX-]*\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return bool(re.search(r"[A-Za-z]{2,}", cleaned))


def compact_translation(text: str, patterns: list[tuple[re.Pattern[str], str]]) -> str:
    translated = text_translator.translate_text(text, patterns)
    translated = translated.replace("\n", " ")
    translated = re.sub(r"\bМИНИМАЛЬНАЯ\b", "МИН.", translated)
    translated = re.sub(r"\bСУММАРНАЯ\b", "СУММ.", translated)
    translated = re.sub(r"\bПОВЕРХНОСТИ\b", "ПОВ.", translated)
    translated = re.sub(r"\bПОВЕРХНОСТЬ\b", "ПОВ.", translated)
    translated = re.sub(r"\s+", " ", translated).strip()
    return translated


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return [text]
    words = text.split()
    if len(words) <= 1:
        return [text]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def fit_text(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int]) -> tuple[ImageFont.ImageFont, list[str]]:
    left, top, right, bottom = box
    max_width = max(10, right - left)
    max_height = max(10, bottom - top)
    for size in range(max_height + 8, 7, -1):
        font = load_font(size)
        lines = wrap_text(draw, text, font, max_width)
        bbox = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=0, align="center")
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= max_width and height <= max_height:
            return font, lines
    font = load_font(8)
    return font, wrap_text(draw, text, font, max_width)


def draw_patch(draw: ImageDraw.ImageDraw, image: Image.Image, box: tuple[int, int, int, int], text: str) -> None:
    left, top, right, bottom = box
    left = max(0, left)
    top = max(0, top)
    right = min(image.width, right)
    bottom = min(image.height, bottom)
    box = (left, top, right, bottom)
    font, lines = fit_text(draw, text, box)
    draw.rectangle(box, fill="white")
    text_value = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), text_value, font=font, spacing=0, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = left + max(0, ((right - left) - text_w) / 2)
    y = top + max(0, ((bottom - top) - text_h) / 2)
    draw.multiline_text((x, y), text_value, fill="black", font=font, spacing=0, align="center")


def extract_line_boxes(image_path: Path) -> list[dict[str, int | str]]:
    tsv = run_tesseract(image_path, tsv=True)
    rows = list(csv.DictReader(io.StringIO(tsv), delimiter="\t"))
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("level") != "5":
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        key = (row["block_num"], row["par_num"], row["line_num"])
        groups.setdefault(key, []).append(row)

    lines: list[dict[str, int | str]] = []
    for words in groups.values():
        left = min(int(w["left"]) for w in words)
        top = min(int(w["top"]) for w in words)
        right = max(int(w["left"]) + int(w["width"]) for w in words)
        bottom = max(int(w["top"]) + int(w["height"]) for w in words)
        text = " ".join(w["text"] for w in words)
        lines.append({"text": text, "left": left, "top": top, "right": right, "bottom": bottom})
    lines.sort(key=lambda item: (int(item["top"]), int(item["left"])))
    return lines


def x_overlap_ratio(a: dict[str, int | str], b: dict[str, int | str]) -> float:
    left = max(int(a["left"]), int(b["left"]))
    right = min(int(a["right"]), int(b["right"]))
    overlap = max(0, right - left)
    width = max(1, min(int(a["right"]) - int(a["left"]), int(b["right"]) - int(b["left"])))
    return overlap / width


def choose_translation_group(
    lines: list[dict[str, int | str]],
    start: int,
    patterns: list[tuple[re.Pattern[str], str]],
) -> tuple[int, str] | None:
    best: tuple[int, str] | None = None
    texts: list[str] = []
    group_box = lines[start]
    for end in range(start, min(len(lines), start + 4)):
        current = lines[end]
        if end > start:
            prev = lines[end - 1]
            gap = int(current["top"]) - int(prev["bottom"])
            if gap > 70 or x_overlap_ratio(group_box, current) < 0.35:
                break
        texts.append(normalize_ocr_text(str(current["text"])))
        candidate = " ".join(texts).strip()
        if not should_translate_line(candidate):
            continue
        translated = compact_translation(candidate, patterns)
        if translated == candidate:
            continue
        if should_translate_line(translated):
            continue
        best = (end, translated)
        group_box = {
            "left": min(int(group_box["left"]), int(current["left"])),
            "top": min(int(group_box["top"]), int(current["top"])),
            "right": max(int(group_box["right"]), int(current["right"])),
            "bottom": max(int(group_box["bottom"]), int(current["bottom"])),
        }
    return best


def paint_translations(image_path: Path, patterns: list[tuple[re.Pattern[str], str]]) -> int:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    changed = 0
    for line in extract_line_boxes(image_path):
        raw_text = str(line["text"])
        cleaned = normalize_ocr_text(raw_text)
        if not should_translate_line(cleaned):
            continue
        translated = compact_translation(cleaned, patterns)
        if translated == cleaned:
            continue
        if should_translate_line(translated):
            continue
        box = (
            max(0, int(line["left"]) - 4),
            max(0, int(line["top"]) - 3),
            min(image.width, int(line["right"]) + 4),
            min(image.height, int(line["bottom"]) + 3),
        )
        draw_patch(draw, image, box, translated)
        changed += 1

    for box, text in MANUAL_PATCHES.get(image_path.name, []):
        draw_patch(draw, image, box, text)
        changed += 1

    if changed:
        image.save(image_path)
    return changed


def suspicious_media(docx_path: Path) -> list[str]:
    results: list[str] = []
    with zipfile.ZipFile(docx_path) as zin, tempfile.TemporaryDirectory() as td:
        for name in zin.namelist():
            if not name.startswith("word/media/") or not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            temp = Path(td) / Path(name).name
            temp.write_bytes(zin.read(name))
            try:
                width, height = Image.open(temp).size
            except Exception:
                continue
            if width >= 750 and height >= 2000:
                results.append(name)
    return results


def patch_docx(input_path: Path, output_path: Path, glossary_path: Path) -> tuple[int, int]:
    patterns = text_translator.build_phrase_patterns(glossary_path.read_text(encoding="utf-8"))
    targets = suspicious_media(input_path)
    changed_images = 0
    changed_lines = 0
    replacements: dict[str, bytes] = {}

    with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(input_path) as zin:
        for name in targets:
            temp = Path(td) / Path(name).name
            temp.write_bytes(zin.read(name))
            count = paint_translations(temp, patterns)
            if count:
                replacements[name] = temp.read_bytes()
                changed_images += 1
                changed_lines += count

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = replacements.get(item.filename, zin.read(item.filename))
                zout.writestr(item, data)

    return changed_images, changed_lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glossary", type=Path, required=True)
    args = parser.parse_args()

    changed_images, changed_lines = patch_docx(args.input, args.output, args.glossary)
    print(f"images changed: {changed_images}")
    print(f"lines repainted: {changed_lines}")


if __name__ == "__main__":
    main()
