from __future__ import annotations

import re

from .pdf_models import FontSpec, PdfSpanStyle

_MONO_HINTS = ("mono", "courier", "consolas", "menlo", "dejavusansmono")
_SERIF_HINTS = ("serif", "times", "cambria", "garamond", "georgia")
_SANS_HINTS = ("sans", "arial", "helvetica", "noto", "ptsans", "ubuntu")


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").strip().lower())


def _classify_family(original_font: str) -> str:
    key = _normalize(original_font)
    if any(hint in key for hint in _MONO_HINTS):
        return "mono"
    if any(hint in key for hint in _SERIF_HINTS):
        return "serif"
    if any(hint in key for hint in _SANS_HINTS):
        return "sans"
    return "sans"


def _choose_default_font_file(
    family: str,
    *,
    default_sans_font: str,
    default_serif_font: str,
    default_mono_font: str,
) -> str:
    if family == "serif":
        return default_serif_font
    if family == "mono":
        return default_mono_font
    return default_sans_font


def _variant_font_file(base_file: str, *, bold: bool, italic: bool) -> str:
    if not base_file:
        return base_file
    stem, dot, suffix = base_file.partition(".")
    suffix = f".{suffix}" if dot else ""
    # Best-effort variant swap for common naming conventions.
    if re.search(r"-(regular|roman)$", stem, flags=re.IGNORECASE):
        stem = re.sub(r"-(regular|roman)$", "", stem, flags=re.IGNORECASE)
    if bold and italic:
        return f"{stem}-BoldItalic{suffix}"
    if bold:
        return f"{stem}-Bold{suffix}"
    if italic:
        return f"{stem}-Italic{suffix}"
    return base_file


def _mapped_family(original_font: str, font_map: dict[str, str]) -> str | None:
    if not font_map:
        return None
    norm_original = _normalize(original_font)
    for source, target in font_map.items():
        if _normalize(source) == norm_original:
            return target
    for source, target in font_map.items():
        if _normalize(source) and _normalize(source) in norm_original:
            return target
    return None


def select_replacement_font(
    original_font: str,
    is_bold: bool,
    is_italic: bool,
    *,
    font_map: dict[str, str] | None = None,
    default_sans_font: str = "NotoSans-Regular.ttf",
    default_serif_font: str = "NotoSerif-Regular.ttf",
    default_mono_font: str = "NotoSansMono-Regular.ttf",
    color_rgb: tuple[int, int, int] | None = None,
) -> FontSpec:
    mapped = _mapped_family(original_font, font_map or {})
    if mapped:
        base_file = mapped
        # If mapping provides a file path, use it as family too for CSS fallback.
        family = mapped.rsplit(".", 1)[0]
    else:
        family_type = _classify_family(original_font)
        family = {
            "serif": "Noto Serif",
            "mono": "Noto Sans Mono",
            "sans": "Noto Sans",
        }[family_type]
        base_file = _choose_default_font_file(
            family_type,
            default_sans_font=default_sans_font,
            default_serif_font=default_serif_font,
            default_mono_font=default_mono_font,
        )

    return FontSpec(
        family=family,
        color_rgb=color_rgb,
        bold=bool(is_bold),
        italic=bool(is_italic),
        font_file=_variant_font_file(base_file, bold=bool(is_bold), italic=bool(is_italic)),
    )


def select_font_for_style(
    style: PdfSpanStyle | None,
    *,
    font_map: dict[str, str] | None = None,
    default_sans_font: str = "NotoSans-Regular.ttf",
    default_serif_font: str = "NotoSerif-Regular.ttf",
    default_mono_font: str = "NotoSansMono-Regular.ttf",
) -> FontSpec:
    if style is None:
        return select_replacement_font(
            "Helvetica",
            False,
            False,
            font_map=font_map,
            default_sans_font=default_sans_font,
            default_serif_font=default_serif_font,
            default_mono_font=default_mono_font,
        )
    return select_replacement_font(
        style.font_name,
        style.bold,
        style.italic,
        font_map=font_map,
        default_sans_font=default_sans_font,
        default_serif_font=default_serif_font,
        default_mono_font=default_mono_font,
        color_rgb=style.color_rgb,
    )

