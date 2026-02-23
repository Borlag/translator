from __future__ import annotations

import html
from pathlib import Path

from .pdf_models import FontSpec, PdfTextBlock

try:
    import fitz  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency at runtime
    fitz = None


def _require_fitz():
    if fitz is None:  # pragma: no cover - runtime guard
        raise RuntimeError("PyMuPDF is required for PDF translation. Install extras: pip install -e '.[pdf]'")
    return fitz


def _rgb_css(color: tuple[int, int, int] | None) -> str:
    if not color:
        return "rgb(0, 0, 0)"
    r, g, b = color
    return f"rgb({int(r)}, {int(g)}, {int(b)})"


def _normalize_insert_result(result: object) -> tuple[float, float]:
    # PyMuPDF docs declare tuple, while stubs may still report float.
    if isinstance(result, tuple) and len(result) >= 2:
        return float(result[0]), float(result[1])
    try:
        spare = float(result)  # type: ignore[arg-type]
    except Exception:
        return -1.0, 0.0
    if spare < 0:
        return spare, 0.0
    return spare, 1.0


def _build_font_archive(font_spec: FontSpec):
    fitz_mod = _require_fitz()
    if not font_spec.font_file:
        return None, font_spec.family
    path = Path(font_spec.font_file)
    if not path.exists():
        return None, font_spec.family
    archive = fitz_mod.Archive(str(path.parent))
    family = path.stem
    return archive, family


def build_bilingual_ocg(doc, *, layer_name: str = "Russian Translation") -> int:
    _require_fitz()
    return int(doc.add_ocg(layer_name, on=1))


def replace_block_text(
    page,
    block: PdfTextBlock,
    translated_text: str,
    font_spec: FontSpec,
    *,
    ocg_xref: int | None = None,
    max_font_shrink_ratio: float = 0.6,
    redact_original: bool = True,
    paint_background: bool = False,
) -> tuple[bool, float]:
    fitz_mod = _require_fitz()
    rect = fitz_mod.Rect(*block.bbox)
    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
        return False, 0.0

    if redact_original:
        page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()

    dominant_size = 10.0
    if block.dominant_style is not None:
        dominant_size = max(6.0, float(block.dominant_style.font_size_pt))

    archive, css_family = _build_font_archive(font_spec)
    css_rules = [
        "div { margin: 0; padding: 0; line-height: 1.15; }",
        f"div {{ font-family: '{css_family}', Helvetica, Arial, sans-serif; }}",
        f"div {{ font-size: {dominant_size:.2f}pt; }}",
        f"div {{ color: {_rgb_css(font_spec.color_rgb)}; }}",
        "div { font-weight: 700; }" if font_spec.bold else "div { font-weight: 400; }",
        "div { font-style: italic; }" if font_spec.italic else "div { font-style: normal; }",
    ]
    if paint_background:
        css_rules.append("div { background-color: rgb(255, 255, 255); }")

    content = html.escape(translated_text or "", quote=False).replace("\n", "<br/>")
    html_text = f"<div>{content}</div>"

    result = page.insert_htmlbox(
        rect,
        html_text,
        css="\n".join(css_rules),
        archive=archive,
        oc=int(ocg_xref or 0),
        scale_low=max(0.1, min(1.0, float(max_font_shrink_ratio))),
        overlay=True,
    )
    spare_height, scale = _normalize_insert_result(result)
    return (spare_height >= 0.0), float(scale)

