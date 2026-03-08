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


def _insert_htmlbox(page, rect, html_text: str, *, css: str, archive, rotate: int, ocg_xref: int | None, scale_low: float):
    return page.insert_htmlbox(
        rect,
        html_text,
        css=css,
        archive=archive,
        rotate=rotate,
        oc=int(ocg_xref or 0),
        scale_low=scale_low,
        overlay=True,
    )


def _probe_insert_htmlbox(
    page,
    rect,
    html_text: str,
    *,
    css: str,
    archive,
    rotate: int,
    ocg_xref: int | None,
    scale_low: float,
) -> tuple[float, float]:
    fitz_mod = _require_fitz()
    page_rect = getattr(page, "rect", None)
    if page_rect is None:
        return _normalize_insert_result(
            _insert_htmlbox(
                page,
                rect,
                html_text,
                css=css,
                archive=archive,
                rotate=rotate,
                ocg_xref=ocg_xref,
                scale_low=scale_low,
            )
        )

    scratch = fitz_mod.open()
    try:
        scratch_page = scratch.new_page(width=float(page_rect.width), height=float(page_rect.height))
        result = _insert_htmlbox(
            scratch_page,
            rect,
            html_text,
            css=css,
            archive=archive,
            rotate=rotate,
            ocg_xref=ocg_xref,
            scale_low=scale_low,
        )
        return _normalize_insert_result(result)
    finally:
        scratch.close()


def replace_block_text(
    page,
    block: PdfTextBlock,
    translated_text: str,
    font_spec: FontSpec,
    *,
    inner_bbox: tuple[float, float, float, float] | None = None,
    text_align: str = "left",
    rotation_deg: float = 0.0,
    line_height_factor: float = 1.05,
    ocg_xref: int | None = None,
    max_font_shrink_ratio: float = 0.6,
    redact_original: bool = True,
    paint_background: bool = False,
) -> tuple[bool, float]:
    fitz_mod = _require_fitz()
    full_rect = fitz_mod.Rect(*block.bbox)
    if full_rect.is_empty or full_rect.width <= 0 or full_rect.height <= 0:
        return False, 0.0

    dominant_size = 10.0
    if block.dominant_style is not None:
        dominant_size = max(6.0, float(block.dominant_style.font_size_pt))

    archive, css_family = _build_font_archive(font_spec)
    align_css = {"left": "left", "center": "center", "right": "right"}.get(str(text_align).strip().lower(), "left")
    rotate = int(round(float(rotation_deg or 0.0))) % 360
    if rotate not in {0, 90, 180, 270}:
        rotate = 0
    line_height = max(0.85, min(1.6, float(line_height_factor or 1.05)))
    css_rules = [
        f"div {{ margin: 0; padding: 0; line-height: {line_height:.2f}; text-align: {align_css}; }}",
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

    candidate_rects = []
    if inner_bbox is not None:
        inner_rect = fitz_mod.Rect(*inner_bbox)
        if not inner_rect.is_empty and inner_rect.width > 0 and inner_rect.height > 0:
            candidate_rects.append(inner_rect)
    candidate_rects.append(full_rect)

    css_text = "\n".join(css_rules)
    scale_low = max(0.1, min(1.0, float(max_font_shrink_ratio)))
    last_scale = 0.0
    for rect in candidate_rects:
        if redact_original:
            spare_height, scale = _probe_insert_htmlbox(
                page,
                rect,
                html_text,
                css=css_text,
                archive=archive,
                rotate=rotate,
                ocg_xref=ocg_xref,
                scale_low=scale_low,
            )
        else:
            spare_height, scale = _normalize_insert_result(
                _insert_htmlbox(
                    page,
                    rect,
                    html_text,
                    css=css_text,
                    archive=archive,
                    rotate=rotate,
                    ocg_xref=ocg_xref,
                    scale_low=scale_low,
                )
            )
        last_scale = float(scale)
        if spare_height < 0.0:
            continue
        if redact_original:
            page.add_redact_annot(full_rect, fill=(1, 1, 1))
            page.apply_redactions()
            result = _insert_htmlbox(
                page,
                rect,
                html_text,
                css=css_text,
                archive=archive,
                rotate=rotate,
                ocg_xref=ocg_xref,
                scale_low=scale_low,
            )
            spare_height, scale = _normalize_insert_result(result)
            last_scale = float(scale)
            if spare_height < 0.0:
                return False, last_scale
        return True, float(scale)
    return False, last_scale

