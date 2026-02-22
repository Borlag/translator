from __future__ import annotations

import math
from pathlib import Path

from .pdf_models import BBox, PdfPage, PdfSpan, PdfSpanStyle, PdfTextBlock

try:
    import fitz  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency at runtime
    fitz = None


def _require_fitz():
    if fitz is None:  # pragma: no cover - runtime guard
        raise RuntimeError("PyMuPDF is required for PDF translation. Install extras: pip install -e '.[pdf]'")
    return fitz


def _coerce_bbox(value: object) -> BBox:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    return (0.0, 0.0, 0.0, 0.0)


def _decode_pdf_color(value: object) -> tuple[int, int, int] | None:
    # PyMuPDF stores span color as an integer 0xRRGGBB.
    try:
        color = int(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if color < 0:
        return None
    r = (color >> 16) & 0xFF
    g = (color >> 8) & 0xFF
    b = color & 0xFF
    return (r, g, b)


def _span_bold(font_name: str, flags: int) -> bool:
    key = font_name.lower()
    if "bold" in key or "black" in key:
        return True
    return bool(flags & 16)


def _span_italic(font_name: str, flags: int) -> bool:
    key = font_name.lower()
    if "italic" in key or "oblique" in key:
        return True
    return bool(flags & 2)


def _rotation_deg(line_dict: dict[str, object]) -> float:
    direction = line_dict.get("dir")
    if not isinstance(direction, (list, tuple)) or len(direction) < 2:
        return 0.0
    try:
        dx = float(direction[0])
        dy = float(direction[1])
    except Exception:
        return 0.0
    if not dx and not dy:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def _extract_drawings(page) -> list[BBox]:
    out: list[BBox] = []
    try:
        drawings = page.get_drawings() or []
    except Exception:
        return out
    for item in drawings:
        rect = item.get("rect") if isinstance(item, dict) else None
        if rect is None:
            continue
        try:
            out.append((float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)))
        except Exception:
            continue
    return out


def extract_pdf_pages(pdf_path: str | Path, *, max_pages: int | None = None) -> list[PdfPage]:
    fitz_mod = _require_fitz()
    path = Path(pdf_path)
    pages: list[PdfPage] = []
    with fitz_mod.open(str(path)) as doc:
        page_limit = len(doc)
        if max_pages is not None:
            page_limit = max(0, min(page_limit, int(max_pages)))

        for page_number in range(page_limit):
            page = doc[page_number]
            payload = page.get_text("dict")
            blocks: list[PdfTextBlock] = []
            has_text = False

            for block in payload.get("blocks", []):
                if int(block.get("type", -1)) != 0:
                    # 0 == text block, other values are images / vector payload.
                    continue

                spans: list[PdfSpan] = []
                lines_text: list[str] = []
                for line in block.get("lines", []):
                    rotation = _rotation_deg(line)
                    line_parts: list[str] = []
                    for span in line.get("spans", []):
                        text = str(span.get("text", ""))
                        if not text:
                            continue
                        if text.strip():
                            has_text = True

                        font_name = str(span.get("font", "Helvetica"))
                        flags = int(span.get("flags", 0))
                        style = PdfSpanStyle(
                            font_name=font_name,
                            font_size_pt=float(span.get("size", 10.0)),
                            color_rgb=_decode_pdf_color(span.get("color")),
                            bold=_span_bold(font_name, flags),
                            italic=_span_italic(font_name, flags),
                        )
                        spans.append(
                            PdfSpan(
                                text=text,
                                bbox=_coerce_bbox(span.get("bbox")),
                                style=style,
                                rotation_deg=rotation,
                            )
                        )
                        line_parts.append(text)
                    if line_parts:
                        lines_text.append("".join(line_parts))

                block_text = "\n".join(lines_text).strip("\n")
                if not block_text.strip():
                    continue

                blocks.append(
                    PdfTextBlock(
                        block_id=len(blocks),
                        bbox=_coerce_bbox(block.get("bbox")),
                        text=block_text,
                        spans=spans,
                    )
                )

            pages.append(
                PdfPage(
                    page_number=page_number,
                    width_pt=float(page.rect.width),
                    height_pt=float(page.rect.height),
                    has_text=has_text,
                    blocks=blocks,
                    drawing_bboxes=_extract_drawings(page),
                )
            )
    return pages

