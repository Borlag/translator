from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import Issue

BBox = tuple[float, float, float, float]


def _union_bboxes(bboxes: list[BBox]) -> BBox:
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


@dataclass(frozen=True)
class PdfSpanStyle:
    font_name: str
    font_size_pt: float
    color_rgb: tuple[int, int, int] | None = None
    bold: bool = False
    italic: bool = False


@dataclass(frozen=True)
class PdfSpan:
    text: str
    bbox: BBox
    style: PdfSpanStyle
    rotation_deg: float = 0.0


@dataclass(frozen=True)
class PdfTextLine:
    text: str
    bbox: BBox
    rotation_deg: float = 0.0


@dataclass
class PdfTextBlock:
    block_id: int
    bbox: BBox
    text: str
    spans: list[PdfSpan] = field(default_factory=list)
    lines: list[PdfTextLine] = field(default_factory=list)
    block_type: str = "body"  # body | header | footer | table_cell
    column_index: int = 0
    table_region_id: int | None = None

    @property
    def dominant_style(self) -> PdfSpanStyle | None:
        if self.spans:
            return self.spans[0].style
        return None

    @property
    def content_bbox(self) -> BBox:
        if self.lines:
            bboxes = [line.bbox for line in self.lines]
            return _union_bboxes(bboxes)
        if self.spans:
            return _union_bboxes([span.bbox for span in self.spans])
        return self.bbox

    @property
    def rotation_deg(self) -> float:
        if self.lines:
            return float(self.lines[0].rotation_deg)
        if self.spans:
            return float(self.spans[0].rotation_deg)
        return 0.0


@dataclass
class PdfPage:
    page_number: int
    width_pt: float
    height_pt: float
    has_text: bool
    blocks: list[PdfTextBlock] = field(default_factory=list)
    drawing_bboxes: list[BBox] = field(default_factory=list)


@dataclass(frozen=True)
class TableRegion:
    region_id: int
    bbox: BBox


@dataclass(frozen=True)
class ColumnRegion:
    column_index: int
    x_min: float
    x_max: float


@dataclass(frozen=True)
class FontSpec:
    family: str
    color_rgb: tuple[int, int, int] | None = None
    bold: bool = False
    italic: bool = False
    font_file: str | None = None


@dataclass
class PdfSegment:
    segment_id: str
    page_number: int
    block_ids: list[int]
    bbox: BBox
    source_text: str
    inner_bbox: BBox | None = None
    target_text: str | None = None
    dominant_style: PdfSpanStyle | None = None
    text_align: str = "left"
    rotation_deg: float = 0.0
    line_height_factor: float = 1.15
    max_target_chars: int | None = None
    context: dict[str, Any] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

