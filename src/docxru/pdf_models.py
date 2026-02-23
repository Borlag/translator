from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import Issue

BBox = tuple[float, float, float, float]


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


@dataclass
class PdfTextBlock:
    block_id: int
    bbox: BBox
    text: str
    spans: list[PdfSpan] = field(default_factory=list)
    block_type: str = "body"  # body | header | footer | table_cell
    column_index: int = 0
    table_region_id: int | None = None

    @property
    def dominant_style(self) -> PdfSpanStyle | None:
        if self.spans:
            return self.spans[0].style
        return None


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
    target_text: str | None = None
    dominant_style: PdfSpanStyle | None = None
    context: dict[str, Any] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

