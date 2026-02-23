from __future__ import annotations

from docxru.pdf_layout import detect_columns, detect_table_regions, group_all_pages, group_blocks_into_segments
from docxru.pdf_models import PdfPage, PdfSpan, PdfSpanStyle, PdfTextBlock


def _block(block_id: int, *, x0: float, y0: float, x1: float, y1: float, text: str) -> PdfTextBlock:
    style = PdfSpanStyle(font_name="Arial", font_size_pt=10.0)
    return PdfTextBlock(
        block_id=block_id,
        bbox=(x0, y0, x1, y1),
        text=text,
        spans=[PdfSpan(text=text, bbox=(x0, y0, x1, y1), style=style)],
    )


def test_group_blocks_into_segments_merges_adjacent_body_blocks():
    page = PdfPage(
        page_number=0,
        width_pt=600,
        height_pt=800,
        has_text=True,
        blocks=[
            _block(0, x0=80, y0=100, x1=300, y1=120, text="Install"),
            _block(1, x0=82, y0=126, x1=305, y1=145, text="bolt"),
            _block(2, x0=80, y0=260, x1=300, y1=280, text="Tighten"),
        ],
    )
    segments = group_blocks_into_segments(page, block_merge_threshold_pt=12.0)
    assert len(segments) == 2
    assert segments[0].source_text == "Install\nbolt"
    assert segments[1].source_text == "Tighten"


def test_detect_columns_finds_two_left_edge_clusters():
    page = PdfPage(
        page_number=0,
        width_pt=600,
        height_pt=800,
        has_text=True,
        blocks=[
            _block(0, x0=60, y0=100, x1=240, y1=120, text="L1"),
            _block(1, x0=65, y0=140, x1=250, y1=160, text="L2"),
            _block(2, x0=340, y0=100, x1=520, y1=120, text="R1"),
            _block(3, x0=345, y0=140, x1=525, y1=160, text="R2"),
        ],
    )
    columns = detect_columns(page, x_threshold_pt=40.0)
    assert len(columns) == 2
    assert columns[0].x_min < columns[1].x_min


def test_detect_columns_ignores_header_footer_noise_clusters():
    page = PdfPage(
        page_number=0,
        width_pt=600,
        height_pt=800,
        has_text=True,
        blocks=[
            _block(0, x0=55, y0=120, x1=545, y1=140, text="Main body line 1 with long text"),
            _block(1, x0=55, y0=155, x1=545, y1=175, text="Main body line 2 with long text"),
            _block(2, x0=55, y0=190, x1=545, y1=210, text="Main body line 3 with long text"),
            _block(3, x0=330, y0=60, x1=520, y1=82, text="Header metadata"),
            _block(4, x0=480, y0=740, x1=560, y1=760, text="Mar 18/2025"),
        ],
    )
    columns = detect_columns(page)
    assert len(columns) == 1


def test_group_all_pages_skips_repeated_margin_headers():
    pages: list[PdfPage] = []
    for idx in range(3):
        pages.append(
            PdfPage(
                page_number=idx,
                width_pt=600,
                height_pt=800,
                has_text=True,
                blocks=[
                    _block(0, x0=210, y0=120, x1=430, y1=142, text="TABLE OF CONTENTS"),
                    _block(1, x0=60, y0=280, x1=540, y1=305, text=f"Body text page {idx + 1}"),
                ],
            )
        )

    segments = group_all_pages(
        pages,
        block_merge_threshold_pt=12.0,
        skip_headers_footers=True,
        table_detection=False,
    )
    assert len(segments) == 3
    assert all("Body text page" in seg.source_text for seg in segments)


def test_detect_table_regions_from_grid_like_blocks():
    page = PdfPage(
        page_number=0,
        width_pt=600,
        height_pt=800,
        has_text=True,
        drawing_bboxes=[(80, 200, 520, 360)],
        blocks=[
            _block(0, x0=100, y0=220, x1=260, y1=240, text="Part No."),
            _block(1, x0=300, y0=220, x1=500, y1=240, text="Description"),
            _block(2, x0=100, y0=270, x1=260, y1=290, text="201587001"),
            _block(3, x0=300, y0=270, x1=500, y1=290, text="Main fitting assembly"),
        ],
    )
    regions = detect_table_regions(page)
    assert len(regions) == 1
