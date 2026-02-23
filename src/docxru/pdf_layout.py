from __future__ import annotations

import re
from collections import defaultdict
from statistics import median

from .pdf_models import BBox, ColumnRegion, PdfPage, PdfSegment, PdfTextBlock, TableRegion

_SPACE_RE = re.compile(r"\s+")
_NON_TEXT_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё:/()., -]+")


def _bbox_union(bboxes: list[BBox]) -> BBox:
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def _bbox_intersects(a: BBox, b: BBox) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _bbox_area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _cluster_count(values: list[float], *, tolerance: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    clusters = 1
    anchor = ordered[0]
    for value in ordered[1:]:
        if abs(value - anchor) <= tolerance:
            anchor = (anchor + value) / 2.0
            continue
        clusters += 1
        anchor = value
    return clusters


def _merge_regions(regions: list[BBox]) -> list[BBox]:
    merged: list[BBox] = []
    for bbox in sorted(regions, key=lambda r: _bbox_area(r), reverse=True):
        inserted = False
        for i, prev in enumerate(merged):
            if not _bbox_intersects(prev, bbox):
                continue
            union = _bbox_union([prev, bbox])
            # Merge only near-overlapping regions; avoid swallowing full-page figure drawings.
            overlap_area = min(_bbox_area(prev), _bbox_area(bbox))
            union_area = _bbox_area(union)
            if union_area <= 0:
                continue
            if overlap_area / union_area < 0.18:
                continue
            merged[i] = union
            inserted = True
            break
        if not inserted:
            merged.append(bbox)
    return merged


def detect_table_regions(page: PdfPage) -> list[TableRegion]:
    if not page.drawing_bboxes:
        return []
    page_area = max(1.0, page.width_pt * page.height_pt)
    candidates: list[BBox] = []
    for bbox in page.drawing_bboxes:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = _bbox_area(bbox)
        if width < 40 or height < 20 or area <= 0:
            continue
        if area / page_area > 0.68:
            continue
        if (max(width, height) / max(1.0, min(width, height))) > 35:
            continue

        hit_blocks: list[PdfTextBlock] = [block for block in page.blocks if _bbox_intersects(block.bbox, bbox)]
        if len(hit_blocks) < 3:
            continue

        centers_x = [((block.bbox[0] + block.bbox[2]) / 2.0) for block in hit_blocks]
        centers_y = [((block.bbox[1] + block.bbox[3]) / 2.0) for block in hit_blocks]
        cols = _cluster_count(centers_x, tolerance=max(14.0, width * 0.04))
        rows = _cluster_count(centers_y, tolerance=max(10.0, height * 0.03))
        if cols < 2 or rows < 2:
            continue

        text_area = sum(_bbox_area(block.bbox) for block in hit_blocks)
        text_fill_ratio = text_area / max(1.0, area)
        if text_fill_ratio < 0.035:
            continue

        candidates.append(bbox)

    merged = _merge_regions(candidates)
    return [TableRegion(region_id=i, bbox=bbox) for i, bbox in enumerate(merged)]


def _primary_column_candidates(page: PdfPage) -> list[PdfTextBlock]:
    top = page.height_pt * 0.08
    bottom = page.height_pt * 0.92
    out: list[PdfTextBlock] = []
    for block in page.blocks:
        width = block.bbox[2] - block.bbox[0]
        text = _SPACE_RE.sub(" ", block.text or "").strip()
        alpha_chars = sum(1 for ch in text if ch.isalpha())
        if alpha_chars < 2:
            continue
        if block.bbox[1] < top or block.bbox[3] > bottom:
            continue
        if width < page.width_pt * 0.16 and len(text) < 18:
            continue
        out.append(block)
    if len(out) >= 4:
        return out
    # Fallback for compact synthetic/test pages.
    fallback: list[PdfTextBlock] = []
    for block in page.blocks:
        width = block.bbox[2] - block.bbox[0]
        text = _SPACE_RE.sub(" ", block.text or "").strip()
        alpha_chars = sum(1 for ch in text if ch.isalpha())
        if alpha_chars < 1:
            continue
        if width < page.width_pt * 0.12 and len(text) < 8:
            continue
        fallback.append(block)
    return fallback


def _single_column_region(page: PdfPage) -> list[ColumnRegion]:
    if not page.blocks:
        return [ColumnRegion(column_index=0, x_min=0.0, x_max=page.width_pt)]
    x_min = min(block.bbox[0] for block in page.blocks)
    x_max = max(block.bbox[2] for block in page.blocks)
    return [ColumnRegion(column_index=0, x_min=x_min, x_max=x_max)]


def detect_columns(page: PdfPage, *, x_threshold_pt: float = 36.0) -> list[ColumnRegion]:
    del x_threshold_pt  # kept for API compatibility
    if not page.blocks:
        return []

    candidates = _primary_column_candidates(page)
    if len(candidates) < 4:
        return _single_column_region(page)

    page_mid = page.width_pt / 2.0
    left = [block for block in candidates if ((block.bbox[0] + block.bbox[2]) / 2.0) <= page_mid]
    right = [block for block in candidates if ((block.bbox[0] + block.bbox[2]) / 2.0) > page_mid]
    total = max(1, len(candidates))
    if len(left) < 2 or len(right) < 2:
        return _single_column_region(page)
    if (len(left) / total) < 0.2 or (len(right) / total) < 0.2:
        return _single_column_region(page)

    left_x0 = median(block.bbox[0] for block in left)
    right_x0 = median(block.bbox[0] for block in right)
    if (right_x0 - left_x0) < page.width_pt * 0.24:
        return _single_column_region(page)

    left_w = median((block.bbox[2] - block.bbox[0]) for block in left)
    right_w = median((block.bbox[2] - block.bbox[0]) for block in right)
    if min(left_w, right_w) < page.width_pt * 0.18:
        return _single_column_region(page)
    width_ratio = max(left_w, right_w) / max(1.0, min(left_w, right_w))
    if width_ratio > 2.4:
        return _single_column_region(page)

    left_x_max = max(block.bbox[2] for block in left)
    right_x_min = min(block.bbox[0] for block in right)
    if (right_x_min - left_x_max) < page.width_pt * 0.03:
        return _single_column_region(page)

    return [
        ColumnRegion(column_index=0, x_min=min(block.bbox[0] for block in left), x_max=left_x_max),
        ColumnRegion(column_index=1, x_min=right_x_min, x_max=max(block.bbox[2] for block in right)),
    ]


def classify_block(
    page: PdfPage,
    block: PdfTextBlock,
    *,
    table_regions: list[TableRegion],
    repeated_margin_blocks: dict[int, str] | None = None,
    header_height_ratio: float = 0.1,
    footer_height_ratio: float = 0.1,
) -> str:
    if repeated_margin_blocks and block.block_id in repeated_margin_blocks:
        return repeated_margin_blocks[block.block_id]

    for region in table_regions:
        if _bbox_intersects(block.bbox, region.bbox):
            return "table_cell"

    top_limit = page.height_pt * max(0.0, float(header_height_ratio))
    bottom_limit = page.height_pt * (1.0 - max(0.0, float(footer_height_ratio)))
    if block.bbox[3] <= top_limit:
        return "header"
    if block.bbox[1] >= bottom_limit:
        return "footer"
    return "body"


def _assign_columns(page: PdfPage, columns: list[ColumnRegion]) -> dict[int, int]:
    if not columns:
        return {block.block_id: 0 for block in page.blocks}
    out: dict[int, int] = {}
    for block in page.blocks:
        center_x = (block.bbox[0] + block.bbox[2]) / 2.0
        best = min(columns, key=lambda c: abs(center_x - ((c.x_min + c.x_max) / 2.0)))
        out[block.block_id] = best.column_index
    return out


def group_blocks_into_segments(
    page: PdfPage,
    *,
    block_merge_threshold_pt: float = 12.0,
    skip_headers_footers: bool = False,
    table_detection: bool = True,
    repeated_margin_blocks: dict[int, str] | None = None,
) -> list[PdfSegment]:
    if not page.blocks:
        return []

    table_regions = detect_table_regions(page) if table_detection else []
    columns = detect_columns(page)
    col_idx = _assign_columns(page, columns)

    for block in page.blocks:
        block.column_index = col_idx.get(block.block_id, 0)
        block.block_type = classify_block(
            page,
            block,
            table_regions=table_regions,
            repeated_margin_blocks=repeated_margin_blocks,
        )
        if block.block_type == "table_cell":
            for region in table_regions:
                if _bbox_intersects(block.bbox, region.bbox):
                    block.table_region_id = region.region_id
                    break

    source_blocks = sorted(
        page.blocks,
        key=lambda b: (b.column_index, b.bbox[1], b.bbox[0]),
    )
    if skip_headers_footers:
        source_blocks = [b for b in source_blocks if b.block_type not in {"header", "footer"}]

    grouped: list[list[PdfTextBlock]] = []
    max_gap = max(4.0, float(block_merge_threshold_pt) * 1.5)
    for block in source_blocks:
        if not grouped:
            grouped.append([block])
            continue
        tail = grouped[-1][-1]
        same_type = tail.block_type == block.block_type
        same_column = tail.column_index == block.column_index
        left_aligned = abs(tail.bbox[0] - block.bbox[0]) <= max(10.0, float(block_merge_threshold_pt) * 2.0)
        vertical_gap = block.bbox[1] - tail.bbox[3]
        if same_type and same_column and left_aligned and vertical_gap <= max_gap:
            grouped[-1].append(block)
        else:
            grouped.append([block])

    segments: list[PdfSegment] = []
    for idx, group in enumerate(grouped):
        group_bboxes = [block.bbox for block in group]
        segment_bbox = _bbox_union(group_bboxes)
        source_text = "\n".join(block.text for block in group if block.text.strip()).strip()
        if not source_text:
            continue

        first = group[0]
        segment = PdfSegment(
            segment_id=f"pdf-p{page.page_number + 1}-s{idx}",
            page_number=page.page_number,
            block_ids=[block.block_id for block in group],
            bbox=segment_bbox,
            source_text=source_text,
            dominant_style=first.dominant_style,
            context={
                "page_number": page.page_number + 1,
                "block_type": first.block_type,
                "in_table": first.block_type == "table_cell",
                "column_index": first.column_index,
            },
        )
        segments.append(segment)

    return segments


def group_all_pages(
    pages: list[PdfPage],
    *,
    block_merge_threshold_pt: float = 12.0,
    skip_headers_footers: bool = False,
    table_detection: bool = True,
) -> list[PdfSegment]:
    repeated_margin_map = _collect_repeated_margin_block_types(pages)
    grouped: list[PdfSegment] = []
    for page in pages:
        grouped.extend(
            group_blocks_into_segments(
                page,
                block_merge_threshold_pt=block_merge_threshold_pt,
                skip_headers_footers=skip_headers_footers,
                table_detection=table_detection,
                repeated_margin_blocks=repeated_margin_map.get(page.page_number),
            )
        )
    # Keep natural reading order page->segment
    grouped.sort(key=lambda seg: (seg.page_number, seg.bbox[1], seg.bbox[0]))
    # Rewrite ids after global ordering for stable resume keys.
    renumbered: list[PdfSegment] = []
    by_page_count: dict[int, int] = defaultdict(int)
    for seg in grouped:
        by_page_count[seg.page_number] += 1
        seg.segment_id = f"pdf-p{seg.page_number + 1}-s{by_page_count[seg.page_number] - 1}"
        renumbered.append(seg)
    return renumbered


def _normalize_margin_text(value: str) -> str:
    text = _SPACE_RE.sub(" ", value or "").strip().lower()
    text = _NON_TEXT_RE.sub("", text)
    return text


def _collect_repeated_margin_block_types(pages: list[PdfPage]) -> dict[int, dict[int, str]]:
    if len(pages) < 2:
        return {}
    min_repeats = 3
    hits: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for page in pages:
        top_limit = page.height_pt * 0.18
        bottom_limit = page.height_pt * 0.82
        for block in page.blocks:
            text = _normalize_margin_text(block.text)
            if len(text) < 8:
                continue
            if block.bbox[3] <= top_limit:
                hits[("header", text)].append((page.page_number, block.block_id))
            elif block.bbox[1] >= bottom_limit:
                hits[("footer", text)].append((page.page_number, block.block_id))

    out: dict[int, dict[int, str]] = defaultdict(dict)
    for (block_type, _text), refs in hits.items():
        page_refs = {page_no for page_no, _ in refs}
        if len(page_refs) < min_repeats:
            continue
        for page_no, block_id in refs:
            out[page_no][block_id] = block_type
    return dict(out)
