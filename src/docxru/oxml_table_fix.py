from __future__ import annotations

from dataclasses import dataclass

from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.table import Table, _Cell


_DEFAULT_PAGE_WIDTH_TWIPS = 11906
_DEFAULT_PAGE_HEIGHT_TWIPS = 16840
_DEFAULT_MARGIN_TWIPS = 1440
_EDGE_TOP_RATIO = 0.18
_EDGE_BOTTOM_RATIO = 0.82
_PAGE_RESET_THRESHOLD_TWIPS = 1200
_DEFAULT_FRAME_HEIGHT_TWIPS = 240
_HEADER_FRAME_TOP_RATIO = 0.26
_FRAME_VERTICAL_TEXT_DISTANCE_TWIPS = 142  # 0.25 cm in twips
_HEADER_FRAME_MIN_Y_TWIPS = 2100
_SERVICE_MANUAL_HEADER_MARKERS: tuple[str, ...] = (
    "руководство по техническому обслуживанию компонентов 32-12-22 стойка основного шасси",
    "руководство по техническому обслуживанию компонентов для деталей № 201587001 и 201587002 стойка основного шасси",
)


def _get_or_add(parent, tag: str):
    child = parent.find(qn(tag))
    if child is None:
        child = OxmlElement(tag.split(":")[-1])
        parent.append(child)
    return child


def set_table_fixed_layout(table: Table) -> None:
    """Force fixed table layout (Word will preserve column widths better)."""
    tbl_pr = table._tbl.tblPr
    tbl_layout = tbl_pr.find(qn("w:tblLayout"))
    if tbl_layout is None:
        tbl_layout = OxmlElement("w:tblLayout")
        tbl_pr.append(tbl_layout)
    tbl_layout.set(qn("w:type"), "fixed")


def set_cell_width_twips(cell: _Cell, width_twips: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:type"), "dxa")
    tc_w.set(qn("w:w"), str(int(width_twips)))


def set_table_grid_widths_twips(table: Table, widths_twips: list[int]) -> None:
    """Set explicit tblGrid column widths. widths_twips length must match table columns."""
    tbl = table._tbl
    grid = tbl.tblGrid
    # Remove existing gridCols
    for gc in list(grid.gridCol_lst):
        grid._element.remove(gc)  # type: ignore[attr-defined]
    for w in widths_twips:
        gc = OxmlElement("w:gridCol")
        gc.set(qn("w:w"), str(int(w)))
        grid._element.append(gc)  # type: ignore[attr-defined]


def remove_exact_tr_height(document) -> int:
    """Remove table row height constraints with w:hRule='exact'."""
    removed = 0
    tr_height_tag = qn("w:trHeight")
    tr_pr_tag = qn("w:trPr")
    h_rule_attr = qn("w:hRule")
    for tr_pr in document.element.iter(tr_pr_tag):
        for tr_height in list(tr_pr):
            if tr_height.tag != tr_height_tag:
                continue
            if str(tr_height.get(h_rule_attr, "")).strip().lower() != "exact":
                continue
            tr_pr.remove(tr_height)
            removed += 1
    return removed


def remove_frame_pr(document) -> int:
    """Remove paragraph frame properties (<w:framePr>) that can pin ABBYY layout."""
    removed = 0
    p_pr_tag = qn("w:pPr")
    frame_pr_tag = qn("w:framePr")
    for p_pr in document.element.iter(p_pr_tag):
        for frame_pr in list(p_pr):
            if frame_pr.tag != frame_pr_tag:
                continue
            p_pr.remove(frame_pr)
            removed += 1
    return removed


def _median_positive(values: list[int], *, default: int) -> int:
    filtered = [int(v) for v in values if int(v) > 0]
    if not filtered:
        return int(default)
    filtered.sort()
    return int(filtered[len(filtered) // 2])


def _resolve_page_metrics_twips(document) -> dict[str, int]:
    pg_sz_tag = qn("w:pgSz")
    pg_mar_tag = qn("w:pgMar")
    page_width_attr = qn("w:w")
    page_height_attr = qn("w:h")
    margin_left_attr = qn("w:left")
    margin_right_attr = qn("w:right")
    margin_top_attr = qn("w:top")
    margin_bottom_attr = qn("w:bottom")

    widths: list[int] = []
    heights: list[int] = []
    margin_left: list[int] = []
    margin_right: list[int] = []
    margin_top: list[int] = []
    margin_bottom: list[int] = []

    for pg_sz in document.element.iter(pg_sz_tag):
        raw_w = pg_sz.get(page_width_attr)
        raw = pg_sz.get(page_height_attr)
        try:
            if raw_w is not None:
                val_w = int(str(raw_w))
                if val_w > 0:
                    widths.append(val_w)
            if raw is not None:
                val_h = int(str(raw))
                if val_h > 0:
                    heights.append(val_h)
        except (TypeError, ValueError):
            continue

    for pg_mar in document.element.iter(pg_mar_tag):
        for attr, bucket in (
            (margin_left_attr, margin_left),
            (margin_right_attr, margin_right),
            (margin_top_attr, margin_top),
            (margin_bottom_attr, margin_bottom),
        ):
            raw = pg_mar.get(attr)
            if raw is None:
                continue
            try:
                value = int(str(raw))
            except (TypeError, ValueError):
                continue
            if value > 0:
                bucket.append(value)

    return {
        "page_width_twips": _median_positive(widths, default=_DEFAULT_PAGE_WIDTH_TWIPS),
        "page_height_twips": _median_positive(heights, default=_DEFAULT_PAGE_HEIGHT_TWIPS),
        "margin_left_twips": _median_positive(margin_left, default=_DEFAULT_MARGIN_TWIPS),
        "margin_right_twips": _median_positive(margin_right, default=_DEFAULT_MARGIN_TWIPS),
        "margin_top_twips": _median_positive(margin_top, default=_DEFAULT_MARGIN_TWIPS),
        "margin_bottom_twips": _median_positive(margin_bottom, default=_DEFAULT_MARGIN_TWIPS),
    }


def _resolve_page_height_twips(document) -> int:
    return int(_resolve_page_metrics_twips(document)["page_height_twips"])


def _is_page_edge_frame(frame_pr, *, page_height_twips: int) -> bool:
    if not _is_page_anchored_frame(frame_pr):
        return False
    raw_y = frame_pr.get(qn("w:y"))
    if raw_y is None:
        return False
    try:
        y_value = int(str(raw_y))
    except (TypeError, ValueError):
        return False

    top_limit = max(720, int(float(page_height_twips) * _EDGE_TOP_RATIO))
    bottom_limit = min(page_height_twips - 720, int(float(page_height_twips) * _EDGE_BOTTOM_RATIO))
    if bottom_limit <= top_limit:
        return False
    return y_value <= top_limit or y_value >= bottom_limit


def _is_page_anchored_frame(frame_pr) -> bool:
    v_anchor = str(frame_pr.get(qn("w:vAnchor"), "")).strip().lower()
    h_anchor = str(frame_pr.get(qn("w:hAnchor"), "")).strip().lower()
    return v_anchor == "page" and h_anchor == "page"


def _try_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _ranges_intersect(left: tuple[int, int], right: tuple[int, int]) -> bool:
    l1, l2 = left
    r1, r2 = right
    return not (l2 <= r1 or r2 <= l1)


@dataclass
class _FrameBlock:
    nodes: list
    y: int
    x: int
    w: int | None
    h: int | None


@dataclass
class _FramedTextBlock:
    signature: tuple[str, ...]
    nodes: list
    text_chunks: list[str]
    y: int | None


def _normalize_text_for_match(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def _is_service_manual_header_frame_text(value: str) -> bool:
    text = _normalize_text_for_match(value)
    if not text:
        return False
    for marker in _SERVICE_MANUAL_HEADER_MARKERS:
        if marker in text:
            return True
    has_core_title = "руководство по техническому обслуживанию компонентов" in text
    has_issue_code = "32-12-22" in text
    has_part_pair = "201587001" in text and "201587002" in text
    has_chassis = "стойка основного шасси" in text
    if has_core_title and (has_issue_code or has_part_pair):
        return True
    return has_core_title and has_chassis


def _collect_framed_text_blocks(document) -> list[_FramedTextBlock]:
    p_pr_tag = qn("w:pPr")
    frame_pr_tag = qn("w:framePr")
    text_tag = qn("w:t")
    signature_attrs = (
        qn("w:x"),
        qn("w:y"),
        qn("w:w"),
        qn("w:h"),
        qn("w:vAnchor"),
        qn("w:hAnchor"),
        qn("w:wrap"),
        qn("w:xAlign"),
        qn("w:yAlign"),
    )

    blocks: list[_FramedTextBlock] = []
    for p_pr in document.element.iter(p_pr_tag):
        frame_pr = p_pr.find(frame_pr_tag)
        if frame_pr is None:
            continue
        signature = tuple(str(frame_pr.get(attr, "") or "") for attr in signature_attrs)
        y_value = _try_int(frame_pr.get(qn("w:y")))
        paragraph = p_pr.getparent()
        text_value = ""
        if paragraph is not None:
            text_value = "".join(str(getattr(node, "text", "") or "") for node in paragraph.iter(text_tag))

        if blocks and blocks[-1].signature == signature:
            blocks[-1].nodes.append(frame_pr)
            if text_value.strip():
                blocks[-1].text_chunks.append(text_value)
            continue

        blocks.append(
            _FramedTextBlock(
                signature=signature,
                nodes=[frame_pr],
                text_chunks=[text_value] if text_value.strip() else [],
                y=y_value,
            )
        )
    return blocks


def normalize_frame_vertical_settings(
    document,
    *,
    vertical_text_distance_twips: int = _FRAME_VERTICAL_TEXT_DISTANCE_TWIPS,
) -> dict[str, int]:
    """Set framePr auto-height globally and apply vertical defaults to non-title blocks."""
    stats = {
        "frame_auto_height_set": 0,
        "frame_vertical_anchor_margin_set": 0,
        "frame_vertical_text_distance_set": 0,
        "frame_header_blocks_detected": 0,
        "frame_header_y_nudged": 0,
    }
    blocks = _collect_framed_text_blocks(document)
    if not blocks:
        return stats

    distance_twips = max(0, int(vertical_text_distance_twips))
    page_height_twips = _resolve_page_height_twips(document)
    title_top_limit = max(720, int(float(page_height_twips) * _HEADER_FRAME_TOP_RATIO))
    h_rule_attr = qn("w:hRule")
    v_anchor_attr = qn("w:vAnchor")
    v_space_attr = qn("w:vSpace")
    y_attr = qn("w:y")
    h_attr = qn("w:h")

    for block in blocks:
        block_text = _normalize_text_for_match(" ".join(block.text_chunks))
        marker_match = _is_service_manual_header_frame_text(block_text)
        top_frame_hint = (
            block.y is not None
            and int(block.y) <= title_top_limit
            and "руководство по техническому обслуживанию компонентов" in block_text
        )
        is_title_block = bool(marker_match or top_frame_hint)
        should_nudge_header = bool(
            is_title_block
            and ("201587001" in block_text and "201587002" in block_text)
        )
        if is_title_block:
            stats["frame_header_blocks_detected"] += 1

        for node in block.nodes:
            if str(node.get(h_rule_attr, "")).strip().lower() != "auto":
                node.set(h_rule_attr, "auto")
                stats["frame_auto_height_set"] += 1
            if is_title_block:
                current_y = _try_int(node.get(y_attr))
                current_h = _try_int(node.get(h_attr))
                is_compact_title_box = current_h is not None and current_h <= 1500
                is_title_y_band = current_y is not None and 1200 <= current_y < _HEADER_FRAME_MIN_Y_TWIPS
                if should_nudge_header and is_title_y_band and is_compact_title_box:
                    node.set(y_attr, str(int(_HEADER_FRAME_MIN_Y_TWIPS)))
                    stats["frame_header_y_nudged"] += 1
                continue
            if str(node.get(v_anchor_attr, "")).strip().lower() != "margin":
                node.set(v_anchor_attr, "margin")
                stats["frame_vertical_anchor_margin_set"] += 1
            if _try_int(node.get(v_space_attr)) != distance_twips:
                node.set(v_space_attr, str(int(distance_twips)))
                stats["frame_vertical_text_distance_set"] += 1
    return stats


def _collect_page_anchored_frame_blocks(document) -> tuple[list[_FrameBlock], dict[str, int]]:
    page_metrics = _resolve_page_metrics_twips(document)
    page_height_twips = int(page_metrics["page_height_twips"])
    blocks: list[_FrameBlock] = []

    p_pr_tag = qn("w:pPr")
    frame_pr_tag = qn("w:framePr")
    y_attr = qn("w:y")
    x_attr = qn("w:x")
    w_attr = qn("w:w")
    h_attr = qn("w:h")

    for p_pr in document.element.iter(p_pr_tag):
        frame_pr = p_pr.find(frame_pr_tag)
        if frame_pr is None:
            continue
        if not _is_page_anchored_frame(frame_pr):
            continue
        if _is_page_edge_frame(frame_pr, page_height_twips=page_height_twips):
            continue

        y = _try_int(frame_pr.get(y_attr))
        if y is None:
            continue
        x = _try_int(frame_pr.get(x_attr))
        w = _try_int(frame_pr.get(w_attr))
        h = _try_int(frame_pr.get(h_attr))
        if w is not None and w <= 0:
            w = None
        if h is not None and h <= 0:
            h = None

        if blocks:
            prev = blocks[-1]
            if prev.y == y and prev.x == int(x or 0) and prev.w == w and prev.h == h:
                prev.nodes.append(frame_pr)
                continue

        blocks.append(
            _FrameBlock(
                nodes=[frame_pr],
                y=int(y),
                x=int(x or 0),
                w=w,
                h=h,
            )
        )
    return blocks, page_metrics


def expand_and_space_page_anchored_frames(
    document,
    *,
    width_factor: float = 1.0,
    height_factor: float = 1.0,
    min_vertical_gap_twips: int = 0,
) -> dict[str, int]:
    """Optionally expand page-anchored framePr boxes and prevent vertical overlap."""
    stats = {
        "frame_width_expanded": 0,
        "frame_height_expanded": 0,
        "frame_vertical_reflowed": 0,
    }
    width_factor = max(1.0, float(width_factor))
    height_factor = max(1.0, float(height_factor))
    min_gap = max(0, int(min_vertical_gap_twips))
    if width_factor <= 1.0 and height_factor <= 1.0 and min_gap <= 0:
        return stats

    blocks, metrics = _collect_page_anchored_frame_blocks(document)
    if not blocks:
        return stats

    page_width = int(metrics["page_width_twips"])
    page_height = int(metrics["page_height_twips"])
    margin_left = int(metrics["margin_left_twips"])
    margin_right = int(metrics["margin_right_twips"])
    margin_top = int(metrics["margin_top_twips"])
    margin_bottom = int(metrics["margin_bottom_twips"])
    content_width = max(720, page_width - margin_left - margin_right)
    content_top = max(0, margin_top)
    content_bottom = max(content_top + 720, page_height - margin_bottom)

    # First pass: enlarge geometry where requested.
    for block in blocks:
        if width_factor > 1.0 and block.w is not None:
            max_w = max(720, int(content_width * 0.98))
            target_w = max(block.w, int(round(float(block.w) * width_factor)))
            new_w = min(max_w, target_w)
            if new_w > block.w:
                for node in block.nodes:
                    node.set(qn("w:w"), str(int(new_w)))
                block.w = int(new_w)
                stats["frame_width_expanded"] += 1

        if height_factor > 1.0:
            base_h = int(block.h if block.h is not None else _DEFAULT_FRAME_HEIGHT_TWIPS)
            max_h = max(_DEFAULT_FRAME_HEIGHT_TWIPS, content_bottom - max(content_top, int(block.y)) - 60)
            target_h = max(base_h, int(round(float(base_h) * height_factor)))
            new_h = min(max_h, target_h)
            if new_h > base_h:
                for node in block.nodes:
                    node.set(qn("w:h"), str(int(new_h)))
                    # Keep frame expandable vertically after growth.
                    node.set(qn("w:hRule"), "atLeast")
                block.h = int(new_h)
                stats["frame_height_expanded"] += 1
            elif block.h is None:
                block.h = base_h

    if min_gap <= 0:
        return stats

    # Second pass: distribute blocks by Y to avoid overlap in the same X band.
    groups: list[list[_FrameBlock]] = []
    current: list[_FrameBlock] = []
    prev_y: int | None = None
    for block in blocks:
        if prev_y is not None and int(block.y) < int(prev_y) - _PAGE_RESET_THRESHOLD_TWIPS:
            if current:
                groups.append(current)
            current = []
        current.append(block)
        prev_y = int(block.y)
    if current:
        groups.append(current)

    for group in groups:
        placed: list[tuple[int, int, int]] = []
        for block in sorted(group, key=lambda item: (int(item.y), int(item.x))):
            block_h = int(block.h if block.h is not None else _DEFAULT_FRAME_HEIGHT_TWIPS)
            block_w = int(block.w if block.w is not None else content_width)
            x1 = int(block.x)
            x2 = int(block.x + block_w)
            new_y = max(content_top, int(block.y))
            for px1, px2, p_bottom in placed:
                if not _ranges_intersect((x1, x2), (px1, px2)):
                    continue
                required_y = int(p_bottom + min_gap)
                if new_y < required_y:
                    new_y = required_y
            max_y = max(content_top, content_bottom - block_h)
            if new_y > max_y:
                new_y = max_y
            if new_y > int(block.y):
                for node in block.nodes:
                    node.set(qn("w:y"), str(int(new_y)))
                block.y = int(new_y)
                stats["frame_vertical_reflowed"] += 1
            placed.append((x1, x2, int(block.y) + block_h))

    return stats


def relax_frame_pr_exact_height(
    document,
    *,
    preserve_page_edge_frames: bool = False,
    preserve_page_anchored_frames: bool = False,
) -> int:
    """Relax <w:framePr w:hRule='exact'> to atLeast while preserving frame geometry."""
    changed = 0
    p_pr_tag = qn("w:pPr")
    frame_pr_tag = qn("w:framePr")
    h_rule_attr = qn("w:hRule")
    page_height_twips = _resolve_page_height_twips(document) if preserve_page_edge_frames else 0
    for p_pr in document.element.iter(p_pr_tag):
        for frame_pr in list(p_pr):
            if frame_pr.tag != frame_pr_tag:
                continue
            h_rule = str(frame_pr.get(h_rule_attr, "")).strip().lower()
            if h_rule != "exact":
                continue
            if preserve_page_anchored_frames and _is_page_anchored_frame(frame_pr):
                continue
            if preserve_page_edge_frames and _is_page_edge_frame(frame_pr, page_height_twips=page_height_twips):
                continue
            frame_pr.set(h_rule_attr, "atLeast")
            changed += 1
    return changed


def relax_exact_line_spacing(
    document,
    *,
    preserve_page_edge_frames: bool = False,
    preserve_page_anchored_frames: bool = False,
) -> int:
    """Relax paragraph spacing lineRule='exact' to lineRule='atLeast'."""
    changed = 0
    p_pr_tag = qn("w:pPr")
    frame_pr_tag = qn("w:framePr")
    spacing_tag = qn("w:spacing")
    line_rule_attr = qn("w:lineRule")
    page_height_twips = _resolve_page_height_twips(document) if preserve_page_edge_frames else 0
    for p_pr in document.element.iter(p_pr_tag):
        frame_pr = p_pr.find(frame_pr_tag)
        for spacing in list(p_pr):
            if spacing.tag != spacing_tag:
                continue
            line_rule = str(spacing.get(line_rule_attr, "")).strip().lower()
            if line_rule != "exact":
                continue
            if (
                preserve_page_anchored_frames
                and frame_pr is not None
                and _is_page_anchored_frame(frame_pr)
            ):
                continue
            if (
                preserve_page_edge_frames
                and frame_pr is not None
                and _is_page_edge_frame(frame_pr, page_height_twips=page_height_twips)
            ):
                continue
            spacing.set(line_rule_attr, "atLeast")
            changed += 1
    return changed


def _local_name(node) -> str:
    tag = str(getattr(node, "tag", ""))
    if not tag:
        return ""
    if "}" in tag:
        return tag.split("}", 1)[1]
    if ":" in tag:
        return tag.split(":", 1)[1]
    return tag


def _contains_non_whitespace_text(node) -> bool:
    text_tag = qn("w:t")
    for text_node in node.iter(text_tag):
        text_value = str(getattr(text_node, "text", "") or "")
        if text_value.strip():
            return True
    return False


def _iter_non_empty_textbox_contents(document):
    txbx_tag = qn("w:txbxContent")
    for txbx_content in document.element.iter(txbx_tag):
        if _contains_non_whitespace_text(txbx_content):
            yield txbx_content


def _iter_related_body_pr_nodes(txbx_content):
    container_hints = {
        "txbx",
        "textbox",
        "shape",
        "wsp",
        "drawing",
        "pict",
    }
    for ancestor in [txbx_content, *list(txbx_content.iterancestors())]:
        if _local_name(ancestor) not in container_hints:
            continue
        for node in ancestor.iter():
            if _local_name(node) == "bodyPr":
                yield node


def _set_body_pr_norm_autofit(body_pr) -> bool:
    no_autofit_nodes = []
    has_norm_autofit = False
    for child in list(body_pr):
        local_name = _local_name(child)
        if local_name == "noAutofit":
            no_autofit_nodes.append(child)
        elif local_name == "normAutofit":
            has_norm_autofit = True

    if not no_autofit_nodes:
        return False

    for node in no_autofit_nodes:
        body_pr.remove(node)
    if not has_norm_autofit:
        body_pr.append(OxmlElement("a:normAutofit"))
    return True


def set_textbox_autofit(document) -> int:
    """Enable TextBody auto-fit for non-empty textboxes by switching to <a:normAutofit/>."""
    updated = 0
    seen_node_ids: set[int] = set()
    for txbx_content in _iter_non_empty_textbox_contents(document):
        for body_pr in _iter_related_body_pr_nodes(txbx_content):
            node_id = id(body_pr)
            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            if _set_body_pr_norm_autofit(body_pr):
                updated += 1
    return updated


def normalize_textbox_insets(document, *, max_inset_emu: int = 25400) -> int:
    """Cap excessive DrawingML textbox insets (a:bodyPr lIns/rIns/tIns/bIns)."""
    cap = max(0, int(max_inset_emu))
    updated = 0
    seen_node_ids: set[int] = set()
    for txbx_content in _iter_non_empty_textbox_contents(document):
        for body_pr in _iter_related_body_pr_nodes(txbx_content):
            node_id = id(body_pr)
            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            changed = False
            for attr in ("lIns", "rIns", "tIns", "bIns"):
                raw = body_pr.get(attr)
                if raw is None:
                    continue
                try:
                    value = int(str(raw))
                except (TypeError, ValueError):
                    continue
                if value <= cap:
                    continue
                body_pr.set(attr, str(cap))
                changed = True
            if changed:
                updated += 1
    return updated


def normalize_table_cell_margins(document, *, max_margin_twips: int = 108) -> int:
    """Cap excessive table-cell margins in w:tcMar to reduce avoidable text loss."""
    normalized = 0
    cap = max(29, int(max_margin_twips))
    tc_mar_tag = qn("w:tcMar")
    margin_tags = {
        qn("w:left"),
        qn("w:right"),
        qn("w:top"),
        qn("w:bottom"),
        qn("w:start"),
        qn("w:end"),
    }
    width_attr = qn("w:w")
    type_attr = qn("w:type")
    for tc_mar in document.element.iter(tc_mar_tag):
        for margin in list(tc_mar):
            if margin.tag not in margin_tags:
                continue
            try:
                value = int(str(margin.get(width_attr)))
            except (TypeError, ValueError):
                continue
            if value <= cap:
                continue
            margin.set(width_attr, str(cap))
            if str(margin.get(type_attr, "")).strip().lower() != "dxa":
                margin.set(type_attr, "dxa")
            normalized += 1
    return normalized


def normalize_abbyy_oxml(
    document,
    *,
    profile: str,
    frame_expand_width_factor: float = 1.0,
    frame_expand_height_factor: float = 1.0,
    frame_vertical_gap_twips: int = 0,
) -> dict[str, int]:
    """Apply optional ABBYY-specific OXML cleanup by profile.

    Profiles:
    - off: no cleanup
    - safe: remove only strict row-height locks (w:trHeight with hRule='exact')
    - aggressive: safe + relax strict frame/line spacing constraints + normalize frame vertical settings
    - full: aggressive + enable textbox auto-fit via <a:normAutofit/> + normalize cell margins
    """
    mode = str(profile or "off").strip().lower()
    if mode not in {"off", "safe", "aggressive", "full"}:
        raise ValueError(f"Unsupported ABBYY profile: {profile!r}")

    stats = {
        "tr_height_exact_removed": 0,
        "frame_pr_removed": 0,
        "frame_pr_exact_relaxed": 0,
        "line_spacing_exact_relaxed": 0,
        "textbox_autofit_updated": 0,
        "textbox_insets_normalized": 0,
        "frame_width_expanded": 0,
        "frame_height_expanded": 0,
        "frame_vertical_reflowed": 0,
        "frame_auto_height_set": 0,
        "frame_vertical_anchor_margin_set": 0,
        "frame_vertical_text_distance_set": 0,
        "frame_header_blocks_detected": 0,
        "frame_header_y_nudged": 0,
        "table_cell_margins_normalized": 0,
    }
    if mode == "off":
        return stats

    stats["tr_height_exact_removed"] = remove_exact_tr_height(document)
    if mode in {"aggressive", "full"}:
        stats["frame_pr_exact_relaxed"] = relax_frame_pr_exact_height(
            document,
            preserve_page_edge_frames=True,
            preserve_page_anchored_frames=True,
        )
        stats["line_spacing_exact_relaxed"] = relax_exact_line_spacing(
            document,
            preserve_page_edge_frames=True,
            preserve_page_anchored_frames=True,
        )
        frame_stats = expand_and_space_page_anchored_frames(
            document,
            width_factor=float(frame_expand_width_factor),
            height_factor=float(frame_expand_height_factor),
            min_vertical_gap_twips=int(frame_vertical_gap_twips),
        )
        for key, value in frame_stats.items():
            stats[key] = int(stats.get(key, 0)) + int(value)
        vertical_stats = normalize_frame_vertical_settings(document)
        for key, value in vertical_stats.items():
            stats[key] = int(stats.get(key, 0)) + int(value)
    if mode == "full":
        stats["textbox_autofit_updated"] = set_textbox_autofit(document)
        stats["textbox_insets_normalized"] = normalize_textbox_insets(document)
        stats["table_cell_margins_normalized"] = normalize_table_cell_margins(document)
    return stats
