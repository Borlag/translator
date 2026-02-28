from __future__ import annotations

from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.table import Table, _Cell


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


def relax_frame_pr_exact_height(document) -> int:
    """Relax <w:framePr w:hRule='exact'> to atLeast while preserving frame geometry."""
    changed = 0
    p_pr_tag = qn("w:pPr")
    frame_pr_tag = qn("w:framePr")
    h_rule_attr = qn("w:hRule")
    for p_pr in document.element.iter(p_pr_tag):
        for frame_pr in list(p_pr):
            if frame_pr.tag != frame_pr_tag:
                continue
            h_rule = str(frame_pr.get(h_rule_attr, "")).strip().lower()
            if h_rule != "exact":
                continue
            frame_pr.set(h_rule_attr, "atLeast")
            changed += 1
    return changed


def relax_exact_line_spacing(document) -> int:
    """Relax paragraph spacing lineRule='exact' to lineRule='atLeast'."""
    changed = 0
    spacing_tag = qn("w:spacing")
    line_rule_attr = qn("w:lineRule")
    for spacing in document.element.iter(spacing_tag):
        line_rule = str(spacing.get(line_rule_attr, "")).strip().lower()
        if line_rule != "exact":
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


def normalize_abbyy_oxml(document, *, profile: str) -> dict[str, int]:
    """Apply optional ABBYY-specific OXML cleanup by profile.

    Profiles:
    - off: no cleanup
    - safe: remove only strict row-height locks (w:trHeight with hRule='exact')
    - aggressive: safe + relax strict frame/line spacing constraints
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
        "table_cell_margins_normalized": 0,
    }
    if mode == "off":
        return stats

    stats["tr_height_exact_removed"] = remove_exact_tr_height(document)
    if mode in {"aggressive", "full"}:
        stats["frame_pr_exact_relaxed"] = relax_frame_pr_exact_height(document)
        stats["line_spacing_exact_relaxed"] = relax_exact_line_spacing(document)
    if mode == "full":
        stats["textbox_autofit_updated"] = set_textbox_autofit(document)
        stats["table_cell_margins_normalized"] = normalize_table_cell_margins(document)
    return stats
