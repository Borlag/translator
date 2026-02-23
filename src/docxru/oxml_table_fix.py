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


def normalize_abbyy_oxml(document, *, profile: str) -> dict[str, int]:
    """Apply optional ABBYY-specific OXML cleanup by profile.

    Profiles:
    - off: no cleanup
    - safe: remove only strict row-height locks (w:trHeight with hRule='exact')
    - aggressive: safe + remove paragraph frame locks (w:framePr)
    """
    mode = str(profile or "off").strip().lower()
    if mode not in {"off", "safe", "aggressive"}:
        raise ValueError(f"Unsupported ABBYY profile: {profile!r}")

    stats = {
        "tr_height_exact_removed": 0,
        "frame_pr_removed": 0,
        "line_spacing_exact_relaxed": 0,
    }
    if mode == "off":
        return stats

    stats["tr_height_exact_removed"] = remove_exact_tr_height(document)
    if mode == "aggressive":
        stats["frame_pr_removed"] = remove_frame_pr(document)
        stats["line_spacing_exact_relaxed"] = relax_exact_line_spacing(document)
    return stats
