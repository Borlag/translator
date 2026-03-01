from __future__ import annotations

from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.table import _Cell

from .config import PipelineConfig
from .models import Issue, Segment, Severity


def reduce_font_size(paragraph, reduction_pt: float = 0.5, *, min_font_pt: float = 6.0) -> bool:
    changed = False
    step = max(0.1, float(reduction_pt))
    floor_pt = max(6.0, float(min_font_pt))
    for run in paragraph.runs:
        current = run.font.size.pt if run.font.size is not None else None
        if current is None:
            continue
        if float(current) <= floor_pt + 1e-6:
            continue
        updated = max(floor_pt, float(current) - step)
        if updated < float(current):
            run.font.size = Pt(updated)
            changed = True

    if changed:
        return True

    # If explicit run sizes are missing, apply a small fallback size to text runs.
    fallback = max(floor_pt, 10.0 - step)
    for run in paragraph.runs:
        if not (run.text or "").strip():
            continue
        current_size = _resolve_run_size_pt(run, paragraph)
        if current_size is not None and float(current_size) <= floor_pt + 1e-6:
            continue
        run.font.size = Pt(fallback)
        changed = True
    return changed


def _resolve_run_size_pt(run, paragraph) -> float | None:
    direct = run.font.size
    if direct is not None:
        return float(direct.pt)

    run_style = getattr(run, "style", None)
    run_style_font = getattr(run_style, "font", None)
    run_style_size = getattr(run_style_font, "size", None)
    if run_style_size is not None:
        return float(run_style_size.pt)

    para_style = getattr(paragraph, "style", None)
    para_style_font = getattr(para_style, "font", None)
    para_style_size = getattr(para_style_font, "size", None)
    if para_style_size is not None:
        return float(para_style_size.pt)

    return None


def apply_global_font_shrink(segments: list[Segment], cfg: PipelineConfig) -> int:
    """Unconditionally reduce font sizes after write-back for translated segments."""
    body_shrink = max(0.0, float(cfg.font_shrink_body_pt))
    table_shrink = max(0.0, float(cfg.font_shrink_table_pt))
    if body_shrink <= 0.0 and table_shrink <= 0.0:
        return 0

    min_font_pt = max(6.0, float(getattr(cfg, "font_shrink_min_font_pt", 6.0)))
    changed_segments = 0
    for seg in segments:
        paragraph = seg.paragraph_ref
        if paragraph is None or seg.target_tagged is None:
            continue

        in_table_like = bool(seg.context.get("in_table") or seg.context.get("in_textbox"))
        shrink = table_shrink if in_table_like else body_shrink
        if shrink <= 0.0:
            continue

        changed = False
        for run in paragraph.runs:
            if not (run.text or "").strip():
                continue
            current_size = _resolve_run_size_pt(run, paragraph)
            if current_size is None:
                continue
            if current_size <= min_font_pt + 1e-6:
                continue
            new_size = max(min_font_pt, current_size - shrink)
            if new_size + 1e-6 >= current_size:
                continue
            run.font.size = Pt(new_size)
            changed = True

        if changed:
            changed_segments += 1
            seg.issues.append(
                Issue(
                    code="global_font_shrink_applied",
                    severity=Severity.INFO,
                    message="Applied unconditional post-writeback font shrink.",
                    details={
                        "segment_id": seg.segment_id,
                        "location": seg.location,
                        "shrink_pt": shrink,
                        "table_or_textbox": in_table_like,
                    },
                )
            )

    return changed_segments


def reduce_cell_spacing(cell: _Cell, factor: float = 0.8) -> bool:
    changed = False
    ratio = min(1.0, max(0.1, float(factor)))
    for paragraph in cell.paragraphs:
        changed = reduce_paragraph_spacing(paragraph, factor=ratio) or changed
    return changed


def reduce_character_spacing(paragraph, twips: int = -10) -> bool:
    changed = False
    for run in paragraph.runs:
        if not (run.text or "").strip():
            continue
        r_pr = run._r.get_or_add_rPr()
        spacing = r_pr.find(qn("w:spacing"))
        if spacing is None:
            spacing = OxmlElement("w:spacing")
            r_pr.append(spacing)
        prev_val = spacing.get(qn("w:val"))
        spacing.set(qn("w:val"), str(int(twips)))
        if prev_val != str(int(twips)):
            changed = True
    return changed


def _paragraph_cell(paragraph) -> _Cell | None:
    parent = getattr(paragraph, "_parent", None)
    if isinstance(parent, _Cell):
        return parent
    return None


def _relax_paragraph_frame_height_rule(paragraph) -> bool:
    p_elm = getattr(paragraph, "_p", None)
    if p_elm is None:
        return False
    p_pr = getattr(p_elm, "pPr", None)
    if p_pr is None:
        return False
    frame_pr = p_pr.find(qn("w:framePr"))
    if frame_pr is None:
        return False
    h_rule_attr = qn("w:hRule")
    h_rule = str(frame_pr.get(h_rule_attr, "")).strip().lower()
    if h_rule != "exact":
        return False
    frame_pr.set(h_rule_attr, "atLeast")
    return True


def _remove_paragraph_frame(paragraph) -> bool:
    p_elm = getattr(paragraph, "_p", None)
    if p_elm is None:
        return False
    p_pr = getattr(p_elm, "pPr", None)
    if p_pr is None:
        return False
    frame_pr = p_pr.find(qn("w:framePr"))
    if frame_pr is None:
        return False
    p_pr.remove(frame_pr)
    return True


def _relax_table_row_exact_height(paragraph) -> bool:
    p_elm = getattr(paragraph, "_p", None)
    if p_elm is None:
        return False
    tr_height_tag = qn("w:trHeight")
    tr_pr_tag = qn("w:trPr")
    h_rule_attr = qn("w:hRule")

    for ancestor in p_elm.iterancestors():
        tag = str(getattr(ancestor, "tag", "")).lower()
        if not tag.endswith("}tr"):
            continue
        tr_pr = ancestor.find(tr_pr_tag)
        if tr_pr is None:
            return False
        changed = False
        for child in list(tr_pr):
            if child.tag != tr_height_tag:
                continue
            if str(child.get(h_rule_attr, "")).strip().lower() != "exact":
                continue
            tr_pr.remove(child)
            changed = True
        return changed
    return False


def reduce_paragraph_spacing(paragraph, factor: float = 0.8) -> bool:
    changed = False
    ratio = min(1.0, max(0.1, float(factor)))
    fmt = paragraph.paragraph_format
    before = fmt.space_before
    after = fmt.space_after
    if before is not None:
        new_before = max(0.0, before.pt * ratio)
        if new_before < before.pt:
            fmt.space_before = Pt(new_before)
            changed = True
    if after is not None:
        new_after = max(0.0, after.pt * ratio)
        if new_after < after.pt:
            fmt.space_after = Pt(new_after)
            changed = True
    if before is None and after is None:
        fmt.space_before = Pt(0.0)
        fmt.space_after = Pt(0.0)
        changed = True
    return changed


def set_single_line_spacing(paragraph) -> bool:
    fmt = paragraph.paragraph_format
    before = fmt.line_spacing
    fmt.line_spacing = 1.0
    return before != fmt.line_spacing


def insert_soft_wraps(paragraph, max_line_chars: int = 34) -> bool:
    if not paragraph.runs:
        return False
    non_empty_runs = [run for run in paragraph.runs if (run.text or "").strip()]
    # Rewriting run text can destroy inline formatting in rich paragraphs,
    # so soft wraps are applied only to simple single-run paragraphs.
    if len(non_empty_runs) != 1:
        return False
    text = "".join(run.text or "" for run in paragraph.runs)
    if len(text.strip()) <= max_line_chars:
        return False
    if "\n" in text:
        return False

    words = text.split()
    if len(words) < 4:
        return False

    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_line_chars:
            current = candidate
            continue
        chunks.append(current)
        current = word
    if current:
        chunks.append(current)
    if len(chunks) <= 1:
        return False

    # Minimal-touch rewrite: keep only first run text and clear rest.
    paragraph.runs[0].text = "\n".join(chunks)
    for run in paragraph.runs[1:]:
        run.text = ""
    return True


def _estimate_overflow_ratio(issue: Issue | None) -> float:
    if issue is None:
        return 1.0
    details = issue.details or {}
    try:
        target_len = float(details.get("target_len", 0) or 0)
        approx_capacity = float(details.get("approx_capacity_chars", 0) or 0)
    except (TypeError, ValueError):
        return 1.0
    if approx_capacity <= 0.0 or target_len <= 0.0:
        return 1.0
    return max(1.0, target_len / approx_capacity)


def _paragraph_average_font_pt(paragraph) -> float | None:
    sizes: list[float] = []
    for run in paragraph.runs:
        if not (run.text or "").strip():
            continue
        current_size = _resolve_run_size_pt(run, paragraph)
        if current_size is None:
            continue
        sizes.append(float(current_size))
    if not sizes:
        return None
    return sum(sizes) / len(sizes)


def _calculate_adaptive_reduction(
    overflow_ratio: float,
    current_font_pt: float,
    *,
    max_reduction: float = 3.0,
) -> float:
    if overflow_ratio <= 1.0:
        return 0.0
    needed = float(current_font_pt) * (1.0 - 1.0 / float(overflow_ratio))
    return min(float(max_reduction), max(0.2, needed))


def _fix_table_overflow(
    seg: Segment,
    cfg: PipelineConfig,
    *,
    issue: Issue | None = None,
    pass_number: int = 1,
) -> bool:
    if seg.paragraph_ref is None:
        return False

    changed = False
    overflow_ratio = _estimate_overflow_ratio(issue)
    cell = _paragraph_cell(seg.paragraph_ref)
    spacing_factor = float(cfg.layout_spacing_factor)
    if pass_number >= 2:
        spacing_factor *= 0.85
    if cell is not None:
        changed = _relax_table_row_exact_height(seg.paragraph_ref) or changed
        if len(cell.paragraphs) > 1:
            spacing_factor *= 0.85
        changed = reduce_cell_spacing(cell, factor=spacing_factor) or changed
        for paragraph in cell.paragraphs:
            if pass_number >= 3 and overflow_ratio >= 1.8:
                char_spacing_twips = -12 if pass_number == 3 else -15
                changed = reduce_character_spacing(paragraph, twips=char_spacing_twips) or changed
            if pass_number >= 3 and overflow_ratio >= 1.8:
                changed = set_single_line_spacing(paragraph) or changed
    else:
        changed = reduce_paragraph_spacing(seg.paragraph_ref, factor=spacing_factor) or changed
        if pass_number >= 3 and overflow_ratio >= 1.8:
            char_spacing_twips = -12 if pass_number == 3 else -15
            changed = reduce_character_spacing(seg.paragraph_ref, twips=char_spacing_twips) or changed
        if pass_number >= 3 and overflow_ratio >= 1.8:
            changed = set_single_line_spacing(seg.paragraph_ref) or changed

    avg_font_pt = _paragraph_average_font_pt(seg.paragraph_ref) or 10.0
    ratio_based_reduction = _calculate_adaptive_reduction(
        overflow_ratio,
        avg_font_pt,
        max_reduction=3.0 if pass_number <= 2 else 3.6,
    )
    table_font_reduction = max(float(cfg.layout_font_reduction_pt), ratio_based_reduction)
    if pass_number >= 3:
        table_font_reduction = max(table_font_reduction, 0.8)
    changed = reduce_font_size(
        seg.paragraph_ref,
        reduction_pt=table_font_reduction,
        min_font_pt=float(getattr(cfg, "layout_min_font_pt", 6.0)),
    ) or changed
    return changed


def _fix_textbox_overflow(
    seg: Segment,
    cfg: PipelineConfig,
    *,
    issue: Issue | None = None,
    pass_number: int = 1,
) -> bool:
    if seg.paragraph_ref is None:
        return False

    changed = False
    overflow_ratio = _estimate_overflow_ratio(issue)
    textbox_spacing_factor = min(0.9, max(0.4, float(cfg.layout_spacing_factor)))
    if pass_number >= 2:
        textbox_spacing_factor *= 0.85
    changed = reduce_paragraph_spacing(seg.paragraph_ref, factor=textbox_spacing_factor) or changed
    if pass_number >= 3 and overflow_ratio >= 1.6:
        char_spacing_twips = -10 if pass_number == 3 else -12
        changed = reduce_character_spacing(seg.paragraph_ref, twips=char_spacing_twips) or changed
    if pass_number >= 3:
        changed = set_single_line_spacing(seg.paragraph_ref) or changed
    avg_font_pt = _paragraph_average_font_pt(seg.paragraph_ref) or 10.0
    ratio_based_reduction = _calculate_adaptive_reduction(
        overflow_ratio,
        avg_font_pt,
        max_reduction=3.0 if pass_number <= 2 else 3.6,
    )
    textbox_font_reduction = max(float(cfg.layout_font_reduction_pt), ratio_based_reduction)
    if pass_number >= 3:
        textbox_font_reduction = max(textbox_font_reduction, 0.7)
    changed = reduce_font_size(
        seg.paragraph_ref,
        reduction_pt=textbox_font_reduction,
        min_font_pt=float(getattr(cfg, "layout_min_font_pt", 6.0)),
    ) or changed
    return changed


def _fix_frame_overflow(
    seg: Segment,
    cfg: PipelineConfig,
    *,
    issue: Issue | None = None,
    pass_number: int = 1,
) -> bool:
    if seg.paragraph_ref is None:
        return False

    changed = False
    overflow_ratio = _estimate_overflow_ratio(issue)
    if (overflow_ratio >= 1.35 and bool(seg.context.get("in_table"))) or pass_number >= 3:
        changed = _remove_paragraph_frame(seg.paragraph_ref) or changed
    changed = _relax_paragraph_frame_height_rule(seg.paragraph_ref) or changed
    frame_spacing_factor = min(0.92, max(0.5, float(cfg.layout_spacing_factor)))
    if pass_number >= 2:
        frame_spacing_factor *= 0.85
    changed = reduce_paragraph_spacing(seg.paragraph_ref, factor=frame_spacing_factor) or changed
    if pass_number >= 3 and overflow_ratio >= 1.5:
        char_spacing_twips = -10 if pass_number == 3 else -12
        changed = reduce_character_spacing(seg.paragraph_ref, twips=char_spacing_twips) or changed
    if pass_number >= 3 and overflow_ratio >= 1.5:
        changed = set_single_line_spacing(seg.paragraph_ref) or changed
    avg_font_pt = _paragraph_average_font_pt(seg.paragraph_ref) or 10.0
    ratio_based_reduction = _calculate_adaptive_reduction(
        overflow_ratio,
        avg_font_pt,
        max_reduction=2.6 if pass_number <= 2 else 3.2,
    )
    frame_font_reduction = max(float(cfg.layout_font_reduction_pt), max(0.4, ratio_based_reduction))
    if pass_number >= 3:
        frame_font_reduction = max(frame_font_reduction, 0.6)
    changed = reduce_font_size(
        seg.paragraph_ref,
        reduction_pt=frame_font_reduction,
        min_font_pt=float(getattr(cfg, "layout_min_font_pt", 6.0)),
    ) or changed
    return changed


def _fix_generic_overflow(seg: Segment, cfg: PipelineConfig, *, pass_number: int = 1) -> bool:
    if seg.paragraph_ref is None:
        return False

    changed = False
    spacing_factor = float(cfg.layout_spacing_factor)
    if pass_number >= 2:
        spacing_factor *= 0.9
    changed = reduce_paragraph_spacing(seg.paragraph_ref, factor=spacing_factor) or changed
    if pass_number >= 3:
        changed = reduce_character_spacing(seg.paragraph_ref, twips=-10) or changed
    base_reduction = max(0.2, float(cfg.layout_font_reduction_pt))
    if pass_number >= 3:
        base_reduction = max(base_reduction, 0.6)
        changed = set_single_line_spacing(seg.paragraph_ref) or changed
    changed = reduce_font_size(
        seg.paragraph_ref,
        reduction_pt=base_reduction,
        min_font_pt=float(getattr(cfg, "layout_min_font_pt", 6.0)),
    ) or changed
    return changed


def fix_expansion_issues(
    segments: list[Segment],
    issues: list[Issue],
    cfg: PipelineConfig,
    *,
    pass_number: int = 1,
) -> int:
    if not segments or not issues:
        return 0

    seg_by_id = {seg.segment_id: seg for seg in segments}
    actionable_codes = {
        "length_ratio_high",
        "layout_table_overflow_risk",
        "layout_textbox_overflow_risk",
        "layout_frame_overflow_risk",
    }
    fixed_segment_ids: set[str] = set()

    for issue in issues:
        if issue.code not in actionable_codes:
            continue
        segment_id = str(issue.details.get("segment_id", "")).strip()
        if not segment_id or segment_id in fixed_segment_ids:
            continue
        seg = seg_by_id.get(segment_id)
        if seg is None or seg.paragraph_ref is None:
            continue

        strategy = "generic"
        if issue.code == "layout_table_overflow_risk" or (
            issue.code == "length_ratio_high" and seg.context.get("in_table")
        ):
            strategy = "table"
            changed = _fix_table_overflow(seg, cfg, issue=issue, pass_number=pass_number)
        elif issue.code == "layout_textbox_overflow_risk" or (
            issue.code == "length_ratio_high" and seg.context.get("in_textbox")
        ):
            strategy = "textbox"
            changed = _fix_textbox_overflow(seg, cfg, issue=issue, pass_number=pass_number)
        elif issue.code == "layout_frame_overflow_risk" or (
            issue.code == "length_ratio_high" and seg.context.get("in_frame")
        ):
            strategy = "frame"
            changed = _fix_frame_overflow(seg, cfg, issue=issue, pass_number=pass_number)
        elif issue.code == "length_ratio_high":
            # Do not apply generic destructive fixes to normal body paragraphs.
            continue
        else:
            changed = _fix_generic_overflow(seg, cfg, pass_number=pass_number)
        if not changed:
            continue

        fixed_segment_ids.add(segment_id)
        seg.issues.append(
            Issue(
                code="layout_auto_fix_applied",
                severity=Severity.INFO,
                message="Applied layout auto-fix (spacing/font reduction).",
                details={
                    "segment_id": seg.segment_id,
                    "location": seg.location,
                    "source_issue_code": issue.code,
                    "strategy": strategy,
                    "pass_number": int(pass_number),
                    "font_reduction_pt": float(cfg.layout_font_reduction_pt),
                    "spacing_factor": float(cfg.layout_spacing_factor),
                },
            )
        )

    return len(fixed_segment_ids)
