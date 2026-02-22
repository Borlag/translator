from __future__ import annotations

from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.table import _Cell

from .config import PipelineConfig
from .models import Issue, Segment, Severity


def reduce_font_size(paragraph, reduction_pt: float = 0.5) -> bool:
    changed = False
    step = max(0.1, float(reduction_pt))
    for run in paragraph.runs:
        current = run.font.size.pt if run.font.size is not None else None
        if current is None:
            continue
        updated = max(6.0, float(current) - step)
        if updated < float(current):
            run.font.size = Pt(updated)
            changed = True

    if changed:
        return True

    # If explicit run sizes are missing, apply a small fallback size to text runs.
    fallback = max(6.0, 10.0 - step)
    for run in paragraph.runs:
        if not (run.text or "").strip():
            continue
        run.font.size = Pt(fallback)
        changed = True
    return changed


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


def insert_soft_wraps(paragraph, max_line_chars: int = 34) -> bool:
    if not paragraph.runs:
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


def fix_expansion_issues(
    segments: list[Segment],
    issues: list[Issue],
    cfg: PipelineConfig,
) -> int:
    if not segments or not issues:
        return 0

    seg_by_id = {seg.segment_id: seg for seg in segments}
    actionable_codes = {
        "length_ratio_high",
        "layout_table_overflow_risk",
        "layout_textbox_overflow_risk",
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

        changed = False
        # First attempt: reduce spacing pressure before touching font size.
        if seg.context.get("in_table"):
            cell = _paragraph_cell(seg.paragraph_ref)
            if cell is not None:
                spacing_factor = float(cfg.layout_spacing_factor)
                if len(cell.paragraphs) > 1:
                    spacing_factor *= 0.85
                changed = reduce_cell_spacing(cell, factor=spacing_factor) or changed
                for paragraph in cell.paragraphs:
                    changed = reduce_character_spacing(paragraph, twips=-10) or changed
        else:
            changed = reduce_paragraph_spacing(seg.paragraph_ref, factor=cfg.layout_spacing_factor) or changed
            changed = reduce_character_spacing(seg.paragraph_ref, twips=-10) or changed

        # Second attempt: reduce explicit run font sizes.
        changed = reduce_font_size(seg.paragraph_ref, reduction_pt=cfg.layout_font_reduction_pt) or changed
        # Last resort: inject soft wraps for long labels.
        changed = insert_soft_wraps(seg.paragraph_ref) or changed
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
                    "font_reduction_pt": float(cfg.layout_font_reduction_pt),
                    "spacing_factor": float(cfg.layout_spacing_factor),
                },
            )
        )

    return len(fixed_segment_ids)
