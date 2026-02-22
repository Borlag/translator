from __future__ import annotations

import re
from collections.abc import Iterable

from docx.oxml.ns import qn

from .config import PipelineConfig
from .models import Issue, Segment, Severity
from .token_shield import strip_bracket_tokens

_SPACE_RE = re.compile(r"\s+")
_EMU_PER_TWIP = 635
_APPROX_CHAR_WIDTH_TWIPS = 120
_APPROX_CHAR_HEIGHT_TWIPS = 220


def _clean_text(value: str) -> str:
    text = strip_bracket_tokens(value or "")
    return _SPACE_RE.sub(" ", text).strip()


def _segment_target_text(seg: Segment) -> str:
    return _clean_text(seg.target_tagged or seg.target_shielded_tagged or "")


def _segment_source_text(seg: Segment) -> str:
    return _clean_text(seg.source_plain or "")


def _try_parse_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _table_cell_width_twips(seg: Segment) -> int | None:
    paragraph = seg.paragraph_ref
    if paragraph is None:
        return None

    try:
        p_elm = paragraph._p
    except Exception:
        return None

    width_attr = qn("w:w")
    for ancestor in p_elm.iterancestors():
        if not str(getattr(ancestor, "tag", "")).lower().endswith("}tc"):
            continue
        for child in ancestor.iterchildren():
            if not str(getattr(child, "tag", "")).lower().endswith("}tcpr"):
                continue
            for grandchild in child.iterchildren():
                if not str(getattr(grandchild, "tag", "")).lower().endswith("}tcw"):
                    continue
                width = _try_parse_int(grandchild.get(width_attr))
                if width is not None and width > 0:
                    return width
    return None


def _textbox_extent_twips(seg: Segment) -> tuple[int | None, int | None]:
    paragraph = seg.paragraph_ref
    if paragraph is None:
        return None, None

    try:
        p_elm = paragraph._p
    except Exception:
        return None, None

    for ancestor in [p_elm, *list(p_elm.iterancestors())]:
        for node in ancestor.iter():
            tag = str(getattr(node, "tag", "")).lower()
            if not tag.endswith("}extent"):
                continue
            cx = _try_parse_int(node.get("cx"))
            cy = _try_parse_int(node.get("cy"))
            if cx is None or cx <= 0 or cy is None or cy <= 0:
                continue
            # 1 twip ~= 635 EMU.
            width_twips = int(cx / _EMU_PER_TWIP)
            height_twips = int(cy / _EMU_PER_TWIP)
            if width_twips > 0 and height_twips > 0:
                return width_twips, height_twips
    return None, None


def check_text_expansion(
    segments: Iterable[Segment],
    *,
    warn_ratio: float = 1.5,
) -> list[Issue]:
    issues: list[Issue] = []
    threshold = max(1.05, float(warn_ratio))
    for seg in segments:
        source_text = _segment_source_text(seg)
        target_text = _segment_target_text(seg)
        if not source_text or not target_text:
            continue
        ratio = len(target_text) / max(1, len(source_text))
        if ratio <= threshold:
            continue
        issues.append(
            Issue(
                code="length_ratio_high",
                severity=Severity.WARN,
                message="Translated text expansion ratio is high for this segment.",
                details={
                    "segment_id": seg.segment_id,
                    "location": seg.location,
                    "ratio": round(ratio, 3),
                    "source_len": len(source_text),
                    "target_len": len(target_text),
                    "warn_ratio": threshold,
                },
            )
        )
    return issues


def check_table_cell_overflow(doc, segments: Iterable[Segment]) -> list[Issue]:
    del doc  # The check currently relies on paragraph refs from segments.
    issues: list[Issue] = []
    for seg in segments:
        if not seg.context.get("in_table"):
            continue
        target_text = _segment_target_text(seg)
        if not target_text:
            continue
        width_twips = _table_cell_width_twips(seg)
        if width_twips is None:
            continue
        approx_capacity = max(1, int(width_twips / _APPROX_CHAR_WIDTH_TWIPS))
        if len(target_text) <= int(approx_capacity * 1.15):
            continue
        issues.append(
            Issue(
                code="layout_table_overflow_risk",
                severity=Severity.WARN,
                message="Table-cell content may overflow the available width.",
                details={
                    "segment_id": seg.segment_id,
                    "location": seg.location,
                    "width_twips": width_twips,
                    "approx_capacity_chars": approx_capacity,
                    "target_len": len(target_text),
                },
            )
        )
    return issues


def check_textbox_overflow(doc, segments: Iterable[Segment]) -> list[Issue]:
    del doc  # The check currently relies on paragraph refs from segments.
    issues: list[Issue] = []
    for seg in segments:
        if not seg.context.get("in_textbox"):
            continue
        source_text = _segment_source_text(seg)
        target_text = _segment_target_text(seg)
        if not source_text or not target_text:
            continue

        ratio = len(target_text) / max(1, len(source_text))
        width_twips, height_twips = _textbox_extent_twips(seg)
        approx_capacity = None
        if width_twips and height_twips:
            area = width_twips * height_twips
            approx_char_area = _APPROX_CHAR_WIDTH_TWIPS * _APPROX_CHAR_HEIGHT_TWIPS
            approx_capacity = max(1, int(area / approx_char_area))
        elif width_twips:
            approx_capacity = max(1, int(width_twips / _APPROX_CHAR_WIDTH_TWIPS))

        overflow = False
        if approx_capacity is not None:
            overflow = len(target_text) > int(approx_capacity * 1.1)
        else:
            overflow = ratio > 1.6 and (len(target_text) - len(source_text)) >= 20
        if not overflow:
            continue

        details = {
            "segment_id": seg.segment_id,
            "location": seg.location,
            "ratio": round(ratio, 3),
            "source_len": len(source_text),
            "target_len": len(target_text),
        }
        if width_twips is not None:
            details["width_twips"] = width_twips
        if height_twips is not None:
            details["height_twips"] = height_twips
        if approx_capacity is not None:
            details["approx_capacity_chars"] = approx_capacity

        issues.append(
            Issue(
                code="layout_textbox_overflow_risk",
                severity=Severity.WARN,
                message="Textbox content may overflow available area.",
                details=details,
            )
        )
    return issues


def validate_layout(doc, segments: list[Segment], cfg: PipelineConfig) -> list[Issue]:
    return [
        *check_text_expansion(segments, warn_ratio=cfg.layout_expansion_warn_ratio),
        *check_table_cell_overflow(doc, segments),
        *check_textbox_overflow(doc, segments),
    ]
