from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Severity(str, Enum):
    ERROR = "error"
    WARN = "warn"
    INFO = "info"


@dataclass(frozen=True)
class RunStyleSnapshot:
    """Subset of run properties that we can safely restore with python-docx.

    Note: Word has many more properties; this snapshot is a pragmatic MVP.
    """

    bold: Optional[bool] = None
    italic: Optional[bool] = None
    underline: Optional[bool] = None
    superscript: Optional[bool] = None
    subscript: Optional[bool] = None
    font_name: Optional[str] = None
    font_size_pt: Optional[float] = None
    color_rgb: Optional[str] = None  # hex without '#', e.g. 'FF0000'
    all_caps: Optional[bool] = None
    small_caps: Optional[bool] = None


@dataclass(frozen=True)
class Span:
    """A merged span of text that shares the same formatting flags."""

    span_id: int
    flags: tuple[str, ...]  # e.g. ("B", "I", "U", "SUP", "SUB")
    source_text: str
    style: RunStyleSnapshot = RunStyleSnapshot()
    # Full <w:rPr> XML snapshot for high-fidelity style restore.
    rpr_xml: str | None = None


@dataclass
class Issue:
    code: str
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Segment:
    """A translation unit: usually a Paragraph inside body/header/footer/table cell."""

    segment_id: str
    location: str
    context: dict[str, Any]
    source_plain: str
    # For writing back:
    paragraph_ref: Any  # python-docx Paragraph (kept as Any to avoid heavy typing)
    # Intermediate
    source_tagged: str | None = None
    spans: list[Span] | None = None
    # Inline run XML placeholders (e.g., drawings, field codes, column/page breaks)
    # placeholder token -> serialized <w:r> XML string
    inline_run_map: dict[str, str] | None = None
    shielded_tagged: str | None = None
    token_map: dict[str, str] | None = None
    target_shielded_tagged: str | None = None
    target_tagged: str | None = None
    issues: list[Issue] = field(default_factory=list)
