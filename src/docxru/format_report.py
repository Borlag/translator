from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable

from .models import Issue, Segment

_OVERFLOW_CODES = {
    "layout_table_overflow_risk",
    "layout_textbox_overflow_risk",
    "layout_frame_overflow_risk",
    "length_ratio_high",
}


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _collect_rows(
    segments: Iterable[Segment],
    *,
    include: set[str],
) -> list[tuple[Segment, Issue]]:
    rows: list[tuple[Segment, Issue]] = []
    for seg in segments:
        for issue in seg.issues:
            if issue.code in include:
                rows.append((seg, issue))
    return rows


def _render_rows(rows: list[tuple[Segment, Issue]]) -> str:
    if not rows:
        return "<tr><td colspan='5'>None</td></tr>"
    out: list[str] = []
    for seg, issue in rows:
        out.append(
            "<tr>"
            + f"<td>{_esc(issue.code)}</td>"
            + f"<td>{_esc(seg.location)}</td>"
            + f"<td>{_esc(issue.message)}</td>"
            + f"<td><pre>{_esc(json.dumps(issue.details, ensure_ascii=False, indent=2))}</pre></td>"
            + f"<td><pre>{_esc(seg.target_tagged or '')}</pre></td>"
            + "</tr>"
        )
    return "".join(out)


def write_format_report(segments: Iterable[Segment], path: Path) -> None:
    segments_list = list(segments)
    shrunk_rows = _collect_rows(
        segments_list,
        include={"global_font_shrink_applied", "layout_auto_fix_applied"},
    )
    spacing_rows = [
        (seg, issue)
        for seg, issue in shrunk_rows
        if issue.code == "layout_auto_fix_applied" and issue.details.get("spacing_factor") is not None
    ]
    abbyy_rows = _collect_rows(segments_list, include={"abbyy_normalization_applied"})
    overflow_rows = _collect_rows(segments_list, include=_OVERFLOW_CODES)
    writeback_rows = _collect_rows(
        segments_list,
        include={"writeback_inplace_fallback", "writeback_inplace_error"},
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>docxru formatting report</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 16px; }}
h1 {{ margin: 0 0 8px 0; }}
h2 {{ margin: 18px 0 8px 0; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
th {{ background: #f6f6f6; text-align: left; }}
pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; }}
.meta {{ color: #666; font-size: 12px; margin-bottom: 12px; }}
</style>
</head>
<body>
<h1>docxru - Formatting Report</h1>
<div class="meta">
Segments: {len(segments_list)} |
font/auto-fix entries: {len(shrunk_rows)} |
remaining overflow entries: {len(overflow_rows)} |
in-place fallback entries: {len(writeback_rows)}
</div>

<h2>Font Shrink / Auto-Fix</h2>
<table>
<thead><tr><th>code</th><th>location</th><th>message</th><th>details</th><th>target</th></tr></thead>
<tbody>{_render_rows(shrunk_rows)}</tbody>
</table>

<h2>Spacing Changes</h2>
<table>
<thead><tr><th>code</th><th>location</th><th>message</th><th>details</th><th>target</th></tr></thead>
<tbody>{_render_rows(spacing_rows)}</tbody>
</table>

<h2>ABBYY Normalization Markers</h2>
<table>
<thead><tr><th>code</th><th>location</th><th>message</th><th>details</th><th>target</th></tr></thead>
<tbody>{_render_rows(abbyy_rows)}</tbody>
</table>

<h2>Remaining Overflow Risks</h2>
<table>
<thead><tr><th>code</th><th>location</th><th>message</th><th>details</th><th>target</th></tr></thead>
<tbody>{_render_rows(overflow_rows)}</tbody>
</table>

<h2>In-Place Writeback Fallbacks</h2>
<table>
<thead><tr><th>code</th><th>location</th><th>message</th><th>details</th><th>target</th></tr></thead>
<tbody>{_render_rows(writeback_rows)}</tbody>
</table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_doc, encoding="utf-8")
