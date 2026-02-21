from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .models import Issue, Segment, Severity


def write_qa_jsonl(segments: Iterable[Segment], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for seg in segments:
            if not seg.issues:
                continue
            for issue in seg.issues:
                rec = {
                    "segment_id": seg.segment_id,
                    "location": seg.location,
                    "code": issue.code,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "details": issue.details,
                    "source": seg.source_plain,
                    "target": seg.target_tagged or "",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_qa_report(segments: Iterable[Segment], path: Path) -> None:
    rows: list[str] = []

    def esc(s: str) -> str:
        return html.escape(s, quote=True)

    for seg in segments:
        if not seg.issues:
            continue
        src = seg.source_plain
        tgt = seg.target_tagged or ""
        for issue in seg.issues:
            rows.append(
                "<tr class='issue' data-sev='{sev}' data-code='{code}'>".format(
                    sev=issue.severity.value, code=esc(issue.code)
                )
                + f"<td class='sev'>{esc(issue.severity.value)}</td>"
                + f"<td class='code'>{esc(issue.code)}</td>"
                + f"<td class='loc'>{esc(seg.location)}</td>"
                + f"<td class='msg'>{esc(issue.message)}</td>"
                + "<td class='details'><details><summary>details</summary><pre>"
                + esc(json.dumps(issue.details, ensure_ascii=False, indent=2))
                + "</pre></details></td>"
                + "<td class='text'><details><summary>text</summary><div class='grid'>"
                + f"<pre class='src'>{esc(src)}</pre>"
                + f"<pre class='tgt'>{esc(tgt)}</pre>"
                + "</div></details></td>"
                + "</tr>"
            )

    html_doc = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<title>docxru QA report</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 16px; }}
h1 {{ margin: 0 0 12px 0; }}
.controls {{ display: flex; gap: 12px; align-items: center; margin: 12px 0; flex-wrap: wrap; }}
input, select {{ padding: 6px 8px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
th {{ background: #f6f6f6; text-align: left; position: sticky; top: 0; }}
tr[data-sev="error"] .sev {{ font-weight: 700; }}
pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
.src {{ background: #fbfbff; padding: 8px; }}
.tgt {{ background: #fbfffb; padding: 8px; }}
.small {{ color: #666; font-size: 12px; }}
</style>
</head>
<body>
<h1>docxru — QA report</h1>
<div class="small">Фильтры работают без отправки данных наружу (локально, JS).</div>

<div class="controls">
  <label>Severity:
    <select id="sev">
      <option value="">all</option>
      <option value="error">error</option>
      <option value="warn">warn</option>
      <option value="info">info</option>
    </select>
  </label>
  <label>Code:
    <input id="code" placeholder="e.g. placeholders_mismatch"/>
  </label>
  <label>Search:
    <input id="q" placeholder="location/text"/>
  </label>
  <button id="reset">reset</button>
</div>

<table>
<thead>
<tr>
  <th>sev</th>
  <th>code</th>
  <th>location</th>
  <th>message</th>
  <th>details</th>
  <th>source/target</th>
</tr>
</thead>
<tbody id="tbody">
{''.join(rows) if rows else '<tr><td colspan="6">No issues 🎉</td></tr>'}
</tbody>
</table>

<script>
const sevSel = document.getElementById('sev');
const codeInp = document.getElementById('code');
const qInp = document.getElementById('q');
const resetBtn = document.getElementById('reset');

function applyFilters() {{
  const sev = sevSel.value.trim();
  const code = codeInp.value.trim().toLowerCase();
  const q = qInp.value.trim().toLowerCase();

  document.querySelectorAll('tr.issue').forEach(tr => {{
    const trSev = tr.getAttribute('data-sev');
    const trCode = tr.getAttribute('data-code').toLowerCase();
    const text = tr.innerText.toLowerCase();

    let ok = true;
    if (sev && trSev !== sev) ok = false;
    if (code && !trCode.includes(code)) ok = false;
    if (q && !text.includes(q)) ok = false;

    tr.style.display = ok ? '' : 'none';
  }});
}}

[sevSel, codeInp, qInp].forEach(el => el.addEventListener('input', applyFilters));
resetBtn.addEventListener('click', () => {{
  sevSel.value = '';
  codeInp.value = '';
  qInp.value = '';
  applyFilters();
}});
</script>
</body>
</html>
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_doc, encoding="utf-8")
