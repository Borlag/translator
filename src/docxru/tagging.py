from __future__ import annotations

import re
from typing import Tuple

from docx.text.paragraph import Paragraph
from docx.text.run import Run

from docx.enum.text import WD_BREAK
from docx.oxml import parse_xml
from docx.oxml.ns import qn

from .models import RunStyleSnapshot, Span

_STYLE_START_RE = re.compile(r"⟦S_(\d+)(?:\|[^⟧]*)?⟧")
_STYLE_END_RE = re.compile(r"⟦/S_(\d+)⟧")

# Inline special tokens that we inject into tagged text.
_BRACKET_TOKEN_RE = re.compile(r"⟦[^⟧]*⟧")

# Allowed simple run child elements; any other content triggers XML-preserve mode (OBJ token).
_ALLOWED_RUN_CHILDREN = {
    "rPr",
    "t",
    "tab",
    "br",
    "cr",
    "noBreakHyphen",
    "softHyphen",
    "lastRenderedPageBreak",
}


def is_supported_paragraph(paragraph: Paragraph) -> bool:
    """Return True if paragraph can be safely rebuilt by docxru.

    Current constraint (safe default): the paragraph must contain only <w:pPr> and <w:r>
    as direct children. Paragraphs with hyperlinks, content controls, bookmarks, etc. are skipped
    to avoid reordering and structure corruption.
    """

    # Some "noise" markers (proofing/bookmarks) are safe to keep; they don't affect visible text and
    # are common in OEM manuals. Treat them as supported so we can translate with full paragraph context.
    allowed_extra = {"bookmarkstart", "bookmarkend", "prooferr"}
    for child in paragraph._p.iterchildren():
        tag = child.tag.lower()
        if tag.endswith("}ppr") or tag.endswith("}r"):
            continue
        local = tag.split("}")[-1]
        if local in allowed_extra:
            continue
        return False
    return True


def _run_needs_xml_preserve(run: Run) -> bool:
    """True if run contains non-text content that python-docx cannot roundtrip via run.text."""
    for child in run._r.iterchildren():
        local = child.tag.split("}")[-1]
        if local in _ALLOWED_RUN_CHILDREN:
            continue
        return True
    return False


def _break_token(br_type: str | None, counters: dict[str, int]) -> str:
    # Normalize br_type -> token prefix
    if br_type is None or br_type in ("textWrapping", "line"):
        name = "BRLINE"
    elif br_type == "column":
        name = "BRCOL"
    elif br_type == "page":
        name = "BRPAGE"
    else:
        # Fallback: keep line break (safe)
        name = "BRLINE"
    counters[name] = counters.get(name, 0) + 1
    return f"⟦{name}_{counters[name]}⟧"


def _extract_run_text_with_special_tokens(run: Run, br_counters: dict[str, int]) -> str:
    """Serialize run content into plain text + special ⟦...⟧ tokens.

    Important: this is used ONLY for 'simple' runs (no drawings/fields/etc).
    """
    out: list[str] = []
    for child in run._r.iterchildren():
        local = child.tag.split("}")[-1]
        if local == "rPr":
            continue
        if local == "t":
            out.append(child.text or "")
            continue
        if local == "tab":
            out.append("\t")
            continue
        if local == "br":
            br_type = child.get(qn("w:type"))
            out.append(_break_token(br_type, br_counters))
            continue
        if local == "cr":
            out.append(_break_token("line", br_counters))
            continue
        if local == "noBreakHyphen":
            # non-breaking hyphen
            out.append("\u2011")
            continue
        if local == "softHyphen":
            out.append("\u00ad")
            continue
        if local == "lastRenderedPageBreak":
            # Word pagination cache marker; not visible content.
            continue
        # Should not happen for simple runs, but don't crash.
        out.append(run.text or "")
    return "".join(out)


def _snapshot_from_run(run: Run) -> RunStyleSnapshot:
    font = run.font
    color_rgb = None
    try:
        if font.color is not None and font.color.rgb is not None:
            color_rgb = str(font.color.rgb)  # e.g. 'FF0000'
    except Exception:
        color_rgb = None

    size_pt = None
    try:
        if font.size is not None:
            size_pt = float(font.size.pt)  # type: ignore[assignment]
    except Exception:
        size_pt = None

    return RunStyleSnapshot(
        bold=run.bold,
        italic=run.italic,
        underline=run.underline if isinstance(run.underline, bool) else None,
        superscript=font.superscript,
        subscript=font.subscript,
        font_name=font.name,
        font_size_pt=size_pt,
        color_rgb=color_rgb,
        all_caps=font.all_caps,
        small_caps=font.small_caps,
    )


def _rpr_xml_from_run(run: Run) -> str | None:
    """Return serialized <w:rPr> for exact style restoration if present."""
    try:
        rpr = run._r.rPr
    except Exception:
        rpr = None
    if rpr is None:
        return None
    try:
        return rpr.xml
    except Exception:
        return None


def _flags_from_run(run: Run) -> tuple[str, ...]:
    flags: list[str] = []
    if run.bold:
        flags.append("B")
    if run.italic:
        flags.append("I")
    if run.underline:
        flags.append("U")
    if run.font.superscript:
        flags.append("SUP")
    if run.font.subscript:
        flags.append("SUB")
    return tuple(flags)


def paragraph_to_tagged(paragraph: Paragraph) -> tuple[str, list[Span], dict[str, str]]:
    """Convert paragraph runs into a single tagged string + span metadata.

    We DO NOT translate per-run. We only use runs to capture formatting spans.

    Output format:
      ⟦S_1|B|I⟧text⟦/S_1⟧⟦S_2⟧plain⟦/S_2⟧...
    """
    # Guard against collisions with our marker alphabet.
    full_text = "".join(r.text for r in paragraph.runs)
    if "⟦" in full_text or "⟧" in full_text:
        raise ValueError(
            "Source paragraph contains reserved marker characters '⟦' or '⟧'. "

            "Please pre-clean the document or change tagging delimiters."
        )

    merged: list[tuple[tuple[str, ...], RunStyleSnapshot, str | None, str]] = []
    inline_run_map: dict[str, str] = {}
    obj_counter = 0
    br_counters: dict[str, int] = {}

    for run in paragraph.runs:
        flags = _flags_from_run(run)
        snap = _snapshot_from_run(run)
        rpr_xml = _rpr_xml_from_run(run)

        if _run_needs_xml_preserve(run):
            # Preserve the entire <w:r> XML as an opaque inline object token.
            obj_counter += 1
            token = f"⟦OBJ_{obj_counter}⟧"
            inline_run_map[token] = run._r.xml
            text = token
        else:
            text = _extract_run_text_with_special_tokens(run, br_counters)

        if text == "":
            continue

        # Merge by visible style snapshot only.
        #
        # In large OEM manuals Word often splits text into many runs that differ only by
        # non-visual rPr details (proofing/language metadata). If we require exact rPr XML
        # equality for merge, translation degrades into near word-by-word mode and quality
        # drops sharply. Keep the first run's rPr XML for restoration and merge by style.
        if merged and merged[-1][0] == flags and merged[-1][1] == snap:
            merged[-1] = (merged[-1][0], merged[-1][1], merged[-1][2], merged[-1][3] + text)
        else:
            merged.append((flags, snap, rpr_xml, text))

    if not merged:
        return ("", [], inline_run_map)

    spans: list[Span] = []
    parts: list[str] = []

    for i, (flags, snap, rpr_xml, text) in enumerate(merged, start=1):
        spans.append(Span(span_id=i, flags=flags, source_text=text, style=snap, rpr_xml=rpr_xml))
        if flags:
            flag_part = "|" + "|".join(flags)
        else:
            flag_part = ""
        parts.append(f"⟦S_{i}{flag_part}⟧")
        parts.append(text)
        parts.append(f"⟦/S_{i}⟧")

    return ("".join(parts), spans, inline_run_map)


def _clear_paragraph_runs(paragraph: Paragraph) -> None:
    # Remove runs safely by removing underlying XML elements.
    for run in list(paragraph.runs):
        try:
            run._element.getparent().remove(run._element)  # type: ignore[attr-defined]
        except Exception:
            # Fallback: set to empty
            run.text = ""


def _apply_rpr_xml(run: Run, rpr_xml: str) -> bool:
    """Apply exact run properties XML. Returns True on success."""
    try:
        r = run._r
        old = r.find(qn("w:rPr"))
        if old is not None:
            r.remove(old)
        new = parse_xml(rpr_xml)
        r.insert(0, new)
        return True
    except Exception:
        return False


def _apply_style(run: Run, span: Span) -> None:
    # Exact XML restore gives the best layout fidelity. Fall back to coarse properties if needed.
    if span.rpr_xml and _apply_rpr_xml(run, span.rpr_xml):
        return

    # Apply formatting flags first (explicit).
    run.bold = "B" in span.flags
    run.italic = "I" in span.flags
    run.underline = "U" in span.flags
    run.font.superscript = "SUP" in span.flags
    run.font.subscript = "SUB" in span.flags

    # Restore snapshot where safe (non-None only).
    s = span.style
    if s.font_name is not None:
        run.font.name = s.font_name
    if s.font_size_pt is not None:
        try:
            from docx.shared import Pt

            run.font.size = Pt(s.font_size_pt)
        except Exception:
            pass
    if s.color_rgb is not None:
        try:
            from docx.shared import RGBColor

            run.font.color.rgb = RGBColor.from_string(s.color_rgb)
        except Exception:
            pass
    if s.all_caps is not None:
        run.font.all_caps = s.all_caps
    if s.small_caps is not None:
        run.font.small_caps = s.small_caps


def tagged_to_runs(
    paragraph: Paragraph,
    tagged_text: str,
    spans: list[Span],
    inline_run_map: dict[str, str] | None = None,
    spans_schema_version: int = 1,
) -> None:
    """Rebuild paragraph runs from translated tagged text using original span styles.

    - Removes all existing runs in the paragraph.
    - Parses only our style tags ⟦S_n...⟧ ... ⟦/S_n⟧.
    - Unknown ⟦...⟧ blocks are treated as plain text.

    Assumption: LLM preserved tag structure. Validator should enforce that.
    """
    if spans_schema_version != 1:
        raise ValueError(f"Unsupported spans_schema_version={spans_schema_version}")

    spans_by_id = {s.span_id: s for s in spans}

    pieces: list[tuple[str, int | None]] = []
    pos = 0
    while True:
        m = _STYLE_START_RE.search(tagged_text, pos)
        if not m:
            if pos < len(tagged_text):
                pieces.append((tagged_text[pos:], None))
            break

        if m.start() > pos:
            pieces.append((tagged_text[pos : m.start()], None))

        span_id = int(m.group(1))
        # Find the corresponding end tag
        end_pat = re.compile(rf"⟦/S_{span_id}⟧")
        m_end = end_pat.search(tagged_text, m.end())
        if not m_end:
            # Treat start tag as literal text
            pieces.append((tagged_text[m.start() : m.end()], None))
            pos = m.end()
            continue

        inner = tagged_text[m.end() : m_end.start()]
        pieces.append((inner, span_id))
        pos = m_end.end()

    _clear_paragraph_runs(paragraph)

    def _emit_text(txt: str, span_id: int | None) -> None:
        if txt == "":
            return
        run = paragraph.add_run(txt)
        if span_id is None:
            return
        span = spans_by_id.get(span_id)
        if span is None:
            return
        _apply_style(run, span)

    def _emit_break(token: str, span_id: int | None) -> bool:
        # token includes brackets, e.g. ⟦BRCOL_1⟧
        inner = token[1:-1]  # strip ⟦ ⟧
        if "_" not in inner:
            return False
        name, _ = inner.rsplit("_", 1)
        name = name.upper()
        if name not in {"BRLINE", "BRCOL", "BRPAGE"}:
            return False
        run = paragraph.add_run("")
        if span_id is not None:
            span = spans_by_id.get(span_id)
            if span is not None:
                _apply_style(run, span)
        if name == "BRCOL":
            run.add_break(WD_BREAK.COLUMN)
        elif name == "BRPAGE":
            run.add_break(WD_BREAK.PAGE)
        else:
            run.add_break(WD_BREAK.LINE)
        return True

    for chunk, span_id in pieces:
        if chunk == "":
            continue

        pos = 0
        for m in _BRACKET_TOKEN_RE.finditer(chunk):
            if m.start() > pos:
                _emit_text(chunk[pos : m.start()], span_id)

            tok = m.group(0)

            if inline_run_map and tok in inline_run_map:
                # Insert preserved run XML.
                try:
                    paragraph._p.append(parse_xml(inline_run_map[tok]))
                except Exception:
                    # Fallback: emit token text to avoid data loss.
                    _emit_text(tok, span_id)
            else:
                if not _emit_break(tok, span_id):
                    # Unknown token: keep as literal text.
                    _emit_text(tok, span_id)

            pos = m.end()

        if pos < len(chunk):
            _emit_text(chunk[pos:], span_id)
