from __future__ import annotations

import re
from typing import Any

from docx.enum.text import WD_BREAK
from docx.oxml import parse_xml
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.text.run import Run

from .models import RunStyleSnapshot, Span

_STYLE_START_RE = re.compile(r"⟦S_(\d+)(?:\|[^⟧]*)?⟧")
_STYLE_END_RE = re.compile(r"⟦/S_(\d+)⟧")
_HYPERLINK_FLAG_RE = re.compile(r"^HREF_(\d+)$")
_HYPERLINK_META_PREFIX = "__DOCXRU_HREF_"

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

    Supported direct children: <w:pPr>, <w:r>, <w:hyperlink> and non-visual markers.
    """

    # Some "noise" markers (proofing/bookmarks) are safe to keep; they don't affect visible text and
    # are common in OEM manuals. Treat them as supported so we can translate with full paragraph context.
    allowed_extra = {"bookmarkstart", "bookmarkend", "prooferr"}
    for child in paragraph._p.iterchildren():
        tag = child.tag.lower()
        if tag.endswith("}ppr") or tag.endswith("}r") or tag.endswith("}hyperlink"):
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


def _hyperlink_meta_key(index: int) -> str:
    return f"{_HYPERLINK_META_PREFIX}{index}__"


def _iter_runs_with_hyperlink_context(
    paragraph: Paragraph,
    inline_run_map: dict[str, str],
) -> list[tuple[Run, str | None]]:
    runs: list[tuple[Run, str | None]] = []
    href_index = 0
    for child in paragraph._p.iterchildren():
        local = child.tag.split("}")[-1].lower()
        if local == "r":
            runs.append((Run(child, paragraph), None))
            continue
        if local != "hyperlink":
            continue

        href_index += 1
        href_flag = f"HREF_{href_index}"
        inline_run_map[_hyperlink_meta_key(href_index)] = child.xml
        for sub in child.iterchildren():
            if sub.tag.split("}")[-1].lower() != "r":
                continue
            runs.append((Run(sub, paragraph), href_flag))
    return runs


def paragraph_to_tagged(paragraph: Paragraph) -> tuple[str, list[Span], dict[str, str]]:
    """Convert paragraph runs into a single tagged string + span metadata.

    We DO NOT translate per-run. We only use runs to capture formatting spans.

    Output format:
      ⟦S_1|B|I⟧text⟦/S_1⟧⟦S_2⟧plain⟦/S_2⟧...
    """
    # Guard against collisions with our marker alphabet.
    full_text = paragraph.text or ""
    if "⟦" in full_text or "⟧" in full_text:
        raise ValueError(
            "Source paragraph contains reserved marker characters '⟦' or '⟧'. "

            "Please pre-clean the document or change tagging delimiters."
        )

    merged: list[tuple[tuple[str, ...], RunStyleSnapshot, str | None, str, list[int]]] = []
    inline_run_map: dict[str, str] = {}
    obj_counter = 0
    br_counters: dict[str, int] = {}

    for run, href_flag in _iter_runs_with_hyperlink_context(paragraph, inline_run_map):
        flags = list(_flags_from_run(run))
        if href_flag:
            flags.append(href_flag)
        flags_tuple = tuple(flags)
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
        if merged and merged[-1][0] == flags_tuple and merged[-1][1] == snap:
            merged[-1] = (
                merged[-1][0],
                merged[-1][1],
                merged[-1][2],
                merged[-1][3] + text,
                [*merged[-1][4], len(text)],
            )
        else:
            merged.append((flags_tuple, snap, rpr_xml, text, [len(text)]))

    if not merged:
        return ("", [], inline_run_map)

    spans: list[Span] = []
    parts: list[str] = []

    for i, (flags, snap, rpr_xml, text, run_lengths) in enumerate(merged, start=1):
        spans.append(
            Span(
                span_id=i,
                flags=flags,
                source_text=text,
                original_run_lengths=tuple(run_lengths),
                style=snap,
                rpr_xml=rpr_xml,
            )
        )
        if flags:
            flag_part = "|" + "|".join(flags)
        else:
            flag_part = ""
        parts.append(f"⟦S_{i}{flag_part}⟧")
        parts.append(text)
        parts.append(f"⟦/S_{i}⟧")

    return ("".join(parts), spans, inline_run_map)


def _clear_paragraph_runs(paragraph: Paragraph) -> None:
    # Remove direct run/hyperlink children but keep paragraph properties and non-visual markers.
    for child in list(paragraph._p.iterchildren()):
        local = child.tag.split("}")[-1].lower()
        if local not in {"r", "hyperlink"}:
            continue
        try:
            paragraph._p.remove(child)
        except Exception:
            continue


def _span_hyperlink_index(span: Span | None) -> int | None:
    if span is None:
        return None
    for flag in span.flags:
        m = _HYPERLINK_FLAG_RE.fullmatch(flag)
        if not m:
            continue
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _build_hyperlink_container(
    paragraph: Paragraph,
    href_index: int,
    inline_run_map: dict[str, str] | None,
) -> Any:
    key = _hyperlink_meta_key(href_index)
    if inline_run_map and key in inline_run_map:
        try:
            container = parse_xml(inline_run_map[key])
            for child in list(container):
                if child.tag.split("}")[-1].lower() == "r":
                    container.remove(child)
            return container
        except Exception:
            pass
    from docx.oxml import OxmlElement

    return OxmlElement("w:hyperlink")


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


def _parse_tagged_pieces(tagged_text: str) -> list[tuple[str, int | None]]:
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
        open_marker = m.group(0)[0]
        close_marker = m.group(0)[-1]
        end_token = f"{open_marker}/S_{span_id}{close_marker}"
        m_end = re.search(re.escape(end_token), tagged_text[m.end() :])
        if not m_end:
            pieces.append((tagged_text[m.start() : m.end()], None))
            pos = m.end()
            continue

        inner_start = m.end()
        inner_end = inner_start + m_end.start()
        pieces.append((tagged_text[inner_start:inner_end], span_id))
        pos = inner_start + m_end.end()
    return pieces


def _run_is_plain_text_only(run: Run) -> bool:
    for child in run._r.iterchildren():
        local = child.tag.split("}")[-1]
        if local in {"rPr", "t"}:
            continue
        return False
    return True


def _collect_span_run_groups(paragraph: Paragraph, spans: list[Span]) -> dict[int, list[Run]] | None:
    merged: list[tuple[tuple[str, ...], RunStyleSnapshot, str, list[Run]]] = []
    inline_run_map: dict[str, str] = {}
    br_counters: dict[str, int] = {}

    for run, href_flag in _iter_runs_with_hyperlink_context(paragraph, inline_run_map):
        if _run_needs_xml_preserve(run):
            return None

        flags = list(_flags_from_run(run))
        if href_flag:
            flags.append(href_flag)
        flags_tuple = tuple(flags)
        snap = _snapshot_from_run(run)
        text = _extract_run_text_with_special_tokens(run, br_counters)

        if text == "":
            continue

        if merged and merged[-1][0] == flags_tuple and merged[-1][1] == snap:
            merged[-1] = (merged[-1][0], merged[-1][1], merged[-1][2] + text, [*merged[-1][3], run])
        else:
            merged.append((flags_tuple, snap, text, [run]))

    if len(merged) != len(spans):
        return None

    out: dict[int, list[Run]] = {}
    for idx, (flags, snap, source_text, runs) in enumerate(merged):
        span = spans[idx]
        if span.flags != flags or span.style != snap or span.source_text != source_text:
            return None
        if _BRACKET_TOKEN_RE.search(source_text):
            return None
        for run in runs:
            if not _run_is_plain_text_only(run):
                return None
        out[span.span_id] = runs
    return out


def _distribute_text_by_lengths(text: str, lengths: tuple[int, ...]) -> list[str]:
    if not lengths:
        return [text]
    if len(lengths) == 1:
        return [text]
    total = sum(max(0, int(value)) for value in lengths)
    if total <= 0:
        return [text] + [""] * (len(lengths) - 1)

    parts: list[str] = []
    cursor = 0
    cumulative = 0
    text_len = len(text)
    for i, length in enumerate(lengths):
        if i == len(lengths) - 1:
            parts.append(text[cursor:])
            break
        cumulative += max(0, int(length))
        target_end = int(round(text_len * (cumulative / total)))
        target_end = max(cursor, min(text_len, target_end))
        parts.append(text[cursor:target_end])
        cursor = target_end
    if len(parts) < len(lengths):
        parts.extend([""] * (len(lengths) - len(parts)))
    return parts


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

    pieces = _parse_tagged_pieces(tagged_text)

    _clear_paragraph_runs(paragraph)
    hyperlink_nodes: dict[int, Any] = {}

    def _bind_run_to_hyperlink(run: Run, href_index: int | None) -> None:
        if href_index is None:
            return
        node = hyperlink_nodes.get(href_index)
        if node is None:
            node = _build_hyperlink_container(paragraph, href_index, inline_run_map)
            paragraph._p.append(node)
            hyperlink_nodes[href_index] = node
        run_node = run._r
        paragraph._p.remove(run_node)
        node.append(run_node)

    def _emit_text(txt: str, span_id: int | None) -> None:
        if txt == "":
            return
        run = paragraph.add_run(txt)
        span = spans_by_id.get(span_id) if span_id is not None else None
        _bind_run_to_hyperlink(run, _span_hyperlink_index(span))
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
        span = spans_by_id.get(span_id) if span_id is not None else None
        _bind_run_to_hyperlink(run, _span_hyperlink_index(span))
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


def tagged_to_runs_inplace(
    paragraph: Paragraph,
    tagged_text: str,
    spans: list[Span],
    inline_run_map: dict[str, str] | None = None,
    spans_schema_version: int = 1,
) -> bool:
    """Try to write translated text into existing runs without rebuilding paragraph XML.

    Returns True when in-place update is applied. Returns False when structure is incompatible
    and caller should fall back to full ``tagged_to_runs`` rebuild.
    """
    if spans_schema_version != 1:
        raise ValueError(f"Unsupported spans_schema_version={spans_schema_version}")

    # Keep signature parity with tagged_to_runs.
    _ = inline_run_map

    pieces = _parse_tagged_pieces(tagged_text)
    expected_span_ids = [span.span_id for span in spans]
    piece_span_ids = [span_id for _, span_id in pieces if span_id is not None]

    # Any free text outside style tags means structure drift; use rebuild path.
    if any(span_id is None and chunk for chunk, span_id in pieces):
        return False
    if piece_span_ids != expected_span_ids:
        return False

    run_groups = _collect_span_run_groups(paragraph, spans)
    if run_groups is None:
        return False

    translated_by_span = {span_id: chunk for chunk, span_id in pieces if span_id is not None}

    for span in spans:
        translated = translated_by_span.get(span.span_id)
        runs = run_groups.get(span.span_id)
        if translated is None or runs is None:
            return False
        if _BRACKET_TOKEN_RE.search(translated):
            return False

        lengths = span.original_run_lengths or tuple(len(run.text or "") for run in runs)
        if len(lengths) != len(runs):
            return False

        parts = _distribute_text_by_lengths(translated, lengths)
        if len(parts) != len(runs):
            return False

        for run, part in zip(runs, parts):
            run.text = part

    return True
