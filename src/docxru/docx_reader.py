from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterator
from typing import Any

from docx.document import Document as DocxDocument
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

from .models import Segment

_TOC_STYLE_HINTS = ("toc", "table of contents", "оглавлен", "содержан")
_TOC_TITLE_RE = re.compile(r"\btable\s+of\s+contents\b", flags=re.IGNORECASE)
_TOC_CONTINUED_RE = re.compile(r"\bcontents?\s*\(continued\)", flags=re.IGNORECASE)
_TOC_PAGE_REF_RE = re.compile(r"\t+\s*[0-9]{1,4}(?:\.[0-9]{1,3})?\s*$")
_TOC_DOTS_PAGE_RE = re.compile(r"(?:\.\s*){6,}[0-9]{1,4}(?:\.[0-9]{1,3})?\s*$")
_logger = logging.getLogger(__name__)


def _looks_like_toc_style(style_name: str) -> bool:
    key = re.sub(r"\s+", " ", style_name or "").strip().lower()
    if not key:
        return False
    return any(hint in key for hint in _TOC_STYLE_HINTS)


def _looks_like_toc_text(text: str) -> bool:
    if not text:
        return False
    flat = re.sub(r"\s+", " ", text).strip()
    if not flat:
        return False
    if _TOC_TITLE_RE.search(flat) or _TOC_CONTINUED_RE.search(flat):
        return True
    return bool(_TOC_PAGE_REF_RE.search(text) or _TOC_DOTS_PAGE_RE.search(flat))


def _stable_segment_id(location: str, source_plain: str) -> str:
    h = hashlib.sha1()
    h.update(location.encode("utf-8"))
    h.update(b"\n")
    h.update(source_plain.encode("utf-8"))
    return h.hexdigest()[:16]


def _parent_element(parent: Any):
    if isinstance(parent, _Cell):
        return parent._tc
    # Document / Header / Footer
    return parent._element.body if hasattr(parent._element, "body") else parent._element


def iter_block_items(parent: Any) -> Iterator[Paragraph | Table]:
    """Yield Paragraph and Table objects in document order for the given parent.

    Parent can be a Document, _Cell, Header, Footer.
    """
    parent_elm = _parent_element(parent)

    for child in parent_elm.iterchildren():
        tag = child.tag.lower()
        if tag.endswith("}p"):
            yield Paragraph(child, parent)
        elif tag.endswith("}tbl"):
            yield Table(child, parent)


def iter_textbox_contents(parent: Any) -> Iterator[Any]:
    """Yield all <w:txbxContent> nodes for the given container."""
    parent_elm = _parent_element(parent)
    for node in parent_elm.iter():
        if node.tag.lower().endswith("}txbxcontent"):
            yield node


def _iter_textbox_paragraphs(parent: Any) -> Iterator[tuple[int, int, Paragraph]]:
    """Yield (textbox index, paragraph index, Paragraph) tuples from textbox content."""
    for txbx_i, txbx_content in enumerate(iter_textbox_contents(parent)):
        p_i = 0
        for child in txbx_content.iterchildren():
            tag = str(getattr(child, "tag", "")).lower()
            if not tag.endswith("}p"):
                continue
            try:
                paragraph = Paragraph(child, parent)
            except Exception as exc:
                _logger.warning(
                    "Skipping malformed textbox paragraph at textbox%d/p%d: %s",
                    txbx_i,
                    p_i,
                    exc,
                )
                p_i += 1
                continue
            yield txbx_i, p_i, paragraph
            p_i += 1


def collect_segments(
    doc: DocxDocument,
    include_headers: bool = False,
    include_footers: bool = False,
) -> list[Segment]:
    segments: list[Segment] = []
    last_heading: str | None = None

    def handle_paragraph(p: Paragraph, location: str, context: dict[str, Any]) -> None:
        nonlocal last_heading
        text = p.text or ""
        if not text.strip():
            return

        style_name = getattr(getattr(p, "style", None), "name", "") or ""
        if "heading" in style_name.lower():
            last_heading = text.strip()

        ctx = dict(context)
        ctx.setdefault("section_header", last_heading)
        if style_name:
            ctx["paragraph_style"] = style_name
        if _looks_like_toc_style(style_name) or _looks_like_toc_text(text):
            ctx["is_toc_entry"] = True

        seg_id = _stable_segment_id(location, text)
        segments.append(
            Segment(
                segment_id=seg_id,
                location=location,
                context=ctx,
                source_plain=text,
                paragraph_ref=p,
            )
        )

    def walk_table(table: Table, base_loc: str, context: dict[str, Any]) -> None:
        for r_i, row in enumerate(table.rows):
            for c_i, cell in enumerate(row.cells):
                cell_loc = f"{base_loc}/r{r_i}/c{c_i}"
                p_i = 0
                t_i = 0
                for item in iter_block_items(cell):
                    if isinstance(item, Paragraph):
                        handle_paragraph(
                            item,
                            f"{cell_loc}/p{p_i}",
                            {**context, "in_table": True, "row_index": r_i, "col_index": c_i},
                        )
                        p_i += 1
                    else:
                        walk_table(
                            item,
                            f"{cell_loc}/t{t_i}",
                            {**context, "in_table": True, "row_index": r_i, "col_index": c_i, "table_index": t_i},
                        )
                        t_i += 1

    def walk_textboxes(container: Any, base_loc: str, context: dict[str, Any]) -> None:
        paragraph_map = {(txbx_i, p_i): p for txbx_i, p_i, p in _iter_textbox_paragraphs(container)}
        for txbx_i, txbx_content in enumerate(iter_textbox_contents(container)):
            p_i = 0
            t_i = 0
            for child in txbx_content.iterchildren():
                tag = str(getattr(child, "tag", "")).lower()
                if tag.endswith("}p"):
                    paragraph = paragraph_map.get((txbx_i, p_i))
                    if paragraph is not None:
                        handle_paragraph(
                            paragraph,
                            f"{base_loc}/textbox{txbx_i}/p{p_i}",
                            {**context, "in_textbox": True},
                        )
                    p_i += 1
                elif tag.endswith("}tbl"):
                    try:
                        textbox_table = Table(child, container)
                    except Exception as exc:
                        _logger.warning(
                            "Skipping malformed textbox table at textbox%d/t%d: %s",
                            txbx_i,
                            t_i,
                            exc,
                        )
                        t_i += 1
                        continue
                    walk_table(
                        textbox_table,
                        f"{base_loc}/textbox{txbx_i}/t{t_i}",
                        {**context, "in_textbox": True, "table_index": t_i},
                    )
                    t_i += 1

    # Body
    tbl_idx = 0
    p_idx = 0
    for item in iter_block_items(doc):
        if isinstance(item, Paragraph):
            handle_paragraph(item, f"body/p{p_idx}", {"part": "body"})
            p_idx += 1
        else:
            walk_table(item, f"body/t{tbl_idx}", {"part": "body", "table_index": tbl_idx})
            tbl_idx += 1
    walk_textboxes(doc, "body", {"part": "body"})

    # Headers/Footers per section
    for s_i, section in enumerate(doc.sections):
        if include_headers:
            header = section.header
            p_i = 0
            t_i = 0
            for item in iter_block_items(header):
                if isinstance(item, Paragraph):
                    handle_paragraph(item, f"header{s_i}/p{p_i}", {"part": "header", "section": s_i})
                    p_i += 1
                else:
                    walk_table(
                        item,
                        f"header{s_i}/t{t_i}",
                        {"part": "header", "section": s_i, "table_index": t_i},
                    )
                    t_i += 1
            walk_textboxes(header, f"header{s_i}", {"part": "header", "section": s_i})

        if include_footers:
            footer = section.footer
            p_i = 0
            t_i = 0
            for item in iter_block_items(footer):
                if isinstance(item, Paragraph):
                    handle_paragraph(item, f"footer{s_i}/p{p_i}", {"part": "footer", "section": s_i})
                    p_i += 1
                else:
                    walk_table(
                        item,
                        f"footer{s_i}/t{t_i}",
                        {"part": "footer", "section": s_i, "table_index": t_i},
                    )
                    t_i += 1
            walk_textboxes(footer, f"footer{s_i}", {"part": "footer", "section": s_i})

    return segments
