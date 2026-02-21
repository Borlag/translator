from __future__ import annotations

import hashlib
from typing import Any, Iterator, Optional

from docx.document import Document as DocxDocument
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

from .models import Segment


def _stable_segment_id(location: str, source_plain: str) -> str:
    h = hashlib.sha1()
    h.update(location.encode("utf-8"))
    h.update(b"\n")
    h.update(source_plain.encode("utf-8"))
    return h.hexdigest()[:16]


def iter_block_items(parent: Any) -> Iterator[Paragraph | Table]:
    """Yield Paragraph and Table objects in document order for the given parent.

    Parent can be a Document, _Cell, Header, Footer.
    """
    if isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        # Document / Header / Footer
        parent_elm = parent._element.body if hasattr(parent._element, "body") else parent._element

    for child in parent_elm.iterchildren():
        tag = child.tag.lower()
        if tag.endswith("}p"):
            yield Paragraph(child, parent)
        elif tag.endswith("}tbl"):
            yield Table(child, parent)


def collect_segments(
    doc: DocxDocument,
    include_headers: bool = False,
    include_footers: bool = False,
) -> list[Segment]:
    segments: list[Segment] = []
    last_heading: Optional[str] = None

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

    return segments
