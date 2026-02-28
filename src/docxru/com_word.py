from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator


_WD_TEXT_FRAME_AUTO_SIZE_SHAPE_TO_FIT_TEXT = 1
_MSO_AUTO_SIZE_SHAPE_TO_FIT_TEXT = 1
_MSO_SHAPE_TYPE_GROUP = 6


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _iter_com_items(collection: Any) -> Iterator[Any]:
    if collection is None:
        return
    count = _safe_int(getattr(collection, "Count", 0), 0)
    for idx in range(1, count + 1):
        item: Any | None = None
        try:
            item = collection.Item(idx)
        except Exception:
            try:
                item = collection(idx)
            except Exception:
                item = None
        if item is not None:
            yield item


def _iter_shapes_recursive(shape: Any) -> Iterator[Any]:
    yield shape
    shape_type = _safe_int(getattr(shape, "Type", -1), -1)
    if shape_type != _MSO_SHAPE_TYPE_GROUP:
        return
    group_items = getattr(shape, "GroupItems", None)
    for child in _iter_com_items(group_items):
        yield from _iter_shapes_recursive(child)


def _iter_document_shapes(doc: Any) -> Iterator[Any]:
    for shape in _iter_com_items(getattr(doc, "Shapes", None)):
        yield from _iter_shapes_recursive(shape)

    for section in _iter_com_items(getattr(doc, "Sections", None)):
        for stories_name in ("Headers", "Footers"):
            stories = getattr(section, stories_name, None)
            for story in _iter_com_items(stories):
                for shape in _iter_com_items(getattr(story, "Shapes", None)):
                    yield from _iter_shapes_recursive(shape)


def _autofit_textbox_shape(
    shape: Any,
    *,
    min_font_size_pt: float,
    max_shrink_steps: int,
    expand_overflowing: bool,
    max_height_growth: float,
) -> dict[str, int]:
    stats = {"textboxes_seen": 0, "textboxes_autofit": 0, "textboxes_shrunk": 0, "textboxes_expanded": 0}
    text_frame = getattr(shape, "TextFrame", None)
    if text_frame is None:
        return stats

    has_text = bool(getattr(text_frame, "HasText", 0))
    if not has_text:
        return stats
    stats["textboxes_seen"] = 1

    did_autofit = False
    try:
        text_frame.AutoSize = _WD_TEXT_FRAME_AUTO_SIZE_SHAPE_TO_FIT_TEXT
        did_autofit = True
    except Exception:
        pass

    text_frame2 = getattr(shape, "TextFrame2", None)
    if text_frame2 is not None:
        try:
            text_frame2.AutoSize = _MSO_AUTO_SIZE_SHAPE_TO_FIT_TEXT
            did_autofit = True
        except Exception:
            pass
    if did_autofit:
        stats["textboxes_autofit"] = 1

    overflowing = bool(getattr(text_frame, "Overflowing", False))
    if not overflowing:
        return stats

    text_range = getattr(text_frame, "TextRange", None)
    font = getattr(text_range, "Font", None) if text_range is not None else None
    if font is None:
        return stats

    shrunk = False
    for _ in range(max(0, int(max_shrink_steps))):
        try:
            size = float(getattr(font, "Size", 0) or 0)
        except Exception:
            break
        if size <= float(min_font_size_pt):
            break
        try:
            font.Size = max(float(min_font_size_pt), size - 1.0)
            shrunk = True
        except Exception:
            break
        try:
            if not bool(getattr(text_frame, "Overflowing", False)):
                break
        except Exception:
            break

    if shrunk:
        stats["textboxes_shrunk"] = 1

    still_overflowing = bool(getattr(text_frame, "Overflowing", False))
    if still_overflowing and expand_overflowing:
        expanded = False
        try:
            original_height = float(getattr(shape, "Height", 0) or 0)
        except Exception:
            original_height = 0.0
        growth_limit = max(1.0, float(max_height_growth))
        max_height = original_height * growth_limit
        if original_height > 0.0 and max_height > original_height:
            current_height = original_height
            for _ in range(24):
                if current_height >= max_height:
                    break
                next_height = min(max_height, current_height + 7.2)
                if next_height <= current_height + 1e-6:
                    break
                try:
                    shape.Height = next_height
                except Exception:
                    break
                current_height = float(getattr(shape, "Height", next_height) or next_height)
                expanded = True
                try:
                    if not bool(getattr(text_frame, "Overflowing", False)):
                        break
                except Exception:
                    break
        if expanded:
            stats["textboxes_expanded"] = 1
    return stats


def _autofit_document_textboxes(
    doc: Any,
    *,
    min_font_size_pt: float,
    max_shrink_steps: int,
    expand_overflowing: bool,
    max_height_growth: float,
) -> dict[str, int]:
    stats = {"textboxes_seen": 0, "textboxes_autofit": 0, "textboxes_shrunk": 0, "textboxes_expanded": 0}
    for shape in _iter_document_shapes(doc):
        part = _autofit_textbox_shape(
            shape,
            min_font_size_pt=min_font_size_pt,
            max_shrink_steps=max_shrink_steps,
            expand_overflowing=expand_overflowing,
            max_height_growth=max_height_growth,
        )
        for key, value in part.items():
            stats[key] += int(value)
    return stats


def update_fields_via_com(
    docx_path: str | Path,
    *,
    autofit_textboxes: bool = True,
    min_font_size_pt: float = 8.0,
    max_shrink_steps: int = 4,
    expand_overflowing: bool = False,
    max_height_growth: float = 1.5,
) -> dict[str, int]:
    """Windows-only: open DOCX in Word via COM and update fields/TOC.

    Requires:
      - Windows
      - Microsoft Word installed
      - pywin32

    Returns operation stats for logging.
    """
    try:
        import win32com.client  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pywin32 is required for COM mode") from e

    stats: dict[str, int] = {
        "fields_updated": 0,
        "tocs_updated": 0,
        "textboxes_seen": 0,
        "textboxes_autofit": 0,
        "textboxes_shrunk": 0,
        "textboxes_expanded": 0,
    }

    p = str(Path(docx_path).resolve())
    word = win32com.client.Dispatch("Word.Application")  # pragma: no cover
    word.Visible = False
    try:
        word.DisplayAlerts = 0
    except Exception:
        pass
    doc = None
    try:
        doc = word.Documents.Open(p)
        stats["fields_updated"] = _safe_int(getattr(getattr(doc, "Fields", None), "Count", 0), 0)
        if stats["fields_updated"] > 0:
            doc.Fields.Update()

        for toc in _iter_com_items(getattr(doc, "TablesOfContents", None)):
            try:
                toc.Update()
                stats["tocs_updated"] += 1
            except Exception:
                continue

        if autofit_textboxes:
            text_stats = _autofit_document_textboxes(
                doc,
                min_font_size_pt=float(min_font_size_pt),
                max_shrink_steps=int(max_shrink_steps),
                expand_overflowing=bool(expand_overflowing),
                max_height_growth=max(1.0, float(max_height_growth)),
            )
            for key, value in text_stats.items():
                stats[key] = stats.get(key, 0) + int(value)

        doc.Save()
    finally:
        try:
            if doc is not None:
                doc.Close(False)
        except Exception:
            pass
        word.Quit()
    return stats
