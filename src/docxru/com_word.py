from __future__ import annotations

from pathlib import Path


def update_fields_via_com(docx_path: str | Path) -> None:
    """Windows-only: open DOCX in Word via COM and update all fields (TOC/PAGEREF).

    Requires:
      - Windows
      - Microsoft Word installed
      - pywin32

    This is OPTIONAL and not used in cross-platform mode.
    """
    try:
        import win32com.client  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pywin32 is required for COM mode") from e

    p = str(Path(docx_path).resolve())
    word = win32com.client.Dispatch("Word.Application")  # pragma: no cover
    word.Visible = False
    try:
        doc = word.Documents.Open(p)
        doc.Fields.Update()
        # Update TOC if present
        for toc in doc.TablesOfContents:
            toc.Update()
        doc.Save()
        doc.Close()
    finally:
        word.Quit()
