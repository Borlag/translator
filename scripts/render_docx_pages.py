from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _convert_docx_to_pdf_with_soffice(input_docx: Path, output_pdf: Path) -> None:
    soffice = shutil.which("soffice")
    if not soffice:
        raise RuntimeError("LibreOffice 'soffice' not found on PATH")

    outdir = output_pdf.parent
    outdir.mkdir(parents=True, exist_ok=True)

    # LibreOffice needs a writable user profile dir; keep it inside outdir to avoid global state.
    profile_dir = outdir / "lo_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_uri = profile_dir.resolve().as_uri()

    cmd = [
        soffice,
        f"-env:UserInstallation={profile_uri}",
        "--headless",
        "--nologo",
        "--nolockcheck",
        "--norestore",
        "--convert-to",
        "pdf",
        "--outdir",
        str(outdir),
        str(input_docx),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"soffice failed ({res.returncode}):\n{res.stdout}\n{res.stderr}")

    generated = outdir / f"{input_docx.stem}.pdf"
    if not generated.exists():
        # Last resort: pick the newest PDF in outdir.
        pdfs = sorted(outdir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not pdfs:
            raise RuntimeError("soffice did not produce a PDF")
        generated = pdfs[0]

    if generated.resolve() != output_pdf.resolve():
        shutil.copyfile(generated, output_pdf)


def _convert_docx_to_pdf_with_word(input_docx: Path, output_pdf: Path) -> None:
    if os.name != "nt":
        raise RuntimeError("Word COM backend is only available on Windows")
    try:
        import pywintypes  # type: ignore
        import win32com.client  # type: ignore
        import pythoncom  # type: ignore
        from win32com.client import gencache  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pywin32 is required for Word COM backend: pip install pywin32") from e

    outdir = output_pdf.parent
    outdir.mkdir(parents=True, exist_ok=True)

    call_rejected = -2147418111

    def retry(fn, tries: int = 30, delay_s: float = 0.25):  # pragma: no cover
        last: Exception | None = None
        for _ in range(tries):
            try:
                return fn()
            except pywintypes.com_error as e:
                last = e
                if e.args and e.args[0] == call_rejected:
                    pythoncom.PumpWaitingMessages()
                    time.sleep(delay_s)
                    continue
                raise
        raise RuntimeError(f"COM call failed after retries: {last}") from last

    wdExportFormatPDF = 17
    word = gencache.EnsureDispatch(win32com.client.DispatchEx("Word.Application"))  # pragma: no cover
    word.Visible = False  # pragma: no cover
    word.DisplayAlerts = 0  # pragma: no cover
    doc = None
    try:  # pragma: no cover
        doc = retry(
            lambda: word.Documents.Open(
                FileName=str(input_docx.resolve()),
                ReadOnly=True,
                AddToRecentFiles=False,
            )
        )
        retry(
            lambda: doc.ExportAsFixedFormat(
                OutputFileName=str(output_pdf.resolve()),
                ExportFormat=wdExportFormatPDF,
                OpenAfterExport=False,
            )
        )
        retry(lambda: doc.Close(False))
        doc = None
    finally:  # pragma: no cover
        try:
            if doc is not None:
                retry(lambda: doc.Close(False), tries=5)
        finally:
            retry(lambda: word.Quit(), tries=5)


def _convert_docx_to_pdf(input_docx: Path, output_pdf: Path, backend: str) -> str:
    backend_norm = backend.strip().lower()
    if backend_norm == "auto":
        if shutil.which("soffice"):
            backend_norm = "soffice"
        elif os.name == "nt":
            backend_norm = "word"
        else:
            backend_norm = "soffice"

    if backend_norm == "soffice":
        _convert_docx_to_pdf_with_soffice(input_docx, output_pdf)
        return "soffice"
    if backend_norm == "word":
        _convert_docx_to_pdf_with_word(input_docx, output_pdf)
        return "word"

    raise ValueError(f"Unknown backend: {backend}")


def _render_pdf_to_pngs(input_pdf: Path, output_dir: Path, *, dpi: int) -> list[Path]:
    try:
        import fitz  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required for PDF rendering: pip install pymupdf") from e

    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(input_pdf))  # pragma: no cover
    out: list[Path] = []
    for i in range(doc.page_count):  # pragma: no cover
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=int(dpi), alpha=False)
        out_path = output_dir / f"page-{i + 1:03d}.png"
        pix.save(str(out_path))
        out.append(out_path)
    return out


def render_docx_pages(
    input_docx: Path,
    output_dir: Path,
    *,
    backend: str = "auto",
    dpi: int = 150,
    keep_pdf: bool = True,
) -> list[Path]:
    input_docx = input_docx.resolve()
    if not input_docx.exists():
        raise FileNotFoundError(str(input_docx))
    if input_docx.suffix.lower() != ".docx":
        raise ValueError(f"Expected a .docx file, got: {input_docx.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{input_docx.stem}.pdf"

    chosen_backend = _convert_docx_to_pdf(input_docx, pdf_path, backend)
    images = _render_pdf_to_pngs(pdf_path, output_dir, dpi=int(dpi))

    meta = {
        "input_docx": str(input_docx),
        "pdf": str(pdf_path.resolve()),
        "backend": chosen_backend,
        "dpi": int(dpi),
        "pages": len(images),
    }
    (output_dir / "render_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if not keep_pdf:
        try:
            pdf_path.unlink()
        except OSError:
            pass

    return images


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render DOCX pages to PNGs (DOCX -> PDF -> PNG).")
    p.add_argument("input", help="Path to .docx")
    p.add_argument("--output-dir", required=True, help="Output directory for PNG pages")
    p.add_argument("--backend", default="auto", choices=["auto", "soffice", "word"], help="DOCX->PDF backend")
    p.add_argument("--dpi", type=int, default=150, help="PNG render DPI (e.g. 120/150/200)")
    p.add_argument("--keep-pdf", action="store_true", help="Keep intermediate PDF (default: true)")
    p.add_argument("--no-keep-pdf", action="store_true", help="Delete intermediate PDF")
    args = p.parse_args(argv)

    keep_pdf = True
    if args.no_keep_pdf:
        keep_pdf = False
    elif args.keep_pdf:
        keep_pdf = True

    pages = render_docx_pages(
        Path(args.input),
        Path(args.output_dir),
        backend=str(args.backend),
        dpi=int(args.dpi),
        keep_pdf=keep_pdf,
    )
    print(f"Rendered pages: {len(pages)} -> {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
