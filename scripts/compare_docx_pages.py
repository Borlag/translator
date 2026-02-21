from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path


def _import_renderer():
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    try:
        from render_docx_pages import render_docx_pages  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Cannot import renderer from {scripts_dir}: {e}") from e
    return render_docx_pages


def _write_index_html(
    output_dir: Path,
    *,
    left_label: str,
    right_label: str,
    left_dir: Path,
    right_dir: Path,
) -> Path:
    left_pages = sorted(left_dir.glob("page-*.png"))
    right_pages = sorted(right_dir.glob("page-*.png"))
    page_count = max(len(left_pages), len(right_pages))

    def rel(p: Path) -> str:
        return p.relative_to(output_dir).as_posix()

    rows: list[str] = []
    for i in range(page_count):
        left_img = rel(left_pages[i]) if i < len(left_pages) else ""
        right_img = rel(right_pages[i]) if i < len(right_pages) else ""
        page_no = i + 1
        rows.append(
            "<div class='row'>"
            f"<div class='cell'><div class='cap'>{html.escape(left_label)} p.{page_no}</div>"
            + (f"<img loading='lazy' src='{html.escape(left_img)}' />" if left_img else "<div class='missing'>missing</div>")
            + "</div>"
            f"<div class='cell'><div class='cap'>{html.escape(right_label)} p.{page_no}</div>"
            + (
                f"<img loading='lazy' src='{html.escape(right_img)}' />"
                if right_img
                else "<div class='missing'>missing</div>"
            )
            + "</div>"
            + "</div>"
        )

    css = """
    :root { --gap: 16px; --bg: #111; --fg: #eee; --muted: #999; }
    body { margin: 0; font-family: Arial, sans-serif; background: var(--bg); color: var(--fg); }
    header { position: sticky; top: 0; background: rgba(17,17,17,0.95); padding: 12px 16px; border-bottom: 1px solid #2a2a2a; }
    header .path { color: var(--muted); font-size: 12px; margin-top: 4px; word-break: break-all; }
    .wrap { padding: 16px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: var(--gap); margin-bottom: 18px; }
    .cell { background: #171717; border: 1px solid #2a2a2a; border-radius: 8px; padding: 10px; }
    .cap { font-size: 12px; color: var(--muted); margin-bottom: 8px; }
    img { width: 100%; height: auto; display: block; border-radius: 4px; }
    .missing { padding: 24px; text-align: center; color: var(--muted); border: 1px dashed #333; border-radius: 6px; }
    @media (max-width: 1000px) { .row { grid-template-columns: 1fr; } }
    """

    html_out = (
        "<!doctype html><html><head><meta charset='utf-8' />"
        "<meta name='viewport' content='width=device-width, initial-scale=1' />"
        f"<title>DOCX Compare: {html.escape(left_label)} vs {html.escape(right_label)}</title>"
        f"<style>{css}</style></head><body>"
        "<header>"
        f"<div><strong>{html.escape(left_label)}</strong> vs <strong>{html.escape(right_label)}</strong></div>"
        f"<div class='path'>{html.escape(str(output_dir.resolve()))}</div>"
        "</header>"
        "<div class='wrap'>"
        + "\n".join(rows)
        + "</div></body></html>"
    )
    out_path = output_dir / "index.html"
    out_path.write_text(html_out, encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render and compare two DOCX files page-by-page in HTML.")
    p.add_argument("--left", required=True, help="Left/original .docx")
    p.add_argument("--right", required=True, help="Right/translated .docx")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--backend", default="auto", choices=["auto", "soffice", "word"], help="DOCX->PDF backend")
    p.add_argument("--dpi", type=int, default=150, help="Render DPI")
    p.add_argument("--left-label", default="Original", help="Left label")
    p.add_argument("--right-label", default="Translated", help="Right label")
    args = p.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    left_dir = output_dir / "left"
    right_dir = output_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    render_docx_pages = _import_renderer()

    render_docx_pages(Path(args.left), left_dir, backend=str(args.backend), dpi=int(args.dpi), keep_pdf=True)
    render_docx_pages(Path(args.right), right_dir, backend=str(args.backend), dpi=int(args.dpi), keep_pdf=True)

    index = _write_index_html(
        output_dir,
        left_label=str(args.left_label),
        right_label=str(args.right_label),
        left_dir=left_dir,
        right_dir=right_dir,
    )
    print(f"Compare index: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

