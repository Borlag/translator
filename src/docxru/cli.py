from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .dashboard_server import serve_dashboard
from .eval import evaluate_batch, write_eval_report
from .pdf_pipeline import translate_pdf
from .pipeline import postformat_docx, run_docx_checker_only, translate_docx
from .structure_check import compare_docx_structure, write_structure_report
from .studio_server import serve_studio


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="docxru", description="DOCX technical aviation EN->RU translator.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("translate", help="Translate DOCX to Russian preserving formatting.")
    t.add_argument("--input", "-i", required=True, help="Path to source .docx")
    t.add_argument("--output", "-o", required=True, help="Path to output .docx")
    t.add_argument("--config", "-c", required=True, help="Path to YAML config")
    t.add_argument("--resume", action="store_true", help="Resume using TM/progress cache.")
    t.add_argument(
        "--checker-only",
        action="store_true",
        help="Skip translation and run checker pass against existing output DOCX (requires --output to exist).",
    )
    t.add_argument(
        "--mode",
        choices=["reflow", "com"],
        default=None,
        help="Override mode from config (reflow|com).",
    )
    t.add_argument("--no-headers", action="store_true", help="Do not translate headers even if enabled.")
    t.add_argument("--no-footers", action="store_true", help="Do not translate footers even if enabled.")
    t.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Translate only first N segments (quick iteration; approximates first pages).",
    )
    t.add_argument(
        "--batch-segments",
        type=int,
        default=None,
        help="Translate up to N nearby segments in one LLM request (OpenAI/Ollama only).",
    )
    t.add_argument(
        "--batch-max-chars",
        type=int,
        default=None,
        help="Soft character cap per batch request payload.",
    )
    t.add_argument(
        "--context-window-chars",
        type=int,
        default=None,
        help="Enable sequential sliding context mode when > 0.",
    )
    t.add_argument(
        "--structured-output",
        choices=["off", "auto", "strict"],
        default=None,
        help="Structured output mode for prompt-based LLM providers.",
    )
    t.add_argument(
        "--glossary-prompt-mode",
        choices=["off", "full", "matched"],
        default=None,
        help="Glossary injection mode for LLM prompts.",
    )
    t.add_argument(
        "--fuzzy-tm",
        action="store_true",
        help="Enable fuzzy TM lookup on top of exact TM.",
    )
    t.add_argument(
        "--checker-openai-batch",
        action="store_true",
        help="Run checker via OpenAI Batch API (async, high-latency, lower cost).",
    )
    t.add_argument(
        "--abbyy-profile",
        choices=["off", "safe", "aggressive", "full"],
        default=None,
        help="Enable optional ABBYY-specific normalization profile.",
    )
    t.add_argument(
        "--formatting-preset",
        choices=["off", "native_docx", "abbyy_standard", "abbyy_aggressive", "auto"],
        default=None,
        help="Override formatting preset from config.",
    )
    t.add_argument("--concurrency", type=int, default=None, help="Override concurrency from config.")
    t.add_argument("--qa", default=None, help="Override QA report HTML path.")
    t.add_argument("--qa-jsonl", default=None, help="Override QA jsonl path.")
    t.add_argument("--history-jsonl", default=None, help="Override translation history jsonl path.")
    t.add_argument("--log", default=None, help="Override log path.")

    pf = sub.add_parser("postformat", help="Run formatting/layout post-process on translated DOCX.")
    pf.add_argument("--input", "-i", required=True, help="Path to translated .docx")
    pf.add_argument("--output", "-o", required=True, help="Path to postformatted .docx")
    pf.add_argument("--config", "-c", required=True, help="Path to YAML config")
    pf.add_argument(
        "--mode",
        choices=["reflow", "com"],
        default=None,
        help="Override mode from config (reflow|com).",
    )
    pf.add_argument(
        "--abbyy-profile",
        choices=["off", "safe", "aggressive", "full"],
        default=None,
        help="Override ABBYY normalization profile.",
    )
    pf.add_argument(
        "--formatting-preset",
        choices=["off", "native_docx", "abbyy_standard", "abbyy_aggressive", "auto"],
        default=None,
        help="Override formatting preset from config.",
    )
    pf.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Process only first N segments (quick iteration mode).",
    )
    pf.add_argument("--log", default=None, help="Override log path.")

    p_pdf = sub.add_parser("translate-pdf", help="Translate PDF to Russian preserving layout.")
    p_pdf.add_argument("--input", "-i", required=True, help="Path to source .pdf")
    p_pdf.add_argument("--output", "-o", required=True, help="Path to output .pdf")
    p_pdf.add_argument("--config", "-c", required=True, help="Path to YAML config")
    p_pdf.add_argument("--resume", action="store_true", help="Resume using TM/progress cache.")
    p_pdf.add_argument("--bilingual", action="store_true", help="Enable bilingual EN/RU OCG layer mode.")
    p_pdf.add_argument("--max-pages", type=int, default=None, help="Translate only first N PDF pages.")
    p_pdf.add_argument("--ocr-fallback", action="store_true", help="Run OCR fallback for scanned pages.")
    p_pdf.add_argument(
        "--checker-openai-batch",
        action="store_true",
        help="Run checker via OpenAI Batch API (async, high-latency, lower cost).",
    )
    p_pdf.add_argument("--qa", default=None, help="Override QA report HTML path.")
    p_pdf.add_argument("--qa-jsonl", default=None, help="Override QA jsonl path.")
    p_pdf.add_argument("--log", default=None, help="Override log path.")

    d = sub.add_parser("dashboard", help="Serve run directory dashboard on localhost.")
    d.add_argument("--dir", required=True, help="Run directory containing dashboard.html and run_status.json")
    d.add_argument("--port", type=int, default=0, help="Port to bind on 127.0.0.1 (0 = random free port)")
    d.add_argument("--open-browser", action="store_true", help="Open dashboard URL in default browser")

    s = sub.add_parser("studio", help="Serve interactive local UI to start translation runs.")
    s.add_argument(
        "--base-dir",
        default="output/studio",
        help="Workspace for UI runs/artifacts (contains runs/<run_id>/...).",
    )
    s.add_argument("--port", type=int, default=0, help="Port to bind on 127.0.0.1 (0 = random free port)")
    s.add_argument("--open-browser", action="store_true", help="Open studio URL in default browser")

    v = sub.add_parser("verify", help="Verify structural invariants between two DOCX files.")
    v.add_argument("--input", "-i", required=True, help="Path to source .docx")
    v.add_argument("--output", "-o", required=True, help="Path to output .docx")
    v.add_argument("--report", default=None, help="Optional JSON report path")
    v.add_argument(
        "--check-text-equality",
        action="store_true",
        help="Also check that visible paragraph text is identical (useful for mock/dry runs).",
    )
    v.add_argument("--max-mismatches", type=int, default=20, help="Max text mismatches to list in report.")

    e = sub.add_parser("eval", help="Run batch evaluation for all DOCX files in a folder.")
    e.add_argument("--input-dir", required=True, help="Folder with source .docx files")
    e.add_argument("--output-dir", required=True, help="Folder for translated files and eval artifacts")
    e.add_argument("--config", "-c", required=True, help="Path to YAML config")
    e.add_argument("--report", default=None, help="JSON report path (default: <output-dir>/eval_report.json)")
    e.add_argument("--threshold-errors", type=int, default=None, help="Fail if total error issues exceed this value.")
    e.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Translate only first N segments per file (quick evaluation mode).",
    )
    e.add_argument("--resume", action="store_true", help="Resume using TM/progress cache.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "translate":
        cfg = load_config(args.config)

        # CLI overrides
        if args.mode is not None:
            cfg = cfg.__class__(**{**cfg.__dict__, "mode": args.mode})
        if args.concurrency is not None:
            cfg = cfg.__class__(**{**cfg.__dict__, "concurrency": int(args.concurrency)})
        if args.qa is not None:
            cfg = cfg.__class__(**{**cfg.__dict__, "qa_report_path": str(args.qa)})
        if args.qa_jsonl is not None:
            cfg = cfg.__class__(**{**cfg.__dict__, "qa_jsonl_path": str(args.qa_jsonl)})
        if args.history_jsonl is not None:
            cfg = cfg.__class__(**{**cfg.__dict__, "translation_history_path": str(args.history_jsonl)})
        if args.log is not None:
            cfg = cfg.__class__(**{**cfg.__dict__, "log_path": str(args.log)})
        if args.batch_segments is not None:
            llm_cfg = cfg.llm.__class__(**{**cfg.llm.__dict__, "batch_segments": int(args.batch_segments)})
            cfg = cfg.__class__(**{**cfg.__dict__, "llm": llm_cfg})
        if args.batch_max_chars is not None:
            llm_cfg = cfg.llm.__class__(**{**cfg.llm.__dict__, "batch_max_chars": int(args.batch_max_chars)})
            cfg = cfg.__class__(**{**cfg.__dict__, "llm": llm_cfg})
        if args.context_window_chars is not None:
            llm_cfg = cfg.llm.__class__(
                **{**cfg.llm.__dict__, "context_window_chars": max(0, int(args.context_window_chars))}
            )
            cfg = cfg.__class__(**{**cfg.__dict__, "llm": llm_cfg})
        if args.structured_output is not None:
            llm_cfg = cfg.llm.__class__(
                **{**cfg.llm.__dict__, "structured_output_mode": str(args.structured_output)}
            )
            cfg = cfg.__class__(**{**cfg.__dict__, "llm": llm_cfg})
        if args.glossary_prompt_mode is not None:
            glossary_mode = str(args.glossary_prompt_mode)
            llm_cfg = cfg.llm.__class__(
                **{
                    **cfg.llm.__dict__,
                    "glossary_prompt_mode": glossary_mode,
                    "glossary_in_prompt": glossary_mode != "off",
                }
            )
            cfg = cfg.__class__(**{**cfg.__dict__, "llm": llm_cfg})
        if args.fuzzy_tm:
            tm_cfg = cfg.tm.__class__(**{**cfg.tm.__dict__, "fuzzy_enabled": True})
            cfg = cfg.__class__(**{**cfg.__dict__, "tm": tm_cfg})
        if args.abbyy_profile is not None:
            cfg = cfg.__class__(**{**cfg.__dict__, "abbyy_profile": str(args.abbyy_profile)})
        if args.formatting_preset is not None:
            cfg = cfg.__class__(**{**cfg.__dict__, "formatting_preset": str(args.formatting_preset)})
        if args.checker_openai_batch:
            checker_cfg = cfg.checker.__class__(**{**cfg.checker.__dict__, "openai_batch_enabled": True})
            cfg = cfg.__class__(**{**cfg.__dict__, "checker": checker_cfg})
        if args.no_headers:
            cfg = cfg.__class__(**{**cfg.__dict__, "include_headers": False})
        if args.no_footers:
            cfg = cfg.__class__(**{**cfg.__dict__, "include_footers": False})

        if args.checker_only and not cfg.checker.enabled:
            checker_cfg = cfg.checker.__class__(**{**cfg.checker.__dict__, "enabled": True})
            cfg = cfg.__class__(**{**cfg.__dict__, "checker": checker_cfg})
        if args.checker_only:
            run_docx_checker_only(
                input_path=Path(args.input),
                output_path=Path(args.output),
                cfg=cfg,
                resume=bool(args.resume),
                max_segments=(int(args.max_segments) if args.max_segments is not None else None),
            )
        else:
            translate_docx(
                input_path=Path(args.input),
                output_path=Path(args.output),
                cfg=cfg,
                resume=bool(args.resume),
                max_segments=(int(args.max_segments) if args.max_segments is not None else None),
            )
        return 0

    if args.cmd == "postformat":
        cfg = load_config(args.config)
        if args.mode is not None:
            cfg = cfg.__class__(**{**cfg.__dict__, "mode": args.mode})
        if args.abbyy_profile is not None:
            cfg = cfg.__class__(**{**cfg.__dict__, "abbyy_profile": str(args.abbyy_profile)})
        if args.formatting_preset is not None:
            cfg = cfg.__class__(**{**cfg.__dict__, "formatting_preset": str(args.formatting_preset)})
        if args.log is not None:
            cfg = cfg.__class__(**{**cfg.__dict__, "log_path": str(args.log)})

        postformat_docx(
            input_path=Path(args.input),
            output_path=Path(args.output),
            cfg=cfg,
            max_segments=(int(args.max_segments) if args.max_segments is not None else None),
        )
        return 0

    if args.cmd == "translate-pdf":
        cfg = load_config(args.config)
        if args.qa is not None:
            cfg = cfg.__class__(**{**cfg.__dict__, "qa_report_path": str(args.qa)})
        if args.qa_jsonl is not None:
            cfg = cfg.__class__(**{**cfg.__dict__, "qa_jsonl_path": str(args.qa_jsonl)})
        if args.log is not None:
            cfg = cfg.__class__(**{**cfg.__dict__, "log_path": str(args.log)})
        if args.max_pages is not None:
            pdf_cfg = cfg.pdf.__class__(**{**cfg.pdf.__dict__, "max_pages": max(0, int(args.max_pages))})
            cfg = cfg.__class__(**{**cfg.__dict__, "pdf": pdf_cfg})
        if args.bilingual:
            pdf_cfg = cfg.pdf.__class__(**{**cfg.pdf.__dict__, "bilingual_mode": True})
            cfg = cfg.__class__(**{**cfg.__dict__, "pdf": pdf_cfg})
        if args.ocr_fallback:
            pdf_cfg = cfg.pdf.__class__(**{**cfg.pdf.__dict__, "ocr_fallback": True})
            cfg = cfg.__class__(**{**cfg.__dict__, "pdf": pdf_cfg})
        if args.checker_openai_batch:
            checker_cfg = cfg.checker.__class__(**{**cfg.checker.__dict__, "openai_batch_enabled": True})
            cfg = cfg.__class__(**{**cfg.__dict__, "checker": checker_cfg})

        translate_pdf(
            input_path=Path(args.input),
            output_path=Path(args.output),
            cfg=cfg,
            resume=bool(args.resume),
        )
        return 0

    if args.cmd == "verify":
        report = compare_docx_structure(
            input_path=Path(args.input),
            output_path=Path(args.output),
            check_text_equality=bool(args.check_text_equality),
            max_text_mismatches=int(args.max_mismatches),
        )

        # Console summary
        basic_diff = report.get("basic_diff", {})
        xml_diff = report.get("word_xml_diff", {})
        marker_hits = report.get("marker_leak_paragraphs", 0)
        text_m = report.get("text_mismatches", [])

        print("DOCX structure check")
        print("Basic diff:", basic_diff)
        print("word/*.xml diff:", xml_diff)
        print("Marker leak paragraphs:", marker_hits)
        if args.check_text_equality:
            print("Text mismatches:", len(text_m))

        if args.report:
            write_structure_report(report, Path(args.report))
            print(f"Report written: {args.report}")
        return 0

    if args.cmd == "dashboard":
        serve_dashboard(
            run_dir=Path(args.dir),
            port=int(args.port),
            open_browser=bool(args.open_browser),
        )
        return 0

    if args.cmd == "studio":
        serve_studio(
            base_dir=Path(args.base_dir),
            port=int(args.port),
            open_browser=bool(args.open_browser),
        )
        return 0

    if args.cmd == "eval":
        cfg = load_config(args.config)
        results = evaluate_batch(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            cfg=cfg,
            max_segments=(int(args.max_segments) if args.max_segments is not None else None),
            resume=bool(args.resume),
        )
        report_path = Path(args.report) if args.report else (Path(args.output_dir) / "eval_report.json")
        summary = write_eval_report(results, report_path)
        print("DOCX eval summary")
        print(summary)
        print(f"Report written: {report_path}")

        if int(summary.get("files_failed", 0)) > 0:
            return 1
        if args.threshold_errors is not None:
            total_errors = int(summary.get("errors_total", 0))
            if total_errors > int(args.threshold_errors):
                print(
                    f"Error threshold exceeded: errors_total={total_errors} > threshold={int(args.threshold_errors)}",
                    file=sys.stderr,
                )
                return 1
        return 0

    print(f"Unknown command: {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
