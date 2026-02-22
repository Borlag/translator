from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .pipeline import translate_docx
from .structure_check import compare_docx_structure, write_structure_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="docxru", description="DOCX technical aviation EN->RU translator.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("translate", help="Translate DOCX to Russian preserving formatting.")
    t.add_argument("--input", "-i", required=True, help="Path to source .docx")
    t.add_argument("--output", "-o", required=True, help="Path to output .docx")
    t.add_argument("--config", "-c", required=True, help="Path to YAML config")
    t.add_argument("--resume", action="store_true", help="Resume using TM/progress cache.")
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
        "--abbyy-profile",
        choices=["off", "safe", "aggressive"],
        default=None,
        help="Enable optional ABBYY-specific normalization profile.",
    )
    t.add_argument("--concurrency", type=int, default=None, help="Override concurrency from config.")
    t.add_argument("--qa", default=None, help="Override QA report HTML path.")
    t.add_argument("--qa-jsonl", default=None, help="Override QA jsonl path.")
    t.add_argument("--history-jsonl", default=None, help="Override translation history jsonl path.")
    t.add_argument("--log", default=None, help="Override log path.")

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
        if args.no_headers:
            cfg = cfg.__class__(**{**cfg.__dict__, "include_headers": False})
        if args.no_footers:
            cfg = cfg.__class__(**{**cfg.__dict__, "include_footers": False})

        translate_docx(
            input_path=Path(args.input),
            output_path=Path(args.output),
            cfg=cfg,
            resume=bool(args.resume),
            max_segments=(int(args.max_segments) if args.max_segments is not None else None),
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

    print(f"Unknown command: {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
