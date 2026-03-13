from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.docxru.checker import (
    CHECKER_SYSTEM_PROMPT,
    apply_checker_suggestions_to_segments,
    filter_checker_suggestions,
    run_llm_checker,
    write_checker_safe_suggestions,
    write_checker_suggestions,
)
from src.docxru.config import load_config
from src.docxru.llm import build_glossary_matchers, build_llm_client
from src.docxru.pipeline import (
    _attach_neighbor_snippets,
    _build_checker_only_docx_segments,
    _build_matched_glossary_context,
)


LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9./()_-]*")
MONTH_RE = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?")
CODE_RE = re.compile(r"(?:[A-Z]{1,4}|[A-Z]{1,8}[0-9][0-9A-Z./-]*|[A-Z]{1,8}[./-][0-9A-Z./-]*)")
SINGLE_LETTER_SUFFIX_RE = re.compile(r"^[A-Z]\)$")
SINGLE_UPPER_RE = re.compile(r"^[A-Z]$")
PARA_REF_RE = re.compile(r"^[A-Z]\.\(\d+\)\.?$")
LOWER_PREFIX_CODE_RE = re.compile(r"^[a-z]\)[A-Z0-9.-]+$")
LOWER_LIST_MARKER_RE = re.compile(r"^[a-z]$")
MAGNIFICATION_RE = re.compile(r"^\d+[xX]\.?$")
TEMP_RANGE_RE = re.compile(r"^(?:\d+)?o[CF]-(?:\d+)?o[CF]\.?$")
ALLOW = {
    "SAFRAN",
    "Safran",
    "Landing",
    "LANDING",
    "Systems",
    "SYSTEMS",
    "UK",
    "LTD",
    "Ltd",
    "LIMITED",
    "CAGE",
    "K0654",
    "Airbus",
    "MLG",
    "ECCN",
    "EAR",
    "Cheltenham",
    "Road",
    "Gloucester",
    "England",
    "Messier-Dowty",
    "Messier-",
    "Limited",
    "www.safran-landing-systems.com",
    "mm",
    "in",
    "oC",
    "oF",
    "Accomet",
    "Araldite",
    "Alocrom",
    "Fibreslip",
    "Loctite",
    "Almen",
    "Mastinox",
    "Messier",
    "Dowty",
    "Molykote",
    "MOLYKOTE",
    "Sermetel",
    "SERMETEL",
    "LOCTITE",
    "MPa",
    "ksi",
    "kg",
    "lb",
    "lbf",
    "MIL",
    "PRF",
    "AMS",
    "ASTM",
    "ISO",
    "MS",
    "NAS",
    "NLG",
    "Titanine",
    "Ardrox",
    "Mobil",
    "JC5A",
    "D40",
    "A320",
    "A330",
    "A340",
    "A380",
    "AN",
    "AO",
    "AP",
    "SB",
    "PRE",
    "POST",
    "REF",
    "TBA",
}
ENGLISH_SIGNAL_WORDS = {
    "and",
    "or",
    "to",
    "the",
    "of",
    "for",
    "with",
    "from",
    "made",
    "between",
    "only",
    "use",
    "apply",
    "install",
    "refer",
    "figure",
    "repair",
    "material",
    "item",
    "grade",
    "class",
    "section",
    "tests",
    "test",
    "hold",
    "joints",
    "flange",
    "internal",
    "diameter",
    "bearing",
    "bush",
    "bushes",
    "sleeve",
    "pad",
    "assembly",
    "tool",
    "alignment",
    "guide",
    "cutter",
    "drift",
    "paint",
    "sealant",
    "plate",
    "cadmium",
    "clearances",
    "fits",
    "reference",
    "letter",
    "degrees",
    "withdrawn",
    "has",
    "been",
    "this",
    "after",
}
CRITICAL_SOURCE_TERM_RE = re.compile(
    r"Press Pad|Press Pad Assembly|Alignment Bar|Guide Bush|Assembly Tool|Drift|Repair Bush|Repair Bushes|"
    r"Repair Sleeve|Transfer Block|Upper Pivot Bracket|Torque Link|Pintle Pin|Material Ref\.? Item|"
    r"Electrically conducting Mastinox|zinc loaded mastinox|Zinc Powder|Loctite grade",
    re.IGNORECASE,
)


def _normalize_token(token: str) -> str:
    normalized = token.strip(".,;:")
    if normalized.startswith("("):
        normalized = normalized[1:]
    if normalized.endswith(")") and "(" not in normalized:
        normalized = normalized[:-1]
    return normalized


def _english_leftovers(text: str) -> list[str]:
    leftovers: list[str] = []
    for token in LATIN_TOKEN_RE.findall(text):
        normalized = _normalize_token(token)
        if not normalized:
            continue
        if token in ALLOW or normalized in ALLOW:
            continue
        if MONTH_RE.fullmatch(normalized):
            continue
        if CODE_RE.fullmatch(normalized):
            continue
        if SINGLE_UPPER_RE.fullmatch(normalized):
            continue
        if SINGLE_LETTER_SUFFIX_RE.fullmatch(token):
            continue
        if PARA_REF_RE.fullmatch(normalized):
            continue
        if LOWER_PREFIX_CODE_RE.fullmatch(normalized):
            continue
        if LOWER_LIST_MARKER_RE.fullmatch(normalized):
            continue
        if MAGNIFICATION_RE.fullmatch(normalized):
            continue
        if TEMP_RANGE_RE.fullmatch(normalized):
            continue
        low = normalized.lower()
        if low in ENGLISH_SIGNAL_WORDS or any(ch.islower() for ch in normalized):
            leftovers.append(token)
    return leftovers


def _segment_target(seg: Any) -> str:
    return str(seg.target_tagged or seg.context.get("checker_target_text") or "").strip()


def _segment_source(seg: Any) -> str:
    return str(seg.source_plain or "").strip()


def _is_candidate(seg: Any) -> bool:
    source = _segment_source(seg)
    target = _segment_target(seg)
    if not source or not target:
        return False
    if "?" in target:
        return True
    if _english_leftovers(target):
        return True
    if CRITICAL_SOURCE_TERM_RE.search(source):
        return True
    return False


def _collect_candidate_indexes(segments: list[Any], neighbor_window: int) -> list[int]:
    picked: set[int] = set()
    for idx, seg in enumerate(segments):
        if not _is_candidate(seg):
            continue
        start = max(0, idx - neighbor_window)
        end = min(len(segments), idx + neighbor_window + 1)
        for j in range(start, end):
            if _segment_target(segments[j]):
                picked.add(j)
    return sorted(picked)


def _summarize_segments(segments: list[Any], *, example_limit: int = 25) -> dict[str, Any]:
    question_segments = 0
    english_segments = 0
    examples: list[dict[str, str]] = []
    for idx, seg in enumerate(segments):
        target = _segment_target(seg)
        if not target:
            continue
        reasons: list[str] = []
        if "?" in target:
            question_segments += 1
            reasons.append("question")
        leftovers = _english_leftovers(target)
        if leftovers:
            english_segments += 1
            reasons.append("english")
        if reasons and len(examples) < example_limit:
            examples.append(
                {
                    "index": str(idx),
                    "location": str(seg.location),
                    "reasons": ", ".join(reasons),
                    "source": _segment_source(seg)[:220],
                    "target": target[:220],
                }
            )
    return {
        "question_segments": question_segments,
        "english_segments": english_segments,
        "examples": examples,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted checker-based cleanup for section 2 translation.")
    parser.add_argument("--input", required=True, help="Path to source English DOCX.")
    parser.add_argument("--output", required=True, help="Path to translated Russian DOCX to fix.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--report-dir", required=True, help="Directory for pass reports and checker outputs.")
    parser.add_argument("--passes", type=int, default=2, help="Maximum cleanup passes.")
    parser.add_argument("--neighbor-window", type=int, default=1, help="Include +/- N nearby segments.")
    parser.add_argument("--chunk-size", type=int, default=10, help="Checker chunk size in segments.")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Minimum checker confidence to auto-apply.")
    parser.add_argument("--checker-model", default="gpt-5.4", help="Checker model to use.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if not output_path.exists():
        shutil.copyfile(input_path, output_path)

    logger = logging.getLogger("manual_fix_part2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(report_dir / "manual_fix_part2.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    cfg = load_config(str(Path(args.config).resolve()))
    checker_cfg = cfg.checker.__class__(
        **{
            **cfg.checker.__dict__,
            "enabled": True,
            "provider": "openai",
            "model": str(args.checker_model),
            "temperature": 0.0,
            "max_output_tokens": max(int(cfg.checker.max_output_tokens), 12000),
            "timeout_s": max(float(cfg.checker.timeout_s), 180.0),
            "retries": max(int(cfg.checker.retries), 2),
            "pages_per_chunk": 1,
            "fallback_segments_per_chunk": max(1, int(args.chunk_size)),
            "openai_batch_enabled": False,
            "only_on_issue_severities": [],
            "only_on_issue_codes": [],
            "auto_apply_safe": False,
            "auto_apply_min_confidence": float(args.min_confidence),
        }
    )
    llm_cfg = cfg.llm.__class__(
        **{
            **cfg.llm.__dict__,
            "provider": "openai",
            "reasoning_effort": "high",
            "prompt_cache_key": "section2-manual-fix-v1",
            "prompt_cache_retention": "24h",
        }
    )
    cfg = cfg.__class__(**{**cfg.__dict__, "checker": checker_cfg, "llm": llm_cfg})

    glossary_text = ""
    if cfg.llm.glossary_path:
        glossary_path = Path(cfg.llm.glossary_path)
        if not glossary_path.is_absolute():
            glossary_path = Path(args.config).resolve().parent / glossary_path
        glossary_text = glossary_path.read_text(encoding="utf-8")
    glossary_matchers = build_glossary_matchers(glossary_text) if glossary_text else ()

    history: list[dict[str, Any]] = []
    previous_signature: tuple[int, int] | None = None

    for pass_no in range(1, max(1, int(args.passes)) + 1):
        doc, segments, alignment_stats = _build_checker_only_docx_segments(
            input_path=input_path,
            output_path=output_path,
            include_headers=cfg.include_headers,
            include_footers=cfg.include_footers,
            logger=logger,
        )
        _attach_neighbor_snippets(segments, cfg)
        if glossary_matchers:
            for seg in segments:
                source = _segment_source(seg)
                if not source:
                    continue
                matched = _build_matched_glossary_context(
                    source,
                    glossary_matchers,
                    limit=min(12, int(cfg.llm.glossary_match_limit)),
                )
                if matched:
                    seg.context["matched_glossary_terms"] = matched

        before_summary = _summarize_segments(segments)
        candidate_indexes = _collect_candidate_indexes(segments, neighbor_window=max(0, int(args.neighbor_window)))
        candidate_segments = [segments[idx] for idx in candidate_indexes]
        pass_dir = report_dir / f"pass_{pass_no}"
        pass_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Pass %d: candidates=%d, question_segments=%d, english_segments=%d",
            pass_no,
            len(candidate_segments),
            int(before_summary["question_segments"]),
            int(before_summary["english_segments"]),
        )

        if not candidate_segments:
            history.append(
                {
                    "pass": pass_no,
                    "alignment": alignment_stats,
                    "before": before_summary,
                    "candidate_segments": 0,
                    "suggestions_total": 0,
                    "safe_suggestions": 0,
                    "applied": 0,
                }
            )
            break

        checker_client = build_llm_client(
            provider="openai",
            model=str(args.checker_model),
            temperature=0.0,
            timeout_s=float(checker_cfg.timeout_s),
            max_output_tokens=int(checker_cfg.max_output_tokens),
            source_lang=cfg.llm.source_lang,
            target_lang=cfg.llm.target_lang,
            base_url=cfg.llm.base_url,
            custom_system_prompt=None,
            glossary_text=glossary_text or None,
            glossary_prompt_text=glossary_text or None,
            prompt_examples_mode="off",
            reasoning_effort=cfg.llm.reasoning_effort,
            prompt_cache_key=cfg.llm.prompt_cache_key,
            prompt_cache_retention=cfg.llm.prompt_cache_retention,
            structured_output_mode="strict",
            base_system_prompt=CHECKER_SYSTEM_PROMPT,
        )

        stats_out: dict[str, Any] = {}
        trace_path = pass_dir / "checker_trace.jsonl"
        edits = run_llm_checker(
            segments=candidate_segments,
            checker_cfg=checker_cfg,
            checker_client=checker_client,
            logger=logger,
            trace_path=trace_path,
            stats_out=stats_out,
        )
        write_checker_suggestions(pass_dir / "checker_suggestions.json", edits)
        safe_edits, skipped = filter_checker_suggestions(
            edits,
            safe_only=True,
            min_confidence=float(args.min_confidence),
        )
        write_checker_safe_suggestions(
            pass_dir / "checker_suggestions_safe.json",
            source_edits=edits,
            safe_edits=safe_edits,
            skipped=skipped,
        )
        apply_summary = apply_checker_suggestions_to_segments(
            segments=segments,
            edits=safe_edits,
            safe_only=True,
            min_confidence=float(args.min_confidence),
            require_current_match=True,
            logger=logger,
        )
        applied = int(apply_summary.get("applied", 0))
        if applied > 0:
            doc.save(str(output_path))

        _, after_segments, _ = _build_checker_only_docx_segments(
            input_path=input_path,
            output_path=output_path,
            include_headers=cfg.include_headers,
            include_footers=cfg.include_footers,
            logger=logger,
        )
        after_summary = _summarize_segments(after_segments)
        _write_json(
            pass_dir / "summary.json",
            {
                "pass": pass_no,
                "alignment": alignment_stats,
                "before": before_summary,
                "after": after_summary,
                "candidate_segments": len(candidate_segments),
                "checker_stats": stats_out,
                "suggestions_total": len(edits),
                "safe_suggestions": len(safe_edits),
                "skipped_suggestions": len(skipped),
                "applied": applied,
                "apply_summary": apply_summary,
            },
        )

        history.append(
            {
                "pass": pass_no,
                "alignment": alignment_stats,
                "before": before_summary,
                "after": after_summary,
                "candidate_segments": len(candidate_segments),
                "checker_stats": stats_out,
                "suggestions_total": len(edits),
                "safe_suggestions": len(safe_edits),
                "skipped_suggestions": len(skipped),
                "applied": applied,
            }
        )

        current_signature = (
            int(after_summary["question_segments"]),
            int(after_summary["english_segments"]),
        )
        if applied == 0:
            logger.info("Pass %d: no safe edits applied, stopping.", pass_no)
            break
        if previous_signature is not None and current_signature >= previous_signature:
            logger.info("Pass %d: suspicious segment counts did not improve, stopping.", pass_no)
            break
        previous_signature = current_signature

    final_doc, final_segments, final_alignment = _build_checker_only_docx_segments(
        input_path=input_path,
        output_path=output_path,
        include_headers=cfg.include_headers,
        include_footers=cfg.include_footers,
        logger=logger,
    )
    del final_doc
    final_summary = _summarize_segments(final_segments, example_limit=50)
    report_payload = {
        "input": str(input_path),
        "output": str(output_path),
        "final_alignment": final_alignment,
        "final_summary": final_summary,
        "passes": history,
    }
    _write_json(report_dir / "report.json", report_payload)
    print(json.dumps(report_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
