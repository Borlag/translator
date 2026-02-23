from __future__ import annotations

import logging
import re
import subprocess
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .checker import (
    CHECKER_SYSTEM_PROMPT,
    filter_checker_suggestions,
    run_llm_checker,
    write_checker_safe_suggestions,
    write_checker_suggestions,
)
from .config import PipelineConfig
from .dashboard_server import ensure_dashboard_html
from .llm import (
    build_glossary_matchers,
    build_hard_glossary_replacements,
    build_llm_client,
    select_matched_glossary_terms,
    supports_repair,
)
from .logging_utils import setup_logging
from .model_sizing import recommend_runtime_model_sizing
from .models import Issue, Segment, Severity
from .pdf_font_map import select_font_for_style
from .pdf_layout import group_all_pages
from .pdf_models import PdfSegment, PdfSpan, PdfTextBlock
from .pdf_reader import extract_pdf_pages
from .pdf_writer import build_bilingual_ocg, replace_block_text
from .pricing import PricingTable, load_pricing_table
from .qa_report import write_qa_jsonl, write_qa_report
from .run_context import resolve_run_paths
from .run_status import RunStatusWriter
from .tm import TMStore, normalize_text, sha256_hex
from .token_shield import shield, shield_terms, strip_bracket_tokens, unshield
from .usage import UsageTotals
from .validator import validate_all, validate_placeholders

try:
    import fitz  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency at runtime
    fitz = None

_LATIN_RE = re.compile(r"[A-Za-z]")
_PDF_TM_RULESET = "2026-02-22-pdf-v1"


def _require_fitz():
    if fitz is None:  # pragma: no cover - runtime guard
        raise RuntimeError("PyMuPDF is required for PDF translation. Install extras: pip install -e '.[pdf]'")
    return fitz


def _read_optional_text(path_value: str | None, logger: logging.Logger, label: str) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    try:
        text = path.read_text(encoding="utf-8-sig").strip()
    except OSError as exc:
        raise RuntimeError(f"Cannot read {label} from '{path}': {exc}") from exc
    if not text:
        logger.warning("%s is configured but empty: %s", label, path)
        return None
    return text


def _compact_context_text(text: str, *, max_chars: int = 220) -> str:
    flat = re.sub(r"\s+", " ", text or "").strip()
    if not flat:
        return ""
    if len(flat) <= max_chars:
        return flat
    return flat[: max_chars - 3].rstrip() + "..."


def _build_repair_payload(source_shielded: str, bad_output: str) -> str:
    return (
        "TASK: REPAIR_MARKERS\n\n"
        f"SOURCE:\n{source_shielded}\n\n"
        f"OUTPUT:\n{bad_output}"
    )


def _to_qa_segment(seg: PdfSegment) -> Segment:
    return Segment(
        segment_id=seg.segment_id,
        location=f"pdf/p{seg.page_number + 1}/{seg.segment_id}",
        context=dict(seg.context),
        source_plain=seg.source_text,
        paragraph_ref=None,
        target_tagged=seg.target_text,
        issues=list(seg.issues),
    )


def _run_ocr_fallback(input_pdf: Path, output_pdf: Path, logger: logging.Logger) -> bool:
    command = [
        "ocrmypdf",
        "--skip-text",
        "--quiet",
        str(input_pdf),
        str(output_pdf),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return output_pdf.exists()
    except FileNotFoundError:
        logger.warning("OCR fallback requested, but 'ocrmypdf' is not installed.")
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        if stderr:
            logger.warning("OCR fallback failed: %s", stderr)
        else:
            logger.warning("OCR fallback failed with exit code %s", exc.returncode)
    return False


def _translate_segment_text(
    seg: PdfSegment,
    *,
    cfg: PipelineConfig,
    llm_client,
    tm: TMStore,
    glossary_terms: tuple[tuple[re.Pattern[str], str], ...],
    glossary_matchers,
    document_glossary: dict[str, str],
    recent_translations: deque[tuple[str, str]],
    resume: bool,
) -> None:
    source_plain = seg.source_text or ""
    if not source_plain.strip():
        return
    if not _LATIN_RE.search(source_plain):
        seg.target_text = source_plain
        seg.issues.append(
            Issue(
                code="skip_no_latin",
                severity=Severity.INFO,
                message="Segment skipped: no Latin text found.",
                details={},
            )
        )
        return

    glossary_prompt_mode = cfg.llm.glossary_prompt_mode.strip().lower()
    glossary_in_prompt = bool(cfg.llm.glossary_in_prompt)
    if glossary_prompt_mode == "matched" and glossary_in_prompt and document_glossary:
        tail_items = list(document_glossary.items())[-max(0, int(cfg.llm.glossary_match_limit)) :]
        seg.context["document_glossary"] = [{"source": s, "target": t} for s, t in tail_items]
    if glossary_prompt_mode == "matched" and glossary_matchers:
        matched = select_matched_glossary_terms(
            source_plain,
            glossary_matchers,
            limit=cfg.llm.glossary_match_limit,
        )
        if matched:
            seg.context["matched_glossary_terms"] = [{"source": s, "target": t} for s, t in matched]
            for source_term, target_term in matched:
                if source_term in document_glossary:
                    document_glossary.pop(source_term, None)
                document_glossary[source_term] = target_term
    elif glossary_matchers and not seg.context.get("matched_glossary_terms"):
        checker_terms = select_matched_glossary_terms(
            source_plain,
            glossary_matchers,
            limit=min(12, cfg.llm.glossary_match_limit),
        )
        if checker_terms:
            seg.context["matched_glossary_terms"] = [{"source": s, "target": t} for s, t in checker_terms]

    context_window_chars = max(0, int(cfg.llm.context_window_chars))
    if context_window_chars > 0:
        if recent_translations:
            seg.context["recent_translations"] = [
                {"source": source, "target": target}
                for source, target in recent_translations
            ]
        seg.context["recent_translations_max_chars"] = context_window_chars

    shielded = source_plain
    token_map: dict[str, str] = {}
    if glossary_terms:
        shielded, glossary_map = shield_terms(
            shielded,
            glossary_terms,
            token_prefix="GLS",
            bridge_break_tokens=False,
        )
        if glossary_map:
            token_map = {**token_map, **glossary_map}
    shielded, pattern_map = shield(shielded, cfg.pattern_set)
    if pattern_map:
        token_map = {**pattern_map, **token_map}

    source_norm = normalize_text(shielded)
    source_hash = sha256_hex(f"{_PDF_TM_RULESET}\n{source_norm}")
    prev_progress = tm.get_progress(seg.segment_id) if resume else None
    if (
        resume
        and prev_progress is not None
        and prev_progress.get("status") in {"ok", "tm"}
        and prev_progress.get("source_hash") == source_hash
    ):
        hit = tm.get_exact(source_hash)
        if hit is not None:
            seg.target_text = hit.target_text
            tm.set_progress(seg.segment_id, "tm", source_hash=source_hash)
            return

    hit = tm.get_exact(source_hash)
    if hit is not None:
        seg.target_text = hit.target_text
        tm.set_progress(seg.segment_id, "tm", source_hash=source_hash)
        return

    try:
        translated_shielded = llm_client.translate(shielded, seg.context)
    except Exception as exc:
        seg.issues.append(
            Issue(
                code="translate_crash",
                severity=Severity.ERROR,
                message=f"Translate crash: {exc}",
                details={},
            )
        )
        tm.set_progress(seg.segment_id, "error", source_hash=source_hash, error=str(exc))
        return

    target_text = unshield(translated_shielded, token_map)
    issues = validate_all(shielded, translated_shielded, source_plain, target_text)
    hard_errors = [issue for issue in issues if issue.severity == Severity.ERROR]
    if hard_errors and supports_repair(llm_client):
        try:
            repaired = llm_client.translate(
                _build_repair_payload(shielded, translated_shielded),
                {**seg.context, "task": "repair"},
            )
            repaired_target = unshield(repaired, token_map)
            repaired_issues = validate_all(shielded, repaired, source_plain, repaired_target)
            if not any(issue.severity == Severity.ERROR for issue in repaired_issues):
                translated_shielded = repaired
                target_text = repaired_target
                issues = repaired_issues
            else:
                issues = repaired_issues
        except Exception as exc:
            issues.append(
                Issue(
                    code="repair_failed",
                    severity=Severity.WARN,
                    message=f"Repair failed: {exc}",
                    details={},
                )
            )

    placeholder_issues = validate_placeholders(shielded, translated_shielded)
    if any(issue.severity == Severity.ERROR for issue in placeholder_issues):
        seg.issues.extend(placeholder_issues)
        seg.target_text = source_plain
        tm.set_progress(seg.segment_id, "error", source_hash=source_hash, error="placeholder mismatch")
        return

    seg.issues.extend(issues)
    seg.target_text = target_text
    tm.put_exact(
        source_hash,
        source_norm,
        target_text,
        meta={
            "source": "pdf",
            "page_number": seg.page_number + 1,
            "segment_id": seg.segment_id,
        },
    )
    tm.set_progress(seg.segment_id, "ok", source_hash=source_hash)

    recent_source = _compact_context_text(strip_bracket_tokens(source_plain), max_chars=220)
    recent_target = _compact_context_text(strip_bracket_tokens(target_text), max_chars=220)
    if recent_source and recent_target:
        recent_translations.append((recent_source, recent_target))


def translate_pdf(
    input_path: Path,
    output_path: Path,
    cfg: PipelineConfig,
    *,
    resume: bool = False,
    max_pages: int | None = None,
    bilingual: bool | None = None,
    ocr_fallback: bool | None = None,
) -> None:
    _require_fitz()
    logger = setup_logging(Path(cfg.log_path))

    bilingual_mode = bool(cfg.pdf.bilingual_mode if bilingual is None else bilingual)
    ocr_enabled = bool(cfg.pdf.ocr_fallback if ocr_fallback is None else ocr_fallback)
    page_limit = max_pages if max_pages is not None else cfg.pdf.max_pages
    if page_limit is not None:
        page_limit = max(0, int(page_limit))

    logger.info("Input PDF: %s", input_path)
    logger.info("Output PDF: %s", output_path)
    logger.info(
        "PDF options: bilingual=%s; ocr_fallback=%s; max_pages=%s; block_merge_threshold_pt=%s; "
        "skip_headers_footers=%s; table_detection=%s; max_font_shrink_ratio=%s",
        bilingual_mode,
        ocr_enabled,
        page_limit if page_limit is not None else "(all)",
        cfg.pdf.block_merge_threshold_pt,
        cfg.pdf.skip_headers_footers,
        cfg.pdf.table_detection,
        cfg.pdf.max_font_shrink_ratio,
    )

    source_pdf = Path(input_path)
    pages = extract_pdf_pages(source_pdf, max_pages=page_limit)
    if ocr_enabled and any(not page.has_text for page in pages):
        logger.info("Detected scanned/empty-text pages. Running OCR fallback pass.")
        with TemporaryDirectory(prefix="docxru_pdf_ocr_") as tmp_dir:
            ocr_pdf = Path(tmp_dir) / "ocr_output.pdf"
            if _run_ocr_fallback(source_pdf, ocr_pdf, logger):
                pages = extract_pdf_pages(ocr_pdf, max_pages=page_limit)
                source_pdf = ocr_pdf
                logger.info("OCR fallback succeeded, using OCR-enhanced PDF stream.")
            else:
                logger.warning("OCR fallback unavailable or failed, continuing with original PDF content.")

            segments = group_all_pages(
                pages,
                block_merge_threshold_pt=cfg.pdf.block_merge_threshold_pt,
                skip_headers_footers=cfg.pdf.skip_headers_footers,
                table_detection=cfg.pdf.table_detection,
            )
            _translate_and_write_pdf(
                source_pdf=source_pdf,
                output_path=output_path,
                segments=segments,
                cfg=cfg,
                logger=logger,
                resume=resume,
                bilingual_mode=bilingual_mode,
            )
            return

    segments = group_all_pages(
        pages,
        block_merge_threshold_pt=cfg.pdf.block_merge_threshold_pt,
        skip_headers_footers=cfg.pdf.skip_headers_footers,
        table_detection=cfg.pdf.table_detection,
    )
    _translate_and_write_pdf(
        source_pdf=source_pdf,
        output_path=output_path,
        segments=segments,
        cfg=cfg,
        logger=logger,
        resume=resume,
        bilingual_mode=bilingual_mode,
    )


def _translate_and_write_pdf(
    *,
    source_pdf: Path,
    output_path: Path,
    segments: list[PdfSegment],
    cfg: PipelineConfig,
    logger: logging.Logger,
    resume: bool,
    bilingual_mode: bool,
) -> None:
    run_paths = resolve_run_paths(cfg, output_path=output_path)
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    ensure_dashboard_html(run_paths.dashboard_html_path.parent, filename=run_paths.dashboard_html_path.name)

    logger.info("Prepared PDF segments: %d", len(segments))
    logger.info("Run dir: %s", run_paths.run_dir)
    logger.info("Run status: %s", run_paths.status_path)
    logger.info("Dashboard HTML: %s", run_paths.dashboard_html_path)

    usage_totals = UsageTotals()
    pricing_table = PricingTable.empty(currency=cfg.pricing.currency)
    if cfg.pricing.enabled:
        if cfg.pricing.pricing_path:
            try:
                pricing_table = load_pricing_table(cfg.pricing.pricing_path, currency=cfg.pricing.currency)
                logger.info("Pricing table loaded: %s", cfg.pricing.pricing_path)
            except Exception as exc:
                logger.warning("Failed to load pricing table (%s): %s", cfg.pricing.pricing_path, exc)
        else:
            logger.warning("pricing.enabled=true but pricing.pricing_path is empty")

    status_writer = RunStatusWriter(
        path=run_paths.status_path,
        run_id=run_paths.run_id,
        total_segments=len(segments),
        flush_every_n_updates=cfg.run.status_flush_every_n_segments,
    )

    def _to_dashboard_link(path: Path) -> str:
        try:
            rel = path.resolve().relative_to(run_paths.run_dir.resolve())
            return rel.as_posix()
        except Exception:
            return str(path)

    status_writer.merge_paths(
        {
            "run_dir": str(run_paths.run_dir),
            "output": str(output_path),
            "qa_report": _to_dashboard_link(run_paths.qa_report_path),
            "qa_jsonl": _to_dashboard_link(run_paths.qa_jsonl_path),
            "dashboard_html": _to_dashboard_link(run_paths.dashboard_html_path),
            "checker_suggestions": _to_dashboard_link(run_paths.checker_suggestions_path),
            "checker_suggestions_safe": _to_dashboard_link(run_paths.checker_suggestions_safe_path),
        }
    )
    status_writer.set_phase("prepare")
    status_writer.set_done(0)
    status_writer.set_usage(usage_totals.snapshot())
    status_writer.write(force=True)

    if not segments:
        # Nothing to translate: copy source bytes.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(source_pdf.read_bytes())
        write_checker_suggestions(run_paths.checker_suggestions_path, [])
        write_checker_safe_suggestions(
            run_paths.checker_suggestions_safe_path,
            source_edits=[],
            safe_edits=[],
            skipped=[],
        )
        status_writer.set_phase("done")
        status_writer.set_done(0)
        status_writer.set_usage(usage_totals.snapshot())
        status_writer.merge_metrics(
            {
                "issues_total": 0,
                "checker_suggestions": 0,
                "checker_safe_suggestions": 0,
            }
        )
        status_writer.write(force=True)
        logger.info("No text segments found. Source PDF copied unchanged.")
        return

    custom_system_prompt = _read_optional_text(cfg.llm.system_prompt_path, logger, "custom system prompt")
    glossary_text = _read_optional_text(cfg.llm.glossary_path, logger, "glossary")
    glossary_prompt_mode = cfg.llm.glossary_prompt_mode.strip().lower()
    glossary_in_prompt = bool(cfg.llm.glossary_in_prompt)
    if glossary_prompt_mode == "off" or not glossary_in_prompt or glossary_prompt_mode == "matched":
        effective_glossary_text = None
    else:
        effective_glossary_text = glossary_text
    effective_llm_max_output_tokens = cfg.llm.max_output_tokens
    effective_checker_cfg = cfg.checker
    if cfg.llm.auto_model_sizing:
        checker_provider_for_sizing = (cfg.checker.provider or cfg.llm.provider).strip()
        checker_model_for_sizing = (cfg.checker.model or cfg.llm.model).strip()
        source_lengths = [len(seg.source_text or "") for seg in segments]
        prompt_chars = len(custom_system_prompt or "") + len(effective_glossary_text or "")
        if glossary_prompt_mode == "matched" and glossary_text:
            prompt_chars += min(1200, len(glossary_text))
        sizing = recommend_runtime_model_sizing(
            provider=cfg.llm.provider,
            model=cfg.llm.model,
            checker_provider=checker_provider_for_sizing,
            checker_model=checker_model_for_sizing,
            source_char_lengths=source_lengths,
            prompt_chars=prompt_chars,
            batch_segments=cfg.llm.batch_segments,
            batch_max_chars=cfg.llm.batch_max_chars,
            max_output_tokens=cfg.llm.max_output_tokens,
            context_window_chars=cfg.llm.context_window_chars,
            checker_pages_per_chunk=cfg.checker.pages_per_chunk,
            checker_fallback_segments_per_chunk=cfg.checker.fallback_segments_per_chunk,
            checker_max_output_tokens=cfg.checker.max_output_tokens,
        )
        effective_llm_max_output_tokens = sizing.max_output_tokens
        effective_checker_cfg = cfg.checker.__class__(
            **{
                **cfg.checker.__dict__,
                "pages_per_chunk": sizing.checker_pages_per_chunk,
                "fallback_segments_per_chunk": sizing.checker_fallback_segments_per_chunk,
                "max_output_tokens": sizing.checker_max_output_tokens,
            }
        )
        for note in sizing.notes:
            logger.info("Model auto-sizing: %s", note)
        logger.info(
            "Model auto-sizing effective values (PDF): llm_max_output_tokens=%d; "
            "checker_pages_per_chunk=%d; checker_fallback_segments_per_chunk=%d; checker_max_output_tokens=%d",
            int(effective_llm_max_output_tokens),
            int(effective_checker_cfg.pages_per_chunk),
            int(effective_checker_cfg.fallback_segments_per_chunk),
            int(effective_checker_cfg.max_output_tokens),
        )

    llm_client = build_llm_client(
        provider=cfg.llm.provider,
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        timeout_s=cfg.llm.timeout_s,
        max_output_tokens=effective_llm_max_output_tokens,
        source_lang=cfg.llm.source_lang,
        target_lang=cfg.llm.target_lang,
        base_url=cfg.llm.base_url,
        custom_system_prompt=custom_system_prompt,
        glossary_text=glossary_text,
        glossary_prompt_text=effective_glossary_text,
        reasoning_effort=cfg.llm.reasoning_effort,
        prompt_cache_key=cfg.llm.prompt_cache_key,
        prompt_cache_retention=cfg.llm.prompt_cache_retention,
        structured_output_mode=cfg.llm.structured_output_mode,
        on_usage=usage_totals.add,
        estimate_cost=(pricing_table.estimate_cost if cfg.pricing.enabled else None),
        pricing_currency=pricing_table.currency,
    )
    glossary_matchers = build_glossary_matchers(glossary_text) if glossary_text else ()
    glossary_terms: tuple[tuple[re.Pattern[str], str], ...] = ()
    if glossary_text and cfg.llm.hard_glossary:
        glossary_terms = build_hard_glossary_replacements(glossary_text)
        logger.info("Hard glossary enabled for PDF translation (%d terms).", len(glossary_terms))

    tm = TMStore(cfg.tm.path)
    document_glossary: dict[str, str] = {}
    recent_translations: deque[tuple[str, str]] = deque(maxlen=3)

    # Neighbor snippets improve local consistency.
    for i, seg in enumerate(segments):
        prev_text = _compact_context_text(segments[i - 1].source_text, max_chars=220) if i > 0 else ""
        next_text = _compact_context_text(segments[i + 1].source_text, max_chars=220) if i + 1 < len(segments) else ""
        if prev_text:
            seg.context["prev_text"] = prev_text
        if next_text:
            seg.context["next_text"] = next_text
        seg.context["part"] = "pdf"

    status_writer.set_phase("translate")
    status_writer.write(force=True)
    for done_segments, seg in enumerate(segments, start=1):
        _translate_segment_text(
            seg,
            cfg=cfg,
            llm_client=llm_client,
            tm=tm,
            glossary_terms=glossary_terms,
            glossary_matchers=glossary_matchers,
            document_glossary=document_glossary,
            recent_translations=recent_translations,
            resume=resume,
        )
        status_writer.set_done(done_segments)
        status_writer.set_usage(usage_totals.snapshot())
        status_writer.merge_metrics({"issues_total": sum(len(s.issues) for s in segments)})
        status_writer.write()
    status_writer.write(force=True)

    status_writer.set_phase("writeback")
    status_writer.write(force=True)
    with fitz.open(str(source_pdf)) as doc:
        ocg_xref: int | None = None
        if bilingual_mode:
            ocg_xref = build_bilingual_ocg(doc, layer_name="Russian Translation")
            logger.info("Bilingual OCG layer created: xref=%s", ocg_xref)

        for seg in segments:
            if not seg.target_text:
                continue
            if any(issue.code == "skip_no_latin" for issue in seg.issues):
                continue
            if any(issue.severity == Severity.ERROR for issue in seg.issues):
                continue
            page = doc[seg.page_number]
            block = PdfTextBlock(
                block_id=0,
                bbox=seg.bbox,
                text=seg.source_text,
                spans=[],
            )
            if seg.dominant_style is not None:
                block.spans = [
                    PdfSpan(
                        text=seg.source_text,
                        bbox=seg.bbox,
                        style=seg.dominant_style,
                    )
                ]
            font_spec = select_font_for_style(
                seg.dominant_style,
                font_map=cfg.pdf.font_map,
                default_sans_font=cfg.pdf.default_sans_font,
                default_serif_font=cfg.pdf.default_serif_font,
                default_mono_font=cfg.pdf.default_mono_font,
            )
            ok, scale = replace_block_text(
                page,
                block,
                seg.target_text,
                font_spec,
                ocg_xref=ocg_xref,
                max_font_shrink_ratio=cfg.pdf.max_font_shrink_ratio,
                redact_original=not bilingual_mode,
                paint_background=bilingual_mode,
            )
            if not ok:
                seg.issues.append(
                    Issue(
                        code="pdf_insert_overflow",
                        severity=Severity.WARN,
                        message="Translated text does not fit segment bbox even after font scaling.",
                        details={"segment_id": seg.segment_id, "bbox": seg.bbox, "scale": scale},
                    )
                )
            elif scale < 0.999:
                seg.issues.append(
                    Issue(
                        code="pdf_font_scaled_down",
                        severity=Severity.INFO,
                        message="Font was scaled down to fit translated text.",
                        details={"segment_id": seg.segment_id, "scale": scale},
                    )
                )

        try:
            doc.subset_fonts()
        except Exception as exc:
            logger.warning("PDF font subsetting failed: %s", exc)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            doc.ez_save(str(output_path))
        except Exception:
            doc.save(str(output_path))

    qa_segments = [_to_qa_segment(seg) for seg in segments]
    checker_edits: list[dict[str, Any]] = []
    if effective_checker_cfg.enabled:
        status_writer.set_phase("checker")
        status_writer.write(force=True)
        checker_provider = (effective_checker_cfg.provider or cfg.llm.provider).strip()
        checker_model = (effective_checker_cfg.model or cfg.llm.model).strip()
        checker_custom_prompt = _read_optional_text(
            effective_checker_cfg.system_prompt_path,
            logger,
            "checker system prompt",
        )
        checker_glossary_text = (
            _read_optional_text(
                effective_checker_cfg.glossary_path,
                logger,
                "checker glossary",
            )
            if effective_checker_cfg.glossary_path
            else glossary_text
        )
        checker_client = build_llm_client(
            provider=checker_provider,
            model=checker_model,
            temperature=effective_checker_cfg.temperature,
            timeout_s=effective_checker_cfg.timeout_s,
            max_output_tokens=effective_checker_cfg.max_output_tokens,
            source_lang=cfg.llm.source_lang,
            target_lang=cfg.llm.target_lang,
            base_url=cfg.llm.base_url,
            custom_system_prompt=checker_custom_prompt,
            glossary_text=checker_glossary_text,
            glossary_prompt_text=checker_glossary_text,
            reasoning_effort=cfg.llm.reasoning_effort,
            prompt_cache_key=cfg.llm.prompt_cache_key,
            prompt_cache_retention=cfg.llm.prompt_cache_retention,
            structured_output_mode="strict",
            base_system_prompt=CHECKER_SYSTEM_PROMPT,
            on_usage=usage_totals.add,
            estimate_cost=(pricing_table.estimate_cost if cfg.pricing.enabled else None),
            pricing_currency=pricing_table.currency,
        )
        checker_edits = run_llm_checker(
            segments=qa_segments,
            checker_cfg=effective_checker_cfg,
            checker_client=checker_client,
            logger=logger,
        )
    write_checker_suggestions(run_paths.checker_suggestions_path, checker_edits)
    safe_checker_edits, safe_checker_skipped = filter_checker_suggestions(
        checker_edits,
        safe_only=True,
        min_confidence=float(effective_checker_cfg.auto_apply_min_confidence),
    )
    write_checker_safe_suggestions(
        run_paths.checker_suggestions_safe_path,
        source_edits=checker_edits,
        safe_edits=safe_checker_edits,
        skipped=safe_checker_skipped,
    )

    status_writer.set_phase("qa")
    status_writer.set_usage(usage_totals.snapshot())
    status_writer.merge_metrics(
        {
            "checker_suggestions": len(checker_edits),
            "checker_safe_suggestions": len(safe_checker_edits),
            "issues_total": sum(len(seg.issues) for seg in qa_segments),
        }
    )
    status_writer.write(force=True)

    qa_html = run_paths.qa_report_path
    qa_jsonl = run_paths.qa_jsonl_path
    write_qa_report(qa_segments, qa_html)
    write_qa_jsonl(qa_segments, qa_jsonl)
    logger.info("PDF saved: %s", output_path)
    logger.info("QA report: %s", qa_html)
    logger.info("QA jsonl: %s", qa_jsonl)

    status_writer.set_phase("done")
    status_writer.set_done(len(segments))
    status_writer.set_usage(usage_totals.snapshot())
    status_writer.merge_metrics(
        {
            "checker_suggestions": len(checker_edits),
            "checker_safe_suggestions": len(safe_checker_edits),
            "issues_total": sum(len(seg.issues) for seg in qa_segments),
        }
    )
    status_writer.write(force=True)
    tm.close()
