from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import CheckerConfig
from .models import Issue, Segment, Severity
from .tagging import paragraph_to_tagged, tagged_to_runs
from .validator import validate_placeholders, validate_style_tokens

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", flags=re.IGNORECASE)
_PDF_PAGE_RE = re.compile(r"^pdf/p(\d+)(?:/|$)", flags=re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_NOCHANGE_INSTRUCTION_RE = re.compile(r"\bno\s+change\s+needed\b|без\s+изменени", flags=re.IGNORECASE)

CHECKER_SYSTEM_PROMPT = """You are a strict EN->RU aviation translation quality checker (CMM/AMM/IPC).
Find only concrete defects that affect correctness, safety, terminology, or technical style.
Do not suggest stylistic rewrites when there is no defect.

Critical rules:
- Preserve all marker tokens exactly (⟦...⟧ / 🦦...🧧) in every suggestion.
- Preserve numbers, units, and identifiers unless they are clearly wrong.
- Procedural steps should use Russian imperative style.
- Terminology must stay technically consistent across nearby segments.
- Untranslated English leftovers and context-token leakage are valid defects.
- Treat issue types strictly:
  - meaning/omission/addition/number_error/terminology/untranslated => real defects.
  - register/style => lower-severity quality issues only when wording is clearly inappropriate for technical documentation.
  - If current translation is acceptable, do not emit an edit.
- If a term appears in glossary_terms_used for a segment, suggested_target must follow that target term.

Return ONLY strict JSON in the requested schema.
"""


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def read_checker_trace_resume_state(path: Path) -> dict[str, Any]:
    requested_roots: set[str] = set()
    terminal_chunks: set[str] = set()
    split_children: dict[str, tuple[str, str]] = {}
    requests_total = 0
    requests_succeeded = 0
    requests_failed = 0
    chunks_failed = 0
    split_events = 0
    suggestions_total = 0
    summary_present = False
    chunks_total = 0
    segments_checked = 0
    translatable_segments = 0

    if not path.exists():
        return {
            "completed_chunk_ids": [],
            "requests_total": 0,
            "requests_succeeded": 0,
            "requests_failed": 0,
            "chunks_failed": 0,
            "split_events": 0,
            "suggestions_total": 0,
            "summary_present": False,
            "chunks_total": 0,
            "segments_checked": 0,
            "translatable_segments": 0,
        }

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            event = str(payload.get("event") or "").strip()
            chunk_id = str(payload.get("chunk_id") or "").strip()
            if event == "request":
                requests_total += 1
                if chunk_id and "." not in chunk_id:
                    requested_roots.add(chunk_id)
            elif event == "response":
                requests_succeeded += 1
                if chunk_id:
                    terminal_chunks.add(chunk_id)
                try:
                    suggestions_total += max(0, int(payload.get("edits_count") or 0))
                except Exception:
                    pass
            elif event == "error":
                requests_failed += 1
            elif event == "failed_chunk":
                chunks_failed += 1
                if chunk_id:
                    terminal_chunks.add(chunk_id)
            elif event == "split":
                split_events += 1
                if chunk_id:
                    split_children[chunk_id] = (f"{chunk_id}.a", f"{chunk_id}.b")
            elif event == "summary":
                summary_present = True
                # Prefer summary counters if present.
                for key in (
                    "chunks_total",
                    "segments_checked",
                    "translatable_segments",
                    "requests_total",
                    "requests_succeeded",
                    "requests_failed",
                    "chunks_failed",
                    "split_events",
                    "suggestions_total",
                ):
                    try:
                        if key in payload:
                            value = int(payload.get(key) or 0)
                            if key == "chunks_total":
                                chunks_total = max(0, value)
                            elif key == "segments_checked":
                                segments_checked = max(0, value)
                            elif key == "translatable_segments":
                                translatable_segments = max(0, value)
                            elif key == "requests_total":
                                requests_total = max(0, value)
                            elif key == "requests_succeeded":
                                requests_succeeded = max(0, value)
                            elif key == "requests_failed":
                                requests_failed = max(0, value)
                            elif key == "chunks_failed":
                                chunks_failed = max(0, value)
                            elif key == "split_events":
                                split_events = max(0, value)
                            elif key == "suggestions_total":
                                suggestions_total = max(0, value)
                    except Exception:
                        continue

    memo: dict[str, bool] = {}

    def _is_complete(chunk_id: str, trail: set[str] | None = None) -> bool:
        cached = memo.get(chunk_id)
        if cached is not None:
            return cached
        if chunk_id in terminal_chunks:
            memo[chunk_id] = True
            return True
        children = split_children.get(chunk_id)
        if not children:
            memo[chunk_id] = False
            return False
        if trail is None:
            trail = set()
        if chunk_id in trail:
            memo[chunk_id] = False
            return False
        left, right = children
        child_trail = set(trail)
        child_trail.add(chunk_id)
        done = _is_complete(left, child_trail) and _is_complete(right, child_trail)
        memo[chunk_id] = done
        return done

    completed_root_chunks = sorted(chunk_id for chunk_id in requested_roots if _is_complete(chunk_id))
    return {
        "completed_chunk_ids": completed_root_chunks,
        "requests_total": requests_total,
        "requests_succeeded": requests_succeeded,
        "requests_failed": requests_failed,
        "chunks_failed": chunks_failed,
        "split_events": split_events,
        "suggestions_total": suggestions_total,
        "summary_present": summary_present,
        "chunks_total": chunks_total,
        "segments_checked": segments_checked,
        "translatable_segments": translatable_segments,
    }


def _extract_json_payload(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        raise RuntimeError("Empty checker response")

    candidates: list[str] = [text]
    for match in _JSON_FENCE_RE.finditer(text):
        inner = (match.group(1) or "").strip()
        if inner:
            candidates.append(inner)
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        candidates.append(text[obj_start : obj_end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("Checker response is not valid JSON")


def _normalize_severity(value: Any) -> Severity:
    raw = str(value or "").strip().lower()
    if raw == "error":
        return Severity.ERROR
    if raw == "info":
        return Severity.INFO
    return Severity.WARN


def _segment_page_number(seg: Segment) -> int | None:
    raw_page = seg.context.get("page_number")
    if isinstance(raw_page, int):
        return raw_page
    if isinstance(raw_page, str) and raw_page.strip().isdigit():
        return int(raw_page.strip())
    # DOCX segment locations (e.g. body/p123) encode paragraph indexes, not real pages.
    # Fallback regex parsing is safe only for explicit PDF locations (pdf/pN/...).
    m = _PDF_PAGE_RE.search((seg.location or "").strip())
    if m:
        return int(m.group(1))
    return None


def _segment_target_text(seg: Segment) -> str:
    target = seg.target_tagged
    if isinstance(target, str) and target.strip():
        return target
    # Fallback for pipelines that do not populate target_tagged.
    fallback = seg.context.get("checker_target_text")
    if isinstance(fallback, str) and fallback.strip():
        return fallback
    return ""


def _compact_text(text: Any, *, max_chars: int = 220) -> str:
    value = _SPACE_RE.sub(" ", str(text or "")).strip()
    if not value:
        return ""
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _segment_context_payload(seg: Segment) -> dict[str, Any]:
    issues = [
        {"code": issue.code, "severity": issue.severity.value}
        for issue in seg.issues[:6]
    ]
    section_header = _compact_text(seg.context.get("section_header"), max_chars=160)
    paragraph_style = _compact_text(seg.context.get("paragraph_style"), max_chars=80)
    prev_source = _compact_text(seg.context.get("prev_text"), max_chars=140)
    next_source = _compact_text(seg.context.get("next_text"), max_chars=140)
    style_norm = paragraph_style.strip().lower()
    heading_like = any(token in style_norm for token in ("heading", "title", "toc", "заголов", "оглавл"))
    payload: dict[str, Any] = {
        "part": str(seg.context.get("part") or ""),
        "section_header": section_header,
        "paragraph_style": paragraph_style,
        "heading_like": heading_like,
        "in_table": bool(seg.context.get("in_table")),
        "in_textbox": bool(seg.context.get("in_textbox")),
        "is_toc_entry": bool(seg.context.get("is_toc_entry")),
        "issues": issues,
    }
    if prev_source:
        payload["prev_source"] = prev_source
    if next_source:
        payload["next_source"] = next_source
    row_index = seg.context.get("row_index")
    col_index = seg.context.get("col_index")
    if isinstance(row_index, int):
        payload["row_index"] = row_index
    if isinstance(col_index, int):
        payload["col_index"] = col_index
    return payload


def _normalize_text_for_compare(text: Any) -> str:
    return _SPACE_RE.sub(" ", str(text or "")).strip()


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except Exception:
        confidence = 0.0
    return max(0.0, min(1.0, confidence))


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    text = str(exc or "").strip().lower()
    if not text:
        return False
    return "timed out" in text or "timeout" in text


def _is_output_limit_error(exc: Exception) -> bool:
    text = str(exc or "").strip().lower()
    if not text:
        return False
    markers = (
        "finish_reason=length",
        "finish_reason = length",
        "max token",
        "max_tokens",
        "max completion tokens",
        "max_completion_tokens",
        "context length",
        "token limit",
        "too many tokens",
        "output too long",
    )
    return any(marker in text for marker in markers)


def _evaluate_checker_edit(
    *,
    edit: dict[str, Any],
    current_target: str,
    safe_only: bool,
    min_confidence: float,
) -> list[str]:
    reasons: list[str] = []
    current = _normalize_text_for_compare(current_target)
    suggested = _normalize_text_for_compare(edit.get("suggested_target"))
    instruction = str(edit.get("instruction") or "").strip()

    if not suggested:
        reasons.append("missing_suggested_target")
    if not instruction:
        reasons.append("missing_instruction")
    if current and suggested and current == suggested:
        reasons.append("no_op")
    if instruction and _NOCHANGE_INSTRUCTION_RE.search(instruction):
        reasons.append("nochange_instruction")

    confidence = _coerce_confidence(edit.get("confidence"))
    if confidence < float(min_confidence):
        reasons.append("low_confidence")

    if safe_only and current and suggested:
        if validate_style_tokens(current, suggested):
            reasons.append("style_tags_mismatch")
        if validate_placeholders(current, suggested):
            reasons.append("placeholders_mismatch")

    return reasons


def _collect_used_glossary_terms(chunk: list[Segment], *, limit: int = 200) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []

    def _add_pair(source: Any, target: Any) -> None:
        src = str(source or "").strip()
        tgt = str(target or "").strip()
        if not src or not tgt:
            return
        key = (src.lower(), tgt.lower())
        if key in seen:
            return
        seen.add(key)
        out.append({"source": src, "target": tgt})

    for seg in chunk:
        matched = seg.context.get("matched_glossary_terms")
        if isinstance(matched, list):
            for item in matched:
                if isinstance(item, dict):
                    _add_pair(item.get("source"), item.get("target"))
        document = seg.context.get("document_glossary")
        if isinstance(document, list):
            for item in document:
                if isinstance(item, dict):
                    _add_pair(item.get("source"), item.get("target"))
        elif isinstance(document, dict):
            for source, target in document.items():
                _add_pair(source, target)
        if len(out) >= limit:
            break
    return out[:limit]


def _should_include_by_issue_filters(seg: Segment, checker_cfg: CheckerConfig) -> bool:
    sev_filter = {item.strip().lower() for item in checker_cfg.only_on_issue_severities if item.strip()}
    code_filter = {item.strip() for item in checker_cfg.only_on_issue_codes if item.strip()}
    if not sev_filter and not code_filter:
        return True
    for issue in seg.issues:
        sev_ok = (not sev_filter) or (issue.severity.value in sev_filter)
        code_ok = (not code_filter) or (issue.code in code_filter)
        if sev_ok and code_ok:
            return True
    return False


def _chunk_segments_by_pages(
    segments: list[Segment],
    *,
    pages_per_chunk: int,
    fallback_segments_per_chunk: int,
) -> list[tuple[str, list[Segment]]]:
    if not segments:
        return []
    pages: list[int | None] = [_segment_page_number(seg) for seg in segments]
    page_aware = sum(1 for item in pages if item is not None)
    if page_aware >= max(1, int(len(segments) * 0.7)):
        by_page: dict[int, list[Segment]] = {}
        for seg, page in zip(segments, pages, strict=False):
            if page is None:
                continue
            by_page.setdefault(page, []).append(seg)
        chunks: list[tuple[str, list[Segment]]] = []
        page_numbers = sorted(by_page.keys())
        window = max(1, int(pages_per_chunk))
        for i in range(0, len(page_numbers), window):
            page_slice = page_numbers[i : i + window]
            items: list[Segment] = []
            for page in page_slice:
                items.extend(by_page.get(page, []))
            if not items:
                continue
            label = f"pages_{page_slice[0]}_{page_slice[-1]}"
            chunks.append((label, items))
        return chunks

    step = max(1, int(fallback_segments_per_chunk))
    chunks = []
    for i in range(0, len(segments), step):
        items = segments[i : i + step]
        label = f"segments_{i + 1}_{i + len(items)}"
        chunks.append((label, items))
    return chunks


def _build_checker_prompt(
    *,
    chunk_id: str,
    source_target_items: list[dict[str, Any]],
    glossary_terms: list[dict[str, str]],
) -> str:
    payload = {
        "chunk_id": chunk_id,
        "glossary_terms_used": glossary_terms,
        "segments": source_target_items,
    }
    return (
        "TASK: CHECK_TRANSLATION_CHUNK\n"
        "Compare SOURCE and TARGET for each segment.\n"
        "Focus on terminology, meaning loss, dangerous ambiguity, and wrong technical phrasing.\n"
        "Use segment context (section_header, paragraph_style, heading_like, part, location, in_table, in_textbox, is_toc_entry, prev_source, next_source) to judge headings/labels/tables correctly.\n"
        "For heading-like segments, keep concise title style and section intent; do not rewrite as full prose sentences.\n"
        "For TOC-style segments, preserve leader dots, numbering, and page references unless SOURCE clearly differs.\n"
        "For tables/textboxes, preserve units, abbreviations, and compact label phrasing.\n"
        "Classify only concrete defects:\n"
        "- meaning: semantic distortion or wrong instruction intent\n"
        "- omission: missing meaningful source content\n"
        "- addition: invented content not present in source\n"
        "- number_error: wrong numeric value/unit/sign/range\n"
        "- terminology: wrong technical term\n"
        "- untranslated: untranslated English leftovers\n"
        "- consistency: conflicting term variant vs nearby context\n"
        "- register/style: non-critical wording/style mismatch\n"
        "If none of the above applies, do not emit an edit.\n"
        "If glossary_terms_used contains a source term for this segment, suggested_target must use the provided target term.\n"
        "If there is no issue, return empty edits list.\n"
        "Return ONLY JSON object with schema:\n"
        "{\n"
        '  "chunk_id": "string",\n'
        '  "edits": [\n'
        "    {\n"
        '      "segment_id": "string",\n'
        '      "location": "string",\n'
        '      "severity": "error|warn|info",\n'
        '      "issue_type": "terminology|meaning|omission|addition|number_error|untranslated|consistency|register|style|other",\n'
        '      "source_excerpt": "string",\n'
        '      "current_target": "string",\n'
        '      "suggested_target": "string",\n'
        '      "instruction": "exact replacement instruction",\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Use only segment_id values from input.\n"
        "- suggested_target must be final replacement text for this segment.\n"
        "- instruction must explicitly say what to replace.\n"
        "- Never remove, reorder, or mutate any marker token like ⟦S_*⟧, ⟦/S_*⟧, ⟦OBJ_*⟧, ⟦BR*_*⟧.\n"
        "- If correction cannot be made without changing marker tokens, do not emit an edit.\n"
        "- Do not return no-op edits (where suggested_target == current_target).\n"
        "- Do not include markdown.\n"
        "INPUT_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def _parse_checker_edits(raw: str, *, allowed_ids: set[str]) -> list[dict[str, Any]]:
    payload = _extract_json_payload(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("Checker response must be a JSON object")
    edits_raw = payload.get("edits", [])
    if not isinstance(edits_raw, list):
        raise RuntimeError("Checker response 'edits' must be a list")
    edits: list[dict[str, Any]] = []
    for item in edits_raw:
        if not isinstance(item, dict):
            continue
        seg_id = str(item.get("segment_id") or "").strip()
        if not seg_id or seg_id not in allowed_ids:
            continue
        location = str(item.get("location") or "").strip()
        issue_type = str(item.get("issue_type") or "other").strip().lower() or "other"
        source_excerpt = str(item.get("source_excerpt") or "").strip()
        current_target = str(item.get("current_target") or "").strip()
        suggested_target = str(item.get("suggested_target") or "").strip()
        instruction = str(item.get("instruction") or "").strip()
        if not suggested_target or not instruction:
            continue
        confidence_raw = item.get("confidence")
        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 0.0
        edits.append(
            {
                "segment_id": seg_id,
                "location": location,
                "severity": _normalize_severity(item.get("severity")).value,
                "issue_type": issue_type,
                "source_excerpt": source_excerpt,
                "current_target": current_target,
                "suggested_target": suggested_target,
                "instruction": instruction,
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )
    return edits


def _run_llm_checker_openai_batch(
    *,
    translatable: list[Segment],
    chunks: list[tuple[str, list[Segment]]],
    checker_cfg: CheckerConfig,
    checker_client: Any,
    logger: logging.Logger,
    trace_path: Path | None = None,
    stats_out: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    chunk_complete_callback: Callable[[str, list[dict[str, Any]]], None] | None = None,
    initial_checked_segments: int = 0,
) -> list[dict[str, Any]]:
    run_batch = getattr(checker_client, "run_checker_batch", None)
    if not callable(run_batch):
        raise RuntimeError("Checker client does not support run_checker_batch()")

    max_segments = max(0, int(checker_cfg.max_segments))
    queued_segments = 0
    checked_segments = max(0, int(initial_checked_segments))
    requests_total = 0
    requests_succeeded = 0
    requests_failed = 0
    chunks_failed = 0
    requests: list[tuple[str, str]] = []
    payload_by_chunk: dict[str, dict[str, Any]] = {}
    seg_by_id = {seg.segment_id: seg for seg in translatable}
    all_edits: list[dict[str, Any]] = []
    stop_due_to_limit = False

    def _emit_trace(event: str, **fields: Any) -> None:
        if trace_path is None:
            return
        payload = {
            "timestamp": _utc_now_iso(),
            "event": event,
            **fields,
        }
        try:
            _append_jsonl(trace_path, payload)
        except Exception:
            logger.debug("Failed to append checker trace event", exc_info=True)

    def _emit_progress(event: str, **fields: Any) -> None:
        if progress_callback is None:
            return
        payload = {
            "mode": "openai_batch",
            "event": event,
            "chunks_total": len(chunks),
            "translatable_segments": len(translatable),
            "segments_queued": queued_segments,
            "segments_checked": checked_segments,
            "requests_total": requests_total,
            "requests_succeeded": requests_succeeded,
            "requests_failed": requests_failed,
            "chunks_failed": chunks_failed,
            "suggestions_total": len(all_edits),
            **fields,
        }
        try:
            progress_callback(payload)
        except Exception:
            logger.debug("Failed to emit checker progress event", exc_info=True)

    _emit_trace(
        "start",
        mode="openai_batch",
        chunks_total=len(chunks),
        translatable_segments=len(translatable),
        max_segments=max_segments,
    )
    _emit_progress("start", max_segments=max_segments, stopped_due_to_limit=False)

    for chunk_id, chunk in chunks:
        candidates = [seg for seg in chunk if _should_include_by_issue_filters(seg, checker_cfg)]
        if not candidates:
            candidates = list(chunk)
        if not candidates:
            continue
        if max_segments > 0:
            left = max_segments - queued_segments
            if left <= 0:
                stop_due_to_limit = True
                break
            candidates = candidates[:left]
            if not candidates:
                stop_due_to_limit = True
                break

        input_items = [
            {
                "segment_id": seg.segment_id,
                "location": seg.location,
                "source": seg.source_plain,
                "target": _segment_target_text(seg),
                "context": _segment_context_payload(seg),
            }
            for seg in candidates
        ]
        if not input_items:
            continue
        allowed_ids = {item["segment_id"] for item in input_items}
        glossary_terms = _collect_used_glossary_terms(candidates)
        prompt = _build_checker_prompt(
            chunk_id=chunk_id,
            source_target_items=input_items,
            glossary_terms=glossary_terms,
        )
        requests.append((chunk_id, prompt))
        payload_by_chunk[chunk_id] = {
            "allowed_ids": allowed_ids,
            "segments_sent": len(input_items),
        }
        queued_segments += len(input_items)
        requests_total += 1
        _emit_trace(
            "request",
            mode="openai_batch",
            chunk_id=chunk_id,
            segments_sent=len(input_items),
            glossary_terms_count=len(glossary_terms),
        )
        _emit_progress(
            "request",
            chunk_id=chunk_id,
            segments_sent=len(input_items),
            glossary_terms_count=len(glossary_terms),
        )

    if not requests:
        summary = {
            "chunks_total": len(chunks),
            "segments_checked": checked_segments,
            "requests_total": requests_total,
            "requests_succeeded": requests_succeeded,
            "requests_failed": requests_failed,
            "chunks_failed": chunks_failed,
            "split_events": 0,
            "suggestions_total": len(all_edits),
            "stopped_due_to_limit": stop_due_to_limit,
            "checker_mode": "openai_batch",
        }
        if stats_out is not None:
            stats_out.update(summary)
        _emit_trace("summary", **summary)
        _emit_progress("summary", **summary)
        return []

    batch_results = run_batch(
        requests=requests,
        completion_window=checker_cfg.openai_batch_completion_window,
        poll_interval_s=checker_cfg.openai_batch_poll_interval_s,
        timeout_s=checker_cfg.openai_batch_timeout_s,
        metadata={"docxru_phase": "checker", "docxru_mode": "openai_batch"},
    )
    if not isinstance(batch_results, dict):
        raise RuntimeError("OpenAI batch checker response must be a mapping")

    for chunk_id, _ in requests:
        result = batch_results.get(chunk_id)
        if result is None:
            requests_failed += 1
            chunks_failed += 1
            _emit_progress(
                "error",
                chunk_id=chunk_id,
                error=f"OpenAI batch checker missing result for chunk: {chunk_id}",
            )
            raise RuntimeError(f"OpenAI batch checker missing result for chunk: {chunk_id}")
        content = getattr(result, "content", None)
        error_text = getattr(result, "error", None)
        if isinstance(result, dict):
            content = result.get("content")
            error_text = result.get("error")
        if error_text:
            requests_failed += 1
            chunks_failed += 1
            _emit_progress(
                "error",
                chunk_id=chunk_id,
                error=f"OpenAI batch checker chunk failed ({chunk_id}): {error_text}",
            )
            raise RuntimeError(f"OpenAI batch checker chunk failed ({chunk_id}): {error_text}")
        if not isinstance(content, str) or not content.strip():
            requests_failed += 1
            chunks_failed += 1
            _emit_progress(
                "error",
                chunk_id=chunk_id,
                error=f"OpenAI batch checker returned empty content for chunk: {chunk_id}",
            )
            raise RuntimeError(f"OpenAI batch checker returned empty content for chunk: {chunk_id}")

        allowed_ids = payload_by_chunk.get(chunk_id, {}).get("allowed_ids", set())
        if not isinstance(allowed_ids, set):
            allowed_ids = set()
        edits = _parse_checker_edits(content, allowed_ids=allowed_ids)
        requests_succeeded += 1
        segments_sent = int(payload_by_chunk.get(chunk_id, {}).get("segments_sent", 0))
        checked_segments += segments_sent
        _emit_trace(
            "response",
            mode="openai_batch",
            chunk_id=chunk_id,
            segments_sent=segments_sent,
            edits_count=len(edits),
        )
        _emit_progress(
            "response",
            chunk_id=chunk_id,
            segments_sent=segments_sent,
            edits_count=len(edits),
            suggestions_total=len(all_edits) + len(edits),
        )
        chunk_attached_edits: list[dict[str, Any]] = []
        for idx, edit in enumerate(edits, start=1):
            seg = seg_by_id.get(edit["segment_id"])
            if seg is None:
                continue
            issue_type = str(edit.get("issue_type") or "other").strip().lower()
            issue_code = f"llm_check_{issue_type}"
            severity = _normalize_severity(edit.get("severity"))
            location = edit.get("location") or seg.location
            seg.issues.append(
                Issue(
                    code=issue_code,
                    severity=severity,
                    message=str(edit.get("instruction") or "LLM checker suggested translation correction."),
                    details={
                        "action": "replace_segment_text",
                        "segment_id": seg.segment_id,
                        "location": location,
                        "suggested_target": edit.get("suggested_target"),
                        "current_target": edit.get("current_target"),
                        "source_excerpt": edit.get("source_excerpt"),
                        "confidence": edit.get("confidence"),
                        "chunk_id": chunk_id,
                        "patch_id": f"{chunk_id}:{idx}:{seg.segment_id}",
                    },
                )
            )
            normalized_edit = {
                "chunk_id": chunk_id,
                "segment_id": seg.segment_id,
                "location": location,
                **edit,
            }
            all_edits.append(normalized_edit)
            chunk_attached_edits.append(normalized_edit)
        if chunk_complete_callback is not None:
            try:
                chunk_complete_callback(chunk_id, chunk_attached_edits)
            except Exception:
                logger.debug("Failed to run checker chunk completion callback", exc_info=True)

    summary = {
        "chunks_total": len(chunks),
        "segments_checked": checked_segments,
        "requests_total": requests_total,
        "requests_succeeded": requests_succeeded,
        "requests_failed": requests_failed,
        "chunks_failed": chunks_failed,
        "split_events": 0,
        "suggestions_total": len(all_edits),
        "stopped_due_to_limit": stop_due_to_limit,
        "checker_mode": "openai_batch",
    }
    if stats_out is not None:
        stats_out.update(summary)
    _emit_trace("summary", **summary)
    _emit_progress("summary", **summary)
    logger.info("LLM checker (openai batch): %d suggestions across %d chunks", len(all_edits), len(chunks))
    return all_edits


def run_llm_checker(
    *,
    segments: list[Segment],
    checker_cfg: CheckerConfig,
    checker_client,
    logger: logging.Logger,
    trace_path: Path | None = None,
    stats_out: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    skip_chunk_ids: set[str] | None = None,
    chunk_complete_callback: Callable[[str, list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    if not checker_cfg.enabled:
        return []

    # Only check segments that already have translated text.
    translatable = [seg for seg in segments if _segment_target_text(seg).strip()]
    if not translatable:
        logger.info("LLM checker: skipped (no translated segments)")
        return []

    chunks = _chunk_segments_by_pages(
        translatable,
        pages_per_chunk=checker_cfg.pages_per_chunk,
        fallback_segments_per_chunk=checker_cfg.fallback_segments_per_chunk,
    )
    if not chunks:
        return []

    skip_ids = {str(item).strip() for item in (skip_chunk_ids or set()) if str(item).strip()}
    skipped_chunks = 0
    skipped_segments = 0
    if skip_ids:
        filtered_chunks: list[tuple[str, list[Segment]]] = []
        for chunk_id, chunk in chunks:
            if chunk_id not in skip_ids:
                filtered_chunks.append((chunk_id, chunk))
                continue
            candidates = [seg for seg in chunk if _should_include_by_issue_filters(seg, checker_cfg)]
            if not candidates:
                candidates = list(chunk)
            skipped_chunks += 1
            skipped_segments += len(candidates)
        if skipped_chunks > 0:
            logger.info(
                "LLM checker resume: skipping %d completed chunks (%d segments).",
                skipped_chunks,
                skipped_segments,
            )
        chunks = filtered_chunks

    if checker_cfg.openai_batch_enabled:
        try:
            return _run_llm_checker_openai_batch(
                translatable=translatable,
                chunks=chunks,
                checker_cfg=checker_cfg,
                checker_client=checker_client,
                logger=logger,
                trace_path=trace_path,
                stats_out=stats_out,
                progress_callback=progress_callback,
                chunk_complete_callback=chunk_complete_callback,
                initial_checked_segments=skipped_segments,
            )
        except Exception as exc:
            logger.warning("LLM checker OpenAI batch mode failed, fallback to sync mode: %s", exc)
            if trace_path is not None:
                try:
                    _append_jsonl(
                        trace_path,
                        {
                            "timestamp": _utc_now_iso(),
                            "event": "batch_fallback_to_sync",
                            "error": str(exc),
                        },
                    )
                except Exception:
                    logger.debug("Failed to append checker trace event", exc_info=True)
            fallback_cfg = checker_cfg.__class__(**{**checker_cfg.__dict__, "openai_batch_enabled": False})
            fallback_stats: dict[str, Any] = {}
            edits = run_llm_checker(
                segments=segments,
                checker_cfg=fallback_cfg,
                checker_client=checker_client,
                logger=logger,
                trace_path=trace_path,
                stats_out=fallback_stats,
                progress_callback=progress_callback,
                skip_chunk_ids=skip_ids,
                chunk_complete_callback=chunk_complete_callback,
            )
            if stats_out is not None:
                stats_out.update(fallback_stats)
                stats_out["batch_mode_fallback"] = True
                stats_out["batch_mode_error"] = str(exc)
            return edits

    max_segments = max(0, int(checker_cfg.max_segments))
    retry_attempts = max(1, int(checker_cfg.retries) + 1)
    max_split_depth = 2
    checked_segments = skipped_segments
    all_edits: list[dict[str, Any]] = []
    seg_by_id = {seg.segment_id: seg for seg in translatable}
    stop_due_to_limit = False
    requests_total = 0
    requests_succeeded = 0
    requests_failed = 0
    chunks_failed = 0
    split_events = 0

    def _emit_trace(event: str, **fields: Any) -> None:
        if trace_path is None:
            return
        payload = {
            "timestamp": _utc_now_iso(),
            "event": event,
            **fields,
        }
        try:
            _append_jsonl(trace_path, payload)
        except Exception:
            logger.debug("Failed to append checker trace event", exc_info=True)

    def _emit_progress(event: str, **fields: Any) -> None:
        if progress_callback is None:
            return
        payload = {
            "mode": "sync",
            "event": event,
            "chunks_total": len(chunks),
            "translatable_segments": len(translatable),
            "segments_checked": checked_segments,
            "requests_total": requests_total,
            "requests_succeeded": requests_succeeded,
            "requests_failed": requests_failed,
            "chunks_failed": chunks_failed,
            "split_events": split_events,
            "suggestions_total": len(all_edits),
            **fields,
        }
        try:
            progress_callback(payload)
        except Exception:
            logger.debug("Failed to emit checker progress event", exc_info=True)

    _emit_trace(
        "start",
        chunks_total=len(chunks),
        translatable_segments=len(translatable),
        retry_attempts=retry_attempts,
        max_segments=max_segments,
    )
    _emit_progress("start", retry_attempts=retry_attempts, max_segments=max_segments, stopped_due_to_limit=False)

    for chunk_id, chunk in chunks:
        candidates = [seg for seg in chunk if _should_include_by_issue_filters(seg, checker_cfg)]
        if not candidates:
            candidates = list(chunk)
        if not candidates:
            continue

        subchunks: list[tuple[str, list[Segment], int]] = [(chunk_id, candidates, 0)]
        while subchunks:
            if max_segments > 0 and checked_segments >= max_segments:
                stop_due_to_limit = True
                break

            current_chunk_id, current_candidates, split_depth = subchunks.pop(0)
            if max_segments > 0:
                left = max_segments - checked_segments
                if left <= 0:
                    stop_due_to_limit = True
                    break
                current_candidates = current_candidates[:left]
                if not current_candidates:
                    stop_due_to_limit = True
                    break

            input_items = [
                {
                    "segment_id": seg.segment_id,
                    "location": seg.location,
                    "source": seg.source_plain,
                    "target": _segment_target_text(seg),
                    "context": _segment_context_payload(seg),
                }
                for seg in current_candidates
            ]
            if not input_items:
                continue
            allowed_ids = {item["segment_id"] for item in input_items}
            glossary_terms = _collect_used_glossary_terms(current_candidates)
            edits: list[dict[str, Any]] | None = None
            last_exc: Exception | None = None
            split_due_to_timeout = False
            split_due_to_output_limit = False
            for attempt in range(1, retry_attempts + 1):
                requests_total += 1
                logger.info(
                    "LLM checker request (%s) attempt %d/%d: segments=%d",
                    current_chunk_id,
                    attempt,
                    retry_attempts,
                    len(input_items),
                )
                _emit_trace(
                    "request",
                    chunk_id=current_chunk_id,
                    attempt=attempt,
                    retry_attempts=retry_attempts,
                    split_depth=split_depth,
                    segments_sent=len(input_items),
                    glossary_terms_count=len(glossary_terms),
                )
                _emit_progress(
                    "request",
                    chunk_id=current_chunk_id,
                    attempt=attempt,
                    retry_attempts=retry_attempts,
                    split_depth=split_depth,
                    segments_sent=len(input_items),
                    glossary_terms_count=len(glossary_terms),
                )
                prompt = _build_checker_prompt(
                    chunk_id=current_chunk_id,
                    source_target_items=input_items,
                    glossary_terms=glossary_terms,
                )
                try:
                    raw = checker_client.translate(prompt, {"task": "checker", "phase": "checker"})
                    edits = _parse_checker_edits(raw, allowed_ids=allowed_ids)
                    requests_succeeded += 1
                    logger.info(
                        "LLM checker response (%s) attempt %d: edits=%d",
                        current_chunk_id,
                        attempt,
                        len(edits),
                    )
                    _emit_trace(
                        "response",
                        chunk_id=current_chunk_id,
                        attempt=attempt,
                        split_depth=split_depth,
                        segments_sent=len(input_items),
                        edits_count=len(edits),
                    )
                    _emit_progress(
                        "response",
                        chunk_id=current_chunk_id,
                        attempt=attempt,
                        split_depth=split_depth,
                        segments_sent=len(input_items),
                        edits_count=len(edits),
                        suggestions_total=len(all_edits) + len(edits),
                    )
                    break
                except Exception as exc:
                    requests_failed += 1
                    last_exc = exc
                    timeout_like = _is_timeout_error(exc)
                    output_limit_like = _is_output_limit_error(exc)
                    can_split_immediately = (
                        (timeout_like or output_limit_like)
                        and len(current_candidates) > 1
                        and split_depth < max_split_depth
                    )
                    _emit_trace(
                        "error",
                        chunk_id=current_chunk_id,
                        attempt=attempt,
                        split_depth=split_depth,
                        segments_sent=len(input_items),
                        will_retry=(attempt < retry_attempts) and not can_split_immediately,
                        timeout_like=timeout_like,
                        output_limit_like=output_limit_like,
                        will_split=can_split_immediately,
                        error=str(exc),
                    )
                    _emit_progress(
                        "error",
                        chunk_id=current_chunk_id,
                        attempt=attempt,
                        split_depth=split_depth,
                        segments_sent=len(input_items),
                        will_retry=(attempt < retry_attempts) and not can_split_immediately,
                        timeout_like=timeout_like,
                        output_limit_like=output_limit_like,
                        will_split=can_split_immediately,
                        error=str(exc),
                    )
                    if can_split_immediately:
                        split_due_to_timeout = timeout_like
                        split_due_to_output_limit = output_limit_like
                        split_reason = "timeout" if timeout_like else "output_limit"
                        logger.warning(
                            "LLM checker chunk (%s) attempt %d/%d hit %s; splitting without further same-size retries: %s",
                            current_chunk_id,
                            attempt,
                            retry_attempts,
                            split_reason,
                            exc,
                        )
                        break
                    if attempt < retry_attempts:
                        logger.warning(
                            "LLM checker chunk retry (%s) %d/%d after failure: %s",
                            current_chunk_id,
                            attempt,
                            retry_attempts - 1,
                            exc,
                        )

            if edits is None:
                if len(current_candidates) > 1 and split_depth < max_split_depth:
                    split_at = len(current_candidates) // 2
                    left_candidates = current_candidates[:split_at]
                    right_candidates = current_candidates[split_at:]
                    if left_candidates and right_candidates:
                        split_events += 1
                        logger.warning(
                            "LLM checker chunk failed (%s): %s; splitting into %d + %d segments",
                            current_chunk_id,
                            last_exc or "unknown error",
                            len(left_candidates),
                            len(right_candidates),
                        )
                        _emit_trace(
                            "split",
                            chunk_id=current_chunk_id,
                            split_depth=split_depth,
                            left_segments=len(left_candidates),
                            right_segments=len(right_candidates),
                            reason=(
                                "timeout"
                                if split_due_to_timeout
                                else ("output_limit" if split_due_to_output_limit else "retry_exhausted")
                            ),
                            error=str(last_exc or "unknown error"),
                        )
                        _emit_progress(
                            "split",
                            chunk_id=current_chunk_id,
                            split_depth=split_depth,
                            left_segments=len(left_candidates),
                            right_segments=len(right_candidates),
                            reason=(
                                "timeout"
                                if split_due_to_timeout
                                else ("output_limit" if split_due_to_output_limit else "retry_exhausted")
                            ),
                            error=str(last_exc or "unknown error"),
                        )
                        subchunks.insert(0, (f"{current_chunk_id}.b", right_candidates, split_depth + 1))
                        subchunks.insert(0, (f"{current_chunk_id}.a", left_candidates, split_depth + 1))
                        continue
                chunks_failed += 1
                logger.warning("LLM checker chunk failed (%s): %s", current_chunk_id, last_exc or "unknown error")
                _emit_trace(
                    "failed_chunk",
                    chunk_id=current_chunk_id,
                    split_depth=split_depth,
                    segments_sent=len(input_items),
                    error=str(last_exc or "unknown error"),
                )
                _emit_progress(
                    "failed_chunk",
                    chunk_id=current_chunk_id,
                    split_depth=split_depth,
                    segments_sent=len(input_items),
                    error=str(last_exc or "unknown error"),
                )
                continue

            checked_segments += len(input_items)
            chunk_attached_edits: list[dict[str, Any]] = []
            for idx, edit in enumerate(edits, start=1):
                seg = seg_by_id.get(edit["segment_id"])
                if seg is None:
                    continue
                issue_type = str(edit.get("issue_type") or "other").strip().lower()
                issue_code = f"llm_check_{issue_type}"
                severity = _normalize_severity(edit.get("severity"))
                location = edit.get("location") or seg.location
                seg.issues.append(
                    Issue(
                        code=issue_code,
                        severity=severity,
                        message=str(edit.get("instruction") or "LLM checker suggested translation correction."),
                        details={
                            "action": "replace_segment_text",
                            "segment_id": seg.segment_id,
                            "location": location,
                            "suggested_target": edit.get("suggested_target"),
                            "current_target": edit.get("current_target"),
                            "source_excerpt": edit.get("source_excerpt"),
                            "confidence": edit.get("confidence"),
                            "chunk_id": current_chunk_id,
                            "patch_id": f"{current_chunk_id}:{idx}:{seg.segment_id}",
                        },
                    )
                )
                normalized_edit = {
                    "chunk_id": current_chunk_id,
                    "segment_id": seg.segment_id,
                    "location": location,
                    **edit,
                }
                all_edits.append(normalized_edit)
                chunk_attached_edits.append(normalized_edit)
            if chunk_complete_callback is not None:
                try:
                    chunk_complete_callback(current_chunk_id, chunk_attached_edits)
                except Exception:
                    logger.debug("Failed to run checker chunk completion callback", exc_info=True)

        if stop_due_to_limit:
            break

    summary = {
        "chunks_total": len(chunks),
        "segments_checked": checked_segments,
        "skipped_chunks": skipped_chunks,
        "skipped_segments": skipped_segments,
        "requests_total": requests_total,
        "requests_succeeded": requests_succeeded,
        "requests_failed": requests_failed,
        "chunks_failed": chunks_failed,
        "split_events": split_events,
        "suggestions_total": len(all_edits),
        "stopped_due_to_limit": stop_due_to_limit,
    }
    if stats_out is not None:
        stats_out.update(summary)
    _emit_trace("summary", **summary)
    _emit_progress("summary", **summary)

    logger.info("LLM checker: %d suggestions across %d chunks", len(all_edits), len(chunks))
    return all_edits


def write_checker_suggestions(path: Path, edits: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": len(edits),
        "edits": edits,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_checker_suggestions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("checker suggestions payload must be a JSON object")
    raw = payload.get("edits")
    if raw is None:
        raw = payload.get("safe_edits")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
    return out


def attach_checker_edits_to_segments(
    *,
    segments: list[Segment],
    edits: list[dict[str, Any]],
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    seg_by_id = {seg.segment_id: seg for seg in segments}
    attached: list[dict[str, Any]] = []
    skipped = 0
    seen_keys: set[tuple[str, str, str, str, str]] = set()

    for idx, raw_edit in enumerate(edits, start=1):
        if not isinstance(raw_edit, dict):
            skipped += 1
            continue
        seg_id = str(raw_edit.get("segment_id") or "").strip()
        if not seg_id:
            skipped += 1
            continue
        seg = seg_by_id.get(seg_id)
        if seg is None:
            skipped += 1
            continue
        chunk_id = str(raw_edit.get("chunk_id") or "resume").strip() or "resume"
        issue_type = str(raw_edit.get("issue_type") or "other").strip().lower() or "other"
        location = str(raw_edit.get("location") or seg.location or "").strip() or seg.location
        instruction = str(raw_edit.get("instruction") or "LLM checker suggested translation correction.").strip()
        suggested_target = str(raw_edit.get("suggested_target") or "").strip()
        dedupe_key = (
            chunk_id,
            seg.segment_id,
            issue_type,
            _normalize_text_for_compare(suggested_target),
            _normalize_text_for_compare(instruction),
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        severity = _normalize_severity(raw_edit.get("severity"))
        issue_code = f"llm_check_{issue_type}"
        seg.issues.append(
            Issue(
                code=issue_code,
                severity=severity,
                message=instruction or "LLM checker suggested translation correction.",
                details={
                    "action": "replace_segment_text",
                    "segment_id": seg.segment_id,
                    "location": location,
                    "suggested_target": raw_edit.get("suggested_target"),
                    "current_target": raw_edit.get("current_target"),
                    "source_excerpt": raw_edit.get("source_excerpt"),
                    "confidence": raw_edit.get("confidence"),
                    "chunk_id": chunk_id,
                    "patch_id": f"{chunk_id}:resume:{idx}:{seg.segment_id}",
                    "resume_attached": True,
                },
            )
        )
        attached.append(
            {
                "chunk_id": chunk_id,
                "segment_id": seg.segment_id,
                "location": location,
                "severity": severity.value,
                "issue_type": issue_type,
                "source_excerpt": str(raw_edit.get("source_excerpt") or ""),
                "current_target": str(raw_edit.get("current_target") or ""),
                "suggested_target": str(raw_edit.get("suggested_target") or ""),
                "instruction": instruction,
                "confidence": _coerce_confidence(raw_edit.get("confidence")),
            }
        )

    if logger is not None and edits:
        logger.info(
            "Checker resume attach: loaded=%d; attached=%d; skipped=%d",
            len(edits),
            len(attached),
            skipped,
        )
    return attached


def filter_checker_suggestions(
    edits: list[dict[str, Any]],
    *,
    safe_only: bool = True,
    min_confidence: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    safe: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    conf_floor = max(0.0, min(1.0, float(min_confidence)))

    for edit in edits:
        current_target = str(edit.get("current_target") or "")
        reasons = _evaluate_checker_edit(
            edit=edit,
            current_target=current_target,
            safe_only=bool(safe_only),
            min_confidence=conf_floor,
        )
        if reasons:
            skipped.append(
                {
                    "segment_id": edit.get("segment_id"),
                    "location": edit.get("location"),
                    "chunk_id": edit.get("chunk_id"),
                    "confidence": _coerce_confidence(edit.get("confidence")),
                    "reasons": reasons,
                }
            )
            continue
        safe.append(edit)
    return safe, skipped


def write_checker_safe_suggestions(
    path: Path,
    *,
    source_edits: list[dict[str, Any]],
    safe_edits: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": len(safe_edits),
        "source_count": len(source_edits),
        "skipped_count": len(skipped),
        "safe_edits": safe_edits,
        "skipped": skipped,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_checker_suggestions_to_segments(
    *,
    segments: list[Segment],
    edits: list[dict[str, Any]],
    safe_only: bool = True,
    min_confidence: float = 0.0,
    require_current_match: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    seg_by_id = {seg.segment_id: seg for seg in segments}
    seg_by_location = {seg.location: seg for seg in segments if seg.location}
    conf_floor = max(0.0, min(1.0, float(min_confidence)))
    skipped_reasons: Counter[str] = Counter()
    applied = 0
    skipped = 0

    for edit in edits:
        segment_id = str(edit.get("segment_id") or "").strip()
        location = str(edit.get("location") or "").strip()
        seg = seg_by_id.get(segment_id)
        if seg is None and location:
            seg = seg_by_location.get(location)
        if seg is None:
            skipped += 1
            skipped_reasons["segment_not_found"] += 1
            continue

        current_target = _segment_target_text(seg)
        if not current_target:
            if seg.paragraph_ref is None:
                skipped += 1
                skipped_reasons["segment_has_no_paragraph_ref"] += 1
                continue
            try:
                tagged, spans, inline_map = paragraph_to_tagged(seg.paragraph_ref)
            except Exception:
                skipped += 1
                skipped_reasons["segment_tagging_failed"] += 1
                continue
            seg.spans = spans
            seg.inline_run_map = inline_map
            current_target = tagged
            seg.target_tagged = tagged

        if seg.spans is None:
            if seg.paragraph_ref is None:
                skipped += 1
                skipped_reasons["segment_spans_missing"] += 1
                continue
            try:
                _, spans, inline_map = paragraph_to_tagged(seg.paragraph_ref)
            except Exception:
                skipped += 1
                skipped_reasons["segment_spans_rebuild_failed"] += 1
                continue
            seg.spans = spans
            seg.inline_run_map = inline_map
        if seg.inline_run_map is None:
            seg.inline_run_map = {}

        edit_current = _normalize_text_for_compare(edit.get("current_target"))
        seg_current = _normalize_text_for_compare(current_target)
        if require_current_match and edit_current and seg_current and edit_current != seg_current:
            skipped += 1
            skipped_reasons["current_target_mismatch"] += 1
            continue

        reasons = _evaluate_checker_edit(
            edit=edit,
            current_target=current_target,
            safe_only=bool(safe_only),
            min_confidence=conf_floor,
        )
        if reasons:
            skipped += 1
            for reason in reasons:
                skipped_reasons[reason] += 1
            continue

        suggested = str(edit.get("suggested_target") or "")
        try:
            tagged_to_runs(
                seg.paragraph_ref,
                suggested,
                seg.spans,
                inline_run_map=seg.inline_run_map,
            )
        except Exception:
            skipped += 1
            skipped_reasons["write_error"] += 1
            continue

        seg.target_tagged = suggested
        if seg.context is not None:
            seg.context["checker_applied"] = True
        applied += 1

    summary = {
        "requested": len(edits),
        "applied": applied,
        "skipped": skipped,
        "safe_only": bool(safe_only),
        "min_confidence": conf_floor,
        "require_current_match": bool(require_current_match),
        "skipped_reasons": dict(skipped_reasons),
    }
    if logger is not None:
        logger.info(
            "Checker suggestions apply: requested=%d; applied=%d; skipped=%d; safe_only=%s",
            len(edits),
            applied,
            skipped,
            bool(safe_only),
        )
        if skipped_reasons:
            logger.info(
                "Checker apply skipped reasons: %s",
                ", ".join(f"{reason}={count}" for reason, count in skipped_reasons.most_common()),
            )
    return summary
