from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import CheckerConfig
from .models import Issue, Segment, Severity

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", flags=re.IGNORECASE)
_PDF_PAGE_RE = re.compile(r"/p(\d+)(?:/|$)", flags=re.IGNORECASE)

CHECKER_SYSTEM_PROMPT = """You are a bilingual EN->RU translation quality checker.
Your job is to find concrete translation defects and suggest exact replacement wording.
Return ONLY strict JSON.
"""


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


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
    m = _PDF_PAGE_RE.search(seg.location or "")
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
        "If there is no issue, return empty edits list.\n"
        "Return ONLY JSON object with schema:\n"
        "{\n"
        '  "chunk_id": "string",\n'
        '  "edits": [\n'
        "    {\n"
        '      "segment_id": "string",\n'
        '      "location": "string",\n'
        '      "severity": "error|warn|info",\n'
        '      "issue_type": "terminology|meaning|consistency|style|other",\n'
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
) -> list[dict[str, Any]]:
    run_batch = getattr(checker_client, "run_checker_batch", None)
    if not callable(run_batch):
        raise RuntimeError("Checker client does not support run_checker_batch()")

    max_segments = max(0, int(checker_cfg.max_segments))
    checked_segments = 0
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

    _emit_trace(
        "start",
        mode="openai_batch",
        chunks_total=len(chunks),
        translatable_segments=len(translatable),
        max_segments=max_segments,
    )

    for chunk_id, chunk in chunks:
        candidates = [seg for seg in chunk if _should_include_by_issue_filters(seg, checker_cfg)]
        if not candidates:
            candidates = list(chunk)
        if not candidates:
            continue
        if max_segments > 0:
            left = max_segments - checked_segments
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
        checked_segments += len(input_items)
        _emit_trace(
            "request",
            mode="openai_batch",
            chunk_id=chunk_id,
            segments_sent=len(input_items),
            glossary_terms_count=len(glossary_terms),
        )

    if not requests:
        summary = {
            "chunks_total": len(chunks),
            "segments_checked": 0,
            "requests_total": 0,
            "requests_succeeded": 0,
            "requests_failed": 0,
            "chunks_failed": 0,
            "split_events": 0,
            "suggestions_total": 0,
            "stopped_due_to_limit": stop_due_to_limit,
            "checker_mode": "openai_batch",
        }
        if stats_out is not None:
            stats_out.update(summary)
        _emit_trace("summary", **summary)
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
            raise RuntimeError(f"OpenAI batch checker missing result for chunk: {chunk_id}")
        content = getattr(result, "content", None)
        error_text = getattr(result, "error", None)
        if isinstance(result, dict):
            content = result.get("content")
            error_text = result.get("error")
        if error_text:
            raise RuntimeError(f"OpenAI batch checker chunk failed ({chunk_id}): {error_text}")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"OpenAI batch checker returned empty content for chunk: {chunk_id}")

        allowed_ids = payload_by_chunk.get(chunk_id, {}).get("allowed_ids", set())
        if not isinstance(allowed_ids, set):
            allowed_ids = set()
        edits = _parse_checker_edits(content, allowed_ids=allowed_ids)
        _emit_trace(
            "response",
            mode="openai_batch",
            chunk_id=chunk_id,
            segments_sent=int(payload_by_chunk.get(chunk_id, {}).get("segments_sent", 0)),
            edits_count=len(edits),
        )
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
            all_edits.append(
                {
                    "chunk_id": chunk_id,
                    "segment_id": seg.segment_id,
                    "location": location,
                    **edit,
                }
            )

    summary = {
        "chunks_total": len(chunks),
        "segments_checked": checked_segments,
        "requests_total": len(requests),
        "requests_succeeded": len(requests),
        "requests_failed": 0,
        "chunks_failed": 0,
        "split_events": 0,
        "suggestions_total": len(all_edits),
        "stopped_due_to_limit": stop_due_to_limit,
        "checker_mode": "openai_batch",
    }
    if stats_out is not None:
        stats_out.update(summary)
    _emit_trace("summary", **summary)
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
            )
            if stats_out is not None:
                stats_out.update(fallback_stats)
                stats_out["batch_mode_fallback"] = True
                stats_out["batch_mode_error"] = str(exc)
            return edits

    max_segments = max(0, int(checker_cfg.max_segments))
    retry_attempts = max(1, int(checker_cfg.retries) + 1)
    max_split_depth = 2
    checked_segments = 0
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

    _emit_trace(
        "start",
        chunks_total=len(chunks),
        translatable_segments=len(translatable),
        retry_attempts=retry_attempts,
        max_segments=max_segments,
    )

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
                }
                for seg in current_candidates
            ]
            if not input_items:
                continue
            allowed_ids = {item["segment_id"] for item in input_items}
            glossary_terms = _collect_used_glossary_terms(current_candidates)
            edits: list[dict[str, Any]] | None = None
            last_exc: Exception | None = None
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
                    break
                except Exception as exc:
                    requests_failed += 1
                    last_exc = exc
                    _emit_trace(
                        "error",
                        chunk_id=current_chunk_id,
                        attempt=attempt,
                        split_depth=split_depth,
                        segments_sent=len(input_items),
                        will_retry=(attempt < retry_attempts),
                        error=str(exc),
                    )
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
                continue

            checked_segments += len(input_items)
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
                all_edits.append(
                    {
                        "chunk_id": current_chunk_id,
                        "segment_id": seg.segment_id,
                        "location": location,
                        **edit,
                    }
                )

        if stop_due_to_limit:
            break

    summary = {
        "chunks_total": len(chunks),
        "segments_checked": checked_segments,
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

    logger.info("LLM checker: %d suggestions across %d chunks", len(all_edits), len(chunks))
    return all_edits


def write_checker_suggestions(path: Path, edits: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": len(edits),
        "edits": edits,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
