from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from .llm import GlossaryMatcher
from .models import Issue, Segment, Severity
from .token_shield import strip_bracket_tokens

_SPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[\W_]+", flags=re.UNICODE)


def _clean_text(value: str) -> str:
    text = strip_bracket_tokens(value or "")
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def _normalize_key(value: str) -> str:
    return _NON_WORD_RE.sub("", (value or "").lower())


def _build_target_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term.strip())
    escaped = escaped.replace(r"\ ", r"(?:\s+)+")
    escaped = escaped.replace(r"\-", r"(?:\s*)-(?:\s*)")
    return re.compile(escaped, flags=re.IGNORECASE)


def _extract_term_translation(
    *,
    source_text: str,
    target_text: str,
    source_term: str,
    target_term: str,
) -> str | None:
    target_pattern = _build_target_pattern(target_term)
    if target_pattern.search(target_text):
        return target_term.strip()

    source_key = _normalize_key(source_text)
    term_key = _normalize_key(source_term)
    # For standalone labels/cell values, keep the whole translated text as a candidate.
    if source_key and term_key and source_key == term_key:
        candidate = target_text.strip(" .,:;!?")
        if candidate:
            return candidate
    return None


def _collect_term_variants(
    segments: Iterable[Segment],
    glossary_matchers: tuple[GlossaryMatcher, ...],
) -> tuple[dict[str, set[str]], dict[str, dict[str, list[str]]]]:
    phrase_map: dict[str, set[str]] = defaultdict(set)
    segment_index: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for seg in segments:
        source_text = _clean_text(seg.source_plain or "")
        target_text = _clean_text(seg.target_tagged or seg.target_shielded_tagged or "")
        if not source_text or not target_text:
            continue

        for source_term, target_term, source_pattern in glossary_matchers:
            if not source_pattern.search(source_text):
                continue
            candidate = _extract_term_translation(
                source_text=source_text,
                target_text=target_text,
                source_term=source_term,
                target_term=target_term,
            )
            if not candidate:
                continue
            source_key = source_term.strip()
            phrase_map[source_key].add(candidate)
            segment_index[source_key][candidate].append(seg.segment_id)

    return dict(phrase_map), dict(segment_index)


def build_phrase_translation_map(
    segments: Iterable[Segment],
    glossary_matchers: tuple[GlossaryMatcher, ...],
) -> dict[str, set[str]]:
    phrase_map, _ = _collect_term_variants(segments, glossary_matchers)
    return phrase_map


def detect_inconsistencies(
    phrase_map: dict[str, set[str]],
    segment_index: dict[str, dict[str, list[str]]] | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    for source_term, targets in sorted(phrase_map.items(), key=lambda item: item[0].lower()):
        norm_targets: dict[str, str] = {}
        for target in targets:
            key = _normalize_key(target)
            if not key:
                continue
            norm_targets.setdefault(key, target.strip())
        if len(norm_targets) < 2:
            continue
        variants = sorted(norm_targets.values())
        details: dict[str, object] = {
            "source_term": source_term,
            "targets": variants,
            "variant_count": len(variants),
        }
        if segment_index is not None:
            counts = {
                target: len(segment_index.get(source_term, {}).get(target, []))
                for target in variants
            }
            sorted_by_count = sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))
            if len(sorted_by_count) >= 2 and sorted_by_count[0][1] > sorted_by_count[1][1]:
                majority_target, majority_count = sorted_by_count[0]
                minority = [item for item in sorted_by_count[1:] if item[1] < majority_count]
                details["majority_target"] = majority_target
                details["majority_count"] = majority_count
                details["minority_targets"] = [
                    {"target": target, "count": count}
                    for target, count in minority
                ]
        issues.append(
            Issue(
                code="consistency_term_variation",
                severity=Severity.WARN,
                message=f"Inconsistent translation variants for term '{source_term}'. Minority variants deviate from majority usage.",
                details=details,
            )
        )
    return issues


def report_consistency(
    segments: Iterable[Segment],
    glossary_matchers: tuple[GlossaryMatcher, ...],
) -> list[Issue]:
    phrase_map, segment_index = _collect_term_variants(segments, glossary_matchers)
    raw_issues = detect_inconsistencies(phrase_map, segment_index)
    if not raw_issues:
        return []

    issues: list[Issue] = []
    for issue in raw_issues:
        source_term = str(issue.details.get("source_term", ""))
        segments_by_target = segment_index.get(source_term, {})
        first_segment_id = ""
        for ids in segments_by_target.values():
            if ids:
                first_segment_id = ids[0]
                break
        details = dict(issue.details)
        details["segments_by_target"] = {
            target: ids for target, ids in sorted(segments_by_target.items(), key=lambda item: item[0].lower())
        }
        if first_segment_id:
            details["segment_id"] = first_segment_id
        issues.append(
            Issue(
                code=issue.code,
                severity=issue.severity,
                message=issue.message,
                details=details,
            )
        )
    return issues

