from __future__ import annotations

import json
import logging
import re
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from docx import Document
from docx.oxml.ns import qn
from tqdm import tqdm

from .config import PipelineConfig
from .consistency import report_consistency
from .docx_reader import collect_segments
from .layout_check import validate_layout
from .layout_fix import fix_expansion_issues
from .llm import (
    build_glossary_matchers,
    build_hard_glossary_replacements,
    build_llm_client,
    select_matched_glossary_terms,
    supports_repair,
)
from .logging_utils import setup_logging
from .models import Issue, Segment, Severity
from .oxml_table_fix import normalize_abbyy_oxml
from .qa_report import write_qa_jsonl, write_qa_report
from .tagging import is_supported_paragraph, paragraph_to_tagged, tagged_to_runs
from .tm import FuzzyTMHit, TMStore, normalize_text, sha256_hex
from .token_shield import BRACKET_TOKEN_RE, shield, shield_terms, strip_bracket_tokens, unshield
from .validator import (
    is_glossary_lemma_check_available,
    validate_all,
    validate_glossary_lemmas,
    validate_numbers,
    validate_placeholders,
)


def _build_repair_payload(source_shielded: str, bad_output: str) -> str:
    return (
        "TASK: REPAIR_MARKERS\n\n"
        f"SOURCE:\n{source_shielded}\n\n"
        f"OUTPUT:\n{bad_output}"
    )


def _build_glossary_retry_payload(
    source_shielded: str,
    bad_output: str,
    missing_terms: list[dict[str, str]],
) -> str:
    terms = "\n".join(
        f"- {item['source']} -> {item['target']}"
        for item in missing_terms
        if item.get("source") and item.get("target")
    )
    if not terms:
        terms = "- (no terms)"
    return (
        "TASK: REWRITE_FOR_GLOSSARY\n\n"
        "Keep marker tokens exactly as in SOURCE.\n"
        "Do not add comments or explanations.\n\n"
        f"SOURCE:\n{source_shielded}\n\n"
        f"OUTPUT:\n{bad_output}\n\n"
        "REQUIRED_TERMS (EN -> RU):\n"
        f"{terms}\n\n"
        "Return only corrected translated text."
    )


def _read_optional_text(path_value: str | None, logger: logging.Logger, label: str) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    try:
        text = path.read_text(encoding="utf-8-sig").strip()
    except OSError as e:
        raise RuntimeError(f"Cannot read {label} from '{path}': {e}") from e
    if not text:
        logger.warning(f"{label} is configured but empty: {path}")
        return None
    logger.info(f"Loaded {label}: {path}")
    return text


_STYLE_START_RE = re.compile(r"⟦S_(\d+)(?:\|[^⟧]*)?⟧")
_PLACEHOLDER_RE = re.compile(r"⟦(?!/?S_)[A-Z][A-Z0-9]*_\d+⟧")
_BRLINE_RE = re.compile(r"⟦BRLINE_\d+⟧")
_LATIN_RE = re.compile(r"[A-Za-z]")
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_LAYOUT_SPLIT_RE = re.compile(r"((?:\.\s*){3,}|\t+)")
_FINAL_CLEANUP_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # Remove zero-width spaces sometimes produced by machine translation (Google).
    (re.compile("\u200b"), ""),
    # Converted manuals often keep standalone English joiners on the title page.
    (re.compile(r"^\s*WITH\s*$", flags=re.IGNORECASE), "С"),
    (re.compile(r"\bNEW/REVISED\b"), "НОВЫЕ/ПЕРЕСМОТРЕННЫЕ"),
    (re.compile(r"\bTable\b"), "Таблица"),
    (re.compile(r"\btable\b"), "таблица"),
    # Legal small-print: tighten wording to avoid cover-page overflow in converted OEM manuals.
    (re.compile(r"\bНастоящий документ\b", flags=re.IGNORECASE), "Документ"),
    (re.compile(r"\bЭтот документ и вся информация\b", flags=re.IGNORECASE), "Документ и информация"),
    (re.compile(r"\bЭтот документ и информация, содержащаяся в нем,\b", flags=re.IGNORECASE), "Документ и его содержание"),
    (re.compile(r"\bявляются\s+исключительной\s+собственностью\b", flags=re.IGNORECASE), "являются собственностью"),
    (re.compile(r"\bсоответствующей\s+дочерней\s+компании\b", flags=re.IGNORECASE), "соответствующей компании"),
    (re.compile(r"^\s*Права на интеллектуальную собственность\b.*", flags=re.IGNORECASE), "Права ИС не предоставляются. Воспроизведение третьим лицам - только с письменного согласия Safran Landing Systems."),
    # Cover title phrasing cleanup.
    (
        re.compile(r"\bС\s+ИЛЛЮСТРИРОВАННЫЙ\s+(?:ПЕРЕЧЕНЬ|СПИСОК)\s+ДЕТАЛЕЙ\b", flags=re.IGNORECASE),
        "С ИЛЛЮСТРИРОВАННЫМ СПИСКОМ ДЕТАЛЕЙ",
    ),
)
_W_T_TAG = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
_W_HYPERLINK_TAG = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}hyperlink"
_W_RUN_TAG = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r"
_TM_RULESET_VERSION = "2026-02-22-consistency-v5-toc-and-morphology"
_BATCH_PROVIDER_ALLOWLIST = {"openai", "ollama"}
_BATCH_MAX_PLACEHOLDER_TOKENS = 12
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", flags=re.IGNORECASE)
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")


def _style_start_tag(span_id: int, flags: tuple[str, ...]) -> str:
    if flags:
        return f"⟦S_{span_id}|{'|'.join(flags)}⟧"
    return f"⟦S_{span_id}⟧"


def _style_end_tag(span_id: int) -> str:
    return f"⟦/S_{span_id}⟧"


def _extract_style_inner_by_id(tagged_text: str) -> dict[int, str]:
    inner_by_id: dict[int, str] = {}
    pos = 0
    while True:
        m = _STYLE_START_RE.search(tagged_text, pos)
        if not m:
            break
        span_id = int(m.group(1))
        end_pat = re.compile(rf"⟦/S_{span_id}⟧")
        m_end = end_pat.search(tagged_text, m.end())
        if not m_end:
            break
        inner_by_id[span_id] = tagged_text[m.end() : m_end.start()]
        pos = m_end.end()
    return inner_by_id


def _protect_bracket_tokens(text: str) -> tuple[str, dict[str, str]]:
    token_map: dict[str, str] = {}

    def _repl(m: re.Match[str]) -> str:
        ph = f"__DOCXRU_BR_{len(token_map) + 1}__"
        token_map[ph] = m.group(0)
        return ph

    protected = BRACKET_TOKEN_RE.sub(_repl, text)
    return protected, token_map


def _restore_bracket_tokens(text: str, token_map: dict[str, str]) -> str:
    out = text
    for ph, token in token_map.items():
        out = out.replace(ph, token).replace(ph.lower(), token).replace(ph.upper(), token)
    return out


def _translate_shielded_fragment(
    text: str,
    llm_client,
    context: dict[str, Any],
) -> tuple[str, list[Issue]]:
    if not text:
        return text, []
    protected, token_map = _protect_bracket_tokens(text)
    try:
        translated = llm_client.translate(protected, context)
    except Exception as e:
        return text, [
            Issue(
                code="llm_error",
                severity=Severity.WARN,
                message=f"LLM ошибка на фрагменте: {e}",
                details={},
            )
        ]
    restored = _restore_bracket_tokens(translated, token_map)
    placeholder_issues = validate_placeholders(text, restored)
    if any(i.severity == Severity.ERROR for i in placeholder_issues):
        return text, placeholder_issues
    return restored, placeholder_issues


def _translate_plain_chunk(
    chunk: str,
    cfg: PipelineConfig,
    llm_client,
    context: dict[str, Any],
    cache: dict[str, str],
    glossary_terms: tuple[tuple[re.Pattern[str], str], ...] = (),
) -> tuple[str, list[Issue]]:
    if not chunk:
        return chunk, []

    if chunk in cache:
        return cache[chunk], []

    shielded = chunk
    token_map: dict[str, str] = {}
    # Apply hard glossary before generic shielding so phrase-level rules can still match
    # raw text fragments that would otherwise be split into DIM/PN placeholders.
    if glossary_terms and _LATIN_RE.search(shielded):
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
    translated_shielded, issues = _translate_shielded_fragment(shielded, llm_client, context)
    translated = unshield(translated_shielded, token_map)

    number_issues = validate_numbers(chunk, translated)
    all_issues = issues + number_issues
    cache[chunk] = translated
    return translated, all_issues


def _translate_toc_like_text(
    text: str,
    cfg: PipelineConfig,
    llm_client,
    context: dict[str, Any],
    cache: dict[str, str],
    glossary_terms: tuple[tuple[re.Pattern[str], str], ...] = (),
) -> tuple[str, list[Issue]]:
    parts = _LAYOUT_SPLIT_RE.split(text)
    out_parts: list[str] = []
    issues: list[Issue] = []

    for part in parts:
        if part == "":
            continue
        if _LAYOUT_SPLIT_RE.fullmatch(part):
            out_parts.append(part)
            continue
        if not _LATIN_RE.search(part):
            out_parts.append(part)
            continue

        m_prefix = re.match(r"^\s*", part)
        m_suffix = re.search(r"\s*$", part)
        prefix = m_prefix.group(0) if m_prefix else ""
        suffix = m_suffix.group(0) if m_suffix else ""
        core_end = len(part) - len(suffix)
        core = part[len(prefix) : core_end]
        if not core:
            out_parts.append(part)
            continue

        translated_core, tr_issues = _translate_plain_chunk(core, cfg, llm_client, context, cache, glossary_terms)
        issues.extend(tr_issues)
        out_parts.append(prefix + translated_core + suffix)

    return "".join(out_parts), issues


def _collect_complex_text_groups(paragraph) -> list[list[Any]]:
    groups: list[list[Any]] = []
    for child in paragraph._p.iterchildren():
        if child.tag == _W_RUN_TAG:
            run_nodes = list(child.iter(_W_T_TAG))
            if run_nodes:
                # Keep run-level boundaries to avoid shifting tabs/page-number alignment.
                groups.append(run_nodes)
            continue

        if child.tag == _W_HYPERLINK_TAG:
            hyperlink_nodes = list(child.iter(_W_T_TAG))
            if hyperlink_nodes:
                groups.append(hyperlink_nodes)
    return groups


def _run_is_safe_for_text_replace(run) -> bool:
    for child in run._r.iterchildren():
        local = child.tag.split("}")[-1]
        if local in {"rPr", "t", "tab", "br", "cr", "noBreakHyphen", "softHyphen", "lastRenderedPageBreak"}:
            continue
        return False
    return True


def _should_translate_segment_text(text: str) -> bool:
    """Heuristic: skip segments that are already in RU or contain no Latin text.

    This avoids:
    - retranslating already-Russian headings (often present in partially translated manuals)
    - rewriting purely numeric / symbol-only paragraphs (page labels, etc.)
    """
    if not text or not text.strip():
        return False
    latin = len(_LATIN_RE.findall(text))
    if latin == 0:
        return False
    cyr = len(_CYRILLIC_RE.findall(text))
    # Mostly Cyrillic with only a few Latin letters (abbreviations, brand names) -> keep as-is.
    return not (cyr >= latin * 3 and latin <= 12)


def _compact_context_text(text: str, *, max_chars: int = 220) -> str:
    flat = re.sub(r"\s+", " ", text or "").strip()
    if not flat:
        return ""
    if len(flat) <= max_chars:
        return flat
    return flat[: max_chars - 3].rstrip() + "..."


def _build_matched_glossary_context(
    text: str,
    glossary_matchers,
    *,
    limit: int,
) -> list[dict[str, str]]:
    matched_pairs = select_matched_glossary_terms(text, glossary_matchers, limit=limit)
    if not matched_pairs:
        return []
    return [{"source": source, "target": target} for source, target in matched_pairs]


def _build_tm_references_context(
    hits: list[FuzzyTMHit],
    *,
    max_chars: int,
) -> list[dict[str, Any]]:
    if not hits:
        return []

    refs: list[dict[str, Any]] = []
    consumed = 0
    budget = max(0, int(max_chars))
    for hit in hits:
        source = _compact_context_text(strip_bracket_tokens(hit.source_norm), max_chars=180)
        target = _compact_context_text(strip_bracket_tokens(hit.target_text), max_chars=180)
        if not source or not target:
            continue
        line = f"{source} => {target}"
        line_cost = len(line) + 1
        if budget > 0 and consumed + line_cost > budget:
            break
        refs.append(
            {
                "source": source,
                "target": target,
                "similarity": round(float(hit.similarity), 4),
            }
        )
        consumed += line_cost
    return refs


def _should_attach_neighbor_context(seg: Segment) -> bool:
    source = (seg.source_plain or "").strip()
    if not source:
        return False
    return bool(_LATIN_RE.search(source))


def _neighbor_context_max_chars(seg: Segment, cfg: PipelineConfig) -> int:
    base = int(cfg.llm.context_window_chars) if cfg.llm.context_window_chars > 0 else 220
    if seg.context.get("in_table"):
        return min(base, 100)
    return base


def _attach_neighbor_snippets(segments: list[Segment], cfg: PipelineConfig) -> None:
    for idx, seg in enumerate(segments):
        if not _should_attach_neighbor_context(seg):
            continue
        max_chars = _neighbor_context_max_chars(seg, cfg)
        prev_text = _compact_context_text(segments[idx - 1].source_plain, max_chars=max_chars) if idx > 0 else ""
        next_text = (
            _compact_context_text(segments[idx + 1].source_plain, max_chars=max_chars)
            if idx + 1 < len(segments)
            else ""
        )
        if prev_text:
            seg.context["prev_text"] = prev_text
        if next_text:
            seg.context["next_text"] = next_text


def _build_document_glossary_context(
    glossary_map: dict[str, str],
    *,
    limit: int,
) -> list[dict[str, str]]:
    if not glossary_map:
        return []
    capped = max(0, int(limit))
    items = list(glossary_map.items())
    if capped > 0:
        items = items[-capped:]
    return [{"source": source, "target": target} for source, target in items]


def _looks_sentence_like_english(text: str) -> bool:
    words = _LATIN_WORD_RE.findall(text or "")
    if len(words) < 8:
        return False
    return bool(re.search(r"[.!?;:]", text or ""))


def _should_apply_hard_glossary_to_segment(seg: Segment) -> bool:
    """Keep hard glossary on compact labels/TOC, relax for long narrative sentences.

    Hard glossary placeholders improve strict term stability, but can lock terms into
    nominative forms inside long RU sentences. We keep locking for TOC/table labels
    and short heading-like text, and relax it for sentence-like body prose.
    """
    if seg.context.get("is_toc_entry"):
        return True
    if seg.context.get("in_table"):
        return True

    source = (seg.source_plain or "").strip()
    if not source:
        return False

    latin_words = _LATIN_WORD_RE.findall(source)
    if len(latin_words) <= 6:
        return True
    if len(latin_words) >= 14:
        return False
    return not _looks_sentence_like_english(source)


def _segment_glossary_terms(
    seg: Segment,
    glossary_terms: tuple[tuple[re.Pattern[str], str], ...],
) -> tuple[tuple[re.Pattern[str], str], ...]:
    if not glossary_terms:
        return ()
    if _should_apply_hard_glossary_to_segment(seg):
        return glossary_terms
    return ()


def _append_recent_translation(
    ring: deque[tuple[str, str]],
    *,
    source_plain: str,
    target_plain: str,
) -> None:
    source = _compact_context_text(strip_bracket_tokens(source_plain), max_chars=220)
    target = _compact_context_text(strip_bracket_tokens(target_plain), max_chars=220)
    if not source or not target:
        return
    ring.append((source, target))


def _apply_final_run_level_cleanup(segments: list[Segment]) -> int:
    changed_runs = 0
    seen_paragraphs: set[int] = set()
    for seg in segments:
        if seg.target_shielded_tagged is None or seg.target_tagged is None:
            continue
        para = seg.paragraph_ref
        key = id(para)
        if key in seen_paragraphs:
            continue
        seen_paragraphs.add(key)

        for run in para.runs:
            if not _run_is_safe_for_text_replace(run):
                continue
            original = run.text or ""
            updated = original
            for pattern, replacement in _FINAL_CLEANUP_RULES:
                updated = pattern.sub(replacement, updated)
            if updated != original:
                run.text = updated
                changed_runs += 1
    return changed_runs


def _write_text_nodes(nodes: list[Any], text: str) -> None:
    if not nodes:
        return
    first = nodes[0]
    first.text = text
    xml_space = qn("xml:space")
    if text.startswith(" ") or text.endswith(" "):
        first.set(xml_space, "preserve")
    elif xml_space in first.attrib:
        del first.attrib[xml_space]
    for node in nodes[1:]:
        node.text = ""


def _translate_complex_paragraph_in_place(
    seg: Segment,
    cfg: PipelineConfig,
    llm_client,
    cache: dict[str, str],
    glossary_terms: tuple[tuple[re.Pattern[str], str], ...] = (),
) -> tuple[bool, list[Issue]]:
    groups = _collect_complex_text_groups(seg.paragraph_ref)
    if not groups:
        return False, []

    changed = False
    issues: list[Issue] = []
    for group_i, nodes in enumerate(groups):
        source_text = "".join(node.text or "" for node in nodes)
        if not source_text.strip():
            continue
        if not _LATIN_RE.search(source_text):
            continue
        ctx = dict(seg.context)
        ctx["complex_group"] = group_i
        translated, tr_issues = _translate_toc_like_text(source_text, cfg, llm_client, ctx, cache, glossary_terms)
        issues.extend(tr_issues)
        if translated != source_text:
            _write_text_nodes(nodes, translated)
            changed = True
    return changed, issues


def _fallback_translate_by_spans(
    seg: Segment,
    cfg: PipelineConfig,
    llm_client,
) -> tuple[str | None, list[Issue]]:
    if not seg.spans or not seg.shielded_tagged:
        return None, []

    source_inner = _extract_style_inner_by_id(seg.shielded_tagged)
    if not source_inner:
        return None, []

    rebuilt: list[str] = []
    issues: list[Issue] = []
    span_count = len(seg.spans)
    for span in seg.spans:
        inner = source_inner.get(span.span_id)
        if inner is None:
            return None, issues

        ctx = dict(seg.context)
        ctx["span_id"] = span.span_id
        ctx["span_total"] = span_count
        translated_inner, tr_issues = _translate_shielded_fragment(inner, llm_client, ctx)
        issues.extend(tr_issues)

        rebuilt.append(_style_start_tag(span.span_id, span.flags))
        rebuilt.append(translated_inner)
        rebuilt.append(_style_end_tag(span.span_id))

    candidate = "".join(rebuilt)
    src_unshielded = unshield(seg.shielded_tagged, seg.token_map or {})
    tgt_unshielded = unshield(candidate, seg.token_map or {})
    src_plain = strip_bracket_tokens(src_unshielded)
    tgt_plain = strip_bracket_tokens(tgt_unshielded)
    final_issues = validate_all(
        source_shielded_tagged=seg.shielded_tagged,
        target_shielded_tagged=candidate,
        source_unshielded_plain=src_plain,
        target_unshielded_plain=tgt_plain,
    )
    issues.extend(final_issues)
    if any(i.severity == Severity.ERROR for i in final_issues):
        return None, issues
    issues.append(
        Issue(
            code="style_fallback_used",
            severity=Severity.INFO,
            message="Применён резервный перевод по span, чтобы сохранить маркеры форматирования.",
            details={},
        )
    )
    return candidate, issues


def _validate_segment_candidate(seg: Segment, out: str) -> list[Issue]:
    src_unshielded = unshield(seg.shielded_tagged or "", seg.token_map or {})
    tgt_unshielded = unshield(out, seg.token_map or {})
    src_plain = strip_bracket_tokens(src_unshielded)
    tgt_plain = strip_bracket_tokens(tgt_unshielded)
    issues = validate_all(
        source_shielded_tagged=seg.shielded_tagged or "",
        target_shielded_tagged=out,
        source_unshielded_plain=src_plain,
        target_unshielded_plain=tgt_plain,
    )
    leak_upper = out.upper()
    if "TASK: REPAIR_MARKERS" in leak_upper or ("SOURCE:" in leak_upper and "OUTPUT:" in leak_upper):
        issues.append(
            Issue(
                code="repair_payload_leak",
                severity=Severity.ERROR,
                message="Repair prompt payload leaked into segment output.",
                details={},
            )
        )
    return issues


def _target_plain_from_candidate(seg: Segment, out: str) -> str:
    target_unshielded = unshield(out, seg.token_map or {})
    return strip_bracket_tokens(target_unshielded)


def _extract_missing_glossary_terms(issues: list[Issue]) -> list[dict[str, str]]:
    missing_terms: list[dict[str, str]] = []
    for issue in issues:
        if issue.code != "glossary_lemma_mismatch":
            continue
        raw_missing = issue.details.get("missing")
        if not isinstance(raw_missing, list):
            continue
        for item in raw_missing:
            if not isinstance(item, dict):
                continue
            source = item.get("source")
            target = item.get("target")
            if isinstance(source, str) and isinstance(target, str):
                missing_terms.append({"source": source, "target": target})
    return missing_terms


def _count_missing_glossary_terms(issues: list[Issue]) -> int:
    return len(_extract_missing_glossary_terms(issues))


def _utc_now_iso() -> str:
    utc = getattr(datetime, "UTC", timezone.utc)  # noqa: UP017
    return datetime.now(utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _tm_text_fingerprint(text: str | None) -> str:
    if not text:
        return "none"
    return sha256_hex(normalize_text(text))


def _build_tm_profile_key(
    cfg: PipelineConfig,
    *,
    custom_system_prompt: str | None,
    glossary_text: str | None,
) -> str:
    return "|".join(
        (
            f"provider={cfg.llm.provider.strip().lower()}",
            f"model={cfg.llm.model.strip()}",
            f"source_lang={cfg.llm.source_lang.strip().lower()}",
            f"target_lang={cfg.llm.target_lang.strip().lower()}",
            f"hard_glossary={int(bool(cfg.llm.hard_glossary))}",
            f"glossary_in_prompt={int(bool(cfg.llm.glossary_in_prompt))}",
            f"glossary_prompt_mode={cfg.llm.glossary_prompt_mode.strip().lower()}",
            f"glossary_match_limit={int(cfg.llm.glossary_match_limit)}",
            f"structured_output_mode={cfg.llm.structured_output_mode.strip().lower()}",
            f"batch_segments={int(cfg.llm.batch_segments)}",
            f"batch_skip_on_brline={int(bool(cfg.llm.batch_skip_on_brline))}",
            f"batch_max_style_tokens={int(cfg.llm.batch_max_style_tokens)}",
            f"context_window_chars={int(cfg.llm.context_window_chars)}",
            f"fuzzy_enabled={int(bool(cfg.tm.fuzzy_enabled))}",
            f"fuzzy_top_k={int(cfg.tm.fuzzy_top_k)}",
            f"fuzzy_min_similarity={cfg.tm.fuzzy_min_similarity:.4f}",
            f"fuzzy_prompt_max_chars={int(cfg.tm.fuzzy_prompt_max_chars)}",
            f"abbyy_profile={cfg.abbyy_profile.strip().lower()}",
            f"glossary_lemma_check={cfg.glossary_lemma_check.strip().lower()}",
            f"layout_check={int(bool(cfg.layout_check))}",
            f"layout_expansion_warn_ratio={float(cfg.layout_expansion_warn_ratio):.3f}",
            f"layout_auto_fix={int(bool(cfg.layout_auto_fix))}",
            f"layout_font_reduction_pt={float(cfg.layout_font_reduction_pt):.3f}",
            f"layout_spacing_factor={float(cfg.layout_spacing_factor):.3f}",
            f"reasoning_effort={(cfg.llm.reasoning_effort or '').strip().lower() or 'default'}",
            f"system_prompt_sha={_tm_text_fingerprint(custom_system_prompt)}",
            f"glossary_sha={_tm_text_fingerprint(glossary_text)}",
        )
    )


def _build_tm_meta(seg: Segment, cfg: PipelineConfig, source_hash: str, *, origin: str) -> dict[str, Any]:
    return {
        "provider": cfg.llm.provider,
        "model": cfg.llm.model,
        "origin": origin,
        "segment_id": seg.segment_id,
        "location": seg.location,
        "part": seg.context.get("part"),
        "section_header": seg.context.get("section_header"),
        "in_table": bool(seg.context.get("in_table")),
        "source_hash": source_hash,
    }


def _build_history_record(
    seg: Segment,
    cfg: PipelineConfig,
    *,
    source_hash: str | None,
    origin: str,
) -> dict[str, Any]:
    source_shielded = seg.shielded_tagged or ""
    target_shielded = seg.target_shielded_tagged or ""
    source_unshielded = unshield(source_shielded, seg.token_map or {})
    target_unshielded = unshield(target_shielded, seg.token_map or {})
    source_plain = strip_bracket_tokens(source_unshielded)
    target_plain = strip_bracket_tokens(target_unshielded)
    return {
        "timestamp_utc": _utc_now_iso(),
        "provider": cfg.llm.provider,
        "model": cfg.llm.model,
        "origin": origin,
        "source_hash": source_hash,
        "segment_id": seg.segment_id,
        "location": seg.location,
        "part": seg.context.get("part"),
        "section_header": seg.context.get("section_header"),
        "in_table": bool(seg.context.get("in_table")),
        "source_text": source_plain,
        "target_text": target_plain,
    }


def _as_warn_issues(issues: list[Issue]) -> list[Issue]:
    normalized: list[Issue] = []
    for issue in issues:
        if issue.severity == Severity.ERROR:
            normalized.append(
                Issue(
                    code=f"{issue.code}_downgraded",
                    severity=Severity.WARN,
                    message=issue.message,
                    details=issue.details,
                )
            )
        else:
            normalized.append(issue)
    return normalized


def _append_history_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _finalize_translation_result(
    *,
    seg: Segment,
    source_hash: str,
    source_norm: str,
    out: str,
    issues: list[Issue],
    cfg: PipelineConfig,
    llm_client,
    tm: TMStore,
    complex_chunk_cache: dict[str, str],
    glossary_terms: tuple[tuple[re.Pattern[str], str], ...],
    llm_translated_segments: set[str],
    recent_translations: deque[tuple[str, str]] | None = None,
) -> int:
    hard_errors = any(i.severity == Severity.ERROR for i in issues)
    if hard_errors:
        changed, fallback_issues = _translate_complex_paragraph_in_place(
            seg,
            cfg,
            llm_client,
            complex_chunk_cache,
            glossary_terms,
        )
        if changed:
            seg.target_shielded_tagged = None
            seg.issues.extend(
                [
                    Issue(
                        code="complex_fallback_after_hard_errors",
                        severity=Severity.WARN,
                        message=(
                            "Segment translated with complex in-place fallback after strict marker "
                            "validation failure."
                        ),
                        details={},
                    ),
                    *_as_warn_issues(fallback_issues),
                ]
            )
            return 1

    seg.target_shielded_tagged = out
    seg.issues.extend(issues)

    # Store to TM only if no hard errors (avoid caching broken markers).
    if not hard_errors:
        llm_translated_segments.add(seg.segment_id)
        tm.put_exact(
            source_hash=source_hash,
            source_norm=source_norm,
            target_text=out,
            meta=_build_tm_meta(seg, cfg, source_hash, origin="llm"),
        )
        tm.set_progress(seg.segment_id, "ok", source_hash=source_hash)
        if recent_translations is not None:
            _append_recent_translation(
                recent_translations,
                source_plain=seg.source_plain,
                target_plain=_target_plain_from_candidate(seg, out),
            )
    else:
        tm.set_progress(
            seg.segment_id,
            "error",
            source_hash=source_hash,
            error="; ".join(i.code for i in issues),
        )
    return 0


def _chunk_translation_jobs(
    jobs: list[tuple[Segment, str, str]],
    *,
    max_segments: int,
    max_chars: int,
) -> list[list[tuple[Segment, str, str]]]:
    if not jobs:
        return []

    seg_limit = max(1, int(max_segments))
    char_limit = max(1, int(max_chars))
    groups: list[list[tuple[Segment, str, str]]] = []
    current: list[tuple[Segment, str, str]] = []
    current_chars = 0

    for job in jobs:
        seg = job[0]
        estimated = len(seg.shielded_tagged or "") + 128
        if current and (len(current) >= seg_limit or current_chars + estimated > char_limit):
            groups.append(current)
            current = []
            current_chars = 0
        current.append(job)
        current_chars += estimated

    if current:
        groups.append(current)
    return groups


def _batch_ineligibility_reasons(seg: Segment, cfg: PipelineConfig) -> list[str]:
    text = str(seg.context.get("_batch_eligibility_text") or seg.shielded_tagged or "")
    reasons: list[str] = []
    if seg.context.get("is_toc_entry"):
        reasons.append("toc_entry")
    if cfg.llm.batch_skip_on_brline and _BRLINE_RE.search(text):
        reasons.append("contains_brline")
    if cfg.llm.batch_max_style_tokens >= 0 and len(_STYLE_START_RE.findall(text)) > cfg.llm.batch_max_style_tokens:
        reasons.append("too_many_style_tokens")
    if len(_PLACEHOLDER_RE.findall(text)) > _BATCH_MAX_PLACEHOLDER_TOKENS:
        reasons.append("too_many_placeholders")
    return reasons


def _collect_issue_code_counts(segments: list[Segment]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for seg in segments:
        for issue in seg.issues:
            counts[issue.code] += 1
    return counts


def _attach_issues_to_segments(segments: list[Segment], issues: list[Issue]) -> int:
    if not segments or not issues:
        return 0
    seg_by_id = {seg.segment_id: seg for seg in segments}
    attached = 0
    for issue in issues:
        segment_id = str(issue.details.get("segment_id", "")).strip()
        if segment_id:
            seg = seg_by_id.get(segment_id)
            if seg is None:
                continue
            seg.issues.append(issue)
            attached += 1
            continue
        segments[0].issues.append(issue)
        attached += 1
    return attached


def _build_batch_translation_prompt(items: list[dict[str, str]]) -> str:
    input_json = json.dumps(items, ensure_ascii=False)
    return (
        "TASK: BATCH_TRANSLATE_SEGMENTS\n"
        "Translate each item.text from English to Russian.\n"
        "Keep marker placeholders and style tags unchanged.\n"
        "Return ONLY valid JSON object (no markdown, no prose) where key=id and value=translated text.\n"
        "Preferred shape:\n"
        '{"<id>":"<translated_text>","<id2>":"<translated_text2>"}\n'
        "Also accepted shape:\n"
        '{"translations":[{"id":"<id>","text":"<translated_text>"}]}\n'
        "Rules:\n"
        "- Keep all IDs exactly as provided.\n"
        "- Do not merge or split items.\n"
        "- Include every item exactly once.\n"
        "INPUT_JSON:\n"
        f"{input_json}"
    )


def _iter_balanced_json_chunks(text: str, *, open_char: str, close_char: str) -> list[str]:
    chunks: list[str] = []
    in_string = False
    escaped = False
    depth = 0
    start_idx = -1

    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == open_char:
            if depth == 0:
                start_idx = idx
            depth += 1
            continue
        if ch == close_char and depth > 0:
            depth -= 1
            if depth == 0 and start_idx >= 0:
                candidate = text[start_idx : idx + 1].strip()
                if candidate:
                    chunks.append(candidate)
                start_idx = -1
    return chunks


def _extract_json_payload(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        raise RuntimeError("Empty batch response")

    candidates: list[str] = []
    seen: set[str] = set()

    def _add_candidate(value: str) -> None:
        candidate = (value or "").strip()
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)

    _add_candidate(text)

    for m in _JSON_FENCE_RE.finditer(text):
        fenced = (m.group(1) or "").strip()
        if fenced:
            _add_candidate(fenced)

    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        _add_candidate(text[obj_start : obj_end + 1])

    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start >= 0 and arr_end > arr_start:
        _add_candidate(text[arr_start : arr_end + 1])

    for chunk in _iter_balanced_json_chunks(text, open_char="{", close_char="}"):
        _add_candidate(chunk)
    for chunk in _iter_balanced_json_chunks(text, open_char="[", close_char="]"):
        _add_candidate(chunk)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("Batch response is not valid JSON")


def _coerce_batch_translation_map(payload: Any) -> dict[str, str]:
    entries: list[dict[str, Any]] | None = None
    if isinstance(payload, dict):
        if "translations" in payload and isinstance(payload["translations"], list):
            entries = [item for item in payload["translations"] if isinstance(item, dict)]
        elif all(isinstance(k, str) and isinstance(v, str) for k, v in payload.items()):
            return {str(k): str(v) for k, v in payload.items()}
    elif isinstance(payload, list):
        entries = [item for item in payload if isinstance(item, dict)]

    if entries is None:
        raise RuntimeError("Batch response has unsupported JSON schema")

    out: dict[str, str] = {}
    for item in entries:
        raw_id = item.get("id")
        raw_text = item.get("text")
        if not isinstance(raw_id, str) or not isinstance(raw_text, str):
            continue
        if raw_id in out:
            raise RuntimeError(f"Batch response has duplicate id: {raw_id}")
        out[raw_id] = raw_text
    return out


def _parse_batch_translation_output(raw: str, expected_ids: list[str]) -> dict[str, str]:
    payload = _extract_json_payload(raw)
    out = _coerce_batch_translation_map(payload)
    expected = set(expected_ids)
    got = set(out.keys())
    missing = sorted(expected - got)
    if missing:
        raise RuntimeError(f"Batch response missing ids: {missing[:5]}")
    extra = sorted(got - expected)
    if extra:
        raise RuntimeError(f"Batch response has unexpected ids: {extra[:5]}")
    return {seg_id: out[seg_id] for seg_id in expected_ids}


def _translate_batch_once(
    llm_client,
    jobs: list[tuple[Segment, str, str]],
) -> dict[str, str]:
    items = [{"id": seg.segment_id, "text": seg.shielded_tagged or ""} for seg, _, _ in jobs]
    prompt = _build_batch_translation_prompt(items)
    first_seg = jobs[0][0]
    context = {
        "task": "batch_translate",
        "part": first_seg.context.get("part"),
        "batch_size": len(jobs),
    }
    raw = llm_client.translate(prompt, context)
    return _parse_batch_translation_output(raw, [item["id"] for item in items])


def _is_batch_json_contract_error(error: Exception) -> bool:
    message = str(error).strip().lower()
    if not message:
        return False
    return any(
        token in message
        for token in (
            "json",
            "schema",
            "missing ids",
            "unexpected ids",
            "duplicate id",
        )
    )


def _translate_batch_group(
    jobs: list[tuple[Segment, str, str]],
    cfg: PipelineConfig,
    llm_client,
    logger: logging.Logger,
) -> list[tuple[Segment, str, str, str, list[Issue]]]:
    if len(jobs) == 1:
        seg, source_hash, source_norm = jobs[0]
        out, issues = _translate_one(seg, cfg, llm_client, source_hash, source_norm, logger)
        return [(seg, source_hash, source_norm, out, issues)]

    try:
        batch_map = _translate_batch_once(llm_client, jobs)
    except Exception as e:
        logger.warning(f"Batch translate failed (size={len(jobs)}), fallback to single-segment: {e}")
        results: list[tuple[Segment, str, str, str, list[Issue]]] = []
        has_json_contract_error = _is_batch_json_contract_error(e)
        for seg, source_hash, source_norm in jobs:
            out, issues = _translate_one(seg, cfg, llm_client, source_hash, source_norm, logger)
            diagnostics: list[Issue] = []
            if has_json_contract_error:
                diagnostics.append(
                    Issue(
                        code="batch_json_schema_violation",
                        severity=Severity.WARN,
                        message=f"Batch JSON/schema validation failed: {e}",
                        details={"batch_size": len(jobs)},
                    )
                )
            issues = diagnostics + [
                Issue(
                    code="batch_fallback_single",
                    severity=Severity.WARN,
                    message=f"Batch translation failed, fallback to single segment: {e}",
                    details={"batch_size": len(jobs)},
                ),
                *issues,
            ]
            results.append((seg, source_hash, source_norm, out, issues))
        return results

    results: list[tuple[Segment, str, str, str, list[Issue]]] = []
    for seg, source_hash, source_norm in jobs:
        candidate = batch_map.get(seg.segment_id)
        if candidate is None:
            out, fallback_issues = _translate_one(seg, cfg, llm_client, source_hash, source_norm, logger)
            issues = [
                Issue(
                    code="batch_missing_segment",
                    severity=Severity.WARN,
                    message="Batch response missed segment id, fallback to single segment translation.",
                    details={"batch_size": len(jobs)},
                ),
                *fallback_issues,
            ]
            results.append((seg, source_hash, source_norm, out, issues))
            continue

        batch_issues = _validate_segment_candidate(seg, candidate)
        if any(i.severity == Severity.ERROR for i in batch_issues):
            out, fallback_issues = _translate_one(seg, cfg, llm_client, source_hash, source_norm, logger)
            issues = [
                Issue(
                    code="batch_validation_fallback",
                    severity=Severity.WARN,
                    message="Batch output failed validation, fallback to single segment translation.",
                    details={"batch_size": len(jobs)},
                ),
                *batch_issues,
                *fallback_issues,
            ]
            results.append((seg, source_hash, source_norm, out, issues))
            continue

        batch_issues.append(
            Issue(
                code="batch_ok",
                severity=Severity.INFO,
                message="Segment translated in grouped batch mode.",
                details={"batch_size": len(jobs)},
            )
        )
        results.append((seg, source_hash, source_norm, candidate, batch_issues))
    return results


def _translate_one(
    seg: Segment,
    cfg: PipelineConfig,
    llm_client,
    source_hash: str,
    source_norm: str,
    logger: logging.Logger,
) -> tuple[str, list[Issue]]:
    """Translate a single segment with retries and marker validation."""
    last_output: str | None = None
    issues: list[Issue] = []
    can_repair = supports_repair(llm_client)
    max_attempts = cfg.llm.retries if can_repair else 1
    glossary_mode = cfg.glossary_lemma_check.strip().lower()
    if glossary_mode not in {"off", "warn", "retry"}:
        glossary_mode = "off"
    glossary_retry_enabled = glossary_mode == "retry" and can_repair
    glossary_retry_done = False

    for attempt in range(1, max_attempts + 1):
        try:
            if attempt == 1:
                out = llm_client.translate(seg.shielded_tagged or "", seg.context)
            else:
                # Repair attempt: do not retranslate, only fix markers.
                ctx = dict(seg.context)
                ctx["task"] = "repair"
                out = llm_client.translate(
                    _build_repair_payload(seg.shielded_tagged or "", last_output or ""),
                    ctx,
                )
        except Exception as e:
            issues = [
                Issue(
                    code="llm_error",
                    severity=Severity.ERROR,
                    message=f"LLM ошибка: {e}",
                    details={"attempt": attempt},
                )
            ]
            last_output = last_output or ""
            continue

        last_output = out

        # Validate markers and numbers
        issues = _validate_segment_candidate(seg, out)

        hard_errors = [i for i in issues if i.severity == Severity.ERROR]
        if not hard_errors:
            glossary_issues = validate_glossary_lemmas(
                _target_plain_from_candidate(seg, out),
                seg.context.get("matched_glossary_terms"),
                mode=glossary_mode,
            )
            if not glossary_issues:
                return out, issues

            base_issues = [*issues, *glossary_issues]
            if not glossary_retry_enabled or glossary_retry_done:
                return out, base_issues

            missing_terms = _extract_missing_glossary_terms(glossary_issues)
            if not missing_terms:
                return out, base_issues

            glossary_retry_done = True
            retry_ctx = dict(seg.context)
            retry_ctx["task"] = "repair"
            retry_ctx["glossary_retry"] = True
            try:
                retry_out = llm_client.translate(
                    _build_glossary_retry_payload(seg.shielded_tagged or "", out, missing_terms),
                    retry_ctx,
                )
            except Exception as e:
                return out, [
                    *base_issues,
                    Issue(
                        code="glossary_retry_llm_error",
                        severity=Severity.WARN,
                        message=f"Glossary-focused rewrite failed: {e}",
                        details={"attempt": attempt},
                    ),
                ]

            retry_marker_issues = _validate_segment_candidate(seg, retry_out)
            retry_hard_errors = [i for i in retry_marker_issues if i.severity == Severity.ERROR]
            if retry_hard_errors:
                return out, [
                    *base_issues,
                    Issue(
                        code="glossary_retry_rejected",
                        severity=Severity.WARN,
                        message="Glossary-focused rewrite failed marker validation; previous output kept.",
                        details={"attempt": attempt},
                    ),
                    *_as_warn_issues(retry_marker_issues),
                ]

            retry_glossary_issues = validate_glossary_lemmas(
                _target_plain_from_candidate(seg, retry_out),
                seg.context.get("matched_glossary_terms"),
                mode=glossary_mode,
            )
            if _count_missing_glossary_terms(retry_glossary_issues) < _count_missing_glossary_terms(glossary_issues):
                return retry_out, [
                    Issue(
                        code="glossary_retry_applied",
                        severity=Severity.INFO,
                        message="Glossary-focused rewrite improved term coverage.",
                        details={"attempt": attempt},
                    ),
                    *retry_marker_issues,
                    *retry_glossary_issues,
                ]

            return out, [
                *base_issues,
                Issue(
                    code="glossary_retry_no_improvement",
                    severity=Severity.INFO,
                    message="Glossary-focused rewrite did not improve term coverage; previous output kept.",
                    details={"attempt": attempt},
                ),
                *retry_marker_issues,
                *_as_warn_issues(retry_glossary_issues),
            ]

        # If we still have marker errors and there is room to retry, go to repair attempt.
        if attempt < max_attempts:
            continue

        # Last-resort recovery: translate span-by-span to salvage marker fidelity.
        fallback_out, fallback_issues = _fallback_translate_by_spans(seg, cfg, llm_client)
        if fallback_out is not None:
            return fallback_out, [
                Issue(
                    code="span_fallback_after_hard_errors",
                    severity=Severity.WARN,
                    message="Segment recovered via span-level fallback after marker validation errors.",
                    details={"attempts": attempt, "supports_repair": can_repair},
                ),
                *fallback_issues,
            ]
        return out, issues

    return last_output or (seg.shielded_tagged or ""), issues


def translate_docx(
    input_path: Path,
    output_path: Path,
    cfg: PipelineConfig,
    *,
    resume: bool = False,
    max_segments: int | None = None,
) -> None:
    logger = setup_logging(Path(cfg.log_path))
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(
        "Mode: "
        f"{cfg.mode}; "
        f"concurrency={cfg.concurrency}; "
        f"headers={cfg.include_headers}; "
        f"footers={cfg.include_footers}; "
        f"structured_output_mode={cfg.llm.structured_output_mode}; "
        f"glossary_in_prompt={cfg.llm.glossary_in_prompt}; "
        f"glossary_prompt_mode={cfg.llm.glossary_prompt_mode}; "
        f"glossary_match_limit={cfg.llm.glossary_match_limit}; "
        f"hard_glossary={cfg.llm.hard_glossary}; "
        f"batch_skip_on_brline={cfg.llm.batch_skip_on_brline}; "
        f"batch_max_style_tokens={cfg.llm.batch_max_style_tokens}; "
        f"context_window_chars={cfg.llm.context_window_chars}; "
        f"reasoning_effort={cfg.llm.reasoning_effort or '(default)'}; "
        f"batch_segments={cfg.llm.batch_segments}; "
        f"batch_max_chars={cfg.llm.batch_max_chars}; "
        f"fuzzy_enabled={cfg.tm.fuzzy_enabled}; "
        f"abbyy_profile={cfg.abbyy_profile}; "
        f"glossary_lemma_check={cfg.glossary_lemma_check}; "
        f"layout_check={cfg.layout_check}; "
        f"layout_expansion_warn_ratio={cfg.layout_expansion_warn_ratio}; "
        f"layout_auto_fix={cfg.layout_auto_fix}; "
        f"layout_font_reduction_pt={cfg.layout_font_reduction_pt}; "
        f"layout_spacing_factor={cfg.layout_spacing_factor}; "
        f"history_jsonl={cfg.translation_history_path or '(off)'}"
    )
    logger.info(f"TM ruleset version: {_TM_RULESET_VERSION}")

    doc = Document(str(input_path))
    segments = collect_segments(doc, include_headers=cfg.include_headers, include_footers=cfg.include_footers)
    if max_segments is not None:
        if max_segments < 0:
            raise ValueError(f"max_segments must be >= 0, got {max_segments}")
        segments = segments[:max_segments]
        logger.info(f"Segment limit enabled: {len(segments)} segments")
    logger.info(f"Segments найдено: {len(segments)}")

    # Attach neighbor snippets to improve local consistency.
    _attach_neighbor_snippets(segments, cfg)

    tm = TMStore(cfg.tm.path)
    if cfg.tm.fuzzy_enabled and not tm.fts_enabled:
        logger.info("Fuzzy TM requested, but SQLite FTS5 is unavailable; continuing with exact-only TM behavior.")
    custom_system_prompt = _read_optional_text(cfg.llm.system_prompt_path, logger, "custom system prompt")
    glossary_text = _read_optional_text(cfg.llm.glossary_path, logger, "glossary")
    if cfg.glossary_lemma_check != "off" and not is_glossary_lemma_check_available():
        logger.info(
            "glossary_lemma_check=%s requested, but pymorphy3 is unavailable; check is skipped.",
            cfg.glossary_lemma_check,
        )
    if cfg.llm.provider.strip().lower() == "google" and custom_system_prompt:
        logger.info("Provider 'google' does not support system prompts; custom prompt is ignored for this run.")
    glossary_terms: tuple[tuple[re.Pattern[str], str], ...] = ()
    glossary_matchers = build_glossary_matchers(glossary_text) if glossary_text else ()
    provider_norm = cfg.llm.provider.strip().lower()
    glossary_in_prompt = bool(cfg.llm.glossary_in_prompt)
    glossary_prompt_mode = cfg.llm.glossary_prompt_mode.strip().lower()
    hard_glossary = bool(cfg.llm.hard_glossary)
    if glossary_text and hard_glossary:
        glossary_terms = build_hard_glossary_replacements(glossary_text)
        logger.info(
            f"Hard glossary enforcement enabled ({len(glossary_terms)} terms); provider={cfg.llm.provider}; "
            f"glossary_in_prompt={glossary_in_prompt}; hard_glossary={hard_glossary}; "
            "scope=adaptive(toc/table/short-labels)"
        )
    if glossary_prompt_mode == "off" or not glossary_in_prompt:
        effective_glossary_text = None
    elif glossary_prompt_mode == "matched":
        # Segment-level term selection: only matched glossary entries are injected into prompts.
        effective_glossary_text = None
        if glossary_text:
            logger.info("Glossary prompt mode 'matched' enabled: prompts will include only matched terms.")
    else:
        effective_glossary_text = glossary_text
    if glossary_text and effective_glossary_text is None:
        if glossary_prompt_mode == "matched" and glossary_in_prompt:
            logger.info(
                "Global glossary prompt injection is disabled; matched segment-level glossary hints are enabled."
            )
        else:
            logger.info(
                "Glossary prompt injection is disabled; only post-translation glossary replacements are active "
                "(unless hard_glossary=true)."
            )
    if glossary_text and provider_norm == "google" and not hard_glossary:
        logger.info(
            "Provider 'google' is running without hard glossary locking; terminology can be less stable "
            "but wording is usually more natural."
        )
    tm_profile_key = _build_tm_profile_key(
        cfg,
        custom_system_prompt=custom_system_prompt,
        glossary_text=glossary_text,
    )
    logger.info(f"TM profile key: {tm_profile_key}")
    llm_client = build_llm_client(
        provider=cfg.llm.provider,
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        timeout_s=cfg.llm.timeout_s,
        max_output_tokens=cfg.llm.max_output_tokens,
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
    )

    # Stage 1: tagging + shielding + TM lookup
    to_translate: list[tuple[Segment, str, str]] = []  # (seg, source_hash, source_norm)
    tm_hits = 0
    resume_hits = 0
    tagging_errors = 0
    complex_translated = 0
    complex_chunk_cache: dict[str, str] = {}
    segment_source_hash: dict[str, str] = {}
    tm_hit_segments: set[str] = set()
    llm_translated_segments: set[str] = set()
    matched_glossary_segments = 0
    fuzzy_reference_segments = 0
    toc_inplace_translated = 0
    document_glossary: dict[str, str] = {}
    progress_cache: dict[str, dict[str, Any]] = {}
    if resume and segments:
        try:
            progress_cache = tm.get_progress_bulk([seg.segment_id for seg in segments])
            logger.info(f"Resume progress cache loaded: {len(progress_cache)} records")
        except Exception as e:
            logger.warning(f"Resume progress bulk load failed; fallback to per-segment lookup: {e}")

    for seg in tqdm(segments, desc="Prepare", unit="seg"):
        prev_progress = progress_cache.get(seg.segment_id) if resume else None

        # Fast-path: if the segment contains no English (Latin) text, do not touch the paragraph at all.
        # This preserves layout and avoids degrading already-RU content in partially translated manuals.
        if not _should_translate_segment_text(seg.source_plain):
            seg.issues.append(
                Issue(
                    code="skip_no_latin",
                    severity=Severity.INFO,
                    message="Сегмент пропущен: нет английского текста (латиницы) или уже в RU — оставлено как в исходнике",
                    details={},
                )
            )
            continue

        if glossary_prompt_mode == "matched" and glossary_in_prompt and document_glossary:
            seg.context["document_glossary"] = _build_document_glossary_context(
                document_glossary,
                limit=cfg.llm.glossary_match_limit,
            )

        if glossary_prompt_mode == "matched" and glossary_matchers:
            matched_terms = _build_matched_glossary_context(
                seg.source_plain,
                glossary_matchers,
                limit=cfg.llm.glossary_match_limit,
            )
            if matched_terms:
                seg.context["matched_glossary_terms"] = matched_terms
                matched_glossary_segments += 1
                for pair in matched_terms:
                    source_term = str(pair.get("source") or "").strip()
                    target_term = str(pair.get("target") or "").strip()
                    if source_term and target_term:
                        if source_term in document_glossary:
                            document_glossary.pop(source_term, None)
                        document_glossary[source_term] = target_term

        seg_glossary_terms = _segment_glossary_terms(seg, glossary_terms)

        # Dedicated TOC flow: preserve tab/page layout and translate column chunks in-place.
        if seg.context.get("is_toc_entry"):
            changed, toc_issues = _translate_complex_paragraph_in_place(
                seg,
                cfg,
                llm_client,
                complex_chunk_cache,
                seg_glossary_terms,
            )
            seg.issues.extend(toc_issues)
            if changed:
                toc_inplace_translated += 1
            continue

        # Safety gate: skip paragraphs that contain complex inline XML (hyperlinks, content controls, etc.)
        # to avoid reordering/corruption when rebuilding runs.
        if not is_supported_paragraph(seg.paragraph_ref):
            changed, complex_issues = _translate_complex_paragraph_in_place(
                seg,
                cfg,
                llm_client,
                complex_chunk_cache,
                seg_glossary_terms,
            )
            seg.issues.extend(complex_issues)
            if changed:
                complex_translated += 1
            else:
                seg.issues.append(
                    Issue(
                        code="skip_complex_paragraph",
                        severity=Severity.INFO,
                        message="Сегмент пропущен: сложная структура абзаца (не только runs) — оставлено как в исходнике",
                        details={},
                    )
                )
            continue
        try:
            tagged, spans, inline_map = paragraph_to_tagged(seg.paragraph_ref)
        except Exception as e:
            seg.issues.append(
                Issue(
                    code="tagging_error",
                    severity=Severity.ERROR,
                    message=f"Tagging ошибка: {e}",
                    details={},
                )
            )
            tagging_errors += 1
            continue

        seg.source_tagged = tagged
        seg.spans = spans
        seg.inline_run_map = inline_map

        # Compute batch-eligibility view before hard glossary shielding so BRLINE decisions
        # are based on original OCR line-break markers.
        batch_eligibility_text, _ = shield(tagged, cfg.pattern_set)
        seg.context["_batch_eligibility_text"] = batch_eligibility_text

        shielded_text = tagged
        token_map: dict[str, str] = {}
        # Apply hard glossary before generic shielding so long phrases with dimensions
        # are not broken by DIM placeholders before glossary matching.
        if seg_glossary_terms and _LATIN_RE.search(shielded_text):
            shielded_text, glossary_map = shield_terms(
                shielded_text,
                seg_glossary_terms,
                token_prefix="GLS",
                bridge_break_tokens=False,
            )
            if glossary_map:
                token_map = {**token_map, **glossary_map}
        shielded_text, pattern_map = shield(shielded_text, cfg.pattern_set)
        if pattern_map:
            token_map = {**pattern_map, **token_map}
        seg.shielded_tagged = shielded_text
        seg.token_map = token_map

        source_norm = normalize_text(shielded_text)
        source_norm_for_hash = f"{_TM_RULESET_VERSION}\n{tm_profile_key}\n{source_norm}"
        source_hash = sha256_hex(source_norm_for_hash)
        segment_source_hash[seg.segment_id] = source_hash

        if (
            resume
            and prev_progress is not None
            and prev_progress.get("status") in {"ok", "tm"}
            and prev_progress.get("source_hash") == source_hash
        ):
            hit = tm.get_exact(source_hash)
            if hit is not None:
                seg.target_shielded_tagged = hit.target_text
                tm_hits += 1
                resume_hits += 1
                tm_hit_segments.add(seg.segment_id)
                continue

        hit = tm.get_exact(source_hash)
        if hit is not None:
            seg.target_shielded_tagged = hit.target_text
            tm_hits += 1
            tm_hit_segments.add(seg.segment_id)
            tm.set_progress(seg.segment_id, "tm", source_hash=source_hash)
        else:
            if cfg.tm.fuzzy_enabled:
                fuzzy_hits = tm.get_fuzzy(
                    source_norm,
                    top_k=cfg.tm.fuzzy_top_k,
                    min_similarity=cfg.tm.fuzzy_min_similarity,
                )
                tm_refs = _build_tm_references_context(
                    fuzzy_hits,
                    max_chars=cfg.tm.fuzzy_prompt_max_chars,
                )
                if tm_refs:
                    seg.context["tm_references"] = tm_refs
                    seg.context["tm_references_max_chars"] = int(cfg.tm.fuzzy_prompt_max_chars)
                    fuzzy_reference_segments += 1
            to_translate.append((seg, source_hash, source_norm))

    logger.info(f"TM hits: {tm_hits}")
    if resume:
        logger.info(f"Resume hits: {resume_hits}")
    logger.info(f"Tagging errors: {tagging_errors}")
    logger.info(f"Complex paragraphs translated in-place: {complex_translated}")
    logger.info(f"TOC segments translated in-place: {toc_inplace_translated}")
    logger.info(f"Segments with matched glossary prompt hints: {matched_glossary_segments}")
    logger.info(f"Segments with fuzzy TM prompt hints: {fuzzy_reference_segments}")
    logger.info(f"LLM segments: {len(to_translate)}")

    # Stage 2: LLM translate (single-segment or grouped batches)
    if to_translate:
        provider_norm = cfg.llm.provider.strip().lower()
        context_window_chars = max(0, int(cfg.llm.context_window_chars))
        context_window_enabled = context_window_chars > 0
        if context_window_enabled:
            logger.info(
                "Context window is enabled (context_window_chars=%d): grouped batches disabled; "
                "running sequential translation with recent translations context.",
                context_window_chars,
            )
            recent_translations: deque[tuple[str, str]] = deque(maxlen=3)
            for seg, source_hash, source_norm in tqdm(to_translate, desc="Translate (sequential)", unit="seg"):
                if recent_translations:
                    seg.context["recent_translations"] = [
                        {"source": source, "target": target}
                        for source, target in recent_translations
                    ]
                else:
                    seg.context.pop("recent_translations", None)
                seg.context["recent_translations_max_chars"] = context_window_chars
                try:
                    out, issues = _translate_one(seg, cfg, llm_client, source_hash, source_norm, logger)
                except Exception as e:
                    seg.issues.append(
                        Issue(
                            code="translate_crash",
                            severity=Severity.ERROR,
                            message=f"Translate crash: {e}",
                            details={},
                        )
                    )
                    tm.set_progress(seg.segment_id, "error", source_hash=source_hash, error=str(e))
                    continue
                complex_translated += _finalize_translation_result(
                    seg=seg,
                    source_hash=source_hash,
                    source_norm=source_norm,
                    out=out,
                    issues=issues,
                    cfg=cfg,
                    llm_client=llm_client,
                    tm=tm,
                    complex_chunk_cache=complex_chunk_cache,
                    glossary_terms=_segment_glossary_terms(seg, glossary_terms),
                    llm_translated_segments=llm_translated_segments,
                    recent_translations=recent_translations,
                )
        else:
            grouped_mode = cfg.llm.batch_segments > 1 and provider_norm in _BATCH_PROVIDER_ALLOWLIST
            if cfg.llm.batch_segments > 1 and not grouped_mode:
                logger.info(
                    f"Grouped batch mode requested, but provider '{cfg.llm.provider}' is not supported; using per-segment mode."
                )

            if grouped_mode:
                batch_eligible: list[tuple[Segment, str, str]] = []
                single_only: list[tuple[tuple[Segment, str, str], list[str]]] = []
                for job in to_translate:
                    reasons = _batch_ineligibility_reasons(job[0], cfg)
                    if reasons:
                        single_only.append((job, reasons))
                    else:
                        batch_eligible.append(job)

                logger.info(
                    f"Batch-eligible segments: {len(batch_eligible)}; "
                    f"forced single by eligibility filter: {len(single_only)}"
                )
                if single_only:
                    reason_counts = Counter(reason for _, reasons in single_only for reason in reasons)
                    logger.info(
                        "Batch ineligibility reasons: "
                        + ", ".join(f"{reason}={count}" for reason, count in reason_counts.most_common())
                    )

                if batch_eligible:
                    grouped_jobs = _chunk_translation_jobs(
                        batch_eligible,
                        max_segments=cfg.llm.batch_segments,
                        max_chars=cfg.llm.batch_max_chars,
                    )
                    logger.info(f"Grouped batches prepared: {len(grouped_jobs)}")
                else:
                    grouped_jobs = []
                    logger.info("No grouped batches prepared after eligibility filtering.")

                if grouped_jobs:
                    batch_workers = min(max(1, cfg.concurrency), len(grouped_jobs))
                    logger.info(f"Grouped batch workers: {batch_workers}")
                    with ThreadPoolExecutor(max_workers=batch_workers) as ex:
                        futures = {
                            ex.submit(_translate_batch_group, jobs, cfg, llm_client, logger): jobs
                            for jobs in grouped_jobs
                        }
                        for fut in tqdm(as_completed(futures), total=len(futures), desc="Translate (batch)", unit="batch"):
                            jobs = futures[fut]
                            try:
                                batch_results = fut.result()
                            except Exception as e:
                                logger.warning(
                                    f"Batch worker crashed (size={len(jobs)}), fallback to single-segment: {e}"
                                )
                                for seg, source_hash, source_norm in jobs:
                                    try:
                                        out, issues = _translate_one(seg, cfg, llm_client, source_hash, source_norm, logger)
                                    except Exception as single_err:
                                        seg.issues.append(
                                            Issue(
                                                code="translate_crash",
                                                severity=Severity.ERROR,
                                                message=f"Translate crash: {single_err}",
                                                details={},
                                            )
                                        )
                                        tm.set_progress(seg.segment_id, "error", source_hash=source_hash, error=str(single_err))
                                        continue
                                    issues = [
                                        Issue(
                                            code="batch_worker_crash_fallback",
                                            severity=Severity.WARN,
                                            message=f"Batch worker crashed, fallback to single segment translation: {e}",
                                            details={"batch_size": len(jobs)},
                                        ),
                                        *issues,
                                    ]
                                    complex_translated += _finalize_translation_result(
                                        seg=seg,
                                        source_hash=source_hash,
                                        source_norm=source_norm,
                                        out=out,
                                        issues=issues,
                                        cfg=cfg,
                                        llm_client=llm_client,
                                        tm=tm,
                                        complex_chunk_cache=complex_chunk_cache,
                                        glossary_terms=_segment_glossary_terms(seg, glossary_terms),
                                        llm_translated_segments=llm_translated_segments,
                                    )
                                continue

                            for seg, source_hash, source_norm, out, issues in batch_results:
                                complex_translated += _finalize_translation_result(
                                    seg=seg,
                                    source_hash=source_hash,
                                    source_norm=source_norm,
                                    out=out,
                                    issues=issues,
                                    cfg=cfg,
                                    llm_client=llm_client,
                                    tm=tm,
                                    complex_chunk_cache=complex_chunk_cache,
                                    glossary_terms=_segment_glossary_terms(seg, glossary_terms),
                                    llm_translated_segments=llm_translated_segments,
                                )

                if single_only:
                    single_workers = min(max(1, cfg.concurrency), len(single_only))
                    logger.info(f"Single-filter workers: {single_workers}")
                    with ThreadPoolExecutor(max_workers=single_workers) as ex:
                        futures = {
                            ex.submit(_translate_one, seg, cfg, llm_client, source_hash, source_norm, logger): (
                                seg,
                                source_hash,
                                source_norm,
                                reasons,
                            )
                            for (seg, source_hash, source_norm), reasons in single_only
                        }
                        for fut in tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc="Translate (single-filter)",
                            unit="seg",
                        ):
                            seg, source_hash, source_norm, reasons = futures[fut]
                            try:
                                out, issues = fut.result()
                            except Exception as e:
                                seg.issues.append(
                                    Issue(
                                        code="translate_crash",
                                        severity=Severity.ERROR,
                                        message=f"Translate crash: {e}",
                                        details={},
                                    )
                                )
                                tm.set_progress(seg.segment_id, "error", source_hash=source_hash, error=str(e))
                                continue

                            issues = [
                                Issue(
                                    code="batch_ineligible_single",
                                    severity=Severity.INFO,
                                    message=(
                                        "Segment excluded from grouped batch by eligibility filter; translated via single-pass mode."
                                    ),
                                    details={"reasons": reasons},
                                ),
                                *issues,
                            ]
                            complex_translated += _finalize_translation_result(
                                seg=seg,
                                source_hash=source_hash,
                                source_norm=source_norm,
                                out=out,
                                issues=issues,
                                cfg=cfg,
                                llm_client=llm_client,
                                tm=tm,
                                complex_chunk_cache=complex_chunk_cache,
                                glossary_terms=_segment_glossary_terms(seg, glossary_terms),
                                llm_translated_segments=llm_translated_segments,
                            )
            else:
                with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as ex:
                    futures = {
                        ex.submit(_translate_one, seg, cfg, llm_client, sh, sn, logger): (seg, sh, sn)
                        for (seg, sh, sn) in to_translate
                    }
                    for fut in tqdm(as_completed(futures), total=len(futures), desc="Translate", unit="seg"):
                        seg, source_hash, source_norm = futures[fut]
                        try:
                            out, issues = fut.result()
                        except Exception as e:
                            seg.issues.append(
                                Issue(
                                    code="translate_crash",
                                    severity=Severity.ERROR,
                                    message=f"Translate crash: {e}",
                                    details={},
                                )
                            )
                            tm.set_progress(seg.segment_id, "error", source_hash=source_hash, error=str(e))
                            continue

                        complex_translated += _finalize_translation_result(
                            seg=seg,
                            source_hash=source_hash,
                            source_norm=source_norm,
                            out=out,
                            issues=issues,
                            cfg=cfg,
                            llm_client=llm_client,
                            tm=tm,
                            complex_chunk_cache=complex_chunk_cache,
                            glossary_terms=_segment_glossary_terms(seg, glossary_terms),
                            llm_translated_segments=llm_translated_segments,
                        )

    # Stage 3: Unshield + write back to DOCX
    written = 0
    for seg in tqdm(segments, desc="Write", unit="seg"):
        if not seg.spans or seg.shielded_tagged is None:
            continue

        if seg.target_shielded_tagged is None:
            # No translation (tagging error etc) — keep source
            continue

        if any(i.severity == Severity.ERROR for i in seg.issues):
            seg.issues.append(
                Issue(
                    code="writeback_skipped_due_to_errors",
                    severity=Severity.INFO,
                    message="Write-back skipped due to hard validation/translation errors; source text preserved.",
                    details={},
                )
            )
            continue

        # Unshield placeholders back to original protected strings
        target_tagged_unshielded = unshield(seg.target_shielded_tagged, seg.token_map or {})
        seg.target_tagged = target_tagged_unshielded

        try:
            tagged_to_runs(
                seg.paragraph_ref,
                target_tagged_unshielded,
                seg.spans,
                inline_run_map=seg.inline_run_map,
            )
            written += 1
        except Exception as e:
            seg.issues.append(
                Issue(
                    code="write_error",
                    severity=Severity.ERROR,
                    message=f"Write-back ошибка: {e}",
                    details={},
                )
            )

    logger.info(f"Written segments: {written}/{len(segments)}; complex in-place: {complex_translated}")

    if glossary_matchers:
        consistency_issues = report_consistency(segments, glossary_matchers)
        attached = _attach_issues_to_segments(segments, consistency_issues)
        if consistency_issues:
            logger.info(f"Consistency check issues: {len(consistency_issues)} (attached={attached})")

    if cfg.layout_check or cfg.layout_auto_fix:
        layout_issues = validate_layout(doc, segments, cfg)
        attached = _attach_issues_to_segments(segments, layout_issues)
        if layout_issues:
            logger.info(f"Layout validation issues: {len(layout_issues)} (attached={attached})")
        if cfg.layout_auto_fix and layout_issues:
            applied_fixes = fix_expansion_issues(segments, layout_issues, cfg)
            logger.info(f"Layout auto-fixes applied: {applied_fixes}")

    cleaned_runs = _apply_final_run_level_cleanup(segments)
    logger.info(f"Final run-level cleanup changes: {cleaned_runs}")
    issue_counts = _collect_issue_code_counts(segments)
    for code in (
        "placeholders_mismatch",
        "style_tags_mismatch",
        "batch_fallback_single",
        "batch_validation_fallback",
    ):
        logger.info(f"Metric {code}: {issue_counts.get(code, 0)}")
    if issue_counts:
        logger.info(
            "Issue codes summary (top 12): "
            + ", ".join(f"{code}={count}" for code, count in issue_counts.most_common(12))
        )

    if cfg.abbyy_profile != "off":
        try:
            oxml_stats = normalize_abbyy_oxml(doc, profile=cfg.abbyy_profile)
            logger.info(
                "ABBYY OXML normalization (%s): trHeight_exact_removed=%d; framePr_removed=%d; "
                "lineSpacing_exact_relaxed=%d",
                cfg.abbyy_profile,
                int(oxml_stats.get("tr_height_exact_removed", 0)),
                int(oxml_stats.get("frame_pr_removed", 0)),
                int(oxml_stats.get("line_spacing_exact_relaxed", 0)),
            )
        except Exception as e:
            logger.warning(f"ABBYY OXML normalization failed: {e}")

    # Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    logger.info("DOCX сохранён")

    if cfg.mode.lower() == "com":
        try:
            from .com_word import update_fields_via_com

            com_stats = update_fields_via_com(output_path)
            logger.info(
                "Word COM post-process: fields_updated=%d; tocs_updated=%d; "
                "textboxes_seen=%d; textboxes_autofit=%d; textboxes_shrunk=%d",
                int(com_stats.get("fields_updated", 0)),
                int(com_stats.get("tocs_updated", 0)),
                int(com_stats.get("textboxes_seen", 0)),
                int(com_stats.get("textboxes_autofit", 0)),
                int(com_stats.get("textboxes_shrunk", 0)),
            )
        except Exception as e:
            logger.warning(f"Word COM post-process failed: {e}")

    # QA outputs
    qa_html = Path(cfg.qa_report_path)
    qa_jsonl = Path(cfg.qa_jsonl_path)
    write_qa_report(segments, qa_html)
    write_qa_jsonl(segments, qa_jsonl)
    logger.info(f"QA report: {qa_html}")
    logger.info(f"QA jsonl: {qa_jsonl}")

    if cfg.translation_history_path:
        history_path = Path(cfg.translation_history_path)
        history_records: list[dict[str, Any]] = []
        for seg in segments:
            if seg.target_shielded_tagged is None or seg.shielded_tagged is None:
                continue
            if any(i.severity == Severity.ERROR for i in seg.issues):
                continue
            source_hash = segment_source_hash.get(seg.segment_id)
            if seg.segment_id in llm_translated_segments:
                origin = "llm"
            elif seg.segment_id in tm_hit_segments:
                origin = "tm"
            else:
                origin = "reuse"
            history_records.append(
                _build_history_record(seg, cfg, source_hash=source_hash, origin=origin)
            )
        _append_history_jsonl(history_path, history_records)
        logger.info(f"Translation history appended: {history_path} ({len(history_records)} records)")

    tm.close()
