from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from docx import Document
from docx.oxml.ns import qn
from tqdm import tqdm

from .config import PipelineConfig
from .docx_reader import collect_segments
from .llm import build_hard_glossary_replacements, build_llm_client, supports_repair
from .logging_utils import setup_logging
from .models import Issue, Segment, Severity
from .qa_report import write_qa_jsonl, write_qa_report
from .tagging import is_supported_paragraph, paragraph_to_tagged, tagged_to_runs
from .tm import TMStore, normalize_text, sha256_hex
from .token_shield import BRACKET_TOKEN_RE, shield, shield_terms, strip_bracket_tokens, unshield
from .validator import validate_all, validate_numbers, validate_placeholders


def _build_repair_payload(source_shielded: str, bad_output: str) -> str:
    return (
        "TASK: REPAIR_MARKERS\n\n"
        f"SOURCE:\n{source_shielded}\n\n"
        f"OUTPUT:\n{bad_output}"
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
_LATIN_RE = re.compile(r"[A-Za-z]")
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_LAYOUT_SPLIT_RE = re.compile(r"((?:\.\s*){3,}|\t+)")
_FINAL_CLEANUP_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # Remove zero-width spaces sometimes produced by machine translation (Google).
    (re.compile("\u200b"), ""),
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
        re.compile(r"\bС\s+ИЛЛЮСТРИРОВАННЫЙ\s+ПЕРЕЧЕНЬ\s+ДЕТАЛЕЙ\b", flags=re.IGNORECASE),
        "С ИЛЛЮСТРИРОВАННЫМ ПЕРЕЧНЕМ ДЕТАЛЕЙ",
    ),
)
_W_T_TAG = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
_W_HYPERLINK_TAG = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}hyperlink"
_W_RUN_TAG = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r"
_TM_RULESET_VERSION = "2026-02-21-consistency-v1"


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
        shielded, glossary_map = shield_terms(shielded, glossary_terms, token_prefix="GLS")
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
    if cyr >= latin * 3 and latin <= 12:
        return False
    return True


def _apply_final_run_level_cleanup(segments: list[Segment]) -> int:
    changed_runs = 0
    seen_paragraphs: set[int] = set()
    for seg in segments:
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

        hard_errors = [i for i in issues if i.severity == Severity.ERROR]
        if not hard_errors:
            return out, issues

        if not can_repair:
            fallback_out, fallback_issues = _fallback_translate_by_spans(seg, cfg, llm_client)
            if fallback_out is not None:
                return fallback_out, fallback_issues

        # If we still have marker errors and there is room to retry, go to repair attempt.
        if attempt < max_attempts:
            continue
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
        f"Mode: {cfg.mode}; concurrency={cfg.concurrency}; headers={cfg.include_headers}; footers={cfg.include_footers}"
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

    tm = TMStore(cfg.tm.path)
    custom_system_prompt = _read_optional_text(cfg.llm.system_prompt_path, logger, "custom system prompt")
    glossary_text = _read_optional_text(cfg.llm.glossary_path, logger, "glossary")
    if cfg.llm.provider.strip().lower() == "google" and custom_system_prompt:
        logger.info("Provider 'google' does not support system prompts; custom prompt is ignored for this run.")
    glossary_terms: tuple[tuple[re.Pattern[str], str], ...] = ()
    if cfg.llm.provider.strip().lower() == "google" and glossary_text:
        glossary_terms = build_hard_glossary_replacements(glossary_text)
        logger.info(f"Hard glossary enforcement enabled for google provider: {len(glossary_terms)} terms")
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
    )

    # Stage 1: tagging + shielding + TM lookup
    to_translate: list[tuple[Segment, str, str]] = []  # (seg, source_hash, source_norm)
    tm_hits = 0
    resume_hits = 0
    tagging_errors = 0
    complex_translated = 0
    complex_chunk_cache: dict[str, str] = {}

    for seg in tqdm(segments, desc="Prepare", unit="seg"):
        prev_progress = tm.get_progress(seg.segment_id) if resume else None

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
            tm.set_progress(seg.segment_id, "skip", source_hash=None)
            continue

        # Safety gate: skip paragraphs that contain complex inline XML (hyperlinks, content controls, etc.)
        # to avoid reordering/corruption when rebuilding runs.
        if not is_supported_paragraph(seg.paragraph_ref):
            changed, complex_issues = _translate_complex_paragraph_in_place(
                seg,
                cfg,
                llm_client,
                complex_chunk_cache,
                glossary_terms,
            )
            seg.issues.extend(complex_issues)
            if changed:
                complex_translated += 1
                tm.set_progress(seg.segment_id, "complex_ok", source_hash=None)
            else:
                seg.issues.append(
                    Issue(
                        code="skip_complex_paragraph",
                        severity=Severity.INFO,
                        message="Сегмент пропущен: сложная структура абзаца (не только runs) — оставлено как в исходнике",
                        details={},
                    )
                )
                tm.set_progress(seg.segment_id, "skip", source_hash=None)
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

        shielded_text = tagged
        token_map: dict[str, str] = {}
        # Apply hard glossary before generic shielding so long phrases with dimensions
        # are not broken by DIM placeholders before glossary matching.
        if glossary_terms and _LATIN_RE.search(shielded_text):
            shielded_text, glossary_map = shield_terms(shielded_text, glossary_terms, token_prefix="GLS")
            if glossary_map:
                token_map = {**token_map, **glossary_map}
        shielded_text, pattern_map = shield(shielded_text, cfg.pattern_set)
        if pattern_map:
            token_map = {**pattern_map, **token_map}
        seg.shielded_tagged = shielded_text
        seg.token_map = token_map

        source_norm = normalize_text(shielded_text)
        source_norm_for_hash = f"{_TM_RULESET_VERSION}\n{source_norm}"
        source_hash = sha256_hex(source_norm_for_hash)

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
                continue

        hit = tm.get_exact(source_hash)
        if hit is not None:
            seg.target_shielded_tagged = hit.target_text
            tm_hits += 1
            tm.set_progress(seg.segment_id, "tm", source_hash=source_hash)
        else:
            to_translate.append((seg, source_hash, source_norm))

    logger.info(f"TM hits: {tm_hits}")
    if resume:
        logger.info(f"Resume hits: {resume_hits}")
    logger.info(f"Tagging errors: {tagging_errors}")
    logger.info(f"Complex paragraphs translated in-place: {complex_translated}")
    logger.info(f"LLM segments: {len(to_translate)}")

    # Stage 2: LLM translate concurrently
    if to_translate:
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

                seg.target_shielded_tagged = out
                seg.issues.extend(issues)

                # Store to TM only if no hard errors (avoid caching broken markers).
                if not any(i.severity == Severity.ERROR for i in issues):
                    tm.put_exact(
                        source_hash=source_hash,
                        source_norm=source_norm,
                        target_text=out,
                        meta={"provider": cfg.llm.provider, "model": cfg.llm.model},
                    )
                    tm.set_progress(seg.segment_id, "ok", source_hash=source_hash)
                else:
                    tm.set_progress(
                        seg.segment_id,
                        "error",
                        source_hash=source_hash,
                        error="; ".join(i.code for i in issues),
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

    cleaned_runs = _apply_final_run_level_cleanup(segments)
    logger.info(f"Final run-level cleanup changes: {cleaned_runs}")

    # Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    logger.info("DOCX сохранён")

    # QA outputs
    qa_html = Path(cfg.qa_report_path)
    qa_jsonl = Path(cfg.qa_jsonl_path)
    write_qa_report(segments, qa_html)
    write_qa_jsonl(segments, qa_jsonl)
    logger.info(f"QA report: {qa_html}")
    logger.info(f"QA jsonl: {qa_jsonl}")

    # Optional COM mode is not invoked automatically here (platform specific).
    if cfg.mode.lower() == "com":
        logger.warning(
            "Mode 'com' выбран: обновление полей Word через COM нужно запускать отдельно (см. src/docxru/com_word.py)."
        )

    tm.close()
