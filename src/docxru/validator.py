from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .models import Issue, Severity
from .token_shield import BRACKET_TOKEN_RE, extract_numbers

_STYLE_TOKEN_BODY_RE = re.compile(r"^/?S_\d+(?:\|[^|]+)*$")
_PLACEHOLDER_BODY_RE = re.compile(r"^[A-Z][A-Z0-9]*_\d+$")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")

_PymorphyMorphAnalyzer: Any | None
try:
    from pymorphy3 import MorphAnalyzer as _PymorphyMorphAnalyzer
except Exception:
    _PymorphyMorphAnalyzer = None

_MORPH_ANALYZER: Any | None = None
_MORPH_INIT_FAILED = False


def _get_morph_analyzer() -> Any | None:
    global _MORPH_ANALYZER, _MORPH_INIT_FAILED
    if _MORPH_INIT_FAILED or _PymorphyMorphAnalyzer is None:
        return None
    if _MORPH_ANALYZER is not None:
        return _MORPH_ANALYZER
    try:
        _MORPH_ANALYZER = _PymorphyMorphAnalyzer()
    except Exception:
        _MORPH_INIT_FAILED = True
        return None
    return _MORPH_ANALYZER


def is_glossary_lemma_check_available() -> bool:
    return _get_morph_analyzer() is not None


def _coerce_glossary_pairs(value: Any) -> list[tuple[str, str]]:
    if value is None:
        return []
    items = list(value) if isinstance(value, (list, tuple)) else [value]

    pairs: list[tuple[str, str]] = []
    for item in items:
        source: Any = None
        target: Any = None
        if isinstance(item, dict):
            source = item.get("source")
            target = item.get("target")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            source, target = item[0], item[1]
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        source_text = source.strip()
        target_text = target.strip()
        if not source_text or not target_text:
            continue
        pairs.append((source_text, target_text))
    return pairs


def _extract_bracket_tokens(text: str) -> list[str]:
    return BRACKET_TOKEN_RE.findall(text or "")


def _token_body(token: str) -> str:
    if len(token) >= 2:
        return token[1:-1]
    return ""


def extract_style_tokens(text: str) -> list[str]:
    out: list[str] = []
    for token in _extract_bracket_tokens(text):
        if _STYLE_TOKEN_BODY_RE.fullmatch(_token_body(token)):
            out.append(token)
    return out


def extract_placeholders(text: str) -> list[str]:
    out: list[str] = []
    for token in _extract_bracket_tokens(text):
        body = _token_body(token)
        if body.startswith("S_") or body.startswith("/S_"):
            continue
        if _PLACEHOLDER_BODY_RE.fullmatch(body):
            out.append(token)
    return out


def validate_placeholders(source: str, target: str) -> list[Issue]:
    src = extract_placeholders(source)
    tgt = extract_placeholders(target)

    src_c = Counter(src)
    tgt_c = Counter(tgt)
    if src_c != tgt_c:
        missing = list((src_c - tgt_c).elements())
        extra = list((tgt_c - src_c).elements())
        return [
            Issue(
                code="placeholders_mismatch",
                severity=Severity.ERROR,
                message="Placeholder tokens were lost/changed/added.",
                details={"missing": missing, "extra": extra},
            )
        ]
    return []


def validate_style_tokens(source: str, target: str) -> list[Issue]:
    src = extract_style_tokens(source)
    tgt = extract_style_tokens(target)
    if src != tgt:
        src_set = set(src)
        tgt_set = set(tgt)
        missing = [t for t in src if t not in tgt_set]
        extra = [t for t in tgt if t not in src_set]
        return [
            Issue(
                code="style_tags_mismatch",
                severity=Severity.ERROR,
                message="Style marker tokens were changed/reordered/removed.",
                details={"missing": missing, "extra": extra, "src_len": len(src), "tgt_len": len(tgt)},
            )
        ]
    return []


def validate_numbers(source_unshielded: str, target_unshielded: str) -> list[Issue]:
    src = extract_numbers(source_unshielded)
    tgt = extract_numbers(target_unshielded)
    if src != tgt:
        return [
            Issue(
                code="numbers_mismatch",
                severity=Severity.WARN,
                message="Numbers/tolerances differ; manual check is recommended.",
                details={"source_numbers": src, "target_numbers": tgt},
            )
        ]
    return []


def validate_length(source_plain: str, target_plain: str, factor_warn: float = 3.0) -> list[Issue]:
    s = len(source_plain.strip())
    t = len(target_plain.strip())
    if s == 0:
        return []
    if t == 0:
        return [
            Issue(
                code="empty_translation",
                severity=Severity.ERROR,
                message="Empty translation output.",
                details={"source_len": s, "target_len": t},
            )
        ]
    ratio = t / max(1, s)
    if ratio > factor_warn:
        return [
            Issue(
                code="length_ratio_high",
                severity=Severity.WARN,
                message="Translation is unexpectedly long.",
                details={"ratio": ratio, "source_len": s, "target_len": t},
            )
        ]
    return []


def _lemmatize_words(text: str, analyzer: Any) -> set[str]:
    lemmas: set[str] = set()
    for token in _WORD_RE.findall(text or ""):
        word = token.strip().lower()
        if not word:
            continue
        try:
            parsed = analyzer.parse(word)
            if parsed:
                normal_form = str(getattr(parsed[0], "normal_form", word)).strip().lower()
                lemmas.add(normal_form or word)
                continue
        except Exception:
            pass
        lemmas.add(word)
    return lemmas


def _contains_target_term_by_lemma(target_text: str, target_term: str, analyzer: Any) -> bool:
    if not target_term:
        return True
    if target_term.lower() in target_text.lower():
        return True
    target_lemmas = _lemmatize_words(target_text, analyzer)
    if not target_lemmas:
        return False
    term_lemmas = _lemmatize_words(target_term, analyzer)
    if not term_lemmas:
        return True
    return term_lemmas.issubset(target_lemmas)


def validate_glossary_lemmas(
    target_plain: str,
    matched_glossary_terms: Any,
    *,
    mode: str = "off",
) -> list[Issue]:
    check_mode = str(mode or "off").strip().lower()
    if check_mode not in {"off", "warn", "retry"}:
        check_mode = "off"
    if check_mode == "off":
        return []

    pairs = _coerce_glossary_pairs(matched_glossary_terms)
    if not pairs:
        return []

    analyzer = _get_morph_analyzer()
    missing: list[dict[str, str]] = []
    target_lower = (target_plain or "").lower()
    if analyzer is None:
        # Fallback when pymorphy3 is unavailable: exact case-insensitive inclusion.
        for source_term, target_term in pairs:
            if target_term.lower() in target_lower:
                continue
            missing.append({"source": source_term, "target": target_term})
    else:
        for source_term, target_term in pairs:
            if _contains_target_term_by_lemma(target_plain, target_term, analyzer):
                continue
            missing.append({"source": source_term, "target": target_term})

    if not missing:
        return []

    return [
        Issue(
            code="glossary_lemma_mismatch",
            severity=Severity.WARN,
            message="Matched glossary terms were not detected in translation output.",
            details={"missing": missing, "mode": check_mode},
        )
    ]


def validate_all(
    source_shielded_tagged: str,
    target_shielded_tagged: str,
    source_unshielded_plain: str,
    target_unshielded_plain: str,
) -> list[Issue]:
    issues: list[Issue] = []
    issues.extend(validate_placeholders(source_shielded_tagged, target_shielded_tagged))
    issues.extend(validate_style_tokens(source_shielded_tagged, target_shielded_tagged))
    issues.extend(validate_numbers(source_unshielded_plain, target_unshielded_plain))
    issues.extend(validate_length(source_unshielded_plain, target_unshielded_plain))
    return issues
