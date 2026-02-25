from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from .models import Issue, Severity
from .token_shield import BRACKET_TOKEN_RE, extract_numbers

_STYLE_TOKEN_BODY_RE = re.compile(r"^/?S_\d+(?:\|[^|]+)*$")
_PLACEHOLDER_BODY_RE = re.compile(r"^[A-Z][A-Z0-9]*_\d+$")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")
_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9/-]*")
_CONTEXT_LEAKAGE_TOKEN_RE = re.compile(
    r"\b(PART|DOC_SECTION|TABLE_CELL|TOC_ENTRY|SECTION|PREV|NEXT|MATCHED_GLOSSARY|DOCUMENT_GLOSSARY|TM_REFERENCES|RECENT_TRANSLATIONS)\b(\s*[:=])?",
    flags=re.IGNORECASE,
)
_DEFAULT_UNTRANSLATED_ALLOWLIST = {
    "amm",
    "api",
    "ata",
    "cmm",
    "docx",
    "en",
    "ipc",
    "kg",
    "kpa",
    "mlg",
    "mm",
    "nlg",
    "nm",
    "oem",
    "pdf",
    "pn",
    "psi",
    "ru",
    "safran",
    "sb",
}
_CONTEXT_TOKENS_REQUIRE_SEPARATOR = {"PART", "SECTION", "PREV", "NEXT"}

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


def validate_short_translation(
    source_plain: str,
    target_plain: str,
    *,
    min_ratio: float = 0.35,
    min_source_chars: int = 24,
) -> list[Issue]:
    source_len = len((source_plain or "").strip())
    target_len = len((target_plain or "").strip())
    min_src = max(1, int(min_source_chars))
    if source_len < min_src:
        return []
    if source_len == 0:
        return []

    ratio = float(target_len) / float(source_len)
    ratio_threshold = max(0.0, min(1.0, float(min_ratio)))
    if ratio >= ratio_threshold:
        return []

    return [
        Issue(
            code="short_translation",
            severity=Severity.WARN,
            message="Translation is suspiciously short relative to source text.",
            details={
                "ratio": round(ratio, 4),
                "ratio_threshold": ratio_threshold,
                "source_len": source_len,
                "target_len": target_len,
                "min_source_chars": min_src,
            },
        )
    ]


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


@lru_cache(maxsize=16)
def _load_allowlist_file(path: str) -> tuple[str, ...]:
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError:
        return ()

    values: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip().lower()
        if line:
            values.append(line)
    return tuple(values)


def _resolve_allowlist(path: str | None, defaults: set[str]) -> set[str]:
    allowlist = set(defaults)
    raw = str(path or "").strip()
    if not raw:
        return allowlist
    allowlist.update(_load_allowlist_file(raw))
    return allowlist


def validate_untranslated_fragments(
    source_plain: str,
    target_plain: str,
    *,
    warn_ratio: float = 0.15,
    min_len: int = 3,
    allowlist_path: str | None = None,
) -> list[Issue]:
    if not source_plain or not target_plain:
        return []

    target_words = _WORD_RE.findall(target_plain or "")
    total_words = len(target_words)
    if total_words == 0:
        return []

    ratio_limit = max(0.0, min(1.0, float(warn_ratio)))
    min_word_len = max(1, int(min_len))
    allowlist = _resolve_allowlist(allowlist_path, _DEFAULT_UNTRANSLATED_ALLOWLIST)

    suspicious: list[str] = []
    for token in _LATIN_WORD_RE.findall(target_plain or ""):
        word = token.strip()
        if len(word) < min_word_len:
            continue
        lowered = word.lower()
        if lowered in allowlist:
            continue
        if any(ch.isdigit() for ch in word):
            continue
        if word.isupper() and len(word) <= 5:
            continue
        suspicious.append(word)

    if not suspicious:
        return []

    ratio = len(suspicious) / max(1, total_words)
    if ratio < ratio_limit:
        return []

    unique_samples: list[str] = []
    seen_samples: set[str] = set()
    for item in suspicious:
        key = item.lower()
        if key in seen_samples:
            continue
        seen_samples.add(key)
        unique_samples.append(item)
        if len(unique_samples) >= 8:
            break

    return [
        Issue(
            code="untranslated_fragments",
            severity=Severity.WARN,
            message="Translation likely contains untranslated Latin fragments.",
            details={
                "ratio": round(ratio, 4),
                "count": len(suspicious),
                "total_words": total_words,
                "samples": unique_samples,
            },
        )
    ]


def validate_repeated_words(
    target_plain: str,
    *,
    phrase_ngram_max: int = 3,
) -> list[Issue]:
    text = str(target_plain or "").strip()
    if not text:
        return []

    repeated_words: list[str] = []
    seen_word: set[str] = set()
    for match in re.finditer(r"\b([А-Яа-яЁё]{2,})\b(?:[\s,;:()\-]+)\1\b", text, flags=re.IGNORECASE):
        word = str(match.group(1) or "").lower()
        if not word or word in seen_word:
            continue
        seen_word.add(word)
        repeated_words.append(word)
        if len(repeated_words) >= 6:
            break

    phrase_hits: list[str] = []
    tokens = [token.lower() for token in re.findall(r"[А-Яа-яЁё]{2,}", text)]
    max_ngram = max(2, int(phrase_ngram_max))
    seen_phrase: set[str] = set()
    for ngram_size in range(2, max_ngram + 1):
        if len(tokens) < ngram_size * 2:
            continue
        for idx in range(0, len(tokens) - (ngram_size * 2) + 1):
            left = tokens[idx : idx + ngram_size]
            right = tokens[idx + ngram_size : idx + (ngram_size * 2)]
            if left != right:
                continue
            phrase = " ".join(left)
            if phrase in seen_phrase:
                continue
            seen_phrase.add(phrase)
            phrase_hits.append(phrase)
            if len(phrase_hits) >= 6:
                break
        if len(phrase_hits) >= 6:
            break

    if not repeated_words and not phrase_hits:
        return []

    return [
        Issue(
            code="repeated_words",
            severity=Severity.WARN,
            message="Translation contains repeated word/phrase artifacts.",
            details={
                "repeated_words": repeated_words,
                "repeated_phrases": phrase_hits,
            },
        )
    ]


def validate_context_leakage(
    target_plain: str,
    *,
    allowlist_path: str | None = None,
) -> list[Issue]:
    text = str(target_plain or "")
    if not text:
        return []

    allowlist = {item.upper() for item in _resolve_allowlist(allowlist_path, set())}
    hits: list[str] = []
    seen: set[str] = set()
    for match in _CONTEXT_LEAKAGE_TOKEN_RE.finditer(text):
        token = str(match.group(1) or "").upper()
        separator = str(match.group(2) or "")
        if token in _CONTEXT_TOKENS_REQUIRE_SEPARATOR and not separator.strip():
            continue
        if token in allowlist or token in seen:
            continue
        seen.add(token)
        hits.append(token)
        if len(hits) >= 8:
            break

    if not hits:
        return []

    return [
        Issue(
            code="context_leakage",
            severity=Severity.WARN,
            message="Prompt/context control tokens leaked into translated text.",
            details={"tokens": hits},
        )
    ]


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
    *,
    short_translation_min_ratio: float = 0.35,
    short_translation_min_source_chars: int = 24,
    untranslated_latin_warn_ratio: float = 0.15,
    untranslated_latin_min_len: int = 3,
    untranslated_latin_allowlist_path: str | None = None,
    repeated_words_check: bool = True,
    repeated_phrase_ngram_max: int = 3,
    context_leakage_check: bool = True,
    context_leakage_allowlist_path: str | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    issues.extend(validate_placeholders(source_shielded_tagged, target_shielded_tagged))
    issues.extend(validate_style_tokens(source_shielded_tagged, target_shielded_tagged))
    issues.extend(validate_numbers(source_unshielded_plain, target_unshielded_plain))
    issues.extend(validate_length(source_unshielded_plain, target_unshielded_plain))
    issues.extend(
        validate_short_translation(
            source_unshielded_plain,
            target_unshielded_plain,
            min_ratio=short_translation_min_ratio,
            min_source_chars=short_translation_min_source_chars,
        )
    )
    issues.extend(
        validate_untranslated_fragments(
            source_unshielded_plain,
            target_unshielded_plain,
            warn_ratio=untranslated_latin_warn_ratio,
            min_len=untranslated_latin_min_len,
            allowlist_path=untranslated_latin_allowlist_path,
        )
    )
    if repeated_words_check:
        issues.extend(
            validate_repeated_words(
                target_unshielded_plain,
                phrase_ngram_max=repeated_phrase_ngram_max,
            )
        )
    if context_leakage_check:
        issues.extend(
            validate_context_leakage(
                target_unshielded_plain,
                allowlist_path=context_leakage_allowlist_path,
            )
        )
    return issues
