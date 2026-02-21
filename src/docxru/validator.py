from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .models import Issue, Severity
from .token_shield import extract_numbers, strip_bracket_tokens

STYLE_TOKEN_RE = re.compile(r"⟦/?S_\d+(?:\|[^⟧]*)?⟧")
PLACEHOLDER_RE = re.compile(r"⟦(?!/?S_)[A-Z][A-Z0-9]*_\d+⟧")


def extract_style_tokens(text: str) -> list[str]:
    return STYLE_TOKEN_RE.findall(text)


def extract_placeholders(text: str) -> list[str]:
    return PLACEHOLDER_RE.findall(text)


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
                message="Плейсхолдеры ⟦...⟧ потеряны/изменены/добавлены",
                details={"missing": missing, "extra": extra},
            )
        ]
    return []


def validate_style_tokens(source: str, target: str) -> list[Issue]:
    src = extract_style_tokens(source)
    tgt = extract_style_tokens(target)
    if src != tgt:
        # More precise diagnostics
        src_set = set(src)
        tgt_set = set(tgt)
        missing = [t for t in src if t not in tgt_set]
        extra = [t for t in tgt if t not in src_set]
        return [
            Issue(
                code="style_tags_mismatch",
                severity=Severity.ERROR,
                message="Маркеры форматирования ⟦S_n...⟧ нарушены (потеря/перестановка)",
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
                message="Числа/толерансы отличаются (проверь вручную)",
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
                message="Пустой перевод",
                details={"source_len": s, "target_len": t},
            )
        ]
    ratio = t / max(1, s)
    if ratio > factor_warn:
        return [
            Issue(
                code="length_ratio_high",
                severity=Severity.WARN,
                message="Слишком длинный перевод (возможный бред/повтор)",
                details={"ratio": ratio, "source_len": s, "target_len": t},
            )
        ]
    return []


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
