from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Pattern

# Any ⟦...⟧ token (style tags and placeholders) must be preserved verbatim.
BRACKET_TOKEN_RE = re.compile(r"⟦[^⟧]*⟧")
_BREAK_TOKEN_RE = re.compile(r"⟦BR(?:LINE|COL|PAGE)_\d+⟧")


@dataclass(frozen=True)
class PatternRule:
    name: str
    pattern: str
    flags: str = ""  # e.g. "I" for IGNORECASE
    description: str = ""


@dataclass(frozen=True)
class PatternSet:
    rules: list[PatternRule]


@dataclass(frozen=True)
class CompiledRule:
    name: str
    regex: Pattern[str]
    description: str = ""


def compile_pattern_set(pattern_set: PatternSet) -> PatternSet:
    # Overload PatternSet to store CompiledRule in `rules` field while keeping type simple.
    compiled: list[PatternRule] = []
    for r in pattern_set.rules:
        # Validate regex early
        re.compile(r.pattern)
        compiled.append(r)
    return PatternSet(compiled)


def _compile_rule(rule: PatternRule) -> CompiledRule:
    flags = 0
    if "I" in rule.flags.upper():
        flags |= re.IGNORECASE
    if "M" in rule.flags.upper():
        flags |= re.MULTILINE
    if "S" in rule.flags.upper():
        flags |= re.DOTALL
    return CompiledRule(name=rule.name, regex=re.compile(rule.pattern, flags=flags), description=rule.description)


def _split_preserving_brackets(text: str) -> list[tuple[str, bool]]:
    """Split text into [(chunk, is_bracket_token)]."""
    out: list[tuple[str, bool]] = []
    pos = 0
    for m in BRACKET_TOKEN_RE.finditer(text):
        if m.start() > pos:
            out.append((text[pos : m.start()], False))
        out.append((m.group(0), True))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], False))
    return out


def shield(text: str, pattern_set: PatternSet) -> tuple[str, dict[str, str]]:
    """Replace protected substrings with deterministic placeholders ⟦NAME_n⟧.

    - Does NOT touch any existing ⟦...⟧ blocks (style tags or already-shielded placeholders).
    - Replacement happens only in plain-text chunks outside ⟦...⟧.
    """
    counters: dict[str, int] = {}
    token_map: dict[str, str] = {}

    rules = [_compile_rule(r) for r in pattern_set.rules]

    chunks = _split_preserving_brackets(text)
    new_chunks: list[str] = []

    for chunk, is_tok in chunks:
        if is_tok or not chunk:
            new_chunks.append(chunk)
            continue

        updated = chunk
        for rule in rules:
            def _repl(m: re.Match[str]) -> str:
                name = rule.name
                counters[name] = counters.get(name, 0) + 1
                placeholder = f"⟦{name}_{counters[name]}⟧"
                token_map[placeholder] = m.group(0)
                return placeholder

            updated = rule.regex.sub(_repl, updated)
        new_chunks.append(updated)

    return ("".join(new_chunks), token_map)


def unshield(text: str, token_map: dict[str, str]) -> str:
    """Reverse shielding. Expects placeholders to be present exactly as keys in token_map."""
    if not token_map:
        return text

    # Replace longer keys first to avoid accidental partial overlaps.
    for ph in sorted(token_map.keys(), key=len, reverse=True):
        text = text.replace(ph, token_map[ph])
    return text


_TOKEN_PREFIX_RE = re.compile(r"^[A-Z][A-Z0-9]*\Z")


def shield_terms(
    text: str,
    replacements: Iterable[tuple[Pattern[str], str]],
    *,
    token_prefix: str = "GLS",
) -> tuple[str, dict[str, str]]:
    """Replace matched terms with deterministic placeholders ⟦PREFIX_n⟧.

    This is useful for "hard glossary" behavior: replace EN terms with placeholders before translation,
    then unshield them to the desired RU equivalents after translation.

    Notes:
    - Does NOT touch style tags/placeholders inside ⟦...⟧.
    - Break tokens ⟦BRLINE_n⟧ / ⟦BRCOL_n⟧ / ⟦BRPAGE_n⟧ are treated as text separators so
      multi-word glossary patterns can match across wrapped lines.
    - Placeholder numbering is based on replacement-rule order (1-based), not match order.
    - The returned token_map maps placeholder -> replacement text, and is suitable for `unshield()`.
    """
    if not text:
        return text, {}

    rules = tuple(replacements)
    if not rules:
        return text, {}

    prefix = token_prefix.strip().upper()
    if not _TOKEN_PREFIX_RE.fullmatch(prefix):
        raise ValueError(f"Invalid token_prefix '{token_prefix}'. Expected [A-Z][A-Z0-9]*.")

    token_map: dict[str, str] = {}
    chunks_raw: list[tuple[str, bool]] = []
    pos = 0
    for m in BRACKET_TOKEN_RE.finditer(text):
        if m.start() > pos:
            chunks_raw.append((text[pos : m.start()], False))
        token = m.group(0)
        # Keep line-break tokens in text stream to allow phrase matching across wraps.
        is_protected_token = _BREAK_TOKEN_RE.fullmatch(token) is None
        chunks_raw.append((token, is_protected_token))
        pos = m.end()
    if pos < len(text):
        chunks_raw.append((text[pos:], False))

    # Merge adjacent non-protected chunks so regex can match across embedded break tokens.
    chunks: list[tuple[str, bool]] = []
    for chunk, is_tok in chunks_raw:
        if chunks and chunks[-1][1] == is_tok:
            prev_chunk, _ = chunks[-1]
            chunks[-1] = (prev_chunk + chunk, is_tok)
        else:
            chunks.append((chunk, is_tok))
    out_chunks: list[str] = []

    for chunk, is_tok in chunks:
        if is_tok or not chunk:
            out_chunks.append(chunk)
            continue

        updated = chunk
        for i, (pattern, replacement) in enumerate(rules, start=1):
            placeholder = f"⟦{prefix}_{i}⟧"
            updated, n = pattern.subn(placeholder, updated)
            if n:
                token_map[placeholder] = replacement
        out_chunks.append(updated)

    return ("".join(out_chunks), token_map)


_NUMBER_RE = re.compile(
    r"""(?x)
    (?:
      (?<![A-Z0-9])            # left boundary (avoid ATA/P/N fragments)
      [-+]?\d+(?:[\.,]\d+)?  # number with optional decimal
      (?:\s*±\s*[-+]?\d+(?:[\.,]\d+)?)?  # optional ± tolerance
      (?![A-Z0-9])             # right boundary
    )
    """
)


def strip_bracket_tokens(text: str) -> str:
    return BRACKET_TOKEN_RE.sub("", text)


def extract_numbers(text: str) -> list[str]:
    """Extract numeric tokens for QA. Keeps order. Normalizes decimal comma to dot."""
    plain = strip_bracket_tokens(text)
    out: list[str] = []
    for m in _NUMBER_RE.finditer(plain):
        tok = m.group(0)
        tok = tok.replace(",", ".")
        tok = re.sub(r"\s+", " ", tok).strip()
        out.append(tok)
    return out
