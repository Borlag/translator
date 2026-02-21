from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Pattern

# Any ⟦...⟧ token (style tags and placeholders) must be preserved verbatim.
BRACKET_TOKEN_RE = re.compile(r"⟦[^⟧]*⟧")


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
