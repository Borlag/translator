from __future__ import annotations

import logging

import docxru.pipeline as pipeline
import docxru.validator as validator
from docxru.config import LLMConfig, PipelineConfig
from docxru.models import Issue, Segment, Severity


class _FakeParse:
    def __init__(self, normal_form: str) -> None:
        self.normal_form = normal_form


class _FakeMorph:
    def __init__(self, lemma_map: dict[str, str]) -> None:
        self.lemma_map = {k.lower(): v.lower() for k, v in lemma_map.items()}

    def parse(self, word: str) -> list[_FakeParse]:
        w = word.lower()
        return [_FakeParse(self.lemma_map.get(w, w))]


class _SequentialClient:
    supports_repair = True

    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    def translate(self, text: str, context: dict[str, object]) -> str:
        self.calls.append({"text": text, "context": dict(context)})
        if not self._outputs:
            raise RuntimeError("No outputs configured")
        return self._outputs.pop(0)


def _glossary_issue() -> list[Issue]:
    return [
        Issue(
            code="glossary_lemma_mismatch",
            severity=Severity.WARN,
            message="Missing glossary term",
            details={"missing": [{"source": "Main Fitting", "target": "корпус стойки"}], "mode": "retry"},
        )
    ]


def _segment_for_retry_test() -> Segment:
    return Segment(
        segment_id="s1",
        location="body/p1",
        context={
            "part": "body",
            "matched_glossary_terms": [{"source": "Main Fitting", "target": "корпус стойки"}],
        },
        source_plain="Install Main Fitting.",
        paragraph_ref=None,
        shielded_tagged="Install Main Fitting.",
    )


def test_validate_glossary_lemmas_detects_missing_term(monkeypatch):
    monkeypatch.setattr(validator, "_get_morph_analyzer", lambda: _FakeMorph({}))
    issues = validator.validate_glossary_lemmas(
        "Установите узел.",
        [{"source": "Main Fitting", "target": "корпус стойки"}],
        mode="warn",
    )
    assert len(issues) == 1
    assert issues[0].code == "glossary_lemma_mismatch"


def test_validate_glossary_lemmas_accepts_inflected_forms(monkeypatch):
    monkeypatch.setattr(
        validator,
        "_get_morph_analyzer",
        lambda: _FakeMorph(
            {
                "нижнего": "нижний",
                "рычага": "рычаг",
                "крутящего": "крутящий",
                "момента": "момент",
            }
        ),
    )
    issues = validator.validate_glossary_lemmas(
        "Установите нижнего рычага крутящего момента.",
        [{"source": "Lower Torque Link", "target": "нижний рычаг крутящего момента"}],
        mode="warn",
    )
    assert issues == []


def test_validate_glossary_lemmas_uses_exact_fallback_without_morphology(monkeypatch):
    monkeypatch.setattr(validator, "_get_morph_analyzer", lambda: None)
    missing_issues = validator.validate_glossary_lemmas(
        "Установите узел.",
        [{"source": "Main Fitting", "target": "корпус стойки"}],
        mode="warn",
    )
    assert len(missing_issues) == 1
    assert missing_issues[0].code == "glossary_lemma_mismatch"

    ok_issues = validator.validate_glossary_lemmas(
        "Установите корпус стойки.",
        [{"source": "Main Fitting", "target": "корпус стойки"}],
        mode="warn",
    )
    assert ok_issues == []


def test_translate_one_glossary_retry_applies_improved_output(monkeypatch):
    def fake_validate(target_plain: str, matched_terms, *, mode: str = "off"):  # noqa: ANN001
        return [] if "корпус стойки" in target_plain.lower() else _glossary_issue()

    monkeypatch.setattr(pipeline, "validate_glossary_lemmas", fake_validate)
    seg = _segment_for_retry_test()
    cfg = PipelineConfig(llm=LLMConfig(retries=1), glossary_lemma_check="retry")
    client = _SequentialClient(["Установите деталь.", "Установите корпус стойки."])

    out, issues = pipeline._translate_one(seg, cfg, client, "hash-1", "norm-1", logging.getLogger("test"))

    assert out == "Установите корпус стойки."
    assert "glossary_retry_applied" in {issue.code for issue in issues}
    assert len(client.calls) == 2
    assert client.calls[1]["context"] == {
        "part": "body",
        "matched_glossary_terms": [{"source": "Main Fitting", "target": "корпус стойки"}],
        "task": "repair",
        "glossary_retry": True,
    }


def test_translate_one_glossary_retry_keeps_original_without_improvement(monkeypatch):
    def fake_validate(target_plain: str, matched_terms, *, mode: str = "off"):  # noqa: ANN001
        return [] if "корпус стойки" in target_plain.lower() else _glossary_issue()

    monkeypatch.setattr(pipeline, "validate_glossary_lemmas", fake_validate)
    seg = _segment_for_retry_test()
    cfg = PipelineConfig(llm=LLMConfig(retries=1), glossary_lemma_check="retry")
    client = _SequentialClient(["Установите деталь.", "Установите компонент."])

    out, issues = pipeline._translate_one(seg, cfg, client, "hash-1", "norm-1", logging.getLogger("test"))

    codes = {issue.code for issue in issues}
    assert out == "Установите деталь."
    assert "glossary_lemma_mismatch" in codes
    assert "glossary_retry_no_improvement" in codes
