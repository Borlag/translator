from __future__ import annotations

from docxru.consistency import (
    build_phrase_translation_map,
    detect_inconsistencies,
    report_consistency,
)
from docxru.llm import build_glossary_matchers
from docxru.models import Segment


def _make_segment(seg_id: str, source: str, target: str) -> Segment:
    return Segment(
        segment_id=seg_id,
        location=f"body/p{seg_id}",
        context={"part": "body"},
        source_plain=source,
        paragraph_ref=None,
        target_tagged=target,
        target_shielded_tagged=target,
    )


def test_consistency_detects_term_variants_for_same_source_term():
    matchers = build_glossary_matchers("Main fitting - Основной фитинг")
    segments = [
        _make_segment("1", "Main fitting", "Основной фитинг"),
        _make_segment("2", "Main fitting", "Главный фитинг"),
    ]

    phrase_map = build_phrase_translation_map(segments, matchers)
    assert phrase_map["Main fitting"] == {"Основной фитинг", "Главный фитинг"}

    issues = detect_inconsistencies(phrase_map)
    assert len(issues) == 1
    assert issues[0].code == "consistency_term_variation"

    report = report_consistency(segments, matchers)
    assert len(report) == 1
    assert report[0].details.get("segment_id") == "1"
    assert report[0].details.get("majority_target") is None
    assert report[0].details.get("minority_targets") is None
    segments_by_target = report[0].details.get("segments_by_target", {})
    assert segments_by_target["Основной фитинг"] == ["1"]
    assert segments_by_target["Главный фитинг"] == ["2"]


def test_consistency_report_marks_minority_variant_against_majority():
    matchers = build_glossary_matchers("Sliding tube - Скользящая трубка")
    segments = [
        _make_segment("1", "Sliding tube", "Скользящая трубка"),
        _make_segment("2", "Sliding tube", "Скользящая трубка"),
        _make_segment("3", "Sliding tube", "Скользящая труба"),
    ]

    report = report_consistency(segments, matchers)

    assert len(report) == 1
    details = report[0].details
    assert details.get("majority_target") == "Скользящая трубка"
    assert details.get("majority_count") == 2
    assert details.get("minority_targets") == [{"target": "Скользящая труба", "count": 1}]


def test_consistency_report_is_empty_when_term_is_uniform():
    matchers = build_glossary_matchers("Sliding tube - Скользящая труба")
    segments = [
        _make_segment("1", "Install sliding tube", "Установите скользящая труба"),
        _make_segment("2", "Remove sliding tube", "Снимите скользящая труба"),
    ]

    assert report_consistency(segments, matchers) == []

