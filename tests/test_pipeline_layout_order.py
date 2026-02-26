from __future__ import annotations

import logging

from docx import Document

import docxru.pipeline as pipeline
from docxru.config import PipelineConfig
from docxru.models import Issue, Segment, Severity


def test_apply_abbyy_and_layout_passes_runs_normalization_before_layout(monkeypatch):
    call_order: list[str] = []

    def _fake_normalize(doc, *, profile):  # noqa: ANN001, ANN202
        del doc
        call_order.append("normalize")
        assert profile == "full"
        return {
            "tr_height_exact_removed": 1,
            "frame_pr_removed": 1,
            "line_spacing_exact_relaxed": 1,
            "textbox_autofit_updated": 1,
        }

    def _fake_validate(doc, segments, cfg):  # noqa: ANN001, ANN202
        del doc, cfg
        call_order.append("validate")
        return [
            Issue(
                code="layout_textbox_overflow_risk",
                severity=Severity.WARN,
                message="overflow",
                details={"segment_id": segments[0].segment_id},
            )
        ]

    def _fake_attach(segments, issues):  # noqa: ANN001, ANN202
        del segments, issues
        call_order.append("attach")
        return 1

    def _fake_fix(segments, issues, cfg):  # noqa: ANN001, ANN202
        del segments, issues, cfg
        call_order.append("fix")
        return 1

    monkeypatch.setattr(pipeline, "normalize_abbyy_oxml", _fake_normalize)
    monkeypatch.setattr(pipeline, "validate_layout", _fake_validate)
    monkeypatch.setattr(pipeline, "_attach_issues_to_segments", _fake_attach)
    monkeypatch.setattr(pipeline, "fix_expansion_issues", _fake_fix)

    doc = Document()
    paragraph = doc.add_paragraph("Text")
    segments = [
        Segment(
            segment_id="s1",
            location="body/p1",
            context={"part": "body", "in_textbox": True},
            source_plain="Text",
            paragraph_ref=paragraph,
            target_tagged="Long translated text",
        )
    ]
    cfg = PipelineConfig(abbyy_profile="full", layout_check=True, layout_auto_fix=True)

    pipeline._apply_abbyy_and_layout_passes(
        doc,
        segments,
        cfg,
        logging.getLogger("test_pipeline_layout_order"),
    )

    assert call_order == ["normalize", "validate", "attach", "fix"]
