from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import PipelineConfig


def default_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: Path
    status_path: Path
    dashboard_html_path: Path
    qa_report_path: Path
    qa_jsonl_path: Path
    checker_suggestions_path: Path
    checker_suggestions_safe_path: Path
    checker_trace_path: Path


def resolve_run_paths(cfg: PipelineConfig, *, output_path: Path) -> RunPaths:
    run_id = (cfg.run.run_id or "").strip() or default_run_id()
    if cfg.run.run_dir:
        run_dir = Path(cfg.run.run_dir).expanduser().resolve() / run_id
    else:
        run_dir = output_path.parent.resolve()
    status_path = (
        Path(cfg.run.status_path).expanduser().resolve()
        if cfg.run.status_path
        else (run_dir / "run_status.json")
    )
    dashboard_html_path = (
        Path(cfg.run.dashboard_html_path).expanduser().resolve()
        if cfg.run.dashboard_html_path
        else (run_dir / "dashboard.html")
    )
    qa_report_raw = Path(cfg.qa_report_path)
    qa_jsonl_raw = Path(cfg.qa_jsonl_path)
    qa_report_path = qa_report_raw if qa_report_raw.is_absolute() else (run_dir / qa_report_raw)
    qa_jsonl_path = qa_jsonl_raw if qa_jsonl_raw.is_absolute() else (run_dir / qa_jsonl_raw)

    checker_path = Path(cfg.checker.output_path)
    if checker_path.is_absolute():
        checker_suggestions_path = checker_path
    else:
        checker_suggestions_path = run_dir / checker_path
    checker_safe_path = Path(cfg.checker.safe_output_path)
    if checker_safe_path.is_absolute():
        checker_suggestions_safe_path = checker_safe_path
    else:
        checker_suggestions_safe_path = run_dir / checker_safe_path
    checker_trace_path = run_dir / "checker_trace.jsonl"

    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        status_path=status_path,
        dashboard_html_path=dashboard_html_path,
        qa_report_path=qa_report_path,
        qa_jsonl_path=qa_jsonl_path,
        checker_suggestions_path=checker_suggestions_path,
        checker_suggestions_safe_path=checker_suggestions_safe_path,
        checker_trace_path=checker_trace_path,
    )
