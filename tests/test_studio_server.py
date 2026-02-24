from __future__ import annotations

import subprocess
import sys
from contextlib import suppress
from pathlib import Path

from docxru.studio_server import (
    StudioRun,
    StudioRunManager,
    _build_studio_html,
    _default_model_for_provider,
    _estimate_grouped_request_count,
    _estimate_request_latency_bounds_seconds,
    _infer_translation_cmd,
    _list_openai_models,
)


def test_infer_translation_cmd_for_docx_and_pdf():
    cmd_docx, out_docx = _infer_translation_cmd(Path("input.docx"))
    cmd_pdf, out_pdf = _infer_translation_cmd(Path("input.pdf"))
    assert cmd_docx == "translate"
    assert out_docx.endswith(".docx")
    assert cmd_pdf == "translate-pdf"
    assert out_pdf.endswith(".pdf")


def test_studio_build_config_payload_includes_checker_settings(tmp_path):
    manager = StudioRunManager(base_dir=tmp_path)
    run_id = "run123"
    run_dir = tmp_path / "runs" / run_id
    payload = manager._build_config_payload(  # noqa: SLF001 - intentional for unit-level validation
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.1,
        max_output_tokens=2000,
        concurrency=3,
        translation_grouping_mode="grouped_fast",
        prompt_path=tmp_path / "prompt.md",
        glossary_path=tmp_path / "glossary.md",
        checker_enabled=True,
        checker_provider="openai",
        checker_model="gpt-4o-mini",
        checker_pages_per_chunk=3,
        checker_fallback_segments_per_chunk=80,
        checker_temperature=0.0,
        checker_max_output_tokens=1500,
        checker_openai_batch_enabled=True,
        checker_auto_apply_safe=True,
        checker_auto_apply_min_confidence=0.8,
        run_base_dir=tmp_path / "runs",
        run_id=run_id,
        run_dir=run_dir,
    )
    assert payload["llm"]["provider"] == "openai"
    assert payload["llm"]["model"] == "gpt-4o-mini"
    assert payload["llm"]["auto_model_sizing"] is True
    assert payload["llm"]["batch_segments"] == 6
    assert payload["llm"]["batch_max_chars"] == 14000
    assert payload["llm"]["context_window_chars"] == 0
    assert payload["checker"]["enabled"] is True
    assert payload["checker"]["provider"] == "openai"
    assert payload["checker"]["model"] == "gpt-4o-mini"
    assert payload["checker"]["pages_per_chunk"] == 3
    assert payload["checker"]["fallback_segments_per_chunk"] == 80
    assert payload["checker"]["openai_batch_enabled"] is True
    assert payload["checker"]["safe_output_path"] == "checker_suggestions_safe.json"
    assert payload["checker"]["auto_apply_safe"] is True
    assert payload["checker"]["auto_apply_min_confidence"] == 0.8
    assert payload["run"]["run_id"] == run_id
    assert payload["run"]["batch_fallback_warn_ratio"] == 0.08
    assert payload["run"]["fail_fast_on_translate_error"] is False
    assert payload["concurrency"] == 3


def test_studio_build_config_payload_can_enable_sequential_context_mode(tmp_path):
    manager = StudioRunManager(base_dir=tmp_path)
    payload = manager._build_config_payload(  # noqa: SLF001 - intentional for unit-level validation
        provider="openai",
        model="gpt-5-mini",
        temperature=0.1,
        max_output_tokens=2000,
        concurrency=2,
        translation_grouping_mode="sequential_context",
        prompt_path=None,
        glossary_path=None,
        checker_enabled=False,
        checker_provider=None,
        checker_model=None,
        checker_pages_per_chunk=3,
        checker_fallback_segments_per_chunk=120,
        checker_temperature=0.0,
        checker_max_output_tokens=2000,
        checker_openai_batch_enabled=False,
        checker_auto_apply_safe=False,
        checker_auto_apply_min_confidence=0.7,
        run_base_dir=tmp_path / "runs",
        run_id="run_seq",
        run_dir=tmp_path / "runs" / "run_seq",
    )
    assert payload["llm"]["batch_segments"] == 1
    assert payload["llm"]["batch_max_chars"] == 12000
    assert payload["llm"]["context_window_chars"] == 600
    assert payload["run"]["batch_fallback_warn_ratio"] == 0.03


def test_studio_build_config_payload_can_enable_grouped_aggressive_mode(tmp_path):
    manager = StudioRunManager(base_dir=tmp_path)
    payload = manager._build_config_payload(  # noqa: SLF001 - intentional for unit-level validation
        provider="openai",
        model="gpt-5-mini",
        temperature=0.1,
        max_output_tokens=2000,
        concurrency=2,
        translation_grouping_mode="grouped_aggressive",
        prompt_path=None,
        glossary_path=None,
        checker_enabled=False,
        checker_provider=None,
        checker_model=None,
        checker_pages_per_chunk=3,
        checker_fallback_segments_per_chunk=120,
        checker_temperature=0.0,
        checker_max_output_tokens=2000,
        checker_openai_batch_enabled=False,
        checker_auto_apply_safe=False,
        checker_auto_apply_min_confidence=0.7,
        run_base_dir=tmp_path / "runs",
        run_id="run_agg",
        run_dir=tmp_path / "runs" / "run_agg",
    )
    assert payload["llm"]["batch_segments"] == 20
    assert payload["llm"]["batch_max_chars"] == 36000
    assert payload["llm"]["context_window_chars"] == 0
    assert payload["llm"]["auto_model_sizing"] is True
    assert payload["llm"]["timeout_s"] == 180.0
    assert payload["run"]["batch_fallback_warn_ratio"] == 0.12


def test_studio_build_config_payload_can_enable_grouped_turbo_mode(tmp_path):
    manager = StudioRunManager(base_dir=tmp_path)
    payload = manager._build_config_payload(  # noqa: SLF001 - intentional for unit-level validation
        provider="openai",
        model="gpt-5-mini",
        temperature=0.1,
        max_output_tokens=2000,
        concurrency=2,
        translation_grouping_mode="grouped_turbo",
        prompt_path=None,
        glossary_path=None,
        checker_enabled=False,
        checker_provider=None,
        checker_model=None,
        checker_pages_per_chunk=3,
        checker_fallback_segments_per_chunk=120,
        checker_temperature=0.0,
        checker_max_output_tokens=2000,
        checker_openai_batch_enabled=False,
        checker_auto_apply_safe=False,
        checker_auto_apply_min_confidence=0.7,
        run_base_dir=tmp_path / "runs",
        run_id="run_turbo",
        run_dir=tmp_path / "runs" / "run_turbo",
    )
    assert payload["llm"]["batch_segments"] == 24
    assert payload["llm"]["batch_max_chars"] == 60_000
    assert payload["llm"]["context_window_chars"] == 0
    assert payload["llm"]["auto_model_sizing"] is True
    assert payload["llm"]["timeout_s"] == 300.0
    assert payload["run"]["batch_fallback_warn_ratio"] == 0.20


def test_default_model_for_provider():
    assert _default_model_for_provider("openai") == "gpt-5-mini"
    assert _default_model_for_provider("ollama") == "qwen2.5:7b"
    assert _default_model_for_provider("google") == "ignored"
    assert _default_model_for_provider("mock") == "mock"


def test_list_openai_models_without_key_returns_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    models = _list_openai_models("")
    assert "gpt-4o-mini" in models


def test_studio_html_contains_checker_trace_widgets():
    html = _build_studio_html()
    assert "checkerTraceLink" in html
    assert "checkerTraceTail" in html
    assert "checkerOpenaiBatch" in html
    assert "checkerAutoApplySafe" in html
    assert "runCheckerBtn" in html
    assert "applyCheckerBtn" in html
    assert "grouped_aggressive" in html
    assert "grouped_turbo" in html
    assert 'value="grouped_turbo" selected' in html
    assert 'name="model" id="modelHidden" value="gpt-5-mini"' in html
    assert "stopRunBtn" in html
    assert "estimateBtn" in html
    assert "estimateHint" in html


def test_estimate_grouped_request_count_respects_segment_and_char_limits():
    lengths = [100, 100, 100, 100, 100]
    assert _estimate_grouped_request_count(lengths, max_segments=2, max_chars=10_000) == 3
    assert _estimate_grouped_request_count(lengths, max_segments=10, max_chars=260) == 5


def test_estimate_request_latency_bounds_openai_gpt5_grouped_is_higher_than_seq():
    seq_low, seq_high = _estimate_request_latency_bounds_seconds("openai", "gpt-5-mini", grouped_mode=False)
    grp_low, grp_high = _estimate_request_latency_bounds_seconds("openai", "gpt-5-mini", grouped_mode=True)
    assert grp_low > seq_low
    assert grp_high > seq_high


def test_estimate_request_latency_bounds_scales_for_large_grouped_batches():
    base_low, base_high = _estimate_request_latency_bounds_seconds(
        "openai",
        "gpt-5-mini",
        grouped_mode=True,
        batch_max_chars=36_000,
    )
    turbo_low, turbo_high = _estimate_request_latency_bounds_seconds(
        "openai",
        "gpt-5-mini",
        grouped_mode=True,
        batch_max_chars=120_000,
    )
    assert turbo_low > base_low
    assert turbo_high > base_high


def test_studio_field_readers_do_not_require_mapping_get(tmp_path):
    manager = StudioRunManager(base_dir=tmp_path)

    class _FakeField:
        def __init__(self, value: str) -> None:
            self.value = value

    class _FakeForm:
        def __init__(self) -> None:
            self._data = {"provider": _FakeField("openai")}

        def __contains__(self, key: str) -> bool:
            return key in self._data

        def __getitem__(self, key: str):
            return self._data[key]

    form = _FakeForm()
    assert manager._read_text_value(form, "provider", "mock") == "openai"  # noqa: SLF001
    assert manager._read_text_value(form, "missing", "mock") == "mock"  # noqa: SLF001


def test_studio_run_manager_can_force_stop_running_translation_process(tmp_path):
    manager = StudioRunManager(base_dir=tmp_path)
    run_id = "run_stop"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    run = StudioRun(
        run_id=run_id,
        run_dir=run_dir,
        source_path=run_dir / "input.docx",
        output_path=run_dir / "translated.docx",
        config_path=run_dir / "config.studio.yaml",
        log_path=run_dir / "studio_process.log",
        status_path=run_dir / "run_status.json",
        command=[sys.executable, "-m", "docxru", "translate"],
        process=process,
        started_at="2026-02-23T00:00:00+00:00",
    )
    manager._runs[run_id] = run  # noqa: SLF001 - unit-level hook
    try:
        payload = manager.stop_run(run_id)
        assert payload["ok"] is True
        assert payload["stopped"] is True
        status = manager.get_status(run_id)
        assert status["state"] == "cancelled"
        assert status["stop_requested_at"]
        assert status["stop_completed_at"]
    finally:
        with suppress(Exception):
            if process.poll() is None:
                process.kill()


def test_studio_run_manager_stop_run_on_finished_process_is_noop(tmp_path):
    manager = StudioRunManager(base_dir=tmp_path)
    run_id = "run_done"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [sys.executable, "-c", "print('ok')"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    process.wait(timeout=10)
    run = StudioRun(
        run_id=run_id,
        run_dir=run_dir,
        source_path=run_dir / "input.docx",
        output_path=run_dir / "translated.docx",
        config_path=run_dir / "config.studio.yaml",
        log_path=run_dir / "studio_process.log",
        status_path=run_dir / "run_status.json",
        command=[sys.executable, "-m", "docxru", "translate"],
        process=process,
        started_at="2026-02-23T00:00:00+00:00",
    )
    manager._runs[run_id] = run  # noqa: SLF001 - unit-level hook
    payload = manager.stop_run(run_id)
    assert payload["ok"] is True
    assert payload["already_finished"] is True
    assert payload["stopped"] is False


def test_studio_manager_loads_existing_run_dirs_on_startup(tmp_path):
    run_id = "run_existing"
    run_dir = tmp_path / "runs" / run_id
    uploads = run_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.studio.yaml").write_text("checker:\n  enabled: true\n", encoding="utf-8")
    (run_dir / "translated.docx").write_bytes(b"docx")
    (uploads / "input.docx").write_bytes(b"src")
    (run_dir / "run_status.json").write_text(
        '{"run_id":"run_existing","phase":"done","started_at":"2026-02-24T00:00:00+00:00"}',
        encoding="utf-8",
    )

    manager = StudioRunManager(base_dir=tmp_path)
    status = manager.get_status(run_id)
    assert status["ok"] is True
    assert status["state"] == "completed"
    assert status["run_id"] == run_id


def test_studio_run_manager_can_start_checker_only_for_finished_run(tmp_path, monkeypatch):
    manager = StudioRunManager(base_dir=tmp_path)
    run_id = "run_checker"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source_path = run_dir / "input.docx"
    source_path.write_bytes(b"source")
    output_path = run_dir / "translated.docx"
    output_path.write_bytes(b"output")
    config_path = run_dir / "config.studio.yaml"
    config_path.write_text("checker:\n  enabled: false\n", encoding="utf-8")
    log_path = run_dir / "studio_process.log"
    log_path.write_text("", encoding="utf-8")

    process = subprocess.Popen(
        [sys.executable, "-c", "print('ok')"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    process.wait(timeout=10)
    run = StudioRun(
        run_id=run_id,
        run_dir=run_dir,
        source_path=source_path,
        output_path=output_path,
        config_path=config_path,
        log_path=log_path,
        status_path=run_dir / "run_status.json",
        command=[sys.executable, "-m", "docxru", "translate"],
        process=process,
        started_at="2026-02-24T00:00:00+00:00",
    )
    manager._runs[run_id] = run  # noqa: SLF001 - unit-level hook

    captured: dict[str, object] = {}

    class _FakeProc:
        def poll(self):
            return None

    def _fake_popen(command, **kwargs):  # noqa: ANN001
        captured["command"] = command
        captured["env"] = kwargs.get("env")
        return _FakeProc()

    monkeypatch.setattr("docxru.studio_server.subprocess.Popen", _fake_popen)

    payload = manager.start_checker_for_run(run_id, openai_api_key="sk-test")
    assert payload["ok"] is True
    assert payload["run_id"] == run_id
    assert payload["state"] == "running"
    command = captured["command"]
    assert isinstance(command, list)
    assert "--checker-only" in command
    assert "--resume" in command
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["OPENAI_API_KEY"] == "sk-test"
