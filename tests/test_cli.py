from __future__ import annotations

from pathlib import Path

from docxru import cli


def _write_min_config(path: Path) -> None:
    path.write_text(
        "llm:\n"
        "  provider: mock\n"
        "  model: mock\n"
        "checker:\n"
        "  enabled: false\n",
        encoding="utf-8",
    )


def test_cli_translate_checker_only_dispatches_to_checker_runner(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    _write_min_config(cfg_path)

    called: dict[str, object] = {}

    def _fake_checker_only(**kwargs):  # noqa: ANN003
        called.update(kwargs)

    def _fake_translate(**kwargs):  # noqa: ANN003
        raise AssertionError("translate_docx should not be called in checker-only mode")

    monkeypatch.setattr(cli, "run_docx_checker_only", _fake_checker_only)
    monkeypatch.setattr(cli, "translate_docx", _fake_translate)

    rc = cli.main(
        [
            "translate",
            "--input",
            str(tmp_path / "in.docx"),
            "--output",
            str(tmp_path / "out.docx"),
            "--config",
            str(cfg_path),
            "--resume",
            "--checker-only",
        ]
    )
    assert rc == 0
    assert called["resume"] is True
    assert called["input_path"] == tmp_path / "in.docx"
    assert called["output_path"] == tmp_path / "out.docx"


def test_cli_translate_default_dispatches_to_translate_docx(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    _write_min_config(cfg_path)

    called: dict[str, object] = {}

    def _fake_checker_only(**kwargs):  # noqa: ANN003
        raise AssertionError("run_docx_checker_only should not be called in default mode")

    def _fake_translate(**kwargs):  # noqa: ANN003
        called.update(kwargs)

    monkeypatch.setattr(cli, "run_docx_checker_only", _fake_checker_only)
    monkeypatch.setattr(cli, "translate_docx", _fake_translate)

    rc = cli.main(
        [
            "translate",
            "--input",
            str(tmp_path / "in.docx"),
            "--output",
            str(tmp_path / "out.docx"),
            "--config",
            str(cfg_path),
        ]
    )
    assert rc == 0
    assert called["resume"] is False
    assert called["input_path"] == tmp_path / "in.docx"
    assert called["output_path"] == tmp_path / "out.docx"


def test_cli_translate_accepts_full_abbyy_profile(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    _write_min_config(cfg_path)

    called: dict[str, object] = {}

    def _fake_checker_only(**kwargs):  # noqa: ANN003
        raise AssertionError("run_docx_checker_only should not be called in default mode")

    def _fake_translate(**kwargs):  # noqa: ANN003
        called.update(kwargs)

    monkeypatch.setattr(cli, "run_docx_checker_only", _fake_checker_only)
    monkeypatch.setattr(cli, "translate_docx", _fake_translate)

    rc = cli.main(
        [
            "translate",
            "--input",
            str(tmp_path / "in.docx"),
            "--output",
            str(tmp_path / "out.docx"),
            "--config",
            str(cfg_path),
            "--abbyy-profile",
            "full",
        ]
    )
    assert rc == 0
    cfg = called["cfg"]
    assert getattr(cfg, "abbyy_profile") == "full"


def test_cli_postformat_dispatches_to_postformat_docx(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    _write_min_config(cfg_path)

    called: dict[str, object] = {}

    def _fake_postformat(**kwargs):  # noqa: ANN003
        called.update(kwargs)

    monkeypatch.setattr(cli, "postformat_docx", _fake_postformat)

    rc = cli.main(
        [
            "postformat",
            "--input",
            str(tmp_path / "translated.docx"),
            "--output",
            str(tmp_path / "final.docx"),
            "--config",
            str(cfg_path),
            "--abbyy-profile",
            "full",
            "--mode",
            "reflow",
            "--max-segments",
            "5",
        ]
    )
    assert rc == 0
    assert called["input_path"] == tmp_path / "translated.docx"
    assert called["output_path"] == tmp_path / "final.docx"
    assert called["max_segments"] == 5
    cfg = called["cfg"]
    assert getattr(cfg, "abbyy_profile") == "full"
    assert getattr(cfg, "mode") == "reflow"
