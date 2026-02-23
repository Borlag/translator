from __future__ import annotations

from docxru.dashboard_server import ensure_dashboard_html


def test_ensure_dashboard_html_writes_template(tmp_path):
    out = ensure_dashboard_html(tmp_path, filename="dashboard.html")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "/api/status" in text
    assert "Open Output Folder" in text
    assert "Open Checker Trace" in text
