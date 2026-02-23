from __future__ import annotations

import json
import subprocess
import sys
import webbrowser
from contextlib import suppress
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _build_dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>docxru dashboard</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --card: #ffffff;
      --text: #152033;
      --muted: #5f6f86;
      --line: #d7dde7;
      --accent: #0d5bd7;
      --ok: #1e8f4b;
      --warn: #b87700;
      --err: #b33232;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Trebuchet MS", "Noto Sans", sans-serif;
      color: var(--text);
      background:
        radial-gradient(1000px 400px at 10% -10%, #dce8ff 0%, transparent 70%),
        radial-gradient(900px 300px at 100% 0%, #ffe9cf 0%, transparent 70%),
        var(--bg);
    }
    .wrap { max-width: 1040px; margin: 20px auto 60px; padding: 0 16px; }
    .head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
    .title { font-size: 28px; font-weight: 800; letter-spacing: 0.3px; margin: 0; }
    .stamp { color: var(--muted); font-size: 13px; }
    .grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-top: 14px; }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px;
      box-shadow: 0 8px 18px rgba(13, 34, 68, 0.05);
    }
    .k { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
    .v { font-size: 24px; font-weight: 800; margin-top: 2px; }
    .mono { font-family: Consolas, "SFMono-Regular", Menlo, monospace; }
    .bar { width: 100%; height: 12px; border-radius: 999px; background: #e4ebf6; overflow: hidden; margin-top: 8px; }
    .bar > i { display: block; height: 100%; width: 0%; background: linear-gradient(90deg, #1d66df, #11a1df); transition: width .3s ease; }
    .links { margin-top: 14px; display: flex; gap: 10px; flex-wrap: wrap; }
    a.btn, button.btn {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      padding: 8px 12px;
      border-radius: 10px;
      text-decoration: none;
      cursor: pointer;
      font-weight: 600;
    }
    button.btn.primary { border-color: #0a4ccf; background: #0d5bd7; color: #fff; }
    .small { font-size: 12px; color: var(--muted); margin-top: 12px; }
    .ok { color: var(--ok); }
    .warn { color: var(--warn); }
    .err { color: var(--err); }
    @media (max-width: 640px) { .title { font-size: 22px; } }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <h1 class="title">docxru Run Dashboard</h1>
      <div id="updated" class="stamp">waiting for status...</div>
    </div>
    <div class="grid">
      <div class="card">
        <div class="k">Phase</div>
        <div id="phase" class="v">-</div>
      </div>
      <div class="card">
        <div class="k">Progress</div>
        <div id="progress" class="v">0%</div>
        <div class="bar"><i id="bar"></i></div>
      </div>
      <div class="card">
        <div class="k">Segments</div>
        <div id="segments" class="v mono">0 / 0</div>
      </div>
      <div class="card">
        <div class="k">ETA</div>
        <div id="eta" class="v mono">-</div>
      </div>
      <div class="card">
        <div class="k">Tokens</div>
        <div id="tokens" class="v mono">0</div>
      </div>
      <div class="card">
        <div class="k">Token I/O</div>
        <div id="tokenIo" class="v mono">0 / 0</div>
      </div>
      <div class="card">
        <div class="k">Cost</div>
        <div id="cost" class="v mono">N/A</div>
      </div>
      <div class="card">
        <div class="k">Checker Edits</div>
        <div id="checker" class="v mono">0</div>
      </div>
      <div class="card">
        <div class="k">Checker Requests</div>
        <div id="checkerReq" class="v mono">0</div>
      </div>
      <div class="card">
        <div class="k">Checker I/O</div>
        <div id="checkerTokenIo" class="v mono">0 / 0</div>
      </div>
      <div class="card">
        <div class="k">Issues Total</div>
        <div id="issues" class="v mono">0</div>
      </div>
    </div>
    <div class="links">
      <a id="qaLink" class="btn" href="qa_report.html" target="_blank" rel="noreferrer">Open QA Report</a>
      <a id="jsonLink" class="btn" href="qa.jsonl" target="_blank" rel="noreferrer">Open QA JSONL</a>
      <a id="checkerLink" class="btn" href="checker_suggestions.json" target="_blank" rel="noreferrer">Open Checker Suggestions</a>
      <a id="checkerTraceLink" class="btn" href="checker_trace.jsonl" target="_blank" rel="noreferrer">Open Checker Trace</a>
      <button id="openFolderBtn" class="btn primary">Open Output Folder</button>
    </div>
    <div class="small">Server listens on localhost only. Refresh interval: 1s.</div>
  </div>
<script>
  function fmtSeconds(v) {
    if (v == null || Number.isNaN(v) || !Number.isFinite(v)) return "-";
    let s = Math.max(0, Math.round(v));
    const h = Math.floor(s / 3600); s -= h * 3600;
    const m = Math.floor(s / 60); s -= m * 60;
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  }

  function fmtNum(v) {
    if (v == null || Number.isNaN(v)) return "0";
    return Number(v).toLocaleString();
  }

  function fmtCost(cost, currency) {
    if (cost == null || Number.isNaN(cost)) return "N/A";
    const c = (currency || "USD").toUpperCase();
    return `${c} ${Number(cost).toFixed(4)}`;
  }

  async function refresh() {
    try {
      const resp = await fetch("/api/status", { cache: "no-store" });
      if (!resp.ok) throw new Error(`status ${resp.status}`);
      const s = await resp.json();
      const metrics = s.metrics || {};
      const usage = s.usage || {};
      const byPhase = usage.by_phase || {};
      const checkerUsage = byPhase.checker || {};
      const paths = s.paths || {};
      const done = Number(s.done_segments || 0);
      const total = Number(s.total_segments || 0);
      const progress = Number(metrics.progress_pct || (total > 0 ? (100 * done / total) : 0));
      const eta = metrics.eta_seconds;
      document.getElementById("phase").textContent = s.phase || "-";
      document.getElementById("progress").textContent = `${progress.toFixed(1)}%`;
      document.getElementById("bar").style.width = `${Math.max(0, Math.min(100, progress))}%`;
      document.getElementById("segments").textContent = `${fmtNum(done)} / ${fmtNum(total)}`;
      document.getElementById("eta").textContent = fmtSeconds(eta);
      document.getElementById("tokens").textContent = fmtNum(usage.total_tokens || 0);
      document.getElementById("tokenIo").textContent =
        `${fmtNum(usage.input_tokens || 0)} / ${fmtNum(usage.output_tokens || 0)}`;
      document.getElementById("cost").textContent = fmtCost(usage.cost, usage.currency);
      document.getElementById("checker").textContent = fmtNum(metrics.checker_suggestions || 0);
      document.getElementById("checkerReq").textContent = fmtNum(metrics.checker_requests_total || 0);
      document.getElementById("checkerTokenIo").textContent =
        `${fmtNum(checkerUsage.input_tokens || 0)} / ${fmtNum(checkerUsage.output_tokens || 0)}`;
      document.getElementById("issues").textContent = fmtNum(metrics.issues_total || 0);
      document.getElementById("updated").textContent = `updated: ${s.updated_at || "-"}`;
      if (paths.qa_report) document.getElementById("qaLink").setAttribute("href", paths.qa_report);
      if (paths.qa_jsonl) document.getElementById("jsonLink").setAttribute("href", paths.qa_jsonl);
      if (paths.checker_suggestions) document.getElementById("checkerLink").setAttribute("href", paths.checker_suggestions);
      if (paths.checker_trace) document.getElementById("checkerTraceLink").setAttribute("href", paths.checker_trace);
    } catch (err) {
      document.getElementById("updated").textContent = `status unavailable: ${err}`;
    }
  }

  document.getElementById("openFolderBtn").addEventListener("click", async () => {
    try {
      await fetch("/api/open-output", { method: "POST" });
    } catch (err) {
      console.error(err);
    }
  });

  refresh();
  setInterval(refresh, 1000);
</script>
</body>
</html>
"""


def ensure_dashboard_html(run_dir: Path, *, filename: str = "dashboard.html") -> Path:
    out = run_dir / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_build_dashboard_html(), encoding="utf-8")
    return out


def _resolve_open_path(run_dir: Path, status_path: Path) -> Path:
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            paths = payload.get("paths", {})
            if isinstance(paths, dict):
                output_file = paths.get("output")
                if isinstance(output_file, str) and output_file.strip():
                    out_path = Path(output_file).expanduser()
                    if out_path.is_absolute():
                        return out_path.parent
                    return (run_dir / out_path).resolve().parent
    except Exception:
        pass
    return run_dir


def _open_path(path: Path) -> None:
    if sys.platform.startswith("win"):
        subprocess.Popen(["explorer.exe", str(path)])
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
        return
    subprocess.Popen(["xdg-open", str(path)])


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, run_dir: Path, **kwargs):
        self._run_dir = run_dir
        self._status_path = run_dir / "run_status.json"
        super().__init__(*args, directory=str(run_dir), **kwargs)

    def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            if self._status_path.exists():
                try:
                    payload = json.loads(self._status_path.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        payload = {"error": "invalid status payload"}
                except Exception as exc:
                    payload = {"error": str(exc)}
            else:
                payload = {"phase": "waiting", "message": "run_status.json not found yet"}
            self._write_json(HTTPStatus.OK, payload)
            return
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/open-output":
            try:
                open_path = _resolve_open_path(self._run_dir, self._status_path)
                _open_path(open_path)
                self._write_json(HTTPStatus.OK, {"ok": True, "path": str(open_path)})
            except Exception as exc:
                self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})


def serve_dashboard(
    *,
    run_dir: Path,
    port: int = 0,
    open_browser: bool = True,
) -> None:
    run_dir = run_dir.resolve()
    ensure_dashboard_html(run_dir)

    def _handler(*args, **kwargs):
        return DashboardRequestHandler(*args, run_dir=run_dir, **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", int(port)), _handler)
    host, real_port = server.server_address
    url = f"http://{host}:{real_port}/dashboard.html"
    print(f"Dashboard server: {url}")
    print(f"Serving run dir: {run_dir}")
    if open_browser:
        with suppress(Exception):
            webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
