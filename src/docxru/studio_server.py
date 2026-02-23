from __future__ import annotations

import cgi
import io
import json
import os
import signal
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml
from docx import Document

from .checker import (
    apply_checker_suggestions_to_segments,
    filter_checker_suggestions,
    read_checker_suggestions,
    write_checker_safe_suggestions,
)
from .config import load_config
from .dashboard_server import ensure_dashboard_html
from .docx_reader import collect_segments
from .model_sizing import recommend_grouped_timeout_s, recommend_runtime_model_sizing
from .pipeline import _should_translate_segment_text
from .tagging import is_supported_paragraph, paragraph_to_tagged

DEFAULT_OPENAI_MODELS: tuple[str, ...] = (
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "o4-mini",
    "o3",
    "o3-mini",
    "o1",
    "o1-mini",
)

_OPENAI_MODEL_PREFIXES: tuple[str, ...] = ("gpt-", "o1", "o3", "o4")


def _now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def _default_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sanitize_filename(name: str, fallback: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return fallback
    safe = "".join(ch for ch in raw if ch.isalnum() or ch in {".", "_", "-", " "}).strip()
    safe = safe.replace(" ", "_")
    if not safe:
        return fallback
    if safe.startswith("."):
        safe = fallback
    return safe


def _to_bool(raw: Any) -> bool:
    text = str(raw or "").strip().lower()
    return text in {"1", "true", "yes", "on", "y"}


def _to_int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except Exception:
        return default


def _to_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except Exception:
        return default


def _infer_translation_cmd(source_path: Path) -> tuple[str, str]:
    ext = source_path.suffix.strip().lower()
    if ext == ".pdf":
        return "translate-pdf", "translated.pdf"
    return "translate", "translated.docx"


def _tail_lines(path: Path, *, max_lines: int = 120) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-max(1, int(max_lines)) :])


def _open_path(path: Path) -> None:
    if sys.platform.startswith("win"):
        subprocess.Popen(["explorer.exe", str(path)])
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
        return
    subprocess.Popen(["xdg-open", str(path)])


def _kill_process_tree(process: subprocess.Popen[str], *, wait_timeout_s: float = 8.0) -> tuple[bool, str]:
    if process.poll() is not None:
        return True, "already_exited"

    pid = int(getattr(process, "pid", 0) or 0)
    if pid <= 0:
        return False, "invalid_pid"

    if sys.platform.startswith("win"):
        with suppress(Exception):
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(1.0, float(wait_timeout_s)),
            )
        with suppress(Exception):
            process.wait(timeout=max(1.0, float(wait_timeout_s)))
        if process.poll() is not None:
            return True, "taskkill"
        with suppress(Exception):
            process.kill()
            process.wait(timeout=1.5)
        if process.poll() is not None:
            return True, "kill_fallback"
        return False, "timeout"

    # POSIX: process is started with start_new_session=True, so its process group is isolated.
    sent_group_term = False
    with suppress(Exception):
        pgid = os.getpgid(pid)
        if pgid > 0:
            os.killpg(pgid, signal.SIGTERM)
            sent_group_term = True
    if not sent_group_term:
        with suppress(Exception):
            process.terminate()
    with suppress(Exception):
        process.wait(timeout=2.0)
    if process.poll() is not None:
        return True, "sigterm_group" if sent_group_term else "terminate"

    sent_group_kill = False
    with suppress(Exception):
        pgid = os.getpgid(pid)
        if pgid > 0:
            os.killpg(pgid, signal.SIGKILL)
            sent_group_kill = True
    if not sent_group_kill:
        with suppress(Exception):
            process.kill()
    with suppress(Exception):
        process.wait(timeout=max(1.0, float(wait_timeout_s)))
    if process.poll() is not None:
        return True, "sigkill_group" if sent_group_kill else "kill"
    return False, "timeout"


def _list_openai_models(api_key: str | None = None) -> list[str]:
    key = (api_key or "").strip() or (os.environ.get("OPENAI_API_KEY", "").strip())
    if not key:
        return list(DEFAULT_OPENAI_MODELS)

    req = urllib.request.Request(
        url="https://api.openai.com/v1/models",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return list(DEFAULT_OPENAI_MODELS)

    items = data.get("data")
    if not isinstance(items, list):
        return list(DEFAULT_OPENAI_MODELS)
    model_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        low = model_id.lower()
        if any(low.startswith(prefix) for prefix in _OPENAI_MODEL_PREFIXES):
            model_ids.add(model_id)
    if not model_ids:
        return list(DEFAULT_OPENAI_MODELS)
    return sorted(model_ids)


def _default_model_for_provider(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p == "openai":
        return "gpt-5-mini"
    if p == "ollama":
        return "qwen2.5:7b"
    if p == "google":
        return "ignored"
    return "mock"


def _translation_grouping_profile(mode: str) -> tuple[str, dict[str, Any], float]:
    key = (mode or "").strip().lower()
    if key == "sequential_context":
        return (
            "sequential_context",
            {
                "batch_segments": 1,
                "batch_max_chars": 12000,
                "context_window_chars": 600,
                "auto_model_sizing": True,
            },
            0.03,
        )
    if key == "grouped_aggressive":
        # Aggressive mode increases throughput by allowing larger grouped requests.
        return (
            "grouped_aggressive",
            {
                "batch_segments": 20,
                "batch_max_chars": 36000,
                "context_window_chars": 0,
                "auto_model_sizing": True,
            },
            0.12,
        )
    if key == "grouped_turbo":
        return (
            "grouped_turbo",
            {
                "batch_segments": 80,
                "batch_max_chars": 120_000,
                "context_window_chars": 0,
                "auto_model_sizing": True,
            },
            0.20,
        )
    # Default studio profile prioritizes throughput for large manuals.
    return (
        "grouped_fast",
        {
            "batch_segments": 6,
            "batch_max_chars": 14000,
            "context_window_chars": 0,
            "auto_model_sizing": True,
        },
        0.08,
    )


def _estimate_grouped_request_count(source_char_lengths: list[int], *, max_segments: int, max_chars: int) -> int:
    if not source_char_lengths:
        return 0
    seg_limit = max(1, int(max_segments))
    char_limit = max(1, int(max_chars))
    groups = 0
    current_seg_count = 0
    current_chars = 0
    for source_len in source_char_lengths:
        estimated = max(1, int(source_len)) + 128
        if current_seg_count > 0 and (current_seg_count >= seg_limit or current_chars + estimated > char_limit):
            groups += 1
            current_seg_count = 0
            current_chars = 0
        current_seg_count += 1
        current_chars += estimated
    if current_seg_count > 0:
        groups += 1
    return groups


def _estimate_request_latency_bounds_seconds(
    provider: str,
    model: str,
    *,
    grouped_mode: bool,
    batch_max_chars: int = 36_000,
) -> tuple[float, float]:
    provider_norm = (provider or "").strip().lower()
    model_norm = (model or "").strip().lower()
    if provider_norm == "mock":
        return (0.01, 0.03)
    if provider_norm == "google":
        return (1.8, 5.0)
    if provider_norm == "ollama":
        return (3.0, 18.0) if grouped_mode else (2.0, 10.0)
    if provider_norm == "openai":
        low, high = (10.0, 24.0) if grouped_mode else (9.0, 17.0)
        if model_norm.startswith("gpt-5"):
            low += 1.0
            high += 3.0
    else:
        low, high = (6.0, 16.0) if grouped_mode else (4.0, 12.0)
    if grouped_mode and int(batch_max_chars) > 36_000:
        scale = min(4.0, float(batch_max_chars) / 36_000.0)
        low *= scale
        high *= scale
    return (low, high)


def _build_studio_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>docxru studio</title>
  <style>
    :root {
      --bg: #f3f5f8;
      --card: #fff;
      --line: #d5dce7;
      --text: #122033;
      --muted: #5f6f86;
      --accent: #1468de;
      --accent2: #129ddf;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Trebuchet MS", "Noto Sans", sans-serif;
      color: var(--text);
      background:
        radial-gradient(1000px 360px at 0% -20%, #dce7ff 0%, transparent 70%),
        radial-gradient(900px 320px at 100% -10%, #ffe5ca 0%, transparent 70%),
        var(--bg);
    }
    .wrap { max-width: 1220px; margin: 20px auto 60px; padding: 0 14px; display: grid; gap: 14px; grid-template-columns: 1.1fr 1fr; }
    @media (max-width: 1024px) { .wrap { grid-template-columns: 1fr; } }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 14px; box-shadow: 0 8px 18px rgba(9, 30, 62, .05); }
    h1 { margin: 0 0 8px; font-size: 27px; letter-spacing: .3px; }
    .muted { color: var(--muted); font-size: 13px; }
    .grid2 { display: grid; gap: 10px; grid-template-columns: 1fr 1fr; }
    .grid3 { display: grid; gap: 10px; grid-template-columns: 1fr 1fr 1fr; }
    @media (max-width: 720px) { .grid2, .grid3 { grid-template-columns: 1fr; } }
    label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; text-transform: uppercase; letter-spacing: .08em; }
    input[type="text"], input[type="number"], input[type="password"], select {
      width: 100%; padding: 9px 10px; border-radius: 10px; border: 1px solid var(--line); background: #fff;
    }
    input[type="file"] { width: 100%; }
    .row { margin-bottom: 10px; }
    .checkline { display: flex; align-items: center; gap: 8px; margin-top: 8px; margin-bottom: 10px; }
    .checkline label { margin: 0; text-transform: none; letter-spacing: normal; font-size: 14px; color: var(--text); }
    .btns { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }
    button, a.btn {
      border: 1px solid var(--line); border-radius: 10px; padding: 9px 12px; background: #fff; color: var(--text); cursor: pointer;
      font-weight: 600; text-decoration: none;
    }
    button.primary { border-color: #0a57d6; background: linear-gradient(90deg, var(--accent), var(--accent2)); color: #fff; }
    button.danger { border-color: #b71f35; background: linear-gradient(90deg, #d8314b, #b91e36); color: #fff; }
    .status-kv { display: grid; grid-template-columns: 150px 1fr; gap: 7px 10px; font-size: 14px; }
    .k { color: var(--muted); }
    .mono { font-family: Consolas, "SFMono-Regular", Menlo, monospace; }
    pre {
      margin: 0; padding: 10px; border: 1px solid var(--line); border-radius: 10px; background: #0d1524; color: #d7deec;
      font-size: 12px; max-height: 360px; overflow: auto;
    }
    .bar { width: 100%; height: 12px; border-radius: 999px; background: #e4ebf6; overflow: hidden; margin-top: 6px; }
    .bar > i { display: block; height: 100%; width: 0%; background: linear-gradient(90deg, #1d66df, #11a1df); transition: width .3s ease; }
    .pill { display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; border: 1px solid var(--line); background: #f8fbff; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>docxru Studio</h1>
      <div class="muted">Run translation from browser: file + glossary + prompt + model selection for translate and checker.</div>
      <form id="runForm">
        <div class="row">
          <label>Source File (.docx/.pdf)</label>
          <input type="file" name="input_file" accept=".docx,.pdf,application/pdf" required />
        </div>
        <div class="grid2">
          <div class="row">
            <label>Glossary File (optional)</label>
            <input type="file" name="glossary_file" accept=".md,.txt,.csv,.json" />
          </div>
          <div class="row">
            <label>System Prompt File (optional)</label>
            <input type="file" name="prompt_file" accept=".md,.txt" />
          </div>
        </div>

        <div class="grid3">
          <div class="row">
            <label>Translate Provider</label>
            <select name="provider" id="providerSelect">
              <option value="openai">openai</option>
              <option value="ollama">ollama</option>
              <option value="google">google</option>
              <option value="mock">mock</option>
            </select>
          </div>
          <div id="translateModelRow" class="row">
            <label>Translate Model (OpenAI)</label>
            <select id="openaiModelSelect">
              <option value="gpt-5-mini">gpt-5-mini</option>
            </select>
            <div class="btns" style="margin-top:6px;">
              <button type="button" id="refreshModelsBtn">Refresh OpenAI models</button>
            </div>
          </div>
          <div class="row">
            <label>Concurrency</label>
            <input type="number" name="concurrency" value="4" min="1" max="64" />
          </div>
        </div>
        <div class="grid2">
          <div class="row">
            <label>Translation Request Grouping (docxru)</label>
            <select name="translation_grouping_mode" id="translationGroupingMode">
              <option value="grouped_turbo" selected>Grouped Requests (turbo, large-context models)</option>
              <option value="grouped_fast">Grouped Requests (recommended, faster)</option>
              <option value="grouped_aggressive">Grouped Requests (aggressive, fastest)</option>
              <option value="sequential_context">Sequential Context Window (slower, max continuity)</option>
            </select>
            <div class="muted">This controls docxru translation request grouping and is separate from OpenAI Batch API.</div>
          </div>
        </div>
        <input type="hidden" name="model" id="modelHidden" value="gpt-5-mini" />

        <div class="grid3">
          <div class="row">
            <label>Temperature</label>
            <input type="number" name="temperature" value="0.1" step="0.1" min="0" max="2" />
          </div>
          <div class="row">
            <label>Max Output Tokens</label>
            <input type="number" name="max_output_tokens" value="2000" min="64" />
          </div>
          <div class="row">
            <label>OpenAI API Key (optional)</label>
            <input type="password" name="openai_api_key" placeholder="sk-..." />
          </div>
        </div>

        <div class="checkline">
          <input id="checkerEnabled" type="checkbox" name="checker_enabled" value="1" />
          <label for="checkerEnabled">Enable LLM checker</label>
        </div>

        <div id="checkerBlock" style="display:none">
          <div class="grid3">
            <div class="row">
              <label>Checker Provider</label>
              <select name="checker_provider" id="checkerProviderSelect">
                <option value="">(same as translate)</option>
                <option value="openai">openai</option>
                <option value="ollama">ollama</option>
                <option value="google">google</option>
                <option value="mock">mock</option>
              </select>
            </div>
            <div id="checkerModelRow" class="row">
              <label>Checker Model (OpenAI)</label>
              <select id="checkerOpenaiModelSelect">
                <option value="gpt-5-mini">gpt-5-mini</option>
              </select>
            </div>
            <div class="row">
              <label>Pages Per Checker Chunk</label>
              <input type="number" name="checker_pages_per_chunk" value="3" min="1" max="20" />
            </div>
          </div>
          <input type="hidden" name="checker_model" id="checkerModelHidden" value="" />
          <div class="grid3">
            <div class="row">
              <label>DOCX Fallback Segments Per Chunk</label>
              <input type="number" name="checker_fallback_segments_per_chunk" value="80" min="1" />
            </div>
            <div class="row">
              <label>Checker Temperature</label>
              <input type="number" name="checker_temperature" value="0.0" step="0.1" min="0" max="2" />
            </div>
            <div class="row">
              <label>Checker Max Output Tokens</label>
              <input type="number" name="checker_max_output_tokens" value="6000" min="64" />
            </div>
          </div>
          <div id="checkerOpenaiBatchRow" class="checkline" style="margin-top:6px; display:none;">
            <input id="checkerOpenaiBatch" type="checkbox" name="checker_openai_batch" value="1" />
            <label for="checkerOpenaiBatch">Use OpenAI Batch API for checker only (async/night mode, separate from translation grouping)</label>
          </div>
          <div class="checkline" style="margin-top:6px;">
            <input id="checkerAutoApplySafe" type="checkbox" name="checker_auto_apply_safe" value="1" />
            <label for="checkerAutoApplySafe">Auto-apply safe checker edits to output DOCX</label>
          </div>
          <div class="row" style="max-width:280px;">
            <label>Checker Auto-Apply Min Confidence</label>
            <input type="number" name="checker_auto_apply_min_confidence" value="0.7" step="0.05" min="0" max="1" />
          </div>
        </div>

        <div class="btns">
          <button id="startBtn" type="submit" class="primary">Start Translation</button>
          <button id="stopRunBtn" type="button" class="danger" disabled>Stop Translation</button>
          <button id="estimateBtn" type="button">Estimate Duration</button>
          <button id="openRunBtn" type="button">Open Run Folder</button>
          <span id="runPill" class="pill">no run</span>
        </div>
        <div id="estimateHint" class="muted" style="margin-top:8px;">Estimate not calculated yet.</div>
      </form>
    </div>

    <div class="card">
      <div class="status-kv">
        <div class="k">State</div><div id="state">idle</div>
        <div class="k">Run ID</div><div id="runId" class="mono">-</div>
        <div class="k">Phase</div><div id="phase">-</div>
        <div class="k">Progress</div><div id="progress">0%</div>
        <div class="k">Segments</div><div id="segments" class="mono">0 / 0</div>
        <div class="k">ETA</div><div id="eta" class="mono">-</div>
        <div class="k">Tokens</div><div id="tokens" class="mono">0</div>
        <div class="k">Token I/O</div><div id="tokenIo" class="mono">0 / 0</div>
        <div class="k">Cost</div><div id="cost" class="mono">N/A</div>
        <div class="k">Checker Token I/O</div><div id="checkerTokenIo" class="mono">0 / 0</div>
        <div class="k">Checker Req</div><div id="checkerReq" class="mono">0 / 0 / 0</div>
        <div class="k">Checker Edits</div><div id="checkerEdits" class="mono">0</div>
        <div class="k">Return Code</div><div id="returnCode">-</div>
      </div>
      <div class="bar"><i id="bar"></i></div>
      <div class="btns" style="margin-top:10px;">
        <a id="dashboardLink" class="btn" href="#" target="_blank" rel="noreferrer">Dashboard</a>
        <a id="qaLink" class="btn" href="#" target="_blank" rel="noreferrer">QA Report</a>
        <a id="checkerLink" class="btn" href="#" target="_blank" rel="noreferrer">Checker JSON</a>
        <a id="checkerSafeLink" class="btn" href="#" target="_blank" rel="noreferrer">Checker Safe JSON</a>
        <a id="checkerTraceLink" class="btn" href="#" target="_blank" rel="noreferrer">Checker Trace</a>
        <a id="outputLink" class="btn" href="#" target="_blank" rel="noreferrer">Output</a>
        <a id="checkedOutputLink" class="btn" href="#" target="_blank" rel="noreferrer">Checked Output</a>
        <button id="applyCheckerBtn" type="button">Apply Checker (Safe)</button>
      </div>
      <div class="muted" style="margin:10px 0 4px;">Live log tail</div>
      <pre id="logTail">(no log yet)</pre>
      <div class="muted" style="margin:10px 0 4px;">Checker trace tail</div>
      <pre id="checkerTraceTail">(no checker trace yet)</pre>
    </div>
  </div>

<script>
  let currentRunId = "";

  const DEFAULT_OPENAI_MODELS = [
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "o4-mini",
    "o3",
    "o3-mini",
    "o1",
    "o1-mini"
  ];

  const providerSelect = document.getElementById("providerSelect");
  const translationGroupingMode = document.getElementById("translationGroupingMode");
  const openaiModelSelect = document.getElementById("openaiModelSelect");
  const translateModelRow = document.getElementById("translateModelRow");
  const modelHidden = document.getElementById("modelHidden");
  const refreshModelsBtn = document.getElementById("refreshModelsBtn");
  const runForm = document.getElementById("runForm");
  const sourceFileInput = document.querySelector("input[name='input_file']");
  const stopRunBtn = document.getElementById("stopRunBtn");
  const estimateBtn = document.getElementById("estimateBtn");
  const estimateHint = document.getElementById("estimateHint");
  const checkerEnabled = document.getElementById("checkerEnabled");
  const checkerBlock = document.getElementById("checkerBlock");
  const checkerProviderSelect = document.getElementById("checkerProviderSelect");
  const checkerModelRow = document.getElementById("checkerModelRow");
  const checkerOpenaiModelSelect = document.getElementById("checkerOpenaiModelSelect");
  const checkerModelHidden = document.getElementById("checkerModelHidden");
  const checkerOpenaiBatchRow = document.getElementById("checkerOpenaiBatchRow");
  const checkerOpenaiBatch = document.getElementById("checkerOpenaiBatch");
  const openaiApiKeyInput = document.querySelector("input[name='openai_api_key']");
  const applyCheckerBtn = document.getElementById("applyCheckerBtn");

  function defaultModelForProvider(provider) {
    const p = (provider || "").toLowerCase();
    if (p === "openai") return "gpt-5-mini";
    if (p === "ollama") return "qwen2.5:7b";
    if (p === "google") return "ignored";
    return "mock";
  }

  function fillModelSelect(selectEl, models, preferredValue) {
    const values = Array.from(new Set((models || []).filter(Boolean)));
    selectEl.innerHTML = "";
    for (const model of values) {
      const opt = document.createElement("option");
      opt.value = model;
      opt.textContent = model;
      selectEl.appendChild(opt);
    }
    if (!values.length) {
      const fallback = document.createElement("option");
      fallback.value = "gpt-5-mini";
      fallback.textContent = "gpt-5-mini";
      selectEl.appendChild(fallback);
    }
    const preferred = preferredValue || selectEl.value || "gpt-5-mini";
    selectEl.value = preferred;
    if (!selectEl.value) {
      selectEl.value = "gpt-5-mini";
    }
  }

  function effectiveCheckerProvider() {
    return (checkerProviderSelect.value || providerSelect.value || "").toLowerCase();
  }

  function syncTranslateModelUI() {
    const provider = (providerSelect.value || "").toLowerCase();
    const isOpenAI = provider === "openai";
    translateModelRow.style.display = isOpenAI ? "" : "none";
    if (isOpenAI) {
      modelHidden.value = openaiModelSelect.value || "gpt-5-mini";
    } else {
      modelHidden.value = defaultModelForProvider(provider);
    }
  }

  function syncCheckerUI() {
    const enabled = checkerEnabled.checked;
    checkerBlock.style.display = enabled ? "block" : "none";
    if (!enabled) {
      checkerModelHidden.value = "";
      return;
    }
    const provider = effectiveCheckerProvider();
    const isOpenAI = provider === "openai";
    checkerModelRow.style.display = isOpenAI ? "" : "none";
    checkerOpenaiBatchRow.style.display = isOpenAI ? "flex" : "none";
    if (!isOpenAI) {
      checkerOpenaiBatch.checked = false;
    }
    if (isOpenAI) {
      checkerModelHidden.value = checkerOpenaiModelSelect.value || (openaiModelSelect.value || "gpt-5-mini");
    } else {
      checkerModelHidden.value = "";
    }
  }

  async function refreshOpenAIModels() {
    const apiKey = (openaiApiKeyInput.value || "").trim();
    let models = DEFAULT_OPENAI_MODELS;
    try {
      const resp = await fetch("/api/openai-models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: apiKey })
      });
      if (resp.ok) {
        const data = await resp.json();
        if (data && Array.isArray(data.models) && data.models.length) {
          models = data.models;
        }
      }
    } catch (err) {
      console.warn(err);
    }
    fillModelSelect(openaiModelSelect, models, openaiModelSelect.value || "gpt-5-mini");
    fillModelSelect(checkerOpenaiModelSelect, models, checkerOpenaiModelSelect.value || "gpt-5-mini");
    syncTranslateModelUI();
    syncCheckerUI();
  }

  checkerEnabled.addEventListener("change", syncCheckerUI);
  checkerProviderSelect.addEventListener("change", syncCheckerUI);
  providerSelect.addEventListener("change", () => {
    syncTranslateModelUI();
    syncCheckerUI();
    estimateHint.textContent = "Estimate may have changed; click Estimate Duration.";
  });
  translationGroupingMode.addEventListener("change", () => {
    estimateHint.textContent = "Estimate may have changed; click Estimate Duration.";
  });
  openaiModelSelect.addEventListener("change", () => {
    syncTranslateModelUI();
    if (!checkerOpenaiModelSelect.value) {
      checkerOpenaiModelSelect.value = openaiModelSelect.value;
    }
    syncCheckerUI();
    estimateHint.textContent = "Estimate may have changed; click Estimate Duration.";
  });
  checkerOpenaiModelSelect.addEventListener("change", () => {
    syncCheckerUI();
    estimateHint.textContent = "Estimate may have changed; click Estimate Duration.";
  });
  sourceFileInput.addEventListener("change", () => {
    estimateHint.textContent = "File selected. Click Estimate Duration.";
  });
  refreshModelsBtn.addEventListener("click", refreshOpenAIModels);
  estimateBtn.addEventListener("click", estimateDuration);

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

  function fmtCost(v, currency) {
    if (v == null || Number.isNaN(v)) return "N/A";
    return `${(currency || "USD").toUpperCase()} ${Number(v).toFixed(4)}`;
  }

  function syncRunButtons(state) {
    const s = (state || "").toLowerCase();
    const canStop = Boolean(currentRunId) && (s === "running" || s === "stopping");
    stopRunBtn.disabled = !canStop;
    stopRunBtn.textContent = s === "stopping" ? "Stopping..." : "Stop Translation";
  }

  async function estimateDuration() {
    if (!sourceFileInput || !sourceFileInput.files || !sourceFileInput.files.length) {
      estimateHint.textContent = "Select a source file to estimate duration.";
      return;
    }
    syncTranslateModelUI();
    syncCheckerUI();
    const fd = new FormData(runForm);
    estimateBtn.disabled = true;
    estimateBtn.textContent = "Estimating...";
    try {
      const resp = await fetch("/api/estimate", { method: "POST", body: fd });
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        estimateHint.textContent = data.error || `Estimate failed: ${resp.status}`;
        return;
      }
      const est = data.estimation || {};
      if ((est.source_kind || "") !== "docx") {
        estimateHint.textContent = est.note || "Estimate is currently available for DOCX only.";
        return;
      }
      const grouped = Boolean(est.grouped_mode_effective);
      const modeLabel = grouped ? "Grouped requests" : "Sequential context";
      const etaLow = fmtSeconds(Number(est.eta_seconds_low || 0));
      const etaHigh = fmtSeconds(Number(est.eta_seconds_high || 0));
      const autoSizing = Boolean(est.auto_model_sizing_effective);
      const fallbackWarnPct = Math.round(Number(est.batch_fallback_warn_ratio || 0) * 1000) / 10;
      estimateHint.textContent =
        `${modeLabel}: ~${etaLow} to ${etaHigh}. ` +
        `LLM segments ~${fmtNum(est.llm_segments_estimate || 0)}, ` +
        `requests ~${fmtNum(est.request_count_estimate || 0)}, ` +
        `effective batch ${fmtNum(est.effective_batch_segments || 1)}x${fmtNum(est.effective_batch_max_chars || 0)} chars, ` +
        `auto sizing ${autoSizing ? "on" : "off"}, ` +
        `fallback warn threshold ${fallbackWarnPct}%.`;
    } catch (err) {
      estimateHint.textContent = String(err);
    } finally {
      estimateBtn.disabled = false;
      estimateBtn.textContent = "Estimate Duration";
    }
  }

  async function refreshStatus() {
    if (!currentRunId) return;
    try {
      const resp = await fetch(`/api/status?run_id=${encodeURIComponent(currentRunId)}`, { cache: "no-store" });
      if (!resp.ok) return;
      const data = await resp.json();
      const status = data.status || {};
      const metrics = status.metrics || {};
      const usage = status.usage || {};
      const byPhase = usage.by_phase || {};
      const checkerUsage = byPhase.checker || {};
      const done = Number(status.done_segments || 0);
      const total = Number(status.total_segments || 0);
      const pct = Number(metrics.progress_pct || (total > 0 ? (100 * done / total) : 0));
      const checkerReqTotal = Number(metrics.checker_requests_total || 0);
      const checkerReqOk = Number(metrics.checker_requests_succeeded || 0);
      const checkerReqFail = Number(metrics.checker_requests_failed || 0);
      document.getElementById("state").textContent = data.state || "-";
      document.getElementById("runId").textContent = data.run_id || "-";
      document.getElementById("phase").textContent = status.phase || "-";
      document.getElementById("progress").textContent = `${pct.toFixed(1)}%`;
      document.getElementById("segments").textContent = `${fmtNum(done)} / ${fmtNum(total)}`;
      document.getElementById("eta").textContent = fmtSeconds(metrics.eta_seconds);
      document.getElementById("tokens").textContent = fmtNum(usage.total_tokens || 0);
      document.getElementById("tokenIo").textContent =
        `${fmtNum(usage.input_tokens || 0)} / ${fmtNum(usage.output_tokens || 0)}`;
      document.getElementById("cost").textContent = fmtCost(usage.cost, usage.currency);
      document.getElementById("checkerTokenIo").textContent =
        `${fmtNum(checkerUsage.input_tokens || 0)} / ${fmtNum(checkerUsage.output_tokens || 0)}`;
      document.getElementById("checkerReq").textContent =
        `${fmtNum(checkerReqTotal)} / ${fmtNum(checkerReqOk)} / ${fmtNum(checkerReqFail)}`;
      document.getElementById("checkerEdits").textContent = fmtNum(metrics.checker_suggestions || 0);
      document.getElementById("returnCode").textContent = data.return_code == null ? "-" : String(data.return_code);
      document.getElementById("bar").style.width = `${Math.max(0, Math.min(100, pct))}%`;
      document.getElementById("logTail").textContent = data.log_tail || "(no log yet)";
      document.getElementById("checkerTraceTail").textContent = data.checker_trace_tail || "(no checker trace yet)";
      const links = data.links || {};
      if (links.dashboard) document.getElementById("dashboardLink").href = links.dashboard;
      if (links.qa_report) document.getElementById("qaLink").href = links.qa_report;
      if (links.checker_suggestions) document.getElementById("checkerLink").href = links.checker_suggestions;
      if (links.checker_suggestions_safe) document.getElementById("checkerSafeLink").href = links.checker_suggestions_safe;
      if (links.checker_trace) document.getElementById("checkerTraceLink").href = links.checker_trace;
      if (links.output) document.getElementById("outputLink").href = links.output;
      if (links.checked_output) document.getElementById("checkedOutputLink").href = links.checked_output;
      const canApplyChecker = Boolean(currentRunId)
        && String(data.state || "").toLowerCase() === "completed"
        && Number(metrics.checker_suggestions || 0) > 0;
      applyCheckerBtn.disabled = !canApplyChecker;
      syncRunButtons(data.state || "");
    } catch (err) {
      console.error(err);
    }
  }

  runForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    syncTranslateModelUI();
    syncCheckerUI();
    const form = e.currentTarget;
    const fd = new FormData(form);
    const btn = document.getElementById("startBtn");
    btn.disabled = true;
    btn.textContent = "Starting...";
    try {
      const resp = await fetch("/api/start", { method: "POST", body: fd });
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        alert(data.error || `Start failed: ${resp.status}`);
        return;
      }
      currentRunId = data.run_id;
      document.getElementById("runPill").textContent = `run: ${currentRunId}`;
      await refreshStatus();
    } catch (err) {
      alert(String(err));
    } finally {
      btn.disabled = false;
      btn.textContent = "Start Translation";
    }
  });

  document.getElementById("openRunBtn").addEventListener("click", async () => {
    if (!currentRunId) return;
    try {
      await fetch(`/api/open-run?run_id=${encodeURIComponent(currentRunId)}`, { method: "POST" });
    } catch (err) {
      console.error(err);
    }
  });

  applyCheckerBtn.addEventListener("click", async () => {
    if (!currentRunId) return;
    applyCheckerBtn.disabled = true;
    applyCheckerBtn.textContent = "Applying...";
    try {
      const resp = await fetch(`/api/apply-checker?run_id=${encodeURIComponent(currentRunId)}&mode=safe`, {
        method: "POST"
      });
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        alert(data.error || `Apply checker failed: ${resp.status}`);
        return;
      }
      const summary = data.summary || {};
      estimateHint.textContent =
        `Checker apply done: applied ${fmtNum(summary.applied || 0)} / ${fmtNum(summary.requested || 0)} edits.`;
      await refreshStatus();
    } catch (err) {
      alert(String(err));
    } finally {
      applyCheckerBtn.textContent = "Apply Checker (Safe)";
      await refreshStatus();
    }
  });

  stopRunBtn.addEventListener("click", async () => {
    if (!currentRunId) return;
    const currentState = (document.getElementById("state").textContent || "").toLowerCase();
    if (currentState !== "running" && currentState !== "stopping") return;
    const confirmed = window.confirm("Force stop current translation and kill process tree?");
    if (!confirmed) return;
    stopRunBtn.disabled = true;
    stopRunBtn.textContent = "Stopping...";
    try {
      const resp = await fetch(`/api/stop-run?run_id=${encodeURIComponent(currentRunId)}`, { method: "POST" });
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        alert(data.error || data.message || `Stop failed: ${resp.status}`);
      } else {
        estimateHint.textContent = "Stop requested. Translation process kill signal sent.";
      }
      await refreshStatus();
    } catch (err) {
      alert(String(err));
      await refreshStatus();
    }
  });

  fillModelSelect(openaiModelSelect, DEFAULT_OPENAI_MODELS, "gpt-5-mini");
  fillModelSelect(checkerOpenaiModelSelect, DEFAULT_OPENAI_MODELS, "gpt-5-mini");
  syncTranslateModelUI();
  syncCheckerUI();
  syncRunButtons("idle");
  applyCheckerBtn.disabled = true;
  refreshOpenAIModels();

  setInterval(refreshStatus, 1000);
</script>
</body>
</html>
"""


@dataclass
class StudioRun:
    run_id: str
    run_dir: Path
    source_path: Path
    output_path: Path
    config_path: Path
    log_path: Path
    status_path: Path
    command: list[str]
    process: subprocess.Popen[str]
    started_at: str
    stop_requested_at: str | None = None
    stop_completed_at: str | None = None
    stop_method: str | None = None


class StudioRunManager:
    def __init__(self, *, base_dir: Path) -> None:
        self.base_dir = base_dir.resolve()
        self.runs_dir = self.base_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._runs: dict[str, StudioRun] = {}

    def _read_file_field(self, form: cgi.FieldStorage, field_name: str, fallback_name: str) -> tuple[str, bytes] | None:
        field = form[field_name] if field_name in form else None
        if field is None:
            return None
        if isinstance(field, list):
            field = field[0]
        if not getattr(field, "file", None):
            return None
        payload = field.file.read()
        if not payload:
            return None
        name = _sanitize_filename(getattr(field, "filename", "") or "", fallback_name)
        return name, payload

    def _read_text_value(self, form: cgi.FieldStorage, field_name: str, default: str = "") -> str:
        field = form[field_name] if field_name in form else None
        if field is None:
            return default
        if isinstance(field, list):
            field = field[0]
        value = getattr(field, "value", default)
        return str(value or default).strip()

    def _build_config_payload(
        self,
        *,
        provider: str,
        model: str,
        temperature: float,
        max_output_tokens: int,
        concurrency: int,
        translation_grouping_mode: str,
        prompt_path: Path | None,
        glossary_path: Path | None,
        checker_enabled: bool,
        checker_provider: str | None,
        checker_model: str | None,
        checker_pages_per_chunk: int,
        checker_fallback_segments_per_chunk: int,
        checker_temperature: float,
        checker_max_output_tokens: int,
        checker_openai_batch_enabled: bool,
        checker_auto_apply_safe: bool,
        checker_auto_apply_min_confidence: float,
        run_base_dir: Path,
        run_id: str,
        run_dir: Path,
    ) -> dict[str, Any]:
        repo_root = _repo_root()
        preset_path = repo_root / "config" / "regex_presets.yaml"
        tm_path = repo_root / "translation_cache.sqlite"
        _grouping_mode, grouping_profile, batch_fallback_warn_ratio = _translation_grouping_profile(
            translation_grouping_mode
        )
        timeout_s = recommend_grouped_timeout_s(
            timeout_s=60.0,
            batch_segments=int(grouping_profile.get("batch_segments", 1) or 1),
            batch_max_chars=int(grouping_profile.get("batch_max_chars", 0) or 0),
        )

        payload: dict[str, Any] = {
            "llm": {
                "provider": provider,
                "model": model,
                "source_lang": "en",
                "target_lang": "ru",
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "retries": 2,
                "timeout_s": timeout_s,
                "system_prompt_path": (str(prompt_path) if prompt_path is not None else None),
                "glossary_path": (str(glossary_path) if glossary_path is not None else None),
                "glossary_prompt_mode": "matched",
                "structured_output_mode": "auto",
                **grouping_profile,
            },
            "tm": {
                "path": str(tm_path),
            },
            "checker": {
                "enabled": checker_enabled,
                "provider": (checker_provider or None),
                "model": (checker_model or None),
                "pages_per_chunk": max(1, checker_pages_per_chunk),
                "fallback_segments_per_chunk": max(1, checker_fallback_segments_per_chunk),
                "temperature": checker_temperature,
                "max_output_tokens": checker_max_output_tokens,
                "openai_batch_enabled": bool(checker_openai_batch_enabled),
                "auto_apply_safe": bool(checker_auto_apply_safe),
                "auto_apply_min_confidence": max(0.0, min(1.0, float(checker_auto_apply_min_confidence))),
                "only_on_issue_severities": ["warn", "error"],
                "output_path": "checker_suggestions.json",
                "safe_output_path": "checker_suggestions_safe.json",
            },
            "pricing": {
                "enabled": False,
                "currency": "USD",
            },
            "run": {
                "run_dir": str(run_base_dir),
                "run_id": run_id,
                "status_path": str(run_dir / "run_status.json"),
                "dashboard_html_path": str(run_dir / "dashboard.html"),
                "status_flush_every_n_segments": 5,
                "batch_fallback_warn_ratio": float(batch_fallback_warn_ratio),
                "fail_fast_on_translate_error": True,
            },
            "concurrency": max(1, concurrency),
            "mode": "reflow",
            "qa_report_path": "qa_report.html",
            "qa_jsonl_path": "qa.jsonl",
            "log_path": str(run_dir / "run.log"),
        }
        if preset_path.exists():
            payload["patterns"] = {
                "preset_file": str(preset_path),
                "preset_name": "default",
            }
        return payload

    def estimate_from_form(self, form: cgi.FieldStorage) -> dict[str, Any]:
        src = self._read_file_field(form, "input_file", "input.docx")
        if src is None:
            raise RuntimeError("Source file is required")
        source_name, source_bytes = src
        source_path = Path(source_name)
        cmd_name, _ = _infer_translation_cmd(source_path)
        if cmd_name != "translate":
            return {
                "ok": True,
                "estimation": {
                    "source_kind": "pdf",
                    "note": "Pre-run ETA is currently available for DOCX only.",
                },
            }

        from docx import Document  # Local import keeps studio startup lightweight.

        doc = Document(io.BytesIO(source_bytes))
        segments = collect_segments(doc, include_headers=False, include_footers=False)

        skip_non_latin = 0
        toc_in_place = 0
        complex_in_place = 0
        llm_segments = 0
        source_lengths: list[int] = []
        for seg in segments:
            if not _should_translate_segment_text(seg.source_plain):
                skip_non_latin += 1
                continue
            if seg.context.get("is_toc_entry"):
                toc_in_place += 1
                continue
            if not is_supported_paragraph(seg.paragraph_ref):
                complex_in_place += 1
                continue
            llm_segments += 1
            source_lengths.append(len(seg.source_plain or ""))

        provider = self._read_text_value(form, "provider", "openai") or "openai"
        model_raw = self._read_text_value(form, "model", "")
        model = model_raw or _default_model_for_provider(provider)
        max_output_tokens = max(64, _to_int(self._read_text_value(form, "max_output_tokens", "2000"), 2000))
        translation_grouping_mode = self._read_text_value(form, "translation_grouping_mode", "grouped_turbo")
        _mode_key, grouping, batch_fallback_warn_ratio = _translation_grouping_profile(translation_grouping_mode)

        checker_provider_raw = self._read_text_value(form, "checker_provider", "")
        checker_provider = checker_provider_raw or provider
        checker_model_raw = self._read_text_value(form, "checker_model", "")
        checker_model = checker_model_raw or model
        checker_pages_per_chunk = max(1, _to_int(self._read_text_value(form, "checker_pages_per_chunk", "3"), 3))
        checker_fallback_segments = max(
            1,
            _to_int(self._read_text_value(form, "checker_fallback_segments_per_chunk", "80"), 80),
        )
        checker_max_output_tokens = max(
            64,
            _to_int(self._read_text_value(form, "checker_max_output_tokens", "6000"), 6000),
        )

        prompt_field = self._read_file_field(form, "prompt_file", "system_prompt.md")
        glossary_field = self._read_file_field(form, "glossary_file", "glossary.md")
        prompt_chars = 0
        if prompt_field is not None:
            prompt_chars += len(prompt_field[1].decode("utf-8", errors="ignore"))
        if glossary_field is not None:
            prompt_chars += len(glossary_field[1].decode("utf-8", errors="ignore"))

        auto_model_sizing = bool(grouping.get("auto_model_sizing", True))
        effective_batch_segments = int(grouping["batch_segments"])
        effective_batch_max_chars = int(grouping["batch_max_chars"])
        effective_max_output_tokens = int(max_output_tokens)
        sizing_notes: list[str] = []
        if auto_model_sizing:
            sizing = recommend_runtime_model_sizing(
                provider=provider,
                model=model,
                checker_provider=checker_provider,
                checker_model=checker_model,
                source_char_lengths=source_lengths,
                prompt_chars=prompt_chars,
                batch_segments=effective_batch_segments,
                batch_max_chars=effective_batch_max_chars,
                max_output_tokens=max_output_tokens,
                context_window_chars=int(grouping["context_window_chars"]),
                checker_pages_per_chunk=checker_pages_per_chunk,
                checker_fallback_segments_per_chunk=checker_fallback_segments,
                checker_max_output_tokens=checker_max_output_tokens,
            )
            effective_batch_segments = int(sizing.batch_segments)
            effective_batch_max_chars = int(sizing.batch_max_chars)
            effective_max_output_tokens = int(sizing.max_output_tokens)
            sizing_notes = list(sizing.notes)
        else:
            sizing_notes = [
                (
                    f"translate profile manual override: batch={effective_batch_segments}x"
                    f"{effective_batch_max_chars}chars max_output_tokens={effective_max_output_tokens}"
                )
            ]

        provider_norm = provider.strip().lower()
        grouped_mode = (
            int(grouping["context_window_chars"]) <= 0
            and effective_batch_segments > 1
            and provider_norm in {"openai", "ollama"}
        )
        if grouped_mode:
            request_count = _estimate_grouped_request_count(
                source_lengths,
                max_segments=effective_batch_segments,
                max_chars=effective_batch_max_chars,
            )
        else:
            request_count = llm_segments

        prepare_seconds = max(2.0, len(segments) / 155.0)
        per_req_low, per_req_high = _estimate_request_latency_bounds_seconds(
            provider,
            model,
            grouped_mode=grouped_mode,
            batch_max_chars=effective_batch_max_chars,
        )
        eta_low = prepare_seconds + (request_count * per_req_low)
        eta_high = prepare_seconds + (request_count * per_req_high)

        return {
            "ok": True,
            "estimation": {
                "source_kind": "docx",
                "source_name": source_path.name,
                "segments_total": len(segments),
                "skip_no_latin_estimate": skip_non_latin,
                "toc_in_place_estimate": toc_in_place,
                "complex_in_place_estimate": complex_in_place,
                "llm_segments_estimate": llm_segments,
                "grouping_mode": translation_grouping_mode,
                "grouped_mode_effective": grouped_mode,
                "auto_model_sizing_effective": auto_model_sizing,
                "effective_batch_segments": int(effective_batch_segments),
                "effective_batch_max_chars": int(effective_batch_max_chars),
                "effective_max_output_tokens": int(effective_max_output_tokens),
                "batch_fallback_warn_ratio": float(batch_fallback_warn_ratio),
                "request_count_estimate": int(request_count),
                "eta_seconds_low": float(eta_low),
                "eta_seconds_high": float(eta_high),
                "notes": sizing_notes,
                "note": (
                    "Rough estimate before TM hits/resume/checker edits; real runtime depends on API latency."
                ),
            },
        }

    def start_from_form(self, form: cgi.FieldStorage) -> dict[str, Any]:
        src = self._read_file_field(form, "input_file", "input.docx")
        if src is None:
            raise RuntimeError("Source file is required")
        source_name, source_bytes = src
        run_id = _default_run_id()
        run_dir = self.runs_dir / run_id
        uploads_dir = run_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        source_path = uploads_dir / source_name
        source_path.write_bytes(source_bytes)

        glossary_field = self._read_file_field(form, "glossary_file", "glossary.md")
        glossary_path: Path | None = None
        if glossary_field is not None:
            glossary_name, glossary_bytes = glossary_field
            glossary_path = uploads_dir / glossary_name
            glossary_path.write_bytes(glossary_bytes)

        prompt_field = self._read_file_field(form, "prompt_file", "system_prompt.md")
        prompt_path: Path | None = None
        if prompt_field is not None:
            prompt_name, prompt_bytes = prompt_field
            prompt_path = uploads_dir / prompt_name
            prompt_path.write_bytes(prompt_bytes)

        provider = self._read_text_value(form, "provider", "openai") or "openai"
        model_raw = self._read_text_value(form, "model", "")
        model = model_raw or _default_model_for_provider(provider)
        temperature = _to_float(self._read_text_value(form, "temperature", "0.1"), 0.1)
        max_output_tokens = max(64, _to_int(self._read_text_value(form, "max_output_tokens", "2000"), 2000))
        concurrency = max(1, _to_int(self._read_text_value(form, "concurrency", "4"), 4))
        translation_grouping_mode = self._read_text_value(form, "translation_grouping_mode", "grouped_turbo")

        checker_enabled = _to_bool(self._read_text_value(form, "checker_enabled", "0"))
        checker_provider_raw = self._read_text_value(form, "checker_provider", "")
        checker_provider = checker_provider_raw or None
        checker_model_raw = self._read_text_value(form, "checker_model", "")
        effective_checker_provider = checker_provider or provider
        if checker_enabled:
            if checker_model_raw:
                checker_model = checker_model_raw
            elif (effective_checker_provider or "").strip().lower() == "openai":
                checker_model = "gpt-5-mini"
            else:
                checker_model = _default_model_for_provider(effective_checker_provider)
        else:
            checker_model = None
        checker_pages_per_chunk = max(1, _to_int(self._read_text_value(form, "checker_pages_per_chunk", "3"), 3))
        checker_fallback_segments = max(
            1,
            _to_int(self._read_text_value(form, "checker_fallback_segments_per_chunk", "80"), 80),
        )
        checker_temperature = _to_float(self._read_text_value(form, "checker_temperature", "0.0"), 0.0)
        checker_max_output_tokens = max(
            64,
            _to_int(self._read_text_value(form, "checker_max_output_tokens", "6000"), 6000),
        )
        checker_openai_batch_enabled = (
            checker_enabled
            and (effective_checker_provider or "").strip().lower() == "openai"
            and _to_bool(self._read_text_value(form, "checker_openai_batch", "0"))
        )
        checker_auto_apply_safe = checker_enabled and _to_bool(
            self._read_text_value(form, "checker_auto_apply_safe", "0")
        )
        checker_auto_apply_min_confidence = max(
            0.0,
            min(1.0, _to_float(self._read_text_value(form, "checker_auto_apply_min_confidence", "0.7"), 0.7)),
        )

        config_payload = self._build_config_payload(
            provider=provider,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            concurrency=concurrency,
            translation_grouping_mode=translation_grouping_mode,
            prompt_path=prompt_path,
            glossary_path=glossary_path,
            checker_enabled=checker_enabled,
            checker_provider=checker_provider,
            checker_model=checker_model,
            checker_pages_per_chunk=checker_pages_per_chunk,
            checker_fallback_segments_per_chunk=checker_fallback_segments,
            checker_temperature=checker_temperature,
            checker_max_output_tokens=checker_max_output_tokens,
            checker_openai_batch_enabled=checker_openai_batch_enabled,
            checker_auto_apply_safe=checker_auto_apply_safe,
            checker_auto_apply_min_confidence=checker_auto_apply_min_confidence,
            run_base_dir=self.runs_dir,
            run_id=run_id,
            run_dir=run_dir,
        )
        config_path = run_dir / "config.studio.yaml"
        config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

        cmd_name, output_name = _infer_translation_cmd(source_path)
        output_path = run_dir / output_name
        command = [
            sys.executable,
            "-m",
            "docxru",
            cmd_name,
            "--input",
            str(source_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
        ]

        env = os.environ.copy()
        openai_key = self._read_text_value(form, "openai_api_key", "")
        if openai_key:
            env["OPENAI_API_KEY"] = openai_key

        log_path = run_dir / "studio_process.log"
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(_repo_root()),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        log_file.close()

        status_path = run_dir / "run_status.json"
        ensure_dashboard_html(run_dir)
        run = StudioRun(
            run_id=run_id,
            run_dir=run_dir,
            source_path=source_path,
            output_path=output_path,
            config_path=config_path,
            log_path=log_path,
            status_path=status_path,
            command=command,
            process=process,
            started_at=_now_utc_iso(),
        )
        with self._lock:
            self._runs[run_id] = run

        return self.get_status(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            run_ids = sorted(self._runs.keys(), reverse=True)
        return [self.get_status(run_id) for run_id in run_ids]

    def get_status(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        return_code = run.process.poll()
        if run.stop_requested_at:
            state = "stopping" if return_code is None else "cancelled"
        elif return_code is None:
            state = "running"
        elif return_code == 0:
            state = "completed"
        else:
            state = "failed"

        status_payload: dict[str, Any] = {}
        if run.status_path.exists():
            with suppress(Exception):
                parsed = json.loads(run.status_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    status_payload = parsed

        links = {
            "run_dir": f"/runs/{run.run_id}/",
            "dashboard": f"/runs/{run.run_id}/dashboard.html",
            "qa_report": f"/runs/{run.run_id}/qa_report.html",
            "qa_jsonl": f"/runs/{run.run_id}/qa.jsonl",
            "checker_suggestions": f"/runs/{run.run_id}/checker_suggestions.json",
            "checker_suggestions_safe": f"/runs/{run.run_id}/checker_suggestions_safe.json",
            "checker_trace": f"/runs/{run.run_id}/checker_trace.jsonl",
            "output": f"/runs/{run.run_id}/{run.output_path.name}",
            "log": f"/runs/{run.run_id}/{run.log_path.name}",
        }
        checked_output = run.run_dir / f"{run.output_path.stem}_checked{run.output_path.suffix}"
        if checked_output.exists():
            links["checked_output"] = f"/runs/{run.run_id}/{checked_output.name}"
        checker_trace_path = run.run_dir / "checker_trace.jsonl"

        return {
            "ok": True,
            "run_id": run.run_id,
            "state": state,
            "return_code": return_code,
            "started_at": run.started_at,
            "stop_requested_at": run.stop_requested_at,
            "stop_completed_at": run.stop_completed_at,
            "stop_method": run.stop_method,
            "command": run.command,
            "run_dir": str(run.run_dir),
            "links": links,
            "status": status_payload,
            "log_tail": _tail_lines(run.log_path),
            "checker_trace_tail": _tail_lines(checker_trace_path),
        }

    def open_run_dir(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        _open_path(run.run_dir)
        return {"ok": True, "path": str(run.run_dir)}

    def apply_checker_suggestions(self, run_id: str, *, safe_only: bool = True) -> dict[str, Any]:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        if run.process.poll() is None:
            raise RuntimeError("Run is still in progress. Wait until completion before applying checker suggestions.")
        if run.output_path.suffix.lower() != ".docx":
            raise RuntimeError("Checker apply is currently supported for DOCX outputs only.")
        if not run.output_path.exists():
            raise RuntimeError(f"Translated output not found: {run.output_path}")

        checker_path = run.run_dir / "checker_suggestions.json"
        checker_safe_path = run.run_dir / "checker_suggestions_safe.json"
        if not checker_path.exists():
            raise RuntimeError(f"Checker suggestions file not found: {checker_path}")

        cfg = load_config(run.config_path)
        all_edits = read_checker_suggestions(checker_path)
        if safe_only:
            if checker_safe_path.exists():
                safe_edits = read_checker_suggestions(checker_safe_path)
            else:
                safe_edits, skipped = filter_checker_suggestions(
                    all_edits,
                    safe_only=True,
                    min_confidence=float(cfg.checker.auto_apply_min_confidence),
                )
                write_checker_safe_suggestions(
                    checker_safe_path,
                    source_edits=all_edits,
                    safe_edits=safe_edits,
                    skipped=skipped,
                )
            edits_to_apply = safe_edits
        else:
            edits_to_apply = all_edits

        doc = Document(str(run.output_path))
        segments = collect_segments(doc, include_headers=cfg.include_headers, include_footers=cfg.include_footers)
        for seg in segments:
            if seg.paragraph_ref is None:
                continue
            with suppress(Exception):
                tagged, spans, inline_map = paragraph_to_tagged(seg.paragraph_ref)
                seg.target_tagged = tagged
                seg.spans = spans
                seg.inline_run_map = inline_map

        summary = apply_checker_suggestions_to_segments(
            segments=segments,
            edits=edits_to_apply,
            safe_only=bool(safe_only),
            min_confidence=float(cfg.checker.auto_apply_min_confidence),
            require_current_match=True,
            logger=None,
        )
        checked_output = run.run_dir / f"{run.output_path.stem}_checked{run.output_path.suffix}"
        checked_output.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(checked_output))
        return {
            "ok": True,
            "run_id": run_id,
            "safe_only": bool(safe_only),
            "summary": summary,
            "checked_output": str(checked_output),
            "links": {
                "checked_output": f"/runs/{run_id}/{checked_output.name}",
            },
        }

    def stop_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise KeyError(run_id)

        if run.process.poll() is not None:
            status = self.get_status(run_id)
            return {
                "ok": True,
                "run_id": run_id,
                "already_finished": True,
                "stopped": False,
                "state": status.get("state"),
            }

        with self._lock:
            run.stop_requested_at = run.stop_requested_at or _now_utc_iso()

        stopped, method = _kill_process_tree(run.process)

        with self._lock:
            run.stop_method = method
            if stopped:
                run.stop_completed_at = _now_utc_iso()

        status = self.get_status(run_id)
        return {
            "ok": bool(stopped),
            "run_id": run_id,
            "already_finished": False,
            "stopped": bool(stopped),
            "state": status.get("state"),
            "method": method,
        }


class StudioRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, manager: StudioRunManager, **kwargs):
        self._manager = manager
        super().__init__(*args, directory=str(manager.base_dir), **kwargs)

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
        if parsed.path in {"/", "/studio"}:
            body = _build_studio_html().encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/runs":
            self._write_json(HTTPStatus.OK, {"ok": True, "runs": self._manager.list_runs()})
            return
        if parsed.path == "/api/status":
            params = parse_qs(parsed.query)
            run_id = (params.get("run_id", [""])[0] or "").strip()
            if not run_id:
                self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "run_id is required"})
                return
            try:
                payload = self._manager.get_status(run_id)
            except KeyError:
                self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown run_id: {run_id}"})
                return
            self._write_json(HTTPStatus.OK, payload)
            return
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/openai-models":
            content_len = int(self.headers.get("Content-Length", "0") or "0")
            payload_raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
            api_key = ""
            with suppress(Exception):
                payload = json.loads(payload_raw.decode("utf-8"))
                if isinstance(payload, dict):
                    api_key = str(payload.get("api_key") or "").strip()
            models = _list_openai_models(api_key)
            self._write_json(HTTPStatus.OK, {"ok": True, "models": models})
            return

        if parsed.path == "/api/estimate":
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "multipart/form-data expected"})
                return
            try:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": content_type,
                    },
                )
                payload = self._manager.estimate_from_form(form)
            except Exception as exc:
                self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
                return
            self._write_json(HTTPStatus.OK, payload)
            return

        if parsed.path == "/api/start":
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "multipart/form-data expected"})
                return
            try:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": content_type,
                    },
                )
                payload = self._manager.start_from_form(form)
            except Exception as exc:
                self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
                return
            self._write_json(HTTPStatus.OK, payload)
            return

        if parsed.path == "/api/stop-run":
            params = parse_qs(parsed.query)
            run_id = (params.get("run_id", [""])[0] or "").strip()
            if not run_id:
                self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "run_id is required"})
                return
            try:
                payload = self._manager.stop_run(run_id)
            except KeyError:
                self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown run_id: {run_id}"})
                return
            except Exception as exc:
                self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
                return
            code = HTTPStatus.OK if payload.get("ok", False) else HTTPStatus.INTERNAL_SERVER_ERROR
            self._write_json(code, payload)
            return

        if parsed.path == "/api/open-run":
            params = parse_qs(parsed.query)
            run_id = (params.get("run_id", [""])[0] or "").strip()
            if not run_id:
                self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "run_id is required"})
                return
            try:
                payload = self._manager.open_run_dir(run_id)
            except KeyError:
                self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown run_id: {run_id}"})
                return
            except Exception as exc:
                self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
                return
            self._write_json(HTTPStatus.OK, payload)
            return

        if parsed.path == "/api/apply-checker":
            params = parse_qs(parsed.query)
            run_id = (params.get("run_id", [""])[0] or "").strip()
            mode = (params.get("mode", ["safe"])[0] or "safe").strip().lower()
            if not run_id:
                self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "run_id is required"})
                return
            safe_only = mode != "all"
            try:
                payload = self._manager.apply_checker_suggestions(run_id, safe_only=safe_only)
            except KeyError:
                self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown run_id: {run_id}"})
                return
            except Exception as exc:
                self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})
                return
            self._write_json(HTTPStatus.OK, payload)
            return

        self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})


def serve_studio(
    *,
    base_dir: Path,
    port: int = 0,
    open_browser: bool = True,
) -> None:
    manager = StudioRunManager(base_dir=base_dir)

    def _handler(*args, **kwargs):
        return StudioRequestHandler(*args, manager=manager, **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", int(port)), _handler)
    host, real_port = server.server_address
    url = f"http://{host}:{real_port}/studio"
    print(f"Studio server: {url}")
    print(f"Base dir: {manager.base_dir}")
    print(f"Runs dir: {manager.runs_dir}")
    if open_browser:
        with suppress(Exception):
            webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
