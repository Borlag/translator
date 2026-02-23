from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_eta_seconds(done: int, total: int, elapsed_s: float) -> float | None:
    if done <= 0 or total <= done or elapsed_s <= 0:
        return None
    rate = done / elapsed_s
    if rate <= 0:
        return None
    left = total - done
    return left / rate


@dataclass
class RunStatusState:
    run_id: str
    started_at: str
    updated_at: str
    phase: str
    total_segments: int
    done_segments: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)


class RunStatusWriter:
    def __init__(
        self,
        *,
        path: Path,
        run_id: str,
        total_segments: int,
        flush_every_n_updates: int = 10,
    ) -> None:
        now = _utc_now_iso()
        self.path = path
        self.state = RunStatusState(
            run_id=run_id,
            started_at=now,
            updated_at=now,
            phase="init",
            total_segments=max(0, int(total_segments)),
        )
        self._lock = Lock()
        self._start_monotonic = time.monotonic()
        self._updates_since_flush = 0
        self._flush_every = max(1, int(flush_every_n_updates))

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.state.phase = str(phase)
            self.state.updated_at = _utc_now_iso()
            self._updates_since_flush += 1

    def set_done(self, done_segments: int) -> None:
        with self._lock:
            done = max(0, int(done_segments))
            self.state.done_segments = done
            self.state.updated_at = _utc_now_iso()
            elapsed = max(0.0, time.monotonic() - self._start_monotonic)
            eta_seconds = _safe_eta_seconds(done, self.state.total_segments, elapsed)
            self.state.metrics["elapsed_seconds"] = elapsed
            self.state.metrics["eta_seconds"] = eta_seconds
            total = self.state.total_segments
            self.state.metrics["progress_pct"] = (100.0 * done / total) if total > 0 else 0.0
            self._updates_since_flush += 1

    def merge_metrics(self, metrics: dict[str, Any]) -> None:
        with self._lock:
            self.state.metrics.update(metrics)
            self.state.updated_at = _utc_now_iso()
            self._updates_since_flush += 1

    def set_usage(self, usage: dict[str, Any]) -> None:
        with self._lock:
            self.state.usage = dict(usage)
            self.state.updated_at = _utc_now_iso()
            self._updates_since_flush += 1

    def merge_paths(self, paths: dict[str, str]) -> None:
        with self._lock:
            for key, value in paths.items():
                if value:
                    self.state.paths[str(key)] = str(value)
            self.state.updated_at = _utc_now_iso()
            self._updates_since_flush += 1

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id": self.state.run_id,
                "started_at": self.state.started_at,
                "updated_at": self.state.updated_at,
                "phase": self.state.phase,
                "total_segments": self.state.total_segments,
                "done_segments": self.state.done_segments,
                "metrics": dict(self.state.metrics),
                "usage": dict(self.state.usage),
                "paths": dict(self.state.paths),
            }

    def write(self, *, force: bool = False) -> None:
        with self._lock:
            if not force and self._updates_since_flush < self._flush_every:
                return
            payload = {
                "run_id": self.state.run_id,
                "started_at": self.state.started_at,
                "updated_at": self.state.updated_at,
                "phase": self.state.phase,
                "total_segments": self.state.total_segments,
                "done_segments": self.state.done_segments,
                "metrics": dict(self.state.metrics),
                "usage": dict(self.state.usage),
                "paths": dict(self.state.paths),
            }
            self._updates_since_flush = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

