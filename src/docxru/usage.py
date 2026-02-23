from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class UsageRecord:
    provider: str
    model: str
    phase: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float | None = None
    currency: str = "USD"
    ts: str = field(default_factory=_utc_now_iso)
    extra: dict[str, Any] = field(default_factory=dict)


class UsageTotals:
    def __init__(self) -> None:
        self._lock = Lock()
        self._records: list[UsageRecord] = []
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._cost_total = 0.0
        self._cost_known = 0
        self._currency = "USD"
        self._by_phase: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _phase_key(phase: str) -> str:
        key = str(phase or "").strip().lower()
        return key if key else "unknown"

    def add(self, record: UsageRecord) -> None:
        with self._lock:
            self._records.append(record)
            in_tokens = max(0, int(record.input_tokens))
            out_tokens = max(0, int(record.output_tokens))
            total_tokens = max(0, int(record.total_tokens))

            self._input_tokens += in_tokens
            self._output_tokens += out_tokens
            self._total_tokens += total_tokens
            if record.cost is not None:
                self._cost_total += float(record.cost)
                self._cost_known += 1
            if record.currency:
                self._currency = str(record.currency)

            phase_key = self._phase_key(record.phase)
            bucket = self._by_phase.get(phase_key)
            if bucket is None:
                bucket = {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_total": 0.0,
                    "cost_records": 0,
                    "currency": self._currency,
                }
                self._by_phase[phase_key] = bucket

            bucket["requests"] = int(bucket["requests"]) + 1
            bucket["input_tokens"] = int(bucket["input_tokens"]) + in_tokens
            bucket["output_tokens"] = int(bucket["output_tokens"]) + out_tokens
            bucket["total_tokens"] = int(bucket["total_tokens"]) + total_tokens
            if record.cost is not None:
                bucket["cost_total"] = float(bucket["cost_total"]) + float(record.cost)
                bucket["cost_records"] = int(bucket["cost_records"]) + 1
            if record.currency:
                bucket["currency"] = str(record.currency)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            by_phase: dict[str, Any] = {}
            for phase, bucket in self._by_phase.items():
                cost_records = int(bucket.get("cost_records", 0))
                by_phase[phase] = {
                    "requests": int(bucket.get("requests", 0)),
                    "input_tokens": int(bucket.get("input_tokens", 0)),
                    "output_tokens": int(bucket.get("output_tokens", 0)),
                    "total_tokens": int(bucket.get("total_tokens", 0)),
                    "cost": (float(bucket.get("cost_total", 0.0)) if cost_records > 0 else None),
                    "currency": str(bucket.get("currency") or self._currency),
                    "cost_records": cost_records,
                }
            return {
                "requests": len(self._records),
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "total_tokens": self._total_tokens,
                "cost": (self._cost_total if self._cost_known > 0 else None),
                "currency": self._currency,
                "cost_records": self._cost_known,
                "by_phase": by_phase,
            }

    def records(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)

