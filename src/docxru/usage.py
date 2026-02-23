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

    def add(self, record: UsageRecord) -> None:
        with self._lock:
            self._records.append(record)
            self._input_tokens += max(0, int(record.input_tokens))
            self._output_tokens += max(0, int(record.output_tokens))
            self._total_tokens += max(0, int(record.total_tokens))
            if record.cost is not None:
                self._cost_total += float(record.cost)
                self._cost_known += 1
            if record.currency:
                self._currency = str(record.currency)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": len(self._records),
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "total_tokens": self._total_tokens,
                "cost": (self._cost_total if self._cost_known > 0 else None),
                "currency": self._currency,
                "cost_records": self._cost_known,
            }

    def records(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)

