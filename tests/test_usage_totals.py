from __future__ import annotations

from docxru.usage import UsageRecord, UsageTotals


def test_usage_totals_snapshot_includes_phase_breakdown():
    totals = UsageTotals()
    totals.add(
        UsageRecord(
            provider="openai",
            model="gpt-5-mini",
            phase="translate",
            input_tokens=120,
            output_tokens=55,
            total_tokens=175,
            cost=0.0105,
            currency="USD",
        )
    )
    totals.add(
        UsageRecord(
            provider="openai",
            model="gpt-5-nano",
            phase="checker",
            input_tokens=40,
            output_tokens=12,
            total_tokens=52,
            cost=0.0012,
            currency="USD",
        )
    )

    snapshot = totals.snapshot()

    assert snapshot["requests"] == 2
    assert snapshot["input_tokens"] == 160
    assert snapshot["output_tokens"] == 67
    assert snapshot["total_tokens"] == 227
    assert snapshot["cost_records"] == 2
    assert snapshot["cost"] == 0.0117

    by_phase = snapshot["by_phase"]
    assert by_phase["translate"]["requests"] == 1
    assert by_phase["translate"]["input_tokens"] == 120
    assert by_phase["translate"]["output_tokens"] == 55
    assert by_phase["translate"]["total_tokens"] == 175
    assert by_phase["translate"]["cost"] == 0.0105
    assert by_phase["translate"]["cost_records"] == 1

    assert by_phase["checker"]["requests"] == 1
    assert by_phase["checker"]["input_tokens"] == 40
    assert by_phase["checker"]["output_tokens"] == 12
    assert by_phase["checker"]["total_tokens"] == 52
    assert by_phase["checker"]["cost"] == 0.0012
    assert by_phase["checker"]["cost_records"] == 1


def test_usage_totals_uses_unknown_phase_for_empty_phase_name():
    totals = UsageTotals()
    totals.add(
        UsageRecord(
            provider="openai",
            model="gpt-4o-mini",
            phase="",
            input_tokens=9,
            output_tokens=3,
            total_tokens=12,
            cost=None,
            currency="USD",
        )
    )

    snapshot = totals.snapshot()
    unknown = snapshot["by_phase"]["unknown"]

    assert unknown["requests"] == 1
    assert unknown["input_tokens"] == 9
    assert unknown["output_tokens"] == 3
    assert unknown["total_tokens"] == 12
    assert unknown["cost_records"] == 0
    assert unknown["cost"] is None
