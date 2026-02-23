from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float


class PricingTable:
    def __init__(self, data: dict[str, dict[str, ModelPricing]], *, currency: str = "USD") -> None:
        self._data = data
        self.currency = currency or "USD"

    @classmethod
    def empty(cls, *, currency: str = "USD") -> "PricingTable":
        return cls({}, currency=currency)

    def get(self, provider: str, model: str) -> ModelPricing | None:
        p = provider.strip().lower()
        m = model.strip().lower()
        provider_map = self._data.get(p)
        if not provider_map:
            return None
        return provider_map.get(m)

    def estimate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float | None:
        price = self.get(provider, model)
        if price is None:
            return None
        in_tokens = max(0, int(input_tokens))
        out_tokens = max(0, int(output_tokens))
        return (in_tokens / 1_000_000.0) * float(price.input_per_million) + (
            out_tokens / 1_000_000.0
        ) * float(price.output_per_million)


def _coerce_rate(value: Any) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Invalid pricing rate value: {value!r}") from exc


def _pick_rate(model_data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in model_data and model_data[key] is not None:
            return _coerce_rate(model_data[key])
    return None


def _parse_pricing_map(raw: dict[str, Any]) -> dict[str, dict[str, ModelPricing]]:
    parsed: dict[str, dict[str, ModelPricing]] = {}
    for provider, provider_data in raw.items():
        if not isinstance(provider_data, dict):
            continue
        pkey = str(provider).strip().lower()
        if not pkey:
            continue
        model_map: dict[str, ModelPricing] = {}
        for model, model_data in provider_data.items():
            if not isinstance(model_data, dict):
                continue
            mkey = str(model).strip().lower()
            if not mkey:
                continue
            in_rate = _pick_rate(model_data, "input_per_million", "in_per_million", "prompt_per_million")
            out_rate = _pick_rate(model_data, "output_per_million", "out_per_million", "completion_per_million")
            if in_rate is None or out_rate is None:
                continue
            model_map[mkey] = ModelPricing(input_per_million=in_rate, output_per_million=out_rate)
        if model_map:
            parsed[pkey] = model_map
    return parsed


def load_pricing_table(path: str | Path, *, currency: str = "USD") -> PricingTable:
    pricing_path = Path(path)
    text = pricing_path.read_text(encoding="utf-8")
    suffix = pricing_path.suffix.strip().lower()
    if suffix == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Pricing file must contain mapping object: {pricing_path}")
    pricing_raw = data.get("pricing", data)
    if not isinstance(pricing_raw, dict):
        raise ValueError(f"Pricing table must be a mapping: {pricing_path}")
    detected_currency = str(data.get("currency", currency)).strip() if isinstance(data, dict) else currency
    parsed = _parse_pricing_map(pricing_raw)
    return PricingTable(parsed, currency=(detected_currency or currency))

