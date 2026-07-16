#!/usr/bin/env python3
"""Create a multi-hop usage settlement workspace with realistic decoys."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SOURCE = '''"""Settle billable usage for a billing period."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal, ROUND_HALF_UP
from typing import Any


def settle_usage(
    meters: Iterable[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    policy: Mapping[str, Any],
    period: Mapping[str, str],
) -> dict[str, Any]:
    """Return the settlement document for the closed UTC period."""
    # Broken production code: sums raw meter quantities at list price,
    # ignores trial exclusion, free tier, promotions, and contract rounding.
    total = Decimal("0")
    lines: list[dict[str, Any]] = []
    for meter in meters:
        sku = str(meter["sku"])
        quantity = Decimal(str(meter["quantity"]))
        unit = Decimal(str(catalog["skus"][sku]["list_price"]))
        amount = (quantity * unit).quantize(Decimal("0.01"))
        lines.append(
            {
                "sku": sku,
                "quantity": float(quantity),
                "unit_price": float(unit),
                "amount": float(amount),
            }
        )
        total += amount
    return {
        "schema_version": "2.0",
        "tenant_id": policy.get("tenant_id", "unknown"),
        "period": dict(period),
        "currency": catalog.get("currency", "USD"),
        "lines": lines,
        "subtotal": float(total),
        "total": float(total),
    }
'''


PUBLIC_TEST = '''from __future__ import annotations

import unittest

from src.settlement import settle_usage


class SettlementPublicTests(unittest.TestCase):
    def test_emits_tenant_and_period(self) -> None:
        meters = [{"sku": "api.calls", "quantity": 10, "trial": False}]
        catalog = {
            "currency": "USD",
            "skus": {"api.calls": {"list_price": "0.01", "tiers": []}},
        }
        policy = {
            "tenant_id": "demo",
            "free_tier": {"api.calls": 0},
            "promotions": [],
        }
        period = {"start": "2026-07-01T00:00:00Z", "end": "2026-07-02T00:00:00Z"}
        result = settle_usage(meters, catalog, policy, period)
        self.assertEqual(result["tenant_id"], "demo")
        self.assertEqual(result["period"], period)

    def test_has_line_items(self) -> None:
        meters = [{"sku": "api.calls", "quantity": 5, "trial": False}]
        catalog = {
            "currency": "USD",
            "skus": {
                "api.calls": {
                    "list_price": "1.00",
                    "tiers": [{"up_to": None, "unit_price": "1.00"}],
                }
            },
        }
        policy = {"tenant_id": "demo", "free_tier": {}, "promotions": []}
        period = {"start": "2026-07-01T00:00:00Z", "end": "2026-07-02T00:00:00Z"}
        result = settle_usage(meters, catalog, policy, period)
        self.assertTrue(result["lines"])


if __name__ == "__main__":
    unittest.main()
'''


CONTRACT = '''# Usage settlement contract v3

Status: authoritative
Owner: Revenue Platform
Applies to: `src.settlement.settle_usage`

Emit one compact JSON object with exactly these top-level keys:
`schema_version`, `tenant_id`, `period`, `currency`, `lines`, `subtotal`,
`total`.

`schema_version` is always the literal string `3.0`.
`tenant_id` comes from policy, never from catalog.
`period` is echoed exactly as supplied (already a closed half-open UTC window).
`currency` comes from catalog.

## Meters

Only meters with `trial` equal to boolean `false` are billable. Trial meters
must be omitted entirely. Group billable meters by `sku` and sum `quantity`
as `Decimal` values (do not use float accumulation).

## Free tier

Before pricing, subtract the free-tier allowance for that sku from the grouped
quantity (policy `free_tier` map, missing skus mean zero free units). Never
reduce a quantity below zero. Free units do not appear as negative lines.

## Tiered pricing

Price remaining quantity with catalog `skus[sku].tiers`, never `list_price`.
Each tier is `{ "up_to": <int|null>, "unit_price": "<decimal-string>" }`.
Tiers are contiguous from zero and ordered; `up_to` is the inclusive upper
bound of cumulative billable units covered by that tier; `null` means open
ended. Consume quantity from lower tiers first (standard progressive tiers).

## Promotions

After lines are priced, apply promotions from policy in listed order. Each
promotion is `{ "sku": "...", "percent_off": <0-100 number> }` and multiplies
that sku's line `amount` by `(1 - percent_off/100)`. Promotions never create
new lines and never apply to free-tier units already removed.

## Rounding and totals

Each line stores:
- `sku` (string)
- `quantity` (JSON number equal to the billable quantity after free tier, as a
  float converted from Decimal)
- `unit_price` must be omitted; instead emit `amount` only for priced lines
- `amount` is the line total after promotions, rounded with
  `Decimal.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)` then emitted
  as a JSON number via `float`.

`subtotal` is the sum of line amounts using Decimal, quantized the same way.
`total` equals `subtotal` (tax is out of scope).
Lines are sorted by sku ascending. Do not mutate inputs. Unknown sku raises
`ValueError`.
'''


README = '''# Billing incident QTA-991

Run public validation with `python3 -m unittest discover -s tests -v`.

Authoritative inputs:
- live meters under `meters/current/`
- product catalog under `catalog/current/`
- tenant policy under `policy/current/`
- settlement contract under `docs/contracts/`
- controller rejection under `evidence/live/`

`archive/` and `catalog/legacy/` are historical noise and must not drive
settlement for the active period.
'''


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def dump(path: Path, value: object) -> None:
    write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: prepare.py WORKSPACE")
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)

    write(root / "src" / "__init__.py", "")
    write(root / "src" / "settlement.py", SOURCE)
    write(root / "tests" / "test_settlement.py", PUBLIC_TEST)
    write(root / "docs" / "contracts" / "settlement-v3.md", CONTRACT)
    write(root / "README.md", README)
    write(
        root / "wllm.toml",
        "[budgets]\ndefault_text = 2500\n\n"
        "[policy]\nexternal_extractors = false\nnetwork = false\n",
    )

    dump(
        root / "meters" / "current" / "batch-001.json",
        [
            {"sku": "api.calls", "quantity": "1200", "trial": False, "window": "2026-07-01"},
            {"sku": "api.calls", "quantity": "300", "trial": True, "window": "2026-07-01"},
            {"sku": "storage.gb", "quantity": "40.5", "trial": False, "window": "2026-07-01"},
            {"sku": "api.calls", "quantity": "800", "trial": False, "window": "2026-07-01"},
            {"sku": "egress.gb", "quantity": "12", "trial": False, "window": "2026-07-01"},
        ],
    )
    dump(
        root / "catalog" / "current" / "products.json",
        {
            "currency": "USD",
            "skus": {
                "api.calls": {
                    "list_price": "0.0200",
                    "tiers": [
                        {"up_to": 1000, "unit_price": "0.0100"},
                        {"up_to": 5000, "unit_price": "0.0080"},
                        {"up_to": None, "unit_price": "0.0050"},
                    ],
                },
                "storage.gb": {
                    "list_price": "0.2500",
                    "tiers": [
                        {"up_to": 50, "unit_price": "0.2000"},
                        {"up_to": None, "unit_price": "0.1500"},
                    ],
                },
                "egress.gb": {
                    "list_price": "0.1200",
                    "tiers": [{"up_to": None, "unit_price": "0.0900"}],
                },
            },
        },
    )
    dump(
        root / "policy" / "current" / "acme-north.json",
        {
            "tenant_id": "acme-north",
            "free_tier": {"api.calls": 500, "storage.gb": 10, "egress.gb": 0},
            "promotions": [
                {"sku": "api.calls", "percent_off": 10},
                {"sku": "egress.gb", "percent_off": 25},
            ],
        },
    )
    dump(
        root / "policy" / "current" / "period.json",
        {"start": "2026-07-01T00:00:00Z", "end": "2026-07-02T00:00:00Z"},
    )
    write(
        root / "evidence" / "live" / "QTA-991.log",
        "2026-07-02T00:05:11Z level=ERROR incident=QTA-991 "
        "code=SETTLEMENT_CONTRACT_V3 tenant=acme-north "
        "detail=schema_version expected=3.0 actual=2.0; "
        "trial_meters_included; list_price_used; free_tier_skipped; "
        "promotion_order_and_rounding_invalid\n",
    )

    # Lexical noise: retired settlements and marketing rate cards.
    for number in range(80):
        year = 2021 + number % 5
        write(
            root / "archive" / str(year) / f"settlement-{number:03d}.md",
            f"# Archived settlement note {number}\n\n"
            "Status: closed historical record\n\n"
            "Used list_price flat billing and included trial meters. "
            "Do not apply this procedure to SETTLEMENT_CONTRACT_V3.\n",
        )
        dump(
            root / "catalog" / "legacy" / f"rate-card-{number:03d}.json",
            {
                "status": "retired",
                "currency": "EUR" if number % 2 else "USD",
                "list_only": True,
                "sku": f"legacy.sku.{number}",
                "list_price": f"{0.01 * (number + 1):.4f}",
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
