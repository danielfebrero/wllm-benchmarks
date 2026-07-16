#!/usr/bin/env python3
"""Externally grade multi-hop usage settlement."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections.abc import Iterable, Mapping
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable


def load_candidate(workspace: Path) -> Callable[..., dict[str, Any]]:
    path = workspace / "src" / "settlement.py"
    spec = importlib.util.spec_from_file_location("candidate_settlement", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load src/settlement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.settle_usage


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def reference(
    meters: Iterable[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    policy: Mapping[str, Any],
    period: Mapping[str, str],
) -> dict[str, Any]:
    grouped: dict[str, Decimal] = {}
    for meter in meters:
        if meter.get("trial") is not False:
            continue
        sku = str(meter["sku"])
        grouped[sku] = grouped.get(sku, Decimal("0")) + Decimal(str(meter["quantity"]))

    free_tier = policy.get("free_tier") or {}
    billable: dict[str, Decimal] = {}
    for sku, quantity in grouped.items():
        free = Decimal(str(free_tier.get(sku, 0)))
        remaining = quantity - free
        if remaining < 0:
            remaining = Decimal("0")
        billable[sku] = remaining

    skus = catalog["skus"]
    amounts: dict[str, Decimal] = {}
    for sku, quantity in billable.items():
        if sku not in skus:
            raise ValueError(f"unknown sku: {sku}")
        remaining = quantity
        previous_up_to = Decimal("0")
        line_amount = Decimal("0")
        for tier in skus[sku]["tiers"]:
            if remaining <= 0:
                break
            up_to = tier["up_to"]
            unit = Decimal(str(tier["unit_price"]))
            if up_to is None:
                span = remaining
            else:
                capacity = Decimal(str(up_to)) - previous_up_to
                if capacity < 0:
                    capacity = Decimal("0")
                span = remaining if remaining <= capacity else capacity
                previous_up_to = Decimal(str(up_to))
            line_amount += span * unit
            remaining -= span
        amounts[sku] = line_amount

    for promo in policy.get("promotions") or []:
        sku = str(promo["sku"])
        if sku not in amounts:
            continue
        percent = Decimal(str(promo["percent_off"]))
        amounts[sku] = amounts[sku] * (Decimal("1") - percent / Decimal("100"))

    lines = []
    subtotal = Decimal("0")
    for sku in sorted(billable):
        amount = _quantize(amounts.get(sku, Decimal("0")))
        quantity = billable[sku]
        lines.append(
            {
                "sku": sku,
                "quantity": float(quantity),
                "amount": float(amount),
            }
        )
        subtotal += amount
    subtotal = _quantize(subtotal)
    return {
        "schema_version": "3.0",
        "tenant_id": policy["tenant_id"],
        "period": dict(period),
        "currency": catalog["currency"],
        "lines": lines,
        "subtotal": float(subtotal),
        "total": float(subtotal),
    }


def fixture_bundle() -> tuple[
    list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, str]
]:
    meters = [
        {"sku": "api.calls", "quantity": "1200", "trial": False},
        {"sku": "api.calls", "quantity": "300", "trial": True},
        {"sku": "storage.gb", "quantity": "40.5", "trial": False},
        {"sku": "api.calls", "quantity": "800", "trial": False},
        {"sku": "egress.gb", "quantity": "12", "trial": False},
    ]
    catalog = {
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
    }
    policy = {
        "tenant_id": "acme-north",
        "free_tier": {"api.calls": 500, "storage.gb": 10, "egress.gb": 0},
        "promotions": [
            {"sku": "api.calls", "percent_off": 10},
            {"sku": "egress.gb", "percent_off": 25},
        ],
    }
    period = {"start": "2026-07-01T00:00:00Z", "end": "2026-07-02T00:00:00Z"}
    return meters, catalog, policy, period


def alternate_bundle() -> tuple[
    list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, str]
]:
    meters = [
        {"sku": "compute.hours", "quantity": "3", "trial": False},
        {"sku": "compute.hours", "quantity": "9", "trial": False},
        {"sku": "compute.hours", "quantity": "100", "trial": True},
        {"sku": "gpu.hours", "quantity": "2.25", "trial": False},
    ]
    catalog = {
        "currency": "EUR",
        "skus": {
            "compute.hours": {
                "list_price": "9.99",
                "tiers": [
                    {"up_to": 5, "unit_price": "2.0000"},
                    {"up_to": None, "unit_price": "1.5000"},
                ],
            },
            "gpu.hours": {
                "list_price": "4.00",
                "tiers": [{"up_to": None, "unit_price": "3.3333"}],
            },
        },
    }
    policy = {
        "tenant_id": "beta-east",
        "free_tier": {"compute.hours": 4},
        "promotions": [{"sku": "gpu.hours", "percent_off": 50}],
    }
    period = {"start": "2026-01-01T00:00:00Z", "end": "2026-01-02T00:00:00Z"}
    return meters, catalog, policy, period


def invoke(
    candidate: Callable[..., dict[str, Any]],
    meters: list[dict[str, Any]],
    catalog: dict[str, Any],
    policy: dict[str, Any],
    period: dict[str, str],
) -> tuple[Any, bool]:
    before = copy.deepcopy((meters, catalog, policy, period))
    result = candidate(meters, catalog, policy, period)
    after = (meters, catalog, policy, period)
    return result, before == after


def preflight() -> dict[str, Any]:
    first = reference(*copy.deepcopy(fixture_bundle()))
    second = reference(*copy.deepcopy(fixture_bundle()))
    encoded = json.dumps(first, sort_keys=True, separators=(",", ":"))
    if first != second:
        raise RuntimeError("reference is not deterministic")
    # Expected arithmetic for fixture (manual sanity):
    # api.calls billable = 1200+800-500 = 1500 → 1000*0.01 + 500*0.008 = 14, *0.9 = 12.6
    # storage.gb billable = 40.5-10 = 30.5 → 30.5*0.2 = 6.1
    # egress.gb billable = 12 → 12*0.09 = 1.08 *0.75 = 0.81
    # total = 19.51
    if first["total"] != 19.51:
        raise RuntimeError(f"unexpected fixture total: {first['total']}")
    return {"deterministic": True, "output_bytes": len(encoded.encode("utf-8"))}


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        print(json.dumps(preflight(), sort_keys=True))
        return 0
    if len(sys.argv) != 2:
        raise SystemExit("usage: grade.py WORKSPACE | grade.py --self-test")

    preflight()
    workspace = Path(sys.argv[1]).resolve()
    total = 12
    try:
        candidate = load_candidate(workspace)
        meters, catalog, policy, period = fixture_bundle()
        result, unchanged = invoke(
            candidate,
            copy.deepcopy(meters),
            copy.deepcopy(catalog),
            copy.deepcopy(policy),
            dict(period),
        )
        expected = reference(*fixture_bundle())

        alt_m, alt_c, alt_p, alt_period = alternate_bundle()
        alternate, alt_unchanged = invoke(
            candidate,
            copy.deepcopy(alt_m),
            copy.deepcopy(alt_c),
            copy.deepcopy(alt_p),
            dict(alt_period),
        )
        alternate_expected = reference(*alternate_bundle())

        unknown_rejected = False
        try:
            bad_meters = [{"sku": "nope", "quantity": "1", "trial": False}]
            candidate(bad_meters, catalog, policy, period)
        except ValueError:
            unknown_rejected = True
        except Exception:
            # Broken candidates often raise KeyError; only ValueError counts.
            unknown_rejected = False

        checks: list[tuple[str, bool]] = [
            ("returns a mapping", isinstance(result, dict)),
            (
                "uses settlement schema 3.0",
                isinstance(result, dict) and result.get("schema_version") == "3.0",
            ),
            (
                "uses policy tenant identity",
                isinstance(result, dict) and result.get("tenant_id") == "acme-north",
            ),
            (
                "excludes trial meters",
                isinstance(result, dict)
                and all(
                    line.get("sku") != "api.calls" or line.get("quantity") == 1500.0
                    for line in result.get("lines") or []
                )
                and sum(
                    1 for line in (result.get("lines") or []) if line.get("sku") == "api.calls"
                )
                == 1,
            ),
            (
                "applies free tier before pricing",
                isinstance(result, dict)
                and any(
                    line.get("sku") == "storage.gb" and line.get("quantity") == 30.5
                    for line in result.get("lines") or []
                ),
            ),
            (
                "uses progressive tiers not list price",
                isinstance(result, dict)
                and any(
                    line.get("sku") == "api.calls" and abs(float(line.get("amount", 0)) - 12.6) < 1e-9
                    for line in result.get("lines") or []
                ),
            ),
            (
                "applies promotions in order",
                isinstance(result, dict)
                and any(
                    line.get("sku") == "egress.gb" and abs(float(line.get("amount", 0)) - 0.81) < 1e-9
                    for line in result.get("lines") or []
                ),
            ),
            (
                "emits exact deterministic document",
                result == expected,
            ),
            (
                "generalizes to alternate tenant",
                alternate == alternate_expected,
            ),
            (
                "sorts lines by sku",
                isinstance(result, dict)
                and [line.get("sku") for line in result.get("lines") or []]
                == sorted(line.get("sku") for line in result.get("lines") or []),
            ),
            ("does not mutate inputs", unchanged and alt_unchanged),
            ("rejects unknown sku", unknown_rejected),
        ]
    except Exception as error:
        print(
            json.dumps(
                {
                    "passed": 0,
                    "total": total,
                    "score": 0.0,
                    "failures": [
                        f"candidate execution failed: {type(error).__name__}: {error}"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0

    failures = [name for name, passed in checks if not passed]
    passed = len(checks) - len(failures)
    print(
        json.dumps(
            {
                "passed": passed,
                "total": len(checks),
                "score": passed / len(checks),
                "failures": failures,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
