#!/usr/bin/env python3
"""Externally grade dependency-aware migration planning."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Callable


def load_candidate(workspace: Path) -> Callable[..., list[str]]:
    path = workspace / "src" / "migration_planner.py"
    spec = importlib.util.spec_from_file_location("candidate_migration_planner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load src/migration_planner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plan_migrations


def reference(
    records: Iterable[Mapping[str, Any]],
    applied: Iterable[str],
    target: str,
) -> list[str]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        migration_id = str(record["id"])
        if migration_id in by_id:
            raise ValueError(f"duplicate migration id: {migration_id}")
        by_id[migration_id] = record
    if target not in by_id:
        raise ValueError(f"unknown target migration: {target}")

    applied_ids = set(applied)
    visiting: set[str] = set()
    emitted: set[str] = set()
    plan: list[str] = []

    def visit(migration_id: str) -> None:
        if migration_id in applied_ids or migration_id in emitted:
            return
        if migration_id in visiting:
            raise ValueError(f"migration dependency cycle at {migration_id}")
        record = by_id.get(migration_id)
        if record is None:
            raise ValueError(f"missing migration parent: {migration_id}")
        visiting.add(migration_id)
        for parent in record.get("parents", []):
            visit(str(parent))
        visiting.remove(migration_id)
        emitted.add(migration_id)
        plan.append(migration_id)

    visit(target)
    return plan


def raises_value_error(call: Callable[[], object]) -> bool:
    try:
        call()
    except ValueError:
        return True
    return False


def invoke(
    candidate: Callable[..., list[str]],
    records: list[dict[str, Any]],
    applied: set[str],
    target: str,
) -> tuple[Any, bool]:
    before = copy.deepcopy((records, applied))
    result = candidate(records, applied, target)
    return result, (records, applied) == before


def active_records() -> list[dict[str, Any]]:
    return [
        {"id": "m000_bootstrap", "parents": []},
        {"id": "m010_accounts", "parents": ["m000_bootstrap"]},
        {"id": "m090_expand_profile", "parents": ["m010_accounts"]},
        {"id": "m020_backfill_profile", "parents": ["m090_expand_profile"]},
        {"id": "m015_audit_export", "parents": ["m010_accounts"]},
        {"id": "m025_preferences", "parents": ["m010_accounts"]},
        {
            "id": "m040_activate_profiles",
            "parents": ["m020_backfill_profile", "m025_preferences"],
        },
    ]


def preflight() -> dict[str, Any]:
    expected = [
        "m010_accounts",
        "m090_expand_profile",
        "m020_backfill_profile",
        "m025_preferences",
        "m040_activate_profiles",
    ]
    first = reference(active_records(), {"m000_bootstrap"}, "m040_activate_profiles")
    second = reference(
        copy.deepcopy(active_records()),
        {"m000_bootstrap"},
        "m040_activate_profiles",
    )
    if first != expected or first != second:
        raise RuntimeError("migration reference is not deterministic")
    return {"deterministic": True, "active_plan_length": len(first)}


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        print(json.dumps(preflight(), sort_keys=True))
        return 0
    if len(sys.argv) != 2:
        raise SystemExit("usage: grade.py WORKSPACE | grade.py --self-test")

    preflight()
    total = 10
    try:
        candidate = load_candidate(Path(sys.argv[1]).resolve())
        records = active_records()
        plan, unchanged = invoke(
            candidate,
            copy.deepcopy(records),
            {"m000_bootstrap"},
            "m040_activate_profiles",
        )
        expected = reference(records, {"m000_bootstrap"}, "m040_activate_profiles")

        alternate_records = [
            {"id": "root", "parents": []},
            {"id": "z_expand", "parents": ["root"]},
            {"id": "a_backfill", "parents": ["z_expand"]},
            {"id": "c_side", "parents": ["root"]},
            {"id": "b_merge", "parents": ["a_backfill", "c_side"]},
            {"id": "aa_unrelated", "parents": ["root"]},
        ]
        alternate, alternate_unchanged = invoke(
            candidate,
            copy.deepcopy(alternate_records),
            {"root"},
            "b_merge",
        )
        alternate_expected = reference(alternate_records, {"root"}, "b_merge")

        duplicate_records = [
            {"id": "root", "parents": []},
            {"id": "root", "parents": []},
        ]
        missing_records = [{"id": "target", "parents": ["missing"]}]
        cyclic_records = [
            {"id": "left", "parents": ["right"]},
            {"id": "right", "parents": ["left"]},
        ]
        pruned_records = [
            {"id": "base", "parents": []},
            {"id": "middle", "parents": ["base"]},
            {"id": "target", "parents": ["middle"]},
        ]

        checks = [
            (
                "returns a migration-id list",
                isinstance(plan, list)
                and all(isinstance(item, str) for item in plan),
            ),
            ("emits the exact target ancestor closure", plan == expected),
            (
                "orders non-monotonic dependencies before children",
                isinstance(plan, list)
                and "m090_expand_profile" in plan
                and "m020_backfill_profile" in plan
                and plan.index("m090_expand_profile") < plan.index("m020_backfill_profile"),
            ),
            (
                "excludes unrelated branches",
                isinstance(plan, list) and "m015_audit_export" not in plan,
            ),
            (
                "honors applied ancestry",
                candidate(copy.deepcopy(pruned_records), {"middle"}, "target")
                == ["target"],
            ),
            (
                "generalizes to a held-out branch merge",
                alternate == alternate_expected,
            ),
            (
                "rejects duplicate identifiers",
                raises_value_error(
                    lambda: candidate(copy.deepcopy(duplicate_records), set(), "root")
                ),
            ),
            (
                "rejects a missing reachable parent",
                raises_value_error(
                    lambda: candidate(copy.deepcopy(missing_records), set(), "target")
                ),
            ),
            (
                "rejects a reachable dependency cycle",
                raises_value_error(
                    lambda: candidate(copy.deepcopy(cyclic_records), set(), "left")
                ),
            ),
            ("does not mutate inputs", unchanged and alternate_unchanged),
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
        return 1

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
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
