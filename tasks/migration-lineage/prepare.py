#!/usr/bin/env python3
"""Create a deterministic migration-lineage incident workspace."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SOURCE = '''"""Plan pending schema migrations."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def load_migrations(directory: Path) -> list[dict[str, Any]]:
    """Load migration metadata files for the deployment command."""
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]


def plan_migrations(
    records: Iterable[Mapping[str, Any]],
    applied: Iterable[str],
    target: str,
) -> list[str]:
    """Return the pending migration identifiers for a target."""
    by_id = {str(record["id"]): record for record in records}
    if target not in by_id:
        raise ValueError(f"unknown target migration: {target}")
    applied_ids = set(applied)
    return [
        migration_id
        for migration_id in sorted(by_id)
        if migration_id not in applied_ids and migration_id <= target
    ]
'''


PUBLIC_TEST = '''from __future__ import annotations

import unittest

from src.migration_planner import plan_migrations


class MigrationPlannerTests(unittest.TestCase):
    def test_linear_history_is_planned_in_order(self) -> None:
        records = [
            {"id": "m001", "parents": []},
            {"id": "m002", "parents": ["m001"]},
            {"id": "m003", "parents": ["m002"]},
        ]
        self.assertEqual(plan_migrations(records, {"m001"}, "m003"), ["m002", "m003"])

    def test_unknown_target_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            plan_migrations([{"id": "m001", "parents": []}], set(), "missing")


if __name__ == "__main__":
    unittest.main()
'''


CONTRACT = '''# Migration planning contract

Status: authoritative
Owner: Data Platform
Applies to: `src.migration_planner.plan_migrations`

Migration identifiers are opaque labels. Filenames and lexical or numeric id
order do not express execution order; only each record's ordered `parents`
list does. To reach a requested target, return its unapplied ancestor closure
followed by the target in deterministic parent-before-child order. Traverse a
record's parents in their declared order, emit every migration at most once,
and exclude migrations that are not ancestors of the target.

An applied migration satisfies that node and its ancestry, so neither it nor
its parents are returned through that branch. Do not mutate supplied records
or applied collections. Reject duplicate identifiers, an unknown target, a
missing parent in the reachable lineage, and a reachable dependency cycle with
`ValueError` rather than constructing a partial plan.
'''


README = '''# MIG-604 profile activation

Run public validation with `python3 -m unittest discover -s tests -v`.

The current migration graph is under `migrations/current`, the deployment
target and applied set under `deployment/current`, the live planner failure
under `evidence/live`, and the operational contract under `docs/runbooks`.
Files under `archive/` document retired migration systems only.
'''


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def dump(path: Path, value: object) -> None:
    write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def migration(root: Path, migration_id: str, parents: list[str], operation: str) -> None:
    dump(
        root / "migrations" / "current" / f"{migration_id}.json",
        {
            "id": migration_id,
            "parents": parents,
            "operation": operation,
            "status": "active",
        },
    )


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: prepare.py WORKSPACE")
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)

    write(root / "src" / "__init__.py", "")
    write(root / "src" / "migration_planner.py", SOURCE)
    write(root / "tests" / "test_migration_planner.py", PUBLIC_TEST)
    write(root / "docs" / "runbooks" / "migration-planning.md", CONTRACT)
    write(root / "README.md", README)
    write(
        root / "wllm.toml",
        "[budgets]\ndefault_text = 2500\n\n"
        "[policy]\nexternal_extractors = false\nnetwork = false\n",
    )

    migration(root, "m000_bootstrap", [], "create migration ledger")
    migration(root, "m010_accounts", ["m000_bootstrap"], "create accounts")
    migration(root, "m090_expand_profile", ["m010_accounts"], "add nullable profile fields")
    migration(root, "m020_backfill_profile", ["m090_expand_profile"], "backfill profile fields")
    migration(root, "m015_audit_export", ["m010_accounts"], "create audit export branch")
    migration(root, "m025_preferences", ["m010_accounts"], "create preference records")
    migration(
        root,
        "m040_activate_profiles",
        ["m020_backfill_profile", "m025_preferences"],
        "enforce profile fields and activate preferences",
    )

    dump(
        root / "deployment" / "current" / "migration-state.json",
        {
            "incident": "MIG-604",
            "target": "m040_activate_profiles",
            "applied": ["m000_bootstrap"],
        },
    )
    write(
        root / "evidence" / "live" / "MIG-604.log",
        "2026-07-14T13:07:19Z level=ERROR incident=MIG-604 "
        "target=m040_activate_profiles code=INVALID_MIGRATION_PLAN "
        "missing_parent=m090_expand_profile unexpected=m015_audit_export\n",
    )

    for number in range(140):
        year = 2019 + number % 7
        migration_id = f"legacy_{number:04d}"
        dump(
            root / "archive" / str(year) / "migrations" / f"{migration_id}.json",
            {
                "id": migration_id,
                "parents": [] if number == 0 else [f"legacy_{number - 1:04d}"],
                "status": "retired historical record",
                "executor": "legacy-sequential-v1",
            },
        )
        if number % 10 == 0:
            write(
                root / "archive" / str(year) / "notes" / f"batch-{number:04d}.md",
                f"# Retired migration batch {number:04d}\n\n"
                "Status: historical only. This executor sorted filenames and "
                "does not describe the current dependency planner.\n",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
