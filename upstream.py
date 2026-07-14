#!/usr/bin/env python3
"""Inspect, validate and deterministically select benchmark suite instances."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_SUITES = ROOT / "suites"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites-dir", type=Path, default=DEFAULT_SUITES)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list registered suites")
    list_parser.add_argument("--json", action="store_true")

    doctor = subparsers.add_parser(
        "doctor", help="validate manifests and inspect local prerequisites"
    )
    doctor.add_argument("suite", nargs="*")
    doctor.add_argument(
        "--strict",
        action="store_true",
        help="also fail when a planned, non-runnable suite lacks dependencies",
    )
    doctor.add_argument("--json", action="store_true")

    select = subparsers.add_parser(
        "select", help="select instances by stable SHA-256 ordering"
    )
    select.add_argument("suite")
    select.add_argument("--count", type=int)
    select.add_argument("--salt")
    select.add_argument(
        "--stratify",
        action="append",
        help="object field; repeat or use comma-separated fields",
    )
    select.add_argument("--output", type=Path)
    return parser


def suite_paths(directory: Path) -> list[Path]:
    return sorted(directory.resolve().glob("*.json"))


def read_suite(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: suite manifest must be a JSON object")
    value["_path"] = str(path.resolve())
    return value


def load_suites(directory: Path) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for path in suite_paths(directory):
        suite = read_suite(path)
        suite_id = suite.get("id")
        if not isinstance(suite_id, str) or not suite_id.strip():
            raise ValueError(f"{path}: missing non-empty string `id`")
        if suite_id in manifests:
            raise ValueError(f"duplicate suite id {suite_id!r}")
        manifests[suite_id] = suite
    return manifests


def selectable_instances(suite: dict[str, Any]) -> list[Any]:
    for key in ("instances", "tasks"):
        value = suite.get(key)
        if isinstance(value, list):
            return value
    return []


def instance_id(instance: Any) -> str:
    if isinstance(instance, str) and instance:
        return instance
    if isinstance(instance, dict):
        for key in ("id", "instance_id", "task_id"):
            value = instance.get(key)
            if isinstance(value, str) and value:
                return value
    raise ValueError(f"instance has no stable string ID: {instance!r}")


def validate_suite(suite: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("id", "kind", "source", "comparability"):
        if not isinstance(suite.get(field), str) or not suite[field].strip():
            errors.append(f"missing non-empty string `{field}`")
    if not isinstance(suite.get("runnable"), bool):
        errors.append("`runnable` must be boolean")
    seen: set[str] = set()
    for instance in selectable_instances(suite):
        try:
            identifier = instance_id(instance)
        except ValueError as error:
            errors.append(str(error))
            continue
        if identifier in seen:
            errors.append(f"duplicate instance ID {identifier!r}")
        seen.add(identifier)
    if suite.get("runnable") and suite.get("id") == "native-core":
        for task in suite.get("tasks") or []:
            if not (ROOT / "tasks" / str(task) / "task.json").is_file():
                errors.append(f"native task is missing: {task}")
    return errors


def dependency_status(suite: dict[str, Any]) -> dict[str, dict[str, bool]]:
    doctor = suite.get("doctor") if isinstance(suite.get("doctor"), dict) else {}
    programs = {
        str(program): shutil.which(str(program)) is not None
        for program in doctor.get("programs") or []
    }
    modules = {
        str(module): importlib.util.find_spec(str(module)) is not None
        for module in doctor.get("python_modules") or []
    }
    return {"programs": programs, "python_modules": modules}


def dependency_failures(status: dict[str, dict[str, bool]]) -> list[str]:
    failures: list[str] = []
    for category, values in status.items():
        failures.extend(
            f"missing {category}: {name}" for name, available in values.items() if not available
        )
    return failures


def split_fields(values: list[str] | None) -> list[str]:
    fields: list[str] = []
    for value in values or []:
        for field in value.split(","):
            field = field.strip()
            if field and field not in fields:
                fields.append(field)
    return fields


def field_value(instance: Any, field: str) -> str:
    if not isinstance(instance, dict):
        return ""
    value: Any = instance
    for part in field.split("."):
        if not isinstance(value, dict):
            return ""
        value = value.get(part)
    return str(value) if value is not None else ""


def sha_rank(salt: str, identifier: str) -> str:
    return hashlib.sha256(f"{salt}\0{identifier}".encode("utf-8")).hexdigest()


def deterministic_select(
    instances: Iterable[Any],
    *,
    count: int | None,
    salt: str,
    strata: list[str],
) -> list[Any]:
    materialized = list(instances)
    if count is None:
        count = len(materialized)
    if count < 1:
        raise ValueError("--count must be at least 1")
    count = min(count, len(materialized))
    buckets: dict[tuple[str, ...], list[Any]] = {}
    for instance in materialized:
        key = tuple(field_value(instance, field) for field in strata)
        buckets.setdefault(key, []).append(instance)
    for values in buckets.values():
        values.sort(key=lambda item: (sha_rank(salt, instance_id(item)), instance_id(item)))
    selected: list[Any] = []
    keys = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def list_command(suites: dict[str, dict[str, Any]], as_json: bool) -> int:
    records = [
        {
            "id": suite_id,
            "kind": suite.get("kind"),
            "runnable": suite.get("runnable"),
            "instances": len(selectable_instances(suite)),
            "comparability": suite.get("comparability"),
        }
        for suite_id, suite in sorted(suites.items())
    ]
    if as_json:
        print(json.dumps(records, indent=2, sort_keys=True))
    else:
        print("ID\tRUNNABLE\tINSTANCES\tKIND")
        for record in records:
            print(
                f"{record['id']}\t{str(record['runnable']).lower()}\t"
                f"{record['instances']}\t{record['kind']}"
            )
    return 0


def doctor_command(
    suites: dict[str, dict[str, Any]],
    requested: list[str],
    *,
    strict: bool,
    as_json: bool,
) -> int:
    unknown = sorted(set(requested) - set(suites))
    if unknown:
        raise ValueError("unknown suites: " + ", ".join(unknown))
    selected = requested or sorted(suites)
    records = []
    failed = False
    for suite_id in selected:
        suite = suites[suite_id]
        errors = validate_suite(suite)
        dependencies = dependency_status(suite)
        missing = dependency_failures(dependencies)
        blocking = bool(errors) or bool(missing and (strict or suite.get("runnable")))
        failed = failed or blocking
        records.append(
            {
                "id": suite_id,
                "runnable": suite.get("runnable"),
                "manifest_errors": errors,
                "dependencies": dependencies,
                "missing_dependencies": missing,
                "blocking": blocking,
            }
        )
    if as_json:
        print(json.dumps(records, indent=2, sort_keys=True))
    else:
        for record in records:
            state = "FAIL" if record["blocking"] else "OK"
            detail = record["manifest_errors"] + record["missing_dependencies"]
            print(f"{state} {record['id']}" + (": " + "; ".join(detail) if detail else ""))
    return 2 if failed else 0


def select_command(
    suite: dict[str, Any],
    *,
    count: int | None,
    salt: str | None,
    strata: list[str],
    output: Path | None,
) -> int:
    instances = selectable_instances(suite)
    if not instances:
        raise ValueError(
            f"suite {suite['id']!r} has no `instances` or `tasks` to select"
        )
    selection = suite.get("selection") if isinstance(suite.get("selection"), dict) else {}
    effective_salt = salt or str(selection.get("salt") or "wllm-bench-v1")
    effective_strata = strata or [str(item) for item in selection.get("strata") or []]
    chosen = deterministic_select(
        instances,
        count=count,
        salt=effective_salt,
        strata=effective_strata,
    )
    result = {
        "schema_version": "1.0",
        "suite": suite["id"],
        "method": "sha256-sorted-stratified" if effective_strata else "sha256-sorted",
        "salt": effective_salt,
        "strata": effective_strata,
        "available": len(instances),
        "selected": len(chosen),
        "instance_ids": [instance_id(instance) for instance in chosen],
        "selection_sha256": hashlib.sha256(
            "\n".join(instance_id(instance) for instance in chosen).encode("utf-8")
        ).hexdigest(),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        print(output)
    else:
        print(encoded, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        suites = load_suites(args.suites_dir)
        if not suites:
            raise ValueError(f"no suite JSON files found under {args.suites_dir}")
        if args.command == "list":
            return list_command(suites, args.json)
        if args.command == "doctor":
            return doctor_command(
                suites, args.suite, strict=args.strict, as_json=args.json
            )
        if args.command == "select":
            if args.suite not in suites:
                raise ValueError(f"unknown suite: {args.suite}")
            return select_command(
                suites[args.suite],
                count=args.count,
                salt=args.salt,
                strata=split_fields(args.stratify),
                output=args.output,
            )
    except ValueError as error:
        parser.error(str(error))
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
