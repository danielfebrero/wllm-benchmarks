#!/usr/bin/env python3
"""Run a reproducible Cartesian matrix of wllm A/B benchmark cells."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import run


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run tasks × agents × efforts × topologies as isolated run.py cells."
        )
    )
    parser.add_argument(
        "--task",
        "--tasks",
        dest="task",
        action="append",
        help="task ID; repeat or use comma-separated values",
    )
    parser.add_argument(
        "--agent",
        "--agents",
        dest="agent",
        action="append",
        help="codex, claude or grok; repeat or use comma-separated values",
    )
    parser.add_argument(
        "--effort",
        "--efforts",
        "--reasoning",
        dest="effort",
        action="append",
        help="effort value; repeat or use comma-separated values",
    )
    parser.add_argument(
        "--topology",
        "--topologies",
        dest="topology",
        action="append",
        help="single or native-multi-agent; repeat or use comma-separated values",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="AGENT=MODEL",
        help="agent-specific model; repeat for multiple agents",
    )
    parser.add_argument(
        "--agent-bin",
        action="append",
        default=[],
        metavar="AGENT=PATH",
        help="agent-specific executable; repeat for multiple agents",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--arm", choices=("both", "baseline", "wllm"), default="both"
    )
    parser.add_argument("--brief-budget", type=int, default=1200)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--wllm-bin", type=Path)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.timeout < 1:
        parser.error("--timeout must be at least 1")
    if args.brief_budget < 256:
        parser.error("--brief-budget must be at least 256")

    args.tasks = split_values(args.task, ["release-evidence"])
    args.agents = split_values(args.agent, ["codex"])
    args.efforts = split_values(args.effort, ["medium"])
    args.topologies = split_values(args.topology, ["single"])
    invalid_agents = sorted(set(args.agents) - set(run.AGENT_DEFAULTS))
    if invalid_agents:
        parser.error("unknown agents: " + ", ".join(invalid_agents))
    invalid_topologies = sorted(set(args.topologies) - set(run.TOPOLOGIES))
    if invalid_topologies:
        parser.error("unknown topologies: " + ", ".join(invalid_topologies))
    args.models = parse_mapping(args.model, "--model", parser)
    args.agent_bins = parse_mapping(args.agent_bin, "--agent-bin", parser)
    for task in args.tasks:
        run.load_task(task)
    return args


def split_values(values: list[str] | None, default: list[str]) -> list[str]:
    if not values:
        return list(default)
    result: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item and item not in result:
                result.append(item)
    return result


def parse_mapping(
    values: Iterable[str], option: str, parser: argparse.ArgumentParser
) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        agent, separator, mapped = value.partition("=")
        if not separator or not agent.strip() or not mapped.strip():
            parser.error(f"{option} expects AGENT=VALUE, got {value!r}")
        agent = agent.strip()
        if agent not in run.AGENT_DEFAULTS:
            parser.error(f"{option} names unknown agent {agent!r}")
        if agent in result:
            parser.error(f"{option} specifies {agent!r} more than once")
        result[agent] = mapped.strip()
    return result


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "cell"


def build_cells(args: argparse.Namespace, matrix_dir: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    combinations = itertools.product(
        args.tasks, args.agents, args.efforts, args.topologies
    )
    for number, (task, agent, effort, topology) in enumerate(combinations, 1):
        model = args.models.get(
            agent, str(run.AGENT_DEFAULTS[agent]["model"])
        )
        label = safe_id(f"{number:03d}-{task}-{agent}-{model}-{effort}-{topology}")
        cell_dir = matrix_dir / "cells" / label
        command = [
            sys.executable,
            str(ROOT / "run.py"),
            "--task",
            task,
            "--agent",
            agent,
            "--model",
            model,
            "--reasoning",
            effort,
            "--topology",
            topology,
            "--runs",
            str(args.runs),
            "--arm",
            args.arm,
            "--brief-budget",
            str(args.brief_budget),
            "--timeout",
            str(args.timeout),
            "--output-dir",
            str(cell_dir / "artifacts"),
        ]
        if agent in args.agent_bins:
            command.extend(("--agent-bin", args.agent_bins[agent]))
        if args.wllm_bin is not None:
            command.extend(("--wllm-bin", str(args.wllm_bin.resolve())))
        if args.no_build:
            command.append("--no-build")
        if args.keep_workspaces:
            command.append("--keep-workspaces")
        cells.append(
            {
                "id": label,
                "number": number,
                "task": task,
                "agent": agent,
                "model": model,
                "effort": effort,
                "topology": topology,
                "cell_dir": str(cell_dir),
                "command": command,
            }
        )
    return cells


def run_cell(cell: dict[str, Any]) -> dict[str, Any]:
    cell_dir = Path(cell["cell_dir"])
    cell_dir.mkdir(parents=True, exist_ok=False)
    (cell_dir / "command.json").write_text(
        json.dumps(cell["command"], indent=2) + "\n", encoding="utf-8"
    )
    started = time.monotonic()
    result = subprocess.run(
        cell["command"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    duration = time.monotonic() - started
    (cell_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (cell_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
    reports = sorted((cell_dir / "artifacts").glob("*/report.json"))
    report_path = reports[-1] if reports else None
    return {
        **{key: value for key, value in cell.items() if key != "cell_dir"},
        "cell_dir": str(cell_dir.relative_to(cell_dir.parents[1])),
        "exit_code": result.returncode,
        "duration_seconds": round(duration, 3),
        "report": (
            str(report_path.relative_to(cell_dir.parents[1]))
            if report_path is not None
            else None
        ),
        "stdout": str((cell_dir / "stdout.log").relative_to(cell_dir.parents[1])),
        "stderr": str((cell_dir / "stderr.log").relative_to(cell_dir.parents[1])),
    }


def write_plan(
    matrix_dir: Path, args: argparse.Namespace, cells: list[dict[str, Any]]
) -> Path:
    plan = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cwd": str(ROOT),
        "runs_per_cell": args.runs,
        "jobs": args.jobs,
        "timing_comparable": args.jobs == 1,
        "cells": [
            {key: value for key, value in cell.items() if key != "cell_dir"}
            for cell in cells
        ],
    }
    path = matrix_dir / "matrix-plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_index(
    matrix_dir: Path,
    args: argparse.Namespace,
    cells: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> Path:
    failures = sum(1 for cell in cells if cell.get("exit_code") not in (None, 0))
    index = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "jobs": args.jobs,
        "timing_comparable": args.jobs == 1,
        "cell_count": len(cells),
        "failed_cells": failures,
        "cells": cells,
    }
    path = matrix_dir / "artifact-index.json"
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime("matrix-%Y%m%dT%H%M%S%fZ")
    matrix_dir = args.output_dir.resolve() / timestamp
    matrix_dir.mkdir(parents=True, exist_ok=False)
    cells = build_cells(args, matrix_dir)
    write_plan(matrix_dir, args, cells)
    if args.dry_run:
        write_index(matrix_dir, args, cells, dry_run=True)
        print(f"Planned {len(cells)} cells: {matrix_dir}")
        return 0

    print(
        f"Running {len(cells)} cells with {args.jobs} worker(s). "
        + (
            "Wall-time comparisons are enabled."
            if args.jobs == 1
            else "Parallel wall times are exploratory, not publication-comparable."
        ),
        file=sys.stderr,
    )
    if args.jobs == 1:
        outcomes = [run_cell(cell) for cell in cells]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            outcomes = list(pool.map(run_cell, cells))
    outcomes.sort(key=lambda cell: int(cell["number"]))
    write_index(matrix_dir, args, outcomes, dry_run=False)
    failures = [cell for cell in outcomes if cell["exit_code"] != 0]
    print(f"Matrix artifacts: {matrix_dir}", file=sys.stderr)
    if failures:
        print(
            "Failed cells: " + ", ".join(str(cell["id"]) for cell in failures),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
