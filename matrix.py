#!/usr/bin/env python3
"""Run a reproducible Cartesian matrix of wllm three-arm benchmark cells."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import run


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results"
CONFIG_SCHEMA_VERSION = "1.0"
CONFIG_METADATA_FIELDS = (
    "selection_salt",
    "cold",
    "retain_failures",
    "model_snapshot_status",
    "cache_regime",
    "machine_regime",
)
MODEL_SNAPSHOT_STATUSES = ("template", "attested-immutable")
CELL_PROCESS_MARGIN_SECONDS = 300
DEFAULT_ARGUMENTS: dict[str, Any] = {
    "tasks": ["release-evidence"],
    "agents": ["codex"],
    "efforts": ["medium"],
    "topologies": ["single"],
    "models": {},
    "agent_bins": {},
    "runs": 1,
    "jobs": 1,
    "arm": "all",
    "brief_budget": 1200,
    "timeout": 900,
    "wllm_bin": None,
    "no_build": False,
    "keep_workspaces": False,
    "output_dir": DEFAULT_OUTPUT,
    "dry_run": False,
    "analysis_plan": None,
}

ATTESTED_EXPLICIT_FIELDS = {
    "tasks",
    "agents",
    "efforts",
    "topologies",
    "models",
    "agent_bins",
    "runs",
    "jobs",
    "arm",
    "brief_budget",
    "timeout",
    "wllm_bin",
    "no_build",
}
ATTESTED_PROTOCOL_CLI_FIELDS = {
    *ATTESTED_EXPLICIT_FIELDS,
    "keep_workspaces",
}
HARNESS_FILES = (
    "run.py",
    "matrix.py",
    "analysis.py",
    "schemas/report.schema.json",
)


class ConfigError(ValueError):
    """Raised when a matrix configuration cannot be loaded safely."""


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value using the benchmark's single canonical encoding."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tree_sha256(root: Path) -> str:
    """Hash a task tree deterministically, excluding interpreter caches."""
    resolved = root.resolve()
    if not resolved.is_dir():
        raise ConfigError(f"task tree is not a directory: {resolved}")
    digest = hashlib.sha256()

    def field(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    def visit(path: Path, relative: str) -> None:
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
        elif stat.S_ISREG(metadata.st_mode):
            kind = "file"
        elif stat.S_ISLNK(metadata.st_mode):
            kind = "symlink"
        else:
            kind = f"special-{stat.S_IFMT(metadata.st_mode):o}"
        field(os.fsencode(relative))
        field(kind.encode("ascii"))
        field(f"{mode:o}".encode("ascii"))
        if kind == "symlink":
            field(os.fsencode(os.readlink(path)))
        elif kind == "file":
            field(str(metadata.st_size).encode("ascii"))
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        elif kind == "directory":
            for child in sorted(path.iterdir(), key=lambda item: os.fsencode(item.name)):
                if child.name == "__pycache__" or child.suffix in {".pyc", ".pyo"}:
                    continue
                child_relative = child.name if relative == "." else f"{relative}/{child.name}"
                visit(child, child_relative)

    visit(resolved, ".")
    return digest.hexdigest()


def _resolved_binary(value: str | Path | None, *, required: bool) -> dict[str, Any]:
    requested = str(value) if value is not None else ""
    candidate: Path | None = None
    if requested:
        expanded = Path(requested).expanduser()
        if expanded.is_absolute() or "/" in requested or "\\" in requested:
            candidate = expanded.resolve()
        else:
            located = shutil.which(requested)
            candidate = Path(located).resolve() if located else None
    if candidate is None or not candidate.is_file() or not os.access(candidate, os.X_OK):
        if required:
            raise ConfigError(
                f"executable {requested!r} could not be resolved to an executable file"
            )
        return {"path": None, "sha256": None}
    return {
        "path": str(candidate),
        "sha256": sha256_file(candidate),
    }


def _protocol_from_values(
    values: dict[str, Any], metadata: dict[str, Any], *, require_resolved: bool
) -> dict[str, Any]:
    agents: dict[str, Any] = {}
    for agent in values["agents"]:
        program = values.get("agent_bins", {}).get(
            agent, str(run.AGENT_DEFAULTS[agent]["binary"])
        )
        agents[agent] = {
            "model": values.get("models", {}).get(
                agent, str(run.AGENT_DEFAULTS[agent]["model"])
            ),
            "binary": _resolved_binary(program, required=require_resolved),
        }
    tasks: dict[str, Any] = {}
    for task in values["tasks"]:
        task_dir, _manifest = run.load_task(task)
        tasks[task] = {
            "path": str(task_dir.resolve()),
            "tree_sha256": tree_sha256(task_dir),
        }
    wllm_value = values.get("wllm_bin")
    return {
        "schema_version": "1.0",
        "matrix": {
            "arm": values["arm"],
            "brief_budget": values["brief_budget"],
            "timeout": values["timeout"],
            "runs": values["runs"],
            "jobs": values["jobs"],
            "no_build": values["no_build"],
            "keep_workspaces": values.get("keep_workspaces", False),
        },
        "cache_regime": metadata.get("cache_regime", "unspecified"),
        "machine_regime": metadata.get("machine_regime", "unspecified"),
        "agents": agents,
        "wllm_binary": _resolved_binary(wllm_value, required=require_resolved),
        "tasks": tasks,
        "harness_files": {
            relative: sha256_file(ROOT / relative) for relative in HARNESS_FILES
        },
    }


def _validate_attested_config(
    values: dict[str, Any], metadata: dict[str, Any]
) -> None:
    missing = sorted(ATTESTED_EXPLICIT_FIELDS - set(values))
    if missing:
        raise ConfigError(
            "model_snapshot_status 'attested-immutable' requires explicit fields: "
            + ", ".join(missing)
        )
    if values["arm"] != "all":
        raise ConfigError("attested-immutable publication requires arm='all'")
    if values["no_build"] is not True:
        raise ConfigError("attested-immutable publication requires no_build=true")
    selected = set(values["agents"])
    configured_bins = set(values["agent_bins"])
    if selected != configured_bins:
        missing_bins = sorted(selected - configured_bins)
        extra_bins = sorted(configured_bins - selected)
        details = []
        if missing_bins:
            details.append("missing binaries for " + ", ".join(missing_bins))
        if extra_bins:
            details.append("binaries for unselected agents " + ", ".join(extra_bins))
        raise ConfigError("agent_bins must cover exactly selected agents: " + "; ".join(details))
    for field in ("cache_regime", "machine_regime"):
        label = metadata.get(field)
        if not isinstance(label, str) or not label.strip() or label.strip().lower() in {
            "template",
            "unspecified",
        }:
            raise ConfigError(
                f"attested-immutable publication requires a non-template {field!r}"
            )


def frozen_execution_protocol(config_path: Path) -> dict[str, Any]:
    """Build the canonical immutable protocol declared by a matrix config."""
    _resolved, values, metadata = load_matrix_config(config_path)
    _validate_attested_config(values, metadata)
    merged = {
        key: list(value) if isinstance(value, list) else dict(value)
        if isinstance(value, dict)
        else value
        for key, value in DEFAULT_ARGUMENTS.items()
        if key != "analysis_plan"
    }
    merged.update(values)
    return _protocol_from_values(merged, metadata, require_resolved=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run tasks x agents x efforts x topologies as isolated run.py cells."
        ),
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON matrix config; explicitly supplied CLI options take precedence",
    )
    parser.add_argument(
        "--task",
        "--tasks",
        dest="tasks",
        action="append",
        help="task ID; repeat or use comma-separated values",
    )
    parser.add_argument(
        "--agent",
        "--agents",
        dest="agents",
        action="append",
        help="codex, claude or grok; repeat or use comma-separated values",
    )
    parser.add_argument(
        "--effort",
        "--efforts",
        "--reasoning",
        dest="efforts",
        action="append",
        help="effort value; repeat or use comma-separated values",
    )
    parser.add_argument(
        "--topology",
        "--topologies",
        dest="topologies",
        action="append",
        help="single or native-multi-agent; repeat or use comma-separated values",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        metavar="AGENT=MODEL",
        help="agent-specific model; repeat for multiple agents",
    )
    parser.add_argument(
        "--agent-bin",
        dest="agent_bins",
        action="append",
        metavar="AGENT=PATH",
        help="agent-specific executable; repeat for multiple agents",
    )
    parser.add_argument("--runs", type=int)
    parser.add_argument("--jobs", type=int)
    parser.add_argument(
        "--arm", choices=("all", "both", "baseline", "brief-only", "wllm")
    )
    parser.add_argument("--brief-budget", type=int)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--wllm-bin", type=Path)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--analysis-plan",
        type=Path,
        help="frozen analysis plan binding this publication execution",
    )
    return parser


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"field {field!r} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ConfigError(f"field {field!r} must be an array of strings")
    if not value:
        raise ConfigError(f"field {field!r} must contain at least one value")
    result = [
        _non_empty_string(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    ]
    duplicates = sorted({item for item in result if result.count(item) > 1})
    if duplicates:
        raise ConfigError(
            f"field {field!r} contains duplicate values: {', '.join(duplicates)}"
        )
    return result


def _string_mapping(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ConfigError(f"field {field!r} must be an object of string values")
    result: dict[str, str] = {}
    for key, mapped in value.items():
        agent = _non_empty_string(key, f"{field} key")
        result[agent] = _non_empty_string(mapped, f"{field}.{agent}")
    return result


def _integer(value: Any, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"field {field!r} must be an integer")
    if value < minimum:
        raise ConfigError(f"field {field!r} must be at least {minimum}")
    return value


def resolve_program_path(value: str, base: Path) -> str:
    """Resolve path-like program values while preserving PATH command names."""
    expanded = Path(value).expanduser()
    if expanded.is_absolute():
        return str(expanded.resolve())
    if "/" in value or "\\" in value or value.startswith("."):
        return str((base / expanded).resolve())
    return value


def load_matrix_config(
    path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(
            f"{resolved}: invalid JSON at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error
    except (OSError, UnicodeError) as error:
        raise ConfigError(f"{resolved}: cannot read config: {error}") from error

    if not isinstance(raw, dict):
        raise ConfigError(f"{resolved}: config must be a JSON object")
    allowed = {
        "schema_version",
        *(set(DEFAULT_ARGUMENTS) - {"analysis_plan"}),
        *CONFIG_METADATA_FIELDS,
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(
            f"{resolved}: unknown config fields: {', '.join(repr(key) for key in unknown)}"
        )
    if "schema_version" not in raw:
        raise ConfigError(f"{resolved}: missing required field 'schema_version'")
    version = _non_empty_string(raw["schema_version"], "schema_version")
    if version != CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            f"{resolved}: unsupported schema_version {version!r}; "
            f"expected {CONFIG_SCHEMA_VERSION!r}"
        )

    values: dict[str, Any] = {}
    for field in ("tasks", "agents", "efforts", "topologies"):
        if field in raw:
            values[field] = _string_list(raw[field], field)
    for field in ("models", "agent_bins"):
        if field in raw:
            values[field] = _string_mapping(raw[field], field)
    for field, minimum in (
        ("runs", 1),
        ("jobs", 1),
        ("brief_budget", 256),
        ("timeout", 1),
    ):
        if field in raw:
            values[field] = _integer(raw[field], field, minimum)
    if "arm" in raw:
        arm = _non_empty_string(raw["arm"], "arm")
        if arm not in ("all", "both", "baseline", "brief-only", "wllm"):
            raise ConfigError(
                "field 'arm' must be one of 'all', 'both', 'baseline', "
                "'brief-only' or 'wllm'"
            )
        values["arm"] = arm
    for field in ("no_build", "keep_workspaces", "dry_run"):
        if field in raw:
            if not isinstance(raw[field], bool):
                raise ConfigError(f"field {field!r} must be boolean")
            values[field] = raw[field]
    for field in ("wllm_bin", "output_dir"):
        if field in raw:
            configured = _non_empty_string(raw[field], field)
            values[field] = (resolved.parent / Path(configured).expanduser()).resolve()
    if "agent_bins" in values:
        values["agent_bins"] = {
            agent: resolve_program_path(program, resolved.parent)
            for agent, program in values["agent_bins"].items()
        }

    metadata: dict[str, Any] = {}
    if "selection_salt" in raw:
        metadata["selection_salt"] = _non_empty_string(
            raw["selection_salt"], "selection_salt"
        )
    if "model_snapshot_status" in raw:
        snapshot_status = _non_empty_string(
            raw["model_snapshot_status"], "model_snapshot_status"
        )
        if snapshot_status not in MODEL_SNAPSHOT_STATUSES:
            raise ConfigError(
                "field 'model_snapshot_status' must be one of "
                + ", ".join(repr(value) for value in MODEL_SNAPSHOT_STATUSES)
            )
        if snapshot_status == "attested-immutable":
            configured_agents = values.get("agents")
            configured_models = values.get("models")
            if configured_agents is None or configured_models is None:
                raise ConfigError(
                    "model_snapshot_status 'attested-immutable' requires explicit "
                    "`agents` and `models` fields"
                )
            missing = sorted(set(configured_agents) - set(configured_models))
            extra = sorted(set(configured_models) - set(configured_agents))
            if missing or extra:
                details: list[str] = []
                if missing:
                    details.append("missing models for " + ", ".join(missing))
                if extra:
                    details.append("models for unselected agents " + ", ".join(extra))
                raise ConfigError(
                    "model snapshot attestation must cover exactly the selected "
                    "agents: " + "; ".join(details)
                )
        metadata["model_snapshot_status"] = snapshot_status
        metadata["model_snapshot_publication_eligible"] = (
            snapshot_status == "attested-immutable"
        )
    for field in ("cold", "retain_failures"):
        if field in raw:
            if not isinstance(raw[field], bool):
                raise ConfigError(f"field {field!r} must be boolean")
            metadata[field] = raw[field]
    for field in ("cache_regime", "machine_regime"):
        if field in raw:
            metadata[field] = _non_empty_string(raw[field], field)
    if metadata.get("model_snapshot_status") == "attested-immutable":
        _validate_attested_config(values, metadata)
    return resolved, values, metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    cli_values = vars(parser.parse_args(argv))
    config_option = cli_values.pop("config", None)
    config_path: Path | None = None
    config_values: dict[str, Any] = {}
    config_metadata: dict[str, Any] = {}
    if config_option is not None:
        try:
            config_path, config_values, config_metadata = load_matrix_config(
                config_option
            )
        except ConfigError as error:
            detail = str(error)
            resolved_option = str(config_option.expanduser().resolve())
            if not detail.startswith(resolved_option):
                detail = f"{resolved_option}: {detail}"
            parser.error(detail)

    merged: dict[str, Any] = {
        key: list(value) if isinstance(value, list) else dict(value)
        if isinstance(value, dict)
        else value
        for key, value in DEFAULT_ARGUMENTS.items()
    }
    merged.update(config_values)
    merged.update(cli_values)
    args = argparse.Namespace(**merged)
    args.config = config_path
    args.config_metadata = config_metadata

    if config_metadata.get("model_snapshot_status") == "attested-immutable":
        invalidating = sorted(set(cli_values) & ATTESTED_PROTOCOL_CLI_FIELDS)
        if invalidating:
            parser.error(
                "CLI overrides for "
                + ", ".join("--" + field.replace("_", "-") for field in invalidating)
                + " invalidate model_snapshot_status 'attested-immutable'; "
                "freeze the complete execution protocol in the config"
            )

    args.tasks = split_values(args.tasks, [])
    args.agents = split_values(args.agents, [])
    args.efforts = split_values(args.efforts, [])
    args.topologies = split_values(args.topologies, [])
    for field in ("tasks", "agents", "efforts", "topologies"):
        if not getattr(args, field):
            parser.error(f"{field} must specify at least one value")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.timeout < 1:
        parser.error("--timeout must be at least 1")
    if args.brief_budget < 256:
        parser.error("--brief-budget must be at least 256")

    invalid_agents = sorted(set(args.agents) - set(run.AGENT_DEFAULTS))
    if invalid_agents:
        parser.error(
            "unknown agents: "
            + ", ".join(invalid_agents)
            + "; known agents: "
            + ", ".join(run.AGENT_DEFAULTS)
        )
    invalid_topologies = sorted(set(args.topologies) - set(run.TOPOLOGIES))
    if invalid_topologies:
        parser.error(
            "unknown topologies: "
            + ", ".join(invalid_topologies)
            + "; known topologies: "
            + ", ".join(run.TOPOLOGIES)
        )
    if isinstance(args.models, dict):
        args.models = dict(args.models)
    else:
        args.models = parse_mapping(args.models, "--model", parser)
    if isinstance(args.agent_bins, dict):
        args.agent_bins = dict(args.agent_bins)
    else:
        args.agent_bins = parse_mapping(args.agent_bins, "--agent-bin", parser)
        invocation_dir = Path.cwd()
        args.agent_bins = {
            agent: resolve_program_path(program, invocation_dir)
            for agent, program in args.agent_bins.items()
        }
    for field, mapping in (("models", args.models), ("agent_bins", args.agent_bins)):
        unknown_agents = sorted(set(mapping) - set(run.AGENT_DEFAULTS))
        if unknown_agents:
            parser.error(
                f"{field} names unknown agents: {', '.join(unknown_agents)}"
            )
    if "wllm_bin" in cli_values and args.wllm_bin is not None:
        args.wllm_bin = args.wllm_bin.expanduser().resolve()
    if "output_dir" in cli_values:
        args.output_dir = args.output_dir.expanduser().resolve()
    if args.analysis_plan is not None:
        args.analysis_plan = args.analysis_plan.expanduser().resolve()
    for task in args.tasks:
        try:
            run.load_task(task)
        except SystemExit as error:
            parser.error(str(error))
    return args


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfigError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a JSON object: {path}")
    return value


def validate_analysis_plan(
    path: Path, *, config_sha256: str, protocol_sha256: str
) -> str:
    """Validate and hash the frozen plan that authorizes an execution."""
    plan = _load_json_object(path, "analysis plan")
    publication = plan.get("publication_protocol")
    if not isinstance(publication, dict):
        raise ConfigError("analysis plan lacks a publication_protocol object")
    if publication.get("matrix_config_sha256") != config_sha256:
        raise ConfigError(
            "analysis plan/config mismatch: matrix_config_sha256 does not match"
        )
    if publication.get("execution_protocol_sha256") != protocol_sha256:
        raise ConfigError(
            "analysis plan/execution mismatch: execution_protocol_sha256 does not match"
        )
    embedded = publication.get("execution_protocol")
    if embedded is not None and canonical_sha256(embedded) != protocol_sha256:
        raise ConfigError(
            "analysis plan execution_protocol body does not match its SHA-256"
        )
    return sha256_file(path)


def _git_command(directory: Path, *arguments: str, text: bool = True) -> Any:
    result = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        capture_output=True,
        text=text,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() if text else result.stderr.decode(errors="replace").strip()
        raise ConfigError(
            f"git {' '.join(arguments)} failed in {directory}: {stderr or 'unknown error'}"
        )
    return result.stdout


def attest_preoutcome_git(config_path: Path, plan_path: Path) -> dict[str, str]:
    """Prove config and plan are clean tracked bytes in the same HEAD commit."""
    roots: list[Path] = []
    for path in (config_path, plan_path):
        output = _git_command(path.parent, "rev-parse", "--show-toplevel")
        roots.append(Path(str(output).strip()).resolve())
    if roots[0] != roots[1]:
        raise ConfigError("publication config and analysis plan must be in the same Git repository")
    root = roots[0]
    for label, path in (("config", config_path), ("analysis plan", plan_path)):
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError as error:
            raise ConfigError(f"{label} is outside its reported Git repository") from error
        _git_command(root, "ls-files", "--error-unmatch", "--", relative)
        status = _git_command(
            root, "status", "--porcelain=v1", "--untracked-files=all", "--", relative
        )
        if str(status).strip():
            raise ConfigError(
                f"{label} must have no index or worktree diff from HEAD: {relative}"
            )
        head_bytes = _git_command(root, "show", f"HEAD:{relative}", text=False)
        try:
            disk_bytes = path.read_bytes()
        except OSError as error:
            raise ConfigError(f"cannot read {label} {path}: {error}") from error
        if head_bytes != disk_bytes:
            raise ConfigError(f"{label} bytes are not identical to HEAD: {relative}")
    commit = str(_git_command(root, "rev-parse", "HEAD")).strip()
    timestamp = str(_git_command(root, "show", "-s", "--format=%cI", "HEAD")).strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ConfigError("Git HEAD did not resolve to a full commit object ID")
    if not timestamp:
        raise ConfigError("Git HEAD commit timestamp is empty")
    return {
        "preoutcome_git_commit": commit,
        "preoutcome_timestamp": timestamp,
    }


def build_execution_context(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze protocol/provenance before any benchmark outcome exists."""
    attested = (
        args.config_metadata.get("model_snapshot_status") == "attested-immutable"
    )
    if attested:
        if args.config is None:
            raise ConfigError("attested-immutable execution requires --config")
        protocol = frozen_execution_protocol(args.config)
    else:
        protocol_values = {
            key: getattr(args, key)
            for key in DEFAULT_ARGUMENTS
            if key != "analysis_plan"
        }
        protocol = _protocol_from_values(
            protocol_values, args.config_metadata, require_resolved=False
        )
    protocol_sha256 = canonical_sha256(protocol)
    config_sha256 = sha256_file(args.config) if args.config is not None else None
    plan_sha256: str | None = None
    git_attestation: dict[str, str | None] = {
        "preoutcome_git_commit": None,
        "preoutcome_timestamp": None,
    }
    if args.analysis_plan is not None:
        if config_sha256 is None:
            raise ConfigError("--analysis-plan requires --config")
        plan_sha256 = validate_analysis_plan(
            args.analysis_plan,
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
        )
    if attested and not args.dry_run:
        if args.analysis_plan is None:
            raise ConfigError(
                "attested-immutable publication execution requires --analysis-plan"
            )
        git_attestation = attest_preoutcome_git(args.config, args.analysis_plan)
    provenance = {
        "execution_id": uuid.uuid4().hex,
        "matrix_config_sha256": config_sha256,
        "analysis_plan_sha256": plan_sha256,
        "execution_protocol_sha256": protocol_sha256,
        **git_attestation,
    }
    return protocol, provenance


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
        arm_count = len(run.arm_names(args.arm))
        cell_timeout_seconds = (
            args.runs * arm_count * args.timeout
            + args.runs
            * (
                run.PREPARE_TIMEOUT_SECONDS
                + arm_count * run.GRADER_TIMEOUT_SECONDS
            )
            + run.BUILD_TIMEOUT_SECONDS
            + 5 * run.PREFLIGHT_TIMEOUT_SECONDS
            + CELL_PROCESS_MARGIN_SECONDS
        )
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
                "cell_timeout_seconds": cell_timeout_seconds,
            }
        )
    return cells


def run_cell(
    cell: dict[str, Any], provenance: dict[str, Any] | None = None
) -> dict[str, Any]:
    cell_dir = Path(cell["cell_dir"])
    cell_dir.mkdir(parents=True, exist_ok=False)
    (cell_dir / "command.json").write_text(
        json.dumps(cell["command"], indent=2) + "\n", encoding="utf-8"
    )
    started = time.monotonic()
    timed_out = False
    try:
        result = run.run_bounded_process_tree(
            cell["command"],
            cwd=ROOT,
            timeout=float(cell["cell_timeout_seconds"]),
        )
        stdout, stderr, exit_code = (
            result.stdout,
            result.stderr,
            result.returncode,
        )
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = run.decode_timeout_stream(error.stdout)
        stderr = run.decode_timeout_stream(error.stderr)
        stderr += (
            "\nMatrix cell exceeded its bounded process timeout of "
            f"{cell['cell_timeout_seconds']} seconds.\n"
        )
        exit_code = 124
    duration = time.monotonic() - started
    reports = sorted((cell_dir / "artifacts").glob("*/report.json"))
    report_path = reports[-1] if reports else None
    report_sha256: str | None = None
    if report_path is not None and provenance is not None:
        try:
            report = _load_json_object(report_path, "cell report")
            report["matrix_provenance"] = {
                **provenance,
                "cell_id": cell["id"],
            }
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            native_index_path = report_path.parent / "artifact-index.json"
            if native_index_path.is_file():
                native_index = _load_json_object(native_index_path, "cell artifact index")
                artifacts = native_index.get("artifacts")
                if not isinstance(artifacts, list):
                    raise ConfigError("cell artifact index lacks an artifacts array")
                report_entry = next(
                    (
                        entry
                        for entry in artifacts
                        if isinstance(entry, dict) and entry.get("path") == "report.json"
                    ),
                    None,
                )
                if report_entry is None:
                    raise ConfigError("cell artifact index does not list report.json")
                report_entry["bytes"] = report_path.stat().st_size
                native_index_path.write_text(
                    json.dumps(native_index, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            report_sha256 = sha256_file(report_path)
        except (ConfigError, OSError, UnicodeError) as error:
            stderr += f"\nCould not inject matrix provenance: {error}\n"
            exit_code = 2
    elif report_path is not None:
        report_sha256 = sha256_file(report_path)
    (cell_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (cell_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    return {
        **{key: value for key, value in cell.items() if key != "cell_dir"},
        "cell_dir": str(cell_dir.relative_to(cell_dir.parents[1])),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": round(duration, 3),
        "report": (
            str(report_path.relative_to(cell_dir.parents[1]))
            if report_path is not None
            else None
        ),
        "report_sha256": report_sha256,
        "stdout": str((cell_dir / "stdout.log").relative_to(cell_dir.parents[1])),
        "stderr": str((cell_dir / "stderr.log").relative_to(cell_dir.parents[1])),
    }


def write_plan(
    matrix_dir: Path,
    args: argparse.Namespace,
    cells: list[dict[str, Any]],
    protocol: dict[str, Any],
    provenance: dict[str, Any],
) -> Path:
    plan = {
        "schema_version": "1.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cwd": str(ROOT),
        "provenance": provenance,
        "execution_protocol": protocol,
        "runs_per_cell": args.runs,
        "jobs": args.jobs,
        "timing_comparable": args.jobs == 1,
        "cells": [
            {key: value for key, value in cell.items() if key != "cell_dir"}
            for cell in cells
        ],
    }
    if args.config is not None:
        plan["configuration"] = {
            "path": str(args.config),
            "schema_version": CONFIG_SCHEMA_VERSION,
            **args.config_metadata,
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
    protocol: dict[str, Any],
    provenance: dict[str, Any],
) -> Path:
    failures = sum(1 for cell in cells if cell.get("exit_code") not in (None, 0))
    index = {
        "schema_version": "1.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "provenance": provenance,
        "execution_protocol": protocol,
        "jobs": args.jobs,
        "timing_comparable": args.jobs == 1,
        "cell_count": len(cells),
        "failed_cells": failures,
        "cells": cells,
    }
    if args.config is not None:
        index["configuration"] = {
            "path": str(args.config),
            "schema_version": CONFIG_SCHEMA_VERSION,
            **args.config_metadata,
        }
    path = matrix_dir / "artifact-index.json"
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        protocol, provenance = build_execution_context(args)
    except ConfigError as error:
        print(f"matrix.py: error: {error}", file=sys.stderr)
        return 2
    timestamp = datetime.now(timezone.utc).strftime("matrix-%Y%m%dT%H%M%S%fZ")
    matrix_dir = args.output_dir.resolve() / timestamp
    matrix_dir.mkdir(parents=True, exist_ok=False)
    cells = build_cells(args, matrix_dir)
    write_plan(matrix_dir, args, cells, protocol, provenance)
    if args.dry_run:
        write_index(
            matrix_dir,
            args,
            cells,
            dry_run=True,
            protocol=protocol,
            provenance=provenance,
        )
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
        outcomes = [run_cell(cell, provenance) for cell in cells]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            outcomes = list(pool.map(run_cell, cells, itertools.repeat(provenance)))
    outcomes.sort(key=lambda cell: int(cell["number"]))
    write_index(
        matrix_dir,
        args,
        outcomes,
        dry_run=False,
        protocol=protocol,
        provenance=provenance,
    )
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
