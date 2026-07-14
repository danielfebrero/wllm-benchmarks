#!/usr/bin/env python3
"""Deterministic paired-bootstrap analysis for wllm benchmark reports."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import matrix


PLAN_SCHEMA_VERSION = "1.1"
ANALYSIS_SCHEMA_VERSION = "1.1"
CELL_KEY_FIELDS = ("task", "agent", "model", "effort", "topology")
REQUIRED_CELL_METRICS = (
    "quality_noninferiority",
    "input_token_upper_ci_below_one",
    "end_to_end_time_upper_ci_below_one",
)
PRIMARY_TASKS = (
    "release-evidence",
    "config-precedence",
    "migration-lineage",
)
PROVENANCE_FIELDS = {
    "execution_id",
    "matrix_config_sha256",
    "analysis_plan_sha256",
    "execution_protocol_sha256",
    "preoutcome_git_commit",
    "preoutcome_timestamp",
}
DECISION_RULE_BASE = {
    "scope": "native-primary-family-cellwise",
    "required_cell_metrics": list(REQUIRED_CELL_METRICS),
    "cell_claim_rule": (
        "primary-cell-only-all-required-metrics-and-complete-coverage-must-pass"
    ),
    "family_rule": "complete_declared_family_no_automatic_global_win",
    "multiplicity_adjustment": "bonferroni-primary-cells",
    "non_primary_cells": "exploratory-no-confirmatory-win",
    "infrastructure_policy": (
        "all_declared_cells_present_once_and_no_infrastructure_invalid_reports"
    ),
}
INFRASTRUCTURE_STATUSES = {
    "invalid",
    "infrastructure_invalid",
    "fixture_changed_by_briefing",
    "fixture_copy_mismatch",
}


class AnalysisError(ValueError):
    """A report or predeclared analysis plan is not analyzable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AnalysisError(f"could not read {label} {resolved}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"{label} must be a JSON object: {resolved}")
    return value


CORE_PLAN_FIELDS = {
    "seed",
    "resamples",
    "alpha",
    "quality_noninferiority_margin",
}


def validate_core_plan(plan: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(CORE_PLAN_FIELDS - set(plan))
    if missing:
        raise AnalysisError(
            "analysis plan is missing: " + ", ".join(repr(item) for item in missing)
        )
    seed = plan["seed"]
    resamples = plan["resamples"]
    alpha = plan["alpha"]
    margin = plan["quality_noninferiority_margin"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise AnalysisError("plan `seed` must be a non-negative integer")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1000:
        raise AnalysisError("plan `resamples` must be an integer of at least 1000")
    for name, value in (("alpha", alpha), ("quality_noninferiority_margin", margin)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AnalysisError(f"plan `{name}` must be numeric")
        if not math.isfinite(float(value)):
            raise AnalysisError(f"plan `{name}` must be finite")
    if not 0.0 < float(alpha) < 0.5:
        raise AnalysisError("plan `alpha` must be strictly between 0 and 0.5")
    if not 0.0 <= float(margin) <= 1.0:
        raise AnalysisError(
            "plan `quality_noninferiority_margin` must be within [0, 1]"
        )
    return {
        "seed": seed,
        "resamples": resamples,
        "alpha": float(alpha),
        "quality_noninferiority_margin": float(margin),
    }


def canonical_cells(cells: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        ({field: str(cell[field]) for field in CELL_KEY_FIELDS} for cell in cells),
        key=lambda cell: tuple(cell[field] for field in CELL_KEY_FIELDS),
    )


def cells_sha256(cells: list[dict[str, str]]) -> str:
    encoded = json.dumps(
        cells, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cell_identifier(cell: dict[str, str]) -> str:
    return " | ".join(f"{field}={cell[field]}" for field in CELL_KEY_FIELDS)


def config_protocol(
    path: Path, *, family_id: str, plan_directory: Path | None = None
) -> dict[str, Any]:
    try:
        resolved, values, metadata = matrix.load_matrix_config(path)
    except matrix.ConfigError as error:
        raise AnalysisError(f"invalid publication matrix config: {error}") from error
    required = {"tasks", "agents", "models", "efforts", "topologies", "runs", "jobs"}
    missing = sorted(required - set(values))
    if missing:
        raise AnalysisError(
            "publication config must explicitly freeze: " + ", ".join(missing)
        )
    if metadata.get("selection_salt") is None:
        raise AnalysisError("publication config must declare `selection_salt`")
    if values.get("arm", "both") != "both":
        raise AnalysisError("publication config must run both arms")
    if int(values["jobs"]) != 1:
        raise AnalysisError("publication config must use jobs=1 for wall-time claims")
    runs = int(values["runs"])
    if runs < 6 or runs % 2:
        raise AnalysisError(
            "publication config must use an even runs value of at least 6"
        )
    models = values["models"]
    cells = canonical_cells(
        {
            "task": task,
            "agent": agent,
            "model": models[agent],
            "effort": effort,
            "topology": topology,
        }
        for task, agent, effort, topology in itertools.product(
            values["tasks"], values["agents"], values["efforts"], values["topologies"]
        )
    )
    primary_cells = canonical_cells(
        {
            "task": task,
            "agent": "codex",
            "model": models.get("codex", ""),
            "effort": "medium",
            "topology": "single",
        }
        for task in PRIMARY_TASKS
    )
    missing_primary = sorted(
        set(cell_identifier(cell) for cell in primary_cells)
        - set(cell_identifier(cell) for cell in cells)
    )
    if not models.get("codex") or missing_primary:
        raise AnalysisError(
            "publication config must contain the three fixed primary cells: "
            + "; ".join(missing_primary or ["codex model is missing"])
        )
    try:
        execution_protocol = matrix.frozen_execution_protocol(resolved)
    except (AttributeError, matrix.ConfigError, OSError, ValueError) as error:
        raise AnalysisError(f"could not freeze matrix execution protocol: {error}") from error
    eligible = metadata.get("model_snapshot_status") == "attested-immutable"
    relative_to = (plan_directory or resolved.parent).resolve()
    return {
        "family_id": family_id,
        "matrix_config_path": os.path.relpath(resolved, relative_to),
        "matrix_config_sha256": sha256_file(resolved),
        "matrix_config_schema_version": matrix.CONFIG_SCHEMA_VERSION,
        "selection_salt": metadata["selection_salt"],
        "model_snapshot_status": metadata.get("model_snapshot_status", "unspecified"),
        "model_snapshot_publication_eligible": eligible,
        "runs_per_cell": runs,
        "cell_key_fields": list(CELL_KEY_FIELDS),
        "expected_cell_count": len(cells),
        "expected_cells_sha256": cells_sha256(cells),
        "expected_cells": cells,
        "primary_cell_count": len(primary_cells),
        "primary_cells_sha256": cells_sha256(primary_cells),
        "primary_cells": primary_cells,
        "execution_protocol": execution_protocol,
        "execution_protocol_sha256": canonical_sha256(execution_protocol),
    }


def fixed_decision_rule(protocol: dict[str, Any], alpha: float) -> dict[str, Any]:
    primary_count = int(protocol["primary_cell_count"])
    return {
        "id": str(protocol["family_id"]),
        **DECISION_RULE_BASE,
        "primary_family_size": primary_count,
        "family_alpha": alpha,
        "confirmatory_cell_alpha": alpha / primary_count,
        "primary_cells_sha256": protocol["primary_cells_sha256"],
    }


def freeze_plan(
    template: dict[str, Any], config_path: Path, *, family_id: str,
    plan_directory: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(family_id, str) or not family_id.strip():
        raise AnalysisError("decision family id must be a non-empty string")
    core = validate_core_plan(template)
    protocol = config_protocol(
        config_path,
        family_id=family_id.strip(),
        plan_directory=plan_directory,
    )
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        **core,
        "publication_protocol": protocol,
        "decision_family": fixed_decision_rule(protocol, float(core["alpha"])),
    }


def load_plan(path: Path) -> dict[str, Any]:
    plan = load_json_object(path, "analysis plan")
    allowed = {
        "schema_version",
        *CORE_PLAN_FIELDS,
        "publication_protocol",
        "decision_family",
    }
    unknown = sorted(set(plan) - allowed)
    if unknown:
        raise AnalysisError(
            "unknown analysis-plan fields: " + ", ".join(repr(item) for item in unknown)
        )
    missing = sorted(allowed - set(plan))
    if missing:
        raise AnalysisError(
            "analysis plan is missing: " + ", ".join(repr(item) for item in missing)
        )
    if plan["schema_version"] != PLAN_SCHEMA_VERSION:
        raise AnalysisError(
            f"unsupported plan schema {plan['schema_version']!r}; "
            f"expected {PLAN_SCHEMA_VERSION!r}; run `analysis.py freeze-plan`"
        )
    core = validate_core_plan(plan)
    protocol = plan["publication_protocol"]
    decision = plan["decision_family"]
    if not isinstance(protocol, dict) or not isinstance(decision, dict):
        raise AnalysisError("publication protocol and decision family must be objects")
    protocol_fields = {
        "family_id",
        "matrix_config_path",
        "matrix_config_sha256",
        "matrix_config_schema_version",
        "selection_salt",
        "model_snapshot_status",
        "model_snapshot_publication_eligible",
        "runs_per_cell",
        "cell_key_fields",
        "expected_cell_count",
        "expected_cells_sha256",
        "expected_cells",
        "primary_cell_count",
        "primary_cells_sha256",
        "primary_cells",
        "execution_protocol",
        "execution_protocol_sha256",
    }
    if set(protocol) != protocol_fields:
        raise AnalysisError("publication protocol has missing or unknown fields")
    if protocol["cell_key_fields"] != list(CELL_KEY_FIELDS):
        raise AnalysisError("publication protocol has an unsupported cell key")
    raw_cells = protocol["expected_cells"]
    if not isinstance(raw_cells, list) or not raw_cells:
        raise AnalysisError("publication protocol must declare expected cells")
    cells: list[dict[str, str]] = []
    for index, cell in enumerate(raw_cells):
        if not isinstance(cell, dict) or set(cell) != set(CELL_KEY_FIELDS):
            raise AnalysisError(f"expected cell {index} has an invalid exact key")
        if any(not isinstance(cell[field], str) or not cell[field] for field in CELL_KEY_FIELDS):
            raise AnalysisError(f"expected cell {index} fields must be non-empty strings")
        cells.append({field: cell[field] for field in CELL_KEY_FIELDS})
    canonical = canonical_cells(cells)
    identifiers = [cell_identifier(cell) for cell in canonical]
    if cells != canonical or len(identifiers) != len(set(identifiers)):
        raise AnalysisError("expected cells must be canonical and unique")
    if protocol["expected_cell_count"] != len(cells):
        raise AnalysisError("expected cell count does not match expected cells")
    if protocol["expected_cells_sha256"] != cells_sha256(cells):
        raise AnalysisError("expected cells SHA-256 mismatch")
    raw_primary = protocol["primary_cells"]
    if not isinstance(raw_primary, list) or not raw_primary:
        raise AnalysisError("publication protocol must declare primary cells")
    primary: list[dict[str, str]] = []
    for index, cell in enumerate(raw_primary):
        if not isinstance(cell, dict) or set(cell) != set(CELL_KEY_FIELDS):
            raise AnalysisError(f"primary cell {index} has an invalid exact key")
        if any(
            not isinstance(cell[field], str) or not cell[field]
            for field in CELL_KEY_FIELDS
        ):
            raise AnalysisError(f"primary cell {index} fields must be non-empty strings")
        primary.append({field: cell[field] for field in CELL_KEY_FIELDS})
    canonical_primary = canonical_cells(primary)
    if primary != canonical_primary or len(primary) != len(
        {cell_identifier(cell) for cell in primary}
    ):
        raise AnalysisError("primary cells must be canonical and unique")
    if len(primary) != len(PRIMARY_TASKS) or {
        cell["task"] for cell in primary
    } != set(PRIMARY_TASKS):
        raise AnalysisError("primary family must contain exactly the three fixed tasks")
    if any(
        cell["agent"] != "codex"
        or cell["effort"] != "medium"
        or cell["topology"] != "single"
        for cell in primary
    ):
        raise AnalysisError("primary cells must be codex/medium/single")
    if not set(cell_identifier(cell) for cell in primary).issubset(identifiers):
        raise AnalysisError("primary cells must be members of the expected family")
    if protocol["primary_cell_count"] != len(primary):
        raise AnalysisError("primary cell count does not match primary cells")
    if protocol["primary_cells_sha256"] != cells_sha256(primary):
        raise AnalysisError("primary cells SHA-256 mismatch")
    execution_protocol = protocol["execution_protocol"]
    if not isinstance(execution_protocol, dict) or not execution_protocol:
        raise AnalysisError("publication protocol must embed an execution protocol")
    if protocol["execution_protocol_sha256"] != canonical_sha256(execution_protocol):
        raise AnalysisError("execution protocol SHA-256 mismatch")
    for field in (
        "matrix_config_sha256",
        "expected_cells_sha256",
        "primary_cells_sha256",
        "execution_protocol_sha256",
    ):
        value = protocol[field]
        if not isinstance(value, str) or len(value) != 64:
            raise AnalysisError(f"publication protocol `{field}` must be a SHA-256")
    if (
        isinstance(protocol["runs_per_cell"], bool)
        or not isinstance(protocol["runs_per_cell"], int)
        or protocol["runs_per_cell"] < 6
        or protocol["runs_per_cell"] % 2
    ):
        raise AnalysisError("publication protocol runs must be even and at least 6")
    expected_decision = fixed_decision_rule(protocol, float(core["alpha"]))
    if decision != expected_decision:
        raise AnalysisError("decision family does not match the fixed cellwise rule")
    config_value = protocol["matrix_config_path"]
    if not isinstance(config_value, str) or not config_value:
        raise AnalysisError("publication protocol matrix config path is invalid")
    rebuilt = config_protocol(
        (path.resolve().parent / config_value).resolve(),
        family_id=str(protocol["family_id"]),
        plan_directory=path.resolve().parent,
    )
    if protocol != rebuilt:
        raise AnalysisError(
            "publication protocol/config drift; regenerate the frozen plan before outcomes"
        )
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        **core,
        "publication_protocol": {
            **protocol,
            "expected_cells": cells,
            "primary_cells": primary,
        },
        "decision_family": decision,
    }


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def nested_number(record: dict[str, Any], path: Iterable[str]) -> float | None:
    value: Any = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return finite_number(value)


def infrastructure_valid(record: dict[str, Any]) -> bool:
    if record.get("valid") is False:
        return False
    return str(record.get("status") or "") not in INFRASTRUCTURE_STATUSES


def timeout_substituted(record: dict[str, Any]) -> bool:
    failure = record.get("failure")
    return bool(
        record.get("timed_out") is True
        or (isinstance(failure, dict) and failure.get("censored") is True)
        or str(record.get("status") or "").endswith("_timeout")
    )


def pair_records(
    report: dict[str, Any],
) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise AnalysisError("report `records` must be a non-empty array")
    key_name = "pair" if all(isinstance(item, dict) and "pair" in item for item in records) else "run"
    if not all(isinstance(item, dict) and key_name in item for item in records):
        raise AnalysisError("records must consistently contain either `run` or `pair`")

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        assert isinstance(record, dict)
        arm = record.get("arm")
        if arm not in ("baseline", "wllm"):
            raise AnalysisError(f"record has invalid arm {arm!r}")
        pair_id = str(record[key_name])
        arms = grouped.setdefault(pair_id, {})
        if arm in arms:
            raise AnalysisError(f"duplicate {arm} record for {key_name} {pair_id}")
        arms[arm] = record

    cluster_key = "instance_id" if key_name == "pair" else "run"
    complete: list[dict[str, Any]] = []
    incomplete_ids: list[str] = []
    invalid_ids: list[str] = []
    for pair_id in sorted(grouped, key=lambda item: (not item.isdigit(), int(item) if item.isdigit() else item)):
        arms = grouped[pair_id]
        if set(arms) != {"baseline", "wllm"}:
            incomplete_ids.append(pair_id)
            continue
        if not all(infrastructure_valid(record) for record in arms.values()):
            invalid_ids.append(pair_id)
            continue
        if cluster_key == "instance_id":
            cluster_ids = [record.get("instance_id") for record in arms.values()]
            if any(not isinstance(item, str) or not item for item in cluster_ids):
                raise AnalysisError(
                    f"RepoQA pair {pair_id} lacks a non-empty `instance_id`"
                )
            if len(set(cluster_ids)) != 1:
                raise AnalysisError(
                    f"RepoQA pair {pair_id} has inconsistent `instance_id` values"
                )
            cluster_id = str(cluster_ids[0])
        else:
            cluster_id = pair_id
        complete.append(
            {
                "id": pair_id,
                "cluster_id": cluster_id,
                "baseline": arms["baseline"],
                "wllm": arms["wllm"],
            }
        )
    accounting = {
        "pair_keys_observed": len(grouped),
        "complete_pairs": len(grouped) - len(incomplete_ids),
        "incomplete_pairs": len(incomplete_ids),
        "incomplete_pair_ids": incomplete_ids,
        "infrastructure_invalid_pairs": len(invalid_ids),
        "infrastructure_invalid_pair_ids": invalid_ids,
        "valid_itt_pairs": len(complete),
        "valid_itt_clusters": len({pair["cluster_id"] for pair in complete}),
    }
    return key_name, cluster_key, complete, accounting


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AnalysisError("cannot calculate a percentile of no values")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metric_seed(seed: int, name: str) -> int:
    encoded = hashlib.sha256(f"{seed}\0{name}".encode("utf-8")).digest()
    return int.from_bytes(encoded[:16], "big")


def paired_bootstrap(
    values: list[float],
    *,
    estimator: Callable[[list[float]], float],
    seed: int,
    name: str,
    resamples: int,
    alpha: float,
) -> tuple[float, float, float]:
    if not values:
        raise AnalysisError(f"metric {name!r} has no analyzable pairs")
    rng = random.Random(metric_seed(seed, name))
    size = len(values)
    draws = [
        estimator([values[rng.randrange(size)] for _ in range(size)])
        for _ in range(resamples)
    ]
    return (
        estimator(values),
        percentile(draws, alpha / 2.0),
        percentile(draws, 1.0 - alpha / 2.0),
    )


def metric_result(
    *,
    name: str,
    observations: list[tuple[str, float]],
    omitted_ids: list[str],
    estimator: Callable[[list[float]], float],
    cluster_key: str,
    all_cluster_ids: set[str],
    plan: dict[str, Any],
) -> dict[str, Any]:
    clustered: dict[str, list[float]] = {}
    for cluster_id, value in observations:
        clustered.setdefault(cluster_id, []).append(value)
    ordered_cluster_ids = sorted(clustered)
    cluster_values = [estimator(clustered[cluster_id]) for cluster_id in ordered_cluster_ids]
    omitted_cluster_ids = sorted(all_cluster_ids - set(clustered))
    common = {
        "pair_count": len(observations),
        "cluster_count": len(cluster_values),
        "cluster_key": cluster_key,
        "pairs_per_cluster": {
            cluster_id: len(clustered[cluster_id]) for cluster_id in ordered_cluster_ids
        },
        "omitted_pairs": len(omitted_ids),
        "omitted_pair_ids": omitted_ids,
        "omitted_clusters": len(omitted_cluster_ids),
        "omitted_cluster_ids": omitted_cluster_ids,
    }
    if not cluster_values:
        return {
            **common,
            "estimate": None,
            "confidence_interval": None,
        }
    estimate, lower, upper = paired_bootstrap(
        cluster_values,
        estimator=estimator,
        seed=int(plan["seed"]),
        name=name,
        resamples=int(plan["resamples"]),
        alpha=float(plan["alpha"]),
    )
    return {
        **common,
        "estimate": estimate,
        "confidence_interval": {
            "method": "clustered-paired-percentile-bootstrap",
            "resampling_unit": cluster_key,
            "within_cluster_estimator": (
                "arithmetic_mean" if estimator is statistics.fmean else "geometric_mean"
            ),
            "level": 1.0 - float(plan["alpha"]),
            "lower": lower,
            "upper": upper,
            "resamples": int(plan["resamples"]),
            "seed": int(plan["seed"]),
        },
    }


def analyze_report(report: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    pair_key, cluster_key, pairs, accounting = pair_records(report)
    all_cluster_ids = {str(pair["cluster_id"]) for pair in pairs}
    quality_values: list[tuple[str, float]] = []
    input_ratios: list[tuple[str, float]] = []
    time_ratios: list[tuple[str, float]] = []
    quality_omitted: list[str] = []
    input_omitted: list[str] = []
    time_omitted: list[str] = []
    timeout_pair_ids: list[str] = []
    timeout_cluster_ids: set[str] = set()

    for pair in pairs:
        pair_id = str(pair["id"])
        cluster_id = str(pair["cluster_id"])
        baseline = pair["baseline"]
        treatment = pair["wllm"]
        baseline_quality = nested_number(baseline, ("grade", "score"))
        treatment_quality = nested_number(treatment, ("grade", "score"))
        if (
            baseline_quality is None
            or treatment_quality is None
            or not 0.0 <= baseline_quality <= 1.0
            or not 0.0 <= treatment_quality <= 1.0
        ):
            quality_omitted.append(pair_id)
        else:
            quality_values.append(
                (cluster_id, treatment_quality - baseline_quality)
            )

        baseline_input = nested_number(baseline, ("usage", "input_tokens"))
        treatment_input = nested_number(treatment, ("usage", "input_tokens"))
        if (
            baseline_input is None
            or treatment_input is None
            or baseline_input <= 0.0
            or treatment_input <= 0.0
        ):
            input_omitted.append(pair_id)
        else:
            input_ratios.append((cluster_id, treatment_input / baseline_input))

        baseline_time = nested_number(baseline, ("duration_seconds",))
        treatment_time = nested_number(treatment, ("duration_seconds",))
        if (
            baseline_time is None
            or treatment_time is None
            or baseline_time <= 0.0
            or treatment_time <= 0.0
        ):
            time_omitted.append(pair_id)
        else:
            time_ratios.append((cluster_id, treatment_time / baseline_time))
            if timeout_substituted(baseline) or timeout_substituted(treatment):
                timeout_pair_ids.append(pair_id)
                timeout_cluster_ids.add(cluster_id)

    quality = metric_result(
        name="quality_mean_delta",
        observations=quality_values,
        omitted_ids=quality_omitted,
        estimator=statistics.fmean,
        cluster_key=cluster_key,
        all_cluster_ids=all_cluster_ids,
        plan=plan,
    )
    if quality["confidence_interval"] is not None:
        quality["noninferiority"] = {
            "margin": float(plan["quality_noninferiority_margin"]),
            "criterion": "lower_ci_greater_than_or_equal_to_negative_margin",
            "met": quality["confidence_interval"]["lower"]
            >= -float(plan["quality_noninferiority_margin"]),
        }
    else:
        quality["noninferiority"] = {
            "margin": float(plan["quality_noninferiority_margin"]),
            "criterion": "lower_ci_greater_than_or_equal_to_negative_margin",
            "met": None,
        }

    input_ratio = metric_result(
        name="input_token_geometric_mean_ratio",
        observations=input_ratios,
        omitted_ids=input_omitted,
        estimator=statistics.geometric_mean,
        cluster_key=cluster_key,
        all_cluster_ids=all_cluster_ids,
        plan=plan,
    )
    time_ratio = metric_result(
        name="end_to_end_time_geometric_mean_ratio",
        observations=time_ratios,
        omitted_ids=time_omitted,
        estimator=statistics.geometric_mean,
        cluster_key=cluster_key,
        all_cluster_ids=all_cluster_ids,
        plan=plan,
    )
    for result in (input_ratio, time_ratio):
        interval = result["confidence_interval"]
        result["efficiency_criterion"] = {
            "criterion": "upper_ci_strictly_below_one",
            "met": interval["upper"] < 1.0 if interval is not None else None,
        }
    time_ratio.update(
        {
            "estimand_label": (
                "descriptive paired end-to-end ratio with timeout substitution; "
                "not survival inference"
            ),
            "timeout_substituted_pairs": len(timeout_pair_ids),
            "timeout_substituted_pair_ids": timeout_pair_ids,
            "timeout_substituted_clusters": len(timeout_cluster_ids),
            "timeout_substituted_cluster_ids": sorted(timeout_cluster_ids),
        }
    )

    return {
        "benchmark": report.get("benchmark"),
        "pair_key": pair_key,
        "cluster_key": cluster_key,
        "cell": {
            key: report.get(key)
            for key in ("agent", "model", "reasoning", "effort", "topology")
            if report.get(key) is not None
        },
        "pair_accounting": accounting,
        "metrics": {
            "quality_mean_delta_wllm_minus_baseline": quality,
            "input_token_geometric_mean_ratio_wllm_over_baseline": input_ratio,
            "end_to_end_time_geometric_mean_ratio_wllm_over_baseline": time_ratio,
        },
    }


def native_report_cell(report: dict[str, Any]) -> dict[str, str]:
    if report.get("benchmark") != "wllm-agent-ab":
        raise AnalysisError(
            "publication analysis accepts only native wllm-agent-ab reports; "
            "use --exploratory for a separately planned diagnostic"
        )
    task = report.get("task")
    effort = report.get("reasoning")
    raw = {
        "task": task.get("id") if isinstance(task, dict) else None,
        "agent": report.get("agent"),
        "model": report.get("model"),
        "effort": effort,
        "topology": report.get("topology"),
    }
    if any(not isinstance(raw[field], str) or not raw[field] for field in CELL_KEY_FIELDS):
        raise AnalysisError("native report lacks its exact task/agent/model/effort/topology key")
    return {field: str(raw[field]) for field in CELL_KEY_FIELDS}


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise AnalysisError(f"{label} must be a 64-digit SHA-256")
    return value.lower()


def validate_matrix_provenance(
    provenance: Any, *, plan: dict[str, Any], plan_sha256: str
) -> dict[str, str]:
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_FIELDS:
        raise AnalysisError(
            "matrix provenance must contain exactly: "
            + ", ".join(sorted(PROVENANCE_FIELDS))
        )
    if any(
        not isinstance(provenance[key], str) or not provenance[key]
        for key in PROVENANCE_FIELDS
    ):
        raise AnalysisError("matrix provenance values must be non-empty strings")
    normalized = {key: provenance[key] for key in PROVENANCE_FIELDS}
    try:
        parsed_execution = uuid.UUID(normalized["execution_id"])
    except (ValueError, AttributeError) as error:
        raise AnalysisError("matrix execution_id must be a UUID") from error
    if parsed_execution.version != 4:
        raise AnalysisError("matrix execution_id must be a random UUIDv4")
    for field in (
        "matrix_config_sha256",
        "analysis_plan_sha256",
        "execution_protocol_sha256",
    ):
        normalized[field] = require_sha256(normalized[field], f"provenance {field}")
    commit = normalized["preoutcome_git_commit"].lower()
    if len(commit) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise AnalysisError("preoutcome_git_commit must be a Git object ID")
    normalized["preoutcome_git_commit"] = commit
    try:
        timestamp = datetime.fromisoformat(
            normalized["preoutcome_timestamp"].replace("Z", "+00:00")
        )
    except ValueError as error:
        raise AnalysisError("preoutcome_timestamp must be ISO-8601") from error
    if timestamp.tzinfo is None:
        raise AnalysisError("preoutcome_timestamp must include a timezone")
    protocol = plan["publication_protocol"]
    expected = {
        "matrix_config_sha256": protocol["matrix_config_sha256"],
        "analysis_plan_sha256": plan_sha256,
        "execution_protocol_sha256": protocol["execution_protocol_sha256"],
    }
    for field, value in expected.items():
        if normalized[field] != value:
            raise AnalysisError(f"matrix provenance {field} does not match frozen plan")
    return normalized


def require_aware_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise AnalysisError(f"{label} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AnalysisError(f"{label} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise AnalysisError(f"{label} must include a timezone")
    return parsed


def confined_report_path(index_root: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise AnalysisError("matrix cell report path must be a non-empty string")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise AnalysisError(f"matrix report path escapes its index: {raw_path!r}")
    resolved = (index_root / relative).resolve()
    try:
        resolved.relative_to(index_root)
    except ValueError as error:
        raise AnalysisError(f"matrix report path escapes its index: {raw_path!r}") from error
    if not resolved.is_file():
        raise AnalysisError(f"matrix report is missing: {resolved}")
    return resolved


def publication_index_reports(
    index_path: Path, *, plan: dict[str, Any], plan_path: Path
) -> tuple[
    list[tuple[Path, dict[str, Any]]],
    list[dict[str, str]],
    dict[str, Any],
]:
    resolved_index = index_path.expanduser().resolve()
    index = load_json_object(resolved_index, "matrix index")
    if index.get("dry_run") is not False:
        raise AnalysisError("publication analysis refuses a dry-run matrix index")
    protocol = plan["publication_protocol"]
    plan_sha = sha256_file(plan_path)
    provenance = validate_matrix_provenance(
        index.get("provenance"), plan=plan, plan_sha256=plan_sha
    )
    preoutcome_time = require_aware_timestamp(
        provenance["preoutcome_timestamp"], "preoutcome_timestamp"
    )
    if require_aware_timestamp(index.get("generated_at"), "matrix generated_at") < preoutcome_time:
        raise AnalysisError("matrix index predates its pre-outcome Git attestation")
    execution_protocol = index.get("execution_protocol")
    if execution_protocol != protocol["execution_protocol"]:
        raise AnalysisError("matrix index execution protocol differs from frozen plan")
    if canonical_sha256(execution_protocol) != provenance["execution_protocol_sha256"]:
        raise AnalysisError("matrix index execution protocol SHA-256 mismatch")
    cells = index.get("cells")
    if not isinstance(cells, list):
        raise AnalysisError("matrix index `cells` must be an array")
    if index.get("cell_count") != len(cells):
        raise AnalysisError("matrix index cell_count does not match cells")
    expected = {
        cell_identifier(cell): cell for cell in protocol["expected_cells"]
    }
    indexed: dict[str, dict[str, Any]] = {}
    cell_ids: set[str] = set()
    for position, cell_record in enumerate(cells):
        if not isinstance(cell_record, dict):
            raise AnalysisError(f"matrix index cell {position} is not an object")
        raw_cell = {field: cell_record.get(field) for field in CELL_KEY_FIELDS}
        if any(
            not isinstance(raw_cell[field], str) or not raw_cell[field]
            for field in CELL_KEY_FIELDS
        ):
            raise AnalysisError(f"matrix index cell {position} lacks an exact cell key")
        cell = {field: str(raw_cell[field]) for field in CELL_KEY_FIELDS}
        identifier = cell_identifier(cell)
        if identifier not in expected:
            raise AnalysisError(f"undeclared matrix-index cell: {identifier}")
        if identifier in indexed:
            raise AnalysisError(f"duplicate matrix-index cell: {identifier}")
        cell_id = cell_record.get("id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in cell_ids:
            raise AnalysisError("matrix index cell IDs must be unique non-empty strings")
        cell_ids.add(cell_id)
        indexed[identifier] = cell_record
    missing_index_cells = sorted(set(expected) - set(indexed))
    if missing_index_cells or len(indexed) != int(protocol["expected_cell_count"]):
        raise AnalysisError(
            "matrix index does not contain the exact frozen cell family"
            + (": " + "; ".join(missing_index_cells) if missing_index_cells else "")
        )

    root = resolved_index.parent.resolve()
    loaded: list[tuple[Path, dict[str, Any]]] = []
    sources: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    for identifier in sorted(indexed):
        cell_record = indexed[identifier]
        report_value = cell_record.get("report")
        digest_value = cell_record.get("report_sha256")
        if report_value is None:
            if digest_value is not None:
                raise AnalysisError("matrix cell without a report cannot declare its SHA")
            continue
        report_path = confined_report_path(root, report_value)
        if report_path in seen_paths:
            raise AnalysisError(f"duplicate matrix report path: {report_path}")
        seen_paths.add(report_path)
        expected_digest = require_sha256(digest_value, "matrix report_sha256")
        observed_digest = sha256_file(report_path)
        if expected_digest != observed_digest:
            raise AnalysisError(f"matrix report SHA-256 mismatch: {report_path}")
        report = load_json_object(report_path, "matrix report")
        if require_aware_timestamp(
            report.get("generated_at"), "report generated_at"
        ) < preoutcome_time:
            raise AnalysisError(
                f"report predates the pre-outcome Git attestation: {report_path}"
            )
        expected_report_provenance = {
            **provenance,
            "cell_id": cell_record["id"],
        }
        if report.get("matrix_provenance") != expected_report_provenance:
            raise AnalysisError(
                f"report matrix provenance mismatch for cell {identifier}"
            )
        if native_report_cell(report) != expected[identifier]:
            raise AnalysisError(f"report cell key mismatch for {identifier}")
        loaded.append((report_path, report))
        sources.append({"path": str(report_path), "sha256": observed_digest})
    index_source = {
        "path": str(resolved_index),
        "sha256": sha256_file(resolved_index),
        "provenance": provenance,
        "execution_protocol_sha256": provenance["execution_protocol_sha256"],
    }
    return loaded, sources, index_source


def cell_claim(
    analysis: dict[str, Any], report: dict[str, Any], *, confirmatory: bool
) -> dict[str, Any]:
    metrics = analysis["metrics"]
    accounting = analysis["pair_accounting"]
    expected_pairs = int(accounting["valid_itt_pairs"])
    expected_clusters = int(accounting["valid_itt_clusters"])
    metric_objects = {
        "quality": metrics["quality_mean_delta_wllm_minus_baseline"],
        "input_tokens": metrics[
            "input_token_geometric_mean_ratio_wllm_over_baseline"
        ],
        "end_to_end_time": metrics[
            "end_to_end_time_geometric_mean_ratio_wllm_over_baseline"
        ],
    }
    coverage = {
        name: metric["pair_count"] == expected_pairs
        and metric["cluster_count"] == expected_clusters
        for name, metric in metric_objects.items()
    }
    token_attestation_required = report.get("topology") == "native-multi-agent"
    attestation = report.get("input_token_coverage")
    token_attestation_complete = not token_attestation_required or (
        isinstance(attestation, dict)
        and attestation.get("scope") == "parent-and-children"
        and attestation.get("complete") is True
    )
    coverage["multi_agent_parent_and_children"] = token_attestation_complete
    complete_coverage = all(coverage.values())
    observed = {
        "quality_noninferiority": metrics[
            "quality_mean_delta_wllm_minus_baseline"
        ]["noninferiority"]["met"],
        "input_token_upper_ci_below_one": metrics[
            "input_token_geometric_mean_ratio_wllm_over_baseline"
        ]["efficiency_criterion"]["met"],
        "end_to_end_time_upper_ci_below_one": metrics[
            "end_to_end_time_geometric_mean_ratio_wllm_over_baseline"
        ]["efficiency_criterion"]["met"],
    }
    win = complete_coverage and all(value is True for value in observed.values())
    return {
        "inference_role": "confirmatory-primary" if confirmatory else "exploratory",
        "metric_coverage_complete": coverage,
        "all_required_metric_coverage_complete": complete_coverage,
        "multi_agent_input_attestation": attestation,
        "required_metrics": observed,
        "wllm_efficiency_win": win if confirmatory else None,
        "confirmatory_wllm_efficiency_win": win if confirmatory else None,
    }


def publication_analysis(
    loaded: list[tuple[Path, dict[str, Any]]], plan: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol = plan["publication_protocol"]
    expected = {
        cell_identifier(cell): cell for cell in protocol["expected_cells"]
    }
    primary_ids = {
        cell_identifier(cell) for cell in protocol["primary_cells"]
    }
    observed: dict[str, tuple[dict[str, str], dict[str, Any]]] = {}
    for _, report in loaded:
        cell = native_report_cell(report)
        identifier = cell_identifier(cell)
        if identifier not in expected:
            raise AnalysisError(f"undeclared publication cell: {identifier}")
        if identifier in observed:
            raise AnalysisError(f"duplicate publication cell: {identifier}")
        observed[identifier] = (cell, report)

    missing = sorted(set(expected) - set(observed))
    analyses: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    nonclaimable_primary: list[dict[str, str]] = []
    for identifier in sorted(observed):
        cell, report = observed[identifier]
        confirmatory = identifier in primary_ids
        cell_plan = dict(plan)
        cell_plan["alpha"] = (
            float(plan["alpha"]) / len(primary_ids)
            if confirmatory
            else float(plan["alpha"])
        )
        analysis = analyze_report(report, cell_plan)
        analysis["cell"] = cell
        analysis["statistical_role"] = (
            "confirmatory-primary" if confirmatory else "exploratory"
        )
        analysis["family_alpha"] = float(plan["alpha"])
        analysis["cell_alpha"] = float(cell_plan["alpha"])
        analysis["cell_decision"] = cell_claim(
            analysis, report, confirmatory=confirmatory
        )
        accounting = analysis["pair_accounting"]
        reasons: list[str] = []
        runs = report.get("runs_requested")
        if runs != protocol["runs_per_cell"]:
            reasons.append("runs_requested_mismatch")
        if report.get("status") != "valid":
            reasons.append("report_status_not_valid")
        if accounting["pair_keys_observed"] != protocol["runs_per_cell"]:
            reasons.append("pair_count_mismatch")
        if accounting["incomplete_pairs"]:
            reasons.append("incomplete_pairs")
        if accounting["infrastructure_invalid_pairs"]:
            reasons.append("infrastructure_invalid_pairs")
        if confirmatory and not analysis["cell_decision"][
            "all_required_metric_coverage_complete"
        ]:
            nonclaimable_primary.append(
                {"cell": identifier, "reason": "incomplete_confirmatory_metric_coverage"}
            )
        if reasons:
            invalid.append({"cell": identifier, "reasons": ",".join(reasons)})
        analyses.append(analysis)

    eligible = bool(protocol["model_snapshot_publication_eligible"])
    ready = eligible and not missing and not invalid and not nonclaimable_primary
    blockers: list[str] = []
    if not eligible:
        blockers.append("model_snapshot_status_not_attested_immutable")
    if missing:
        blockers.append("missing_declared_cells")
    if invalid:
        blockers.append("infrastructure_invalid_or_incomplete_cells")
    if nonclaimable_primary:
        blockers.append("primary_metric_coverage_incomplete")
    assessment = {
        "family_id": protocol["family_id"],
        "decision_scope": "cellwise; no automatic pooled or global win",
        "expected_cells": len(expected),
        "reported_cells": len(observed),
        "missing_cells": len(missing),
        "missing_cell_ids": missing,
        "invalid_cells": len(invalid),
        "invalid_cell_details": invalid,
        "model_snapshot_publication_eligible": eligible,
        "primary_cells": len(primary_ids),
        "primary_cell_ids": sorted(primary_ids),
        "primary_family_alpha": float(plan["alpha"]),
        "primary_confirmatory_cell_alpha": float(plan["alpha"])
        / len(primary_ids),
        "nonclaimable_primary_cells": len(nonclaimable_primary),
        "nonclaimable_primary_cell_details": nonclaimable_primary,
        "publication_ready": ready,
        "publication_blockers": blockers,
        "cell_wins": sum(
            analysis["cell_decision"]["confirmatory_wllm_efficiency_win"] is True
            for analysis in analyses
        ),
        "confirmatory_primary_wins": sum(
            analysis["cell_decision"]["confirmatory_wllm_efficiency_win"] is True
            for analysis in analyses
        ),
        "exploratory_cells": len(expected) - len(primary_ids),
        "global_win": None,
    }
    return analyses, assessment


def write_json_result(value: dict[str, Any], destination: Path | None) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination is None:
        print(encoded, end="")
        return
    resolved = destination.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(encoded, encoding="utf-8")
    print(resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "freeze-plan", help="bind statistical parameters to a frozen matrix config"
    )
    freeze.add_argument("--template", type=Path, required=True)
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--family-id", default="native-publication-v1")
    freeze.add_argument("--output", type=Path, required=True)

    run_command = commands.add_parser("run", help="analyze completed reports")
    run_command.add_argument("--plan", type=Path, required=True)
    run_command.add_argument("--report", type=Path, action="append", default=[])
    run_command.add_argument("--reports-root", type=Path, action="append", default=[])
    run_command.add_argument(
        "--matrix-index",
        type=Path,
        action="append",
        default=[],
        help="single attested artifact-index.json required for publication mode",
    )
    run_command.add_argument("--output", type=Path)
    run_command.add_argument(
        "--exploratory",
        action="store_true",
        help="analyze without a publishable family decision",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze-plan":
            template = load_json_object(args.template, "analysis-plan template")
            plan = freeze_plan(
                template,
                args.config,
                family_id=args.family_id,
                plan_directory=args.output.expanduser().resolve().parent,
            )
            write_json_result(plan, args.output)
            return 0

        plan_path = args.plan.expanduser().resolve()
        if args.exploratory:
            raw_plan = load_json_object(plan_path, "analysis plan")
            if raw_plan.get("schema_version") != PLAN_SCHEMA_VERSION:
                raise AnalysisError(
                    f"exploratory plan schema must be {PLAN_SCHEMA_VERSION!r}"
                )
            plan = {"schema_version": PLAN_SCHEMA_VERSION, **validate_core_plan(raw_plan)}
        else:
            plan = load_plan(plan_path)

        if args.exploratory:
            if args.matrix_index:
                raise AnalysisError("--matrix-index is reserved for publication mode")
            report_paths = [path.expanduser().resolve() for path in args.report]
            for root in args.reports_root:
                report_paths.extend(root.expanduser().resolve().rglob("report.json"))
            report_paths.sort(key=str)
            if not report_paths:
                raise AnalysisError(
                    "exploratory mode requires at least one --report or --reports-root"
                )
            loaded = [
                (path, load_json_object(path, "benchmark report"))
                for path in report_paths
            ]
            sources = [
                {"path": str(path), "sha256": sha256_file(path)}
                for path, _ in loaded
            ]
            matrix_index_source = None
            analyses = [analyze_report(report, plan) for _, report in loaded]
            assessment = {
                "publication_ready": False,
                "publication_blockers": ["explicit_exploratory_mode"],
                "global_win": None,
            }
            return_code = 0
        else:
            if args.report or args.reports_root:
                raise AnalysisError(
                    "publication mode refuses --report/--reports-root; "
                    "supply exactly one --matrix-index"
                )
            if len(args.matrix_index) != 1:
                raise AnalysisError(
                    "publication mode requires exactly one --matrix-index"
                )
            loaded, sources, matrix_index_source = publication_index_reports(
                args.matrix_index[0], plan=plan, plan_path=plan_path
            )
            analyses, assessment = publication_analysis(loaded, plan)
            return_code = 0 if assessment["publication_ready"] else 2
        output = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "analysis_plan": plan,
            "analysis_plan_sha256": sha256_file(plan_path),
            "matrix_index": matrix_index_source,
            "source_reports": sources,
            "publication_assessment": assessment,
            "analyses": analyses,
        }
        write_json_result(output, args.output)
        return return_code
    except AnalysisError as error:
        print(f"analysis: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
