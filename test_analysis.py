from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import analysis


PLAN = {
    "schema_version": "1.1",
    "seed": 17,
    "resamples": 1000,
    "alpha": 0.05,
    "quality_noninferiority_margin": 0.1,
}


def native_record(
    run: int,
    arm: str,
    *,
    score: float,
    duration: float,
    tokens: int | None,
    valid: bool = True,
    status: str = "valid",
    timed_out: bool = False,
) -> dict[str, object]:
    return {
        "run": run,
        "arm": arm,
        "valid": valid,
        "status": status,
        "timed_out": timed_out,
        "duration_seconds": duration,
        "usage": {"input_tokens": tokens},
        "grade": {"score": score},
    }


class AnalysisTests(unittest.TestCase):
    def write_executable(self, path: Path) -> Path:
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    def write_publication_config(
        self, root: Path, *, tasks: list[str] | None = None, status: str = "attested-immutable"
    ) -> Path:
        agent_binary = self.write_executable(root / "codex-test-binary")
        wllm_binary = self.write_executable(root / "wllm-test-binary")
        path = root / "publication.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "selection_salt": "frozen-test-family",
                    "model_snapshot_status": status,
                    "tasks": tasks or list(analysis.PRIMARY_TASKS),
                    "agents": ["codex"],
                    "models": {"codex": "immutable-model-snapshot"},
                    "agent_bins": {"codex": str(agent_binary)},
                    "efforts": ["medium"],
                    "topologies": ["single"],
                    "arm": "both",
                    "brief_budget": 1200,
                    "timeout": 900,
                    "runs": 6,
                    "jobs": 1,
                    "wllm_bin": str(wllm_binary),
                    "no_build": True,
                    "cache_regime": "cold-new-provider-session-per-arm",
                    "machine_regime": "dedicated-test-host",
                }
            ),
            encoding="utf-8",
        )
        return path

    def publication_report(
        self,
        task: str = "release-evidence",
        *,
        topology: str = "single",
    ) -> dict[str, object]:
        records = []
        for run in range(1, 7):
            records.extend(
                [
                    native_record(run, "baseline", score=1, duration=10, tokens=100),
                    native_record(run, "wllm", score=1, duration=5, tokens=50),
                ]
            )
        return {
            "benchmark": "wllm-agent-ab",
            "generated_at": "2026-07-14T01:00:00+00:00",
            "status": "valid",
            "task": {"id": task},
            "agent": "codex",
            "model": "immutable-model-snapshot",
            "reasoning": "medium",
            "topology": topology,
            "runs_requested": 6,
            "records": records,
        }

    def write_frozen_plan(
        self, root: Path, config: Path, *, family_id: str = "test-family"
    ) -> tuple[Path, dict[str, object]]:
        plan = analysis.freeze_plan(
            PLAN, config, family_id=family_id, plan_directory=root
        )
        path = root / "plan.json"
        path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
        return path, plan

    def write_matrix_index(
        self,
        root: Path,
        plan_path: Path,
        plan: dict[str, object],
        reports: dict[str, dict[str, object]],
    ) -> tuple[Path, dict[str, object]]:
        protocol = plan["publication_protocol"]
        assert isinstance(protocol, dict)
        provenance = {
            "execution_id": "12345678-1234-4234-8234-123456789abc",
            "matrix_config_sha256": protocol["matrix_config_sha256"],
            "analysis_plan_sha256": analysis.sha256_file(plan_path),
            "execution_protocol_sha256": protocol["execution_protocol_sha256"],
            "preoutcome_git_commit": "a" * 40,
            "preoutcome_timestamp": "2026-07-14T00:00:00+00:00",
        }
        index_cells = []
        for number, cell in enumerate(protocol["expected_cells"], 1):
            assert isinstance(cell, dict)
            identifier = analysis.cell_identifier(cell)
            cell_id = f"cell-{number:03d}"
            indexed = {**cell, "id": cell_id}
            report = reports.get(identifier)
            if report is None:
                indexed.update({"report": None, "report_sha256": None})
            else:
                report["matrix_provenance"] = {**provenance, "cell_id": cell_id}
                relative = Path("cells") / cell_id / "report.json"
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(report, sort_keys=True), encoding="utf-8"
                )
                indexed.update(
                    {
                        "report": str(relative),
                        "report_sha256": analysis.sha256_file(destination),
                    }
                )
            index_cells.append(indexed)
        index = {
            "schema_version": "1.1",
            "generated_at": "2026-07-14T02:00:00+00:00",
            "dry_run": False,
            "cell_count": len(index_cells),
            "provenance": provenance,
            "execution_protocol": protocol["execution_protocol"],
            "cells": index_cells,
        }
        path = root / "artifact-index.json"
        path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
        return path, index

    def complete_publication_reports(
        self, plan: dict[str, object]
    ) -> dict[str, dict[str, object]]:
        protocol = plan["publication_protocol"]
        assert isinstance(protocol, dict)
        result = {}
        for cell in protocol["expected_cells"]:
            assert isinstance(cell, dict)
            report = self.publication_report(
                str(cell["task"]), topology=str(cell["topology"])
            )
            report["agent"] = cell["agent"]
            report["model"] = cell["model"]
            report["reasoning"] = cell["effort"]
            result[analysis.cell_identifier(cell)] = report
        return result

    def test_native_keeps_itt_and_excludes_infrastructure_invalid_pairs(self) -> None:
        report = {
            "benchmark": "wllm-agent-ab",
            "agent": "codex",
            "model": "model",
            "reasoning": "medium",
            "topology": "single",
            "records": [
                native_record(1, "baseline", score=1, duration=10, tokens=100),
                native_record(1, "wllm", score=1, duration=5, tokens=50),
                native_record(2, "baseline", score=1, duration=10, tokens=100),
                native_record(
                    2,
                    "wllm",
                    score=0,
                    duration=20,
                    tokens=None,
                    status="outcome_failure",
                    timed_out=True,
                ),
                native_record(3, "baseline", score=1, duration=10, tokens=100),
                native_record(
                    3,
                    "wllm",
                    score=0,
                    duration=1,
                    tokens=1,
                    valid=False,
                    status="invalid",
                ),
            ],
        }
        result = analysis.analyze_report(report, PLAN)
        accounting = result["pair_accounting"]
        self.assertEqual(accounting["valid_itt_pairs"], 2)
        self.assertEqual(accounting["infrastructure_invalid_pair_ids"], ["3"])
        metrics = result["metrics"]
        quality = metrics["quality_mean_delta_wllm_minus_baseline"]
        self.assertEqual(quality["pair_count"], 2)
        self.assertEqual(quality["cluster_count"], 2)
        self.assertEqual(quality["cluster_key"], "run")
        self.assertEqual(quality["estimate"], -0.5)
        input_ratio = metrics[
            "input_token_geometric_mean_ratio_wllm_over_baseline"
        ]
        self.assertEqual(input_ratio["pair_count"], 1)
        self.assertEqual(input_ratio["cluster_count"], 1)
        self.assertEqual(input_ratio["omitted_pair_ids"], ["2"])
        time_ratio = metrics[
            "end_to_end_time_geometric_mean_ratio_wllm_over_baseline"
        ]
        self.assertEqual(time_ratio["pair_count"], 2)
        self.assertEqual(time_ratio["cluster_count"], 2)
        self.assertEqual(time_ratio["timeout_substituted_pair_ids"], ["2"])
        self.assertIn("not survival inference", time_ratio["estimand_label"])

    def test_repoqa_uses_pair_key_and_known_fixture_status_is_invalid(self) -> None:
        records = []
        for pair, status in ((1, "completed"), (2, "completed")):
            records.extend(
                [
                    {
                        "pair": pair,
                        "instance_id": "python::repo::needle",
                        "arm": "baseline",
                        "status": "completed",
                        "duration_seconds": 8,
                        "usage": {"input_tokens": 16000},
                        "grade": {"score": 1.0},
                    },
                    {
                        "pair": pair,
                        "instance_id": "python::repo::needle",
                        "arm": "wllm",
                        "status": (
                            "fixture_changed_by_briefing" if pair == 2 else status
                        ),
                        "duration_seconds": 4,
                        "usage": {"input_tokens": 1000},
                        "grade": {"score": 1.0},
                    },
                ]
            )
        result = analysis.analyze_report(
            {"benchmark": "repoqa-snf-derived-agent-ab", "records": records}, PLAN
        )
        self.assertEqual(result["pair_key"], "pair")
        self.assertEqual(result["cluster_key"], "instance_id")
        self.assertEqual(result["pair_accounting"]["valid_itt_pairs"], 1)
        self.assertEqual(result["pair_accounting"]["valid_itt_clusters"], 1)
        self.assertEqual(
            result["pair_accounting"]["infrastructure_invalid_pair_ids"], ["2"]
        )

    def test_repoqa_aggregates_repetitions_then_bootstraps_instances(self) -> None:
        records = []
        pair_number = 0
        for instance_id, repetitions, treatment_score, ratio in (
            ("python::repo-a::needle", 4, 1.0, 0.5),
            ("python::repo-b::needle", 1, 0.0, 2.0),
        ):
            for _ in range(repetitions):
                pair_number += 1
                records.extend(
                    [
                        {
                            "pair": pair_number,
                            "instance_id": instance_id,
                            "arm": "baseline",
                            "status": "completed",
                            "duration_seconds": 10.0,
                            "usage": {"input_tokens": 100},
                            "grade": {"score": 0.0 if treatment_score else 1.0},
                        },
                        {
                            "pair": pair_number,
                            "instance_id": instance_id,
                            "arm": "wllm",
                            "status": "completed",
                            "duration_seconds": 10.0 * ratio,
                            "usage": {"input_tokens": int(100 * ratio)},
                            "grade": {"score": treatment_score},
                        },
                    ]
                )
        result = analysis.analyze_report(
            {"benchmark": "repoqa-snf-derived-agent-ab", "records": records}, PLAN
        )
        metrics = result["metrics"]
        quality = metrics["quality_mean_delta_wllm_minus_baseline"]
        self.assertEqual(quality["pair_count"], 5)
        self.assertEqual(quality["cluster_count"], 2)
        self.assertEqual(quality["pairs_per_cluster"]["python::repo-a::needle"], 4)
        self.assertEqual(quality["estimate"], 0.0)
        for name in (
            "input_token_geometric_mean_ratio_wllm_over_baseline",
            "end_to_end_time_geometric_mean_ratio_wllm_over_baseline",
        ):
            metric = metrics[name]
            self.assertEqual(metric["pair_count"], 5)
            self.assertEqual(metric["cluster_count"], 2)
            self.assertAlmostEqual(metric["estimate"], 1.0)

    def test_repoqa_requires_consistent_instance_cluster(self) -> None:
        report = {
            "benchmark": "repoqa-snf-derived-agent-ab",
            "records": [
                {
                    "pair": 1,
                    "instance_id": "instance-a",
                    "arm": "baseline",
                    "status": "completed",
                },
                {
                    "pair": 1,
                    "instance_id": "instance-b",
                    "arm": "wllm",
                    "status": "completed",
                },
            ],
        }
        with self.assertRaises(analysis.AnalysisError):
            analysis.analyze_report(report, PLAN)

    def test_bootstrap_is_deterministic_and_metric_specific(self) -> None:
        values = [0.5, 0.75, 1.0, 1.25]
        first = analysis.paired_bootstrap(
            values,
            estimator=analysis.statistics.geometric_mean,
            seed=9,
            name="ratio",
            resamples=1000,
            alpha=0.05,
        )
        second = analysis.paired_bootstrap(
            values,
            estimator=analysis.statistics.geometric_mean,
            seed=9,
            name="ratio",
            resamples=1000,
            alpha=0.05,
        )
        self.assertEqual(first, second)

    def test_multi_agent_claim_requires_complete_parent_and_child_tokens(self) -> None:
        report = self.publication_report(topology="native-multi-agent")
        analyzed = analysis.analyze_report(report, PLAN)
        unattested = analysis.cell_claim(analyzed, report, confirmatory=True)
        self.assertFalse(unattested["all_required_metric_coverage_complete"])
        self.assertFalse(unattested["confirmatory_wllm_efficiency_win"])

        report["input_token_coverage"] = {
            "scope": "parent-and-children",
            "complete": True,
        }
        attested = analysis.cell_claim(analyzed, report, confirmatory=True)
        self.assertTrue(attested["all_required_metric_coverage_complete"])
        self.assertTrue(attested["confirmatory_wllm_efficiency_win"])

    def test_cli_output_is_byte_deterministic(self) -> None:
        report = {
            "benchmark": "wllm-agent-ab",
            "records": [
                native_record(1, "baseline", score=1, duration=10, tokens=100),
                native_record(1, "wllm", score=1, duration=5, tokens=50),
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "plan.json"
            report_path = root / "report.json"
            first = root / "first.json"
            second = root / "second.json"
            plan_path.write_text(json.dumps(PLAN), encoding="utf-8")
            report_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(
                analysis.main(
                    [
                        "run",
                        "--plan",
                        str(plan_path),
                        "--exploratory",
                        "--report",
                        str(report_path),
                        "--output",
                        str(first),
                    ]
                ),
                0,
            )
            self.assertEqual(
                analysis.main(
                    [
                        "run",
                        "--plan",
                        str(plan_path),
                        "--exploratory",
                        "--report",
                        str(report_path),
                        "--output",
                        str(second),
                    ]
                ),
                0,
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_plan_validation_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            path.write_text(
                json.dumps({**PLAN, "post_hoc_override": True}), encoding="utf-8"
            )
            with self.assertRaises(analysis.AnalysisError):
                analysis.load_plan(path)

    def test_freeze_plan_binds_exact_config_and_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "template.json"
            frozen = root / "frozen.json"
            config = self.write_publication_config(root)
            template.write_text(json.dumps(PLAN), encoding="utf-8")
            self.assertEqual(
                analysis.main(
                    [
                        "freeze-plan",
                        "--template",
                        str(template),
                        "--config",
                        str(config),
                        "--output",
                        str(frozen),
                    ]
                ),
                0,
            )
            loaded = analysis.load_plan(frozen)
            protocol = loaded["publication_protocol"]
            self.assertEqual(protocol["expected_cell_count"], 3)
            self.assertEqual(protocol["primary_cell_count"], 3)
            self.assertEqual(
                protocol["execution_protocol_sha256"],
                analysis.canonical_sha256(protocol["execution_protocol"]),
            )
            self.assertTrue(protocol["model_snapshot_publication_eligible"])
            config.write_text(config.read_text() + "\n", encoding="utf-8")
            with self.assertRaises(analysis.AnalysisError):
                analysis.load_plan(frozen)

    def test_publication_requires_every_declared_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.write_publication_config(root)
            output_path = root / "analysis.json"
            plan_path, plan = self.write_frozen_plan(root, config)
            reports = self.complete_publication_reports(plan)
            reports.pop(next(iter(reports)))
            index_path, _ = self.write_matrix_index(root, plan_path, plan, reports)
            self.assertEqual(
                analysis.main(
                    [
                        "run",
                        "--plan",
                        str(plan_path),
                        "--matrix-index",
                        str(index_path),
                        "--output",
                        str(output_path),
                    ]
                ),
                2,
            )
            index_path, index = self.write_matrix_index(
                root, plan_path, plan, reports
            )
            index["dry_run"] = True
            index_path.write_text(json.dumps(index), encoding="utf-8")
            self.assertEqual(
                analysis.main(
                    [
                        "run",
                        "--plan",
                        str(plan_path),
                        "--matrix-index",
                        str(index_path),
                    ]
                ),
                2,
            )
            assessment = json.loads(output_path.read_text())["publication_assessment"]
            self.assertFalse(assessment["publication_ready"])
            self.assertEqual(assessment["missing_cells"], 1)

    def test_publication_rejects_duplicate_and_undeclared_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.write_publication_config(root)
            plan_path, plan = self.write_frozen_plan(root, config)
            reports = self.complete_publication_reports(plan)
            index_path, index = self.write_matrix_index(root, plan_path, plan, reports)
            index["cells"].append(dict(index["cells"][0]))
            index["cells"][-1]["id"] = "duplicate-cell"
            index["cell_count"] = len(index["cells"])
            index_path.write_text(json.dumps(index), encoding="utf-8")
            self.assertEqual(
                analysis.main(
                    [
                        "run",
                        "--plan",
                        str(plan_path),
                        "--matrix-index",
                        str(index_path),
                    ]
                ),
                2,
            )
            index["cells"] = index["cells"][:-1]
            index["cell_count"] = len(index["cells"])
            index["cells"][0]["task"] = "not-declared"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            self.assertEqual(
                analysis.main(
                    [
                        "run",
                        "--plan",
                        str(plan_path),
                        "--matrix-index",
                        str(index_path),
                    ]
                ),
                2,
            )

    def test_publication_refuses_arbitrary_reports_and_tampered_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.write_publication_config(root)
            plan_path, plan = self.write_frozen_plan(root, config)
            reports = self.complete_publication_reports(plan)
            index_path, index = self.write_matrix_index(root, plan_path, plan, reports)
            arbitrary = root / "arbitrary.json"
            arbitrary.write_text(json.dumps(self.publication_report()), encoding="utf-8")
            self.assertEqual(
                analysis.main(
                    [
                        "run",
                        "--plan",
                        str(plan_path),
                        "--report",
                        str(arbitrary),
                    ]
                ),
                2,
            )
            index["provenance"]["analysis_plan_sha256"] = "0" * 64
            index_path.write_text(json.dumps(index), encoding="utf-8")
            self.assertEqual(
                analysis.main(
                    [
                        "run",
                        "--plan",
                        str(plan_path),
                        "--matrix-index",
                        str(index_path),
                    ]
                ),
                2,
            )

    def test_publication_rejects_report_escape_hash_and_provenance_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.write_publication_config(root)
            plan_path, plan = self.write_frozen_plan(root, config)
            reports = self.complete_publication_reports(plan)
            index_path, index = self.write_matrix_index(root, plan_path, plan, reports)
            original = json.loads(json.dumps(index))
            index["cells"][0]["report"] = "../escape.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            self.assertEqual(
                analysis.main(
                    ["run", "--plan", str(plan_path), "--matrix-index", str(index_path)]
                ),
                2,
            )
            index = json.loads(json.dumps(original))
            report_path = root / index["cells"][0]["report"]
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["matrix_provenance"]["execution_id"] = (
                "87654321-4321-4321-8321-cba987654321"
            )
            report_path.write_text(json.dumps(report), encoding="utf-8")
            index["cells"][0]["report_sha256"] = analysis.sha256_file(report_path)
            index_path.write_text(json.dumps(index), encoding="utf-8")
            self.assertEqual(
                analysis.main(
                    ["run", "--plan", str(plan_path), "--matrix-index", str(index_path)]
                ),
                2,
            )
            index = json.loads(json.dumps(original))
            index["cells"][0]["report_sha256"] = "0" * 64
            index_path.write_text(json.dumps(index), encoding="utf-8")
            self.assertEqual(
                analysis.main(
                    ["run", "--plan", str(plan_path), "--matrix-index", str(index_path)]
                ),
                2,
            )

    def test_primary_bonferroni_exploratory_role_and_complete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.write_publication_config(
                root, tasks=[*analysis.PRIMARY_TASKS, "single-file-control"]
            )
            plan_path, plan = self.write_frozen_plan(root, config)
            reports = self.complete_publication_reports(plan)
            index_path, _ = self.write_matrix_index(root, plan_path, plan, reports)
            output = root / "analysis.json"
            self.assertEqual(
                analysis.main(
                    [
                        "run",
                        "--plan",
                        str(plan_path),
                        "--matrix-index",
                        str(index_path),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(result["publication_assessment"]["publication_ready"])
            for cell in result["analyses"]:
                if cell["cell"]["task"] in analysis.PRIMARY_TASKS:
                    self.assertEqual(cell["statistical_role"], "confirmatory-primary")
                    self.assertAlmostEqual(cell["cell_alpha"], PLAN["alpha"] / 3)
                    self.assertTrue(
                        cell["cell_decision"]["confirmatory_wllm_efficiency_win"]
                    )
                else:
                    self.assertEqual(cell["statistical_role"], "exploratory")
                    self.assertIsNone(
                        cell["cell_decision"]["confirmatory_wllm_efficiency_win"]
                    )
            # Remove one input-token observation from a primary report and rebuild
            # the attested index: it must withhold readiness and the cell claim.
            primary_id = next(
                identifier
                for identifier in reports
                if "task=release-evidence" in identifier
            )
            reports[primary_id]["records"][1]["usage"]["input_tokens"] = None
            clean_root = root / "incomplete"
            clean_root.mkdir()
            config_copy = clean_root / config.name
            config_copy.write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
            incomplete_plan_path, incomplete_plan = self.write_frozen_plan(
                clean_root, config_copy
            )
            incomplete_index, _ = self.write_matrix_index(
                clean_root, incomplete_plan_path, incomplete_plan, reports
            )
            incomplete_output = clean_root / "analysis.json"
            self.assertEqual(
                analysis.main(
                    [
                        "run",
                        "--plan",
                        str(incomplete_plan_path),
                        "--matrix-index",
                        str(incomplete_index),
                        "--output",
                        str(incomplete_output),
                    ]
                ),
                2,
            )
            assessment = json.loads(
                incomplete_output.read_text(encoding="utf-8")
            )["publication_assessment"]
            self.assertEqual(assessment["nonclaimable_primary_cells"], 1)


if __name__ == "__main__":
    unittest.main()
