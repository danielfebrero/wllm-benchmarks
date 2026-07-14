from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import matrix


class MatrixTests(unittest.TestCase):
    def write_executable(self, path: Path) -> Path:
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    def attested_config(self, directory: Path) -> tuple[Path, dict[str, object]]:
        self.write_executable(directory / "codex")
        self.write_executable(directory / "wllm")
        payload: dict[str, object] = {
            "schema_version": "1.0",
            "selection_salt": "test-family",
            "model_snapshot_status": "attested-immutable",
            "cache_regime": "cold-new-provider-session-per-arm",
            "machine_regime": "dedicated-test-host",
            "tasks": ["release-evidence"],
            "agents": ["codex"],
            "models": {"codex": "provider-snapshot-codex"},
            "agent_bins": {"codex": "./codex"},
            "efforts": ["medium"],
            "topologies": ["single"],
            "arm": "both",
            "brief_budget": 1200,
            "timeout": 900,
            "runs": 6,
            "jobs": 1,
            "wllm_bin": "./wllm",
            "no_build": True,
            "keep_workspaces": False,
        }
        path = directory / "publication.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path, payload

    def parse_error(self, config: object) -> str:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                matrix.parse_args(["--config", str(path)])
            return stderr.getvalue()

    def test_cartesian_product_and_agent_models_are_explicit(self) -> None:
        args = matrix.parse_args(
            [
                "--task",
                "release-evidence,webhook-rotation",
                "--agent",
                "codex,claude",
                "--effort",
                "low,high",
                "--topology",
                "single,native-multi-agent",
                "--model",
                "claude=claude-test",
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            cells = matrix.build_cells(args, Path(temporary))
        self.assertEqual(len(cells), 16)
        claude = [cell for cell in cells if cell["agent"] == "claude"]
        codex = [cell for cell in cells if cell["agent"] == "codex"]
        self.assertTrue(all(cell["model"] == "claude-test" for cell in claude))
        self.assertTrue(all(cell["model"] == "gpt-5.6-sol" for cell in codex))
        self.assertTrue(all(str(matrix.ROOT / "run.py") in cell["command"] for cell in cells))

    def test_dry_run_writes_exact_plan_without_launching_agents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(matrix.subprocess, "run") as subprocess_run:
                exit_code = matrix.main(
                    [
                        "--task",
                        "release-evidence",
                        "--agent",
                        "codex,grok",
                        "--dry-run",
                        "--output-dir",
                        temporary,
                    ]
                )
            self.assertEqual(exit_code, 0)
            subprocess_run.assert_not_called()
            root = next(Path(temporary).iterdir())
            plan = json.loads((root / "matrix-plan.json").read_text(encoding="utf-8"))
            index = json.loads((root / "artifact-index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(plan["cells"]), 2)
            self.assertEqual(index["cell_count"], 2)
            self.assertTrue(index["timing_comparable"])
            self.assertEqual(plan["execution_protocol"], index["execution_protocol"])
            self.assertEqual(plan["provenance"], index["provenance"])
            self.assertEqual(
                plan["provenance"]["execution_protocol_sha256"],
                matrix.canonical_sha256(plan["execution_protocol"]),
            )
            self.assertEqual(len(plan["provenance"]["execution_id"]), 32)
            for cell in plan["cells"]:
                self.assertIn("--agent", cell["command"])
                self.assertIn("--model", cell["command"])

    def test_cell_process_timeout_is_bounded_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            matrix_dir = Path(temporary) / "matrix"
            cell_dir = matrix_dir / "cells" / "001-test"
            cell = {
                "id": "001-test",
                "number": 1,
                "task": "webhook-rotation",
                "agent": "codex",
                "model": "test-model",
                "effort": "medium",
                "topology": "single",
                "cell_dir": str(cell_dir),
                "command": ["python3", "run.py"],
                "cell_timeout_seconds": 17,
            }
            timeout = matrix.subprocess.TimeoutExpired(
                cell["command"], 17, output="partial", stderr="diagnostic"
            )
            with mock.patch.object(
                matrix.run, "run_bounded_process_tree", side_effect=timeout
            ) as bounded_run:
                outcome = matrix.run_cell(cell)
            self.assertEqual(outcome["exit_code"], 124)
            self.assertTrue(outcome["timed_out"])
            self.assertIn(
                "bounded process timeout",
                (cell_dir / "stderr.log").read_text(encoding="utf-8"),
            )
            self.assertEqual(bounded_run.call_args.kwargs["timeout"], 17.0)

    def test_all_checked_in_configs_load(self) -> None:
        local = matrix.parse_args(["--config", "configs/local-matrix.json"])
        publication = matrix.parse_args(["--config", "configs/publication.json"])
        smoke = matrix.parse_args(["--config", "configs/smoke.json"])

        self.assertEqual(local.runs, 2)
        self.assertEqual(len(local.tasks), 5)
        self.assertEqual(local.agents, ["codex", "claude", "grok"])
        self.assertEqual(publication.runs, 6)
        self.assertEqual(
            publication.config_metadata,
            {
                "selection_salt": "wllm-bench-v1",
                "model_snapshot_status": "template",
                "model_snapshot_publication_eligible": False,
                "cache_regime": "template",
                "machine_regime": "template",
            },
        )
        self.assertEqual(smoke.tasks, ["webhook-rotation", "single-file-control"])

    def test_explicit_cli_fields_replace_config_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_dir = Path(temporary) / "config"
            config_dir.mkdir()
            config = config_dir / "matrix.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "tasks": ["webhook-rotation"],
                        "agents": ["claude"],
                        "efforts": ["low"],
                        "topologies": ["native-multi-agent"],
                        "models": {"claude": "configured-model"},
                        "runs": 3,
                        "jobs": 2,
                        "wllm_bin": "../bin/wllm",
                        "output_dir": "artifacts",
                    }
                ),
                encoding="utf-8",
            )
            args = matrix.parse_args(
                [
                    "--config",
                    str(config),
                    "--tasks",
                    "release-evidence,config-precedence",
                    "--agents",
                    "codex",
                    "--model",
                    "codex=cli-model",
                    "--runs",
                    "7",
                    "--output-dir",
                    "cli-results",
                ]
            )

            self.assertEqual(
                args.tasks, ["release-evidence", "config-precedence"]
            )
            self.assertEqual(args.agents, ["codex"])
            self.assertEqual(args.efforts, ["low"])
            self.assertEqual(args.topologies, ["native-multi-agent"])
            self.assertEqual(args.models, {"codex": "cli-model"})
            self.assertEqual(args.runs, 7)
            self.assertEqual(args.jobs, 2)
            self.assertEqual(args.wllm_bin, (config_dir / "../bin/wllm").resolve())
            self.assertEqual(args.output_dir, (Path.cwd() / "cli-results").resolve())

    def test_config_schema_types_and_names_are_rejected_clearly(self) -> None:
        cases = (
            (
                {"schema_version": "2.0"},
                "unsupported schema_version '2.0'",
            ),
            (
                {"schema_version": "1.0", "agents": "codex"},
                "field 'agents' must be an array of strings",
            ),
            (
                {"schema_version": "1.0", "agents": ["unknown"]},
                "unknown agents: unknown",
            ),
            (
                {"schema_version": "1.0", "topologies": ["swarm"]},
                "unknown topologies: swarm",
            ),
            (
                {"schema_version": "1.0", "runz": 2},
                "unknown config fields: 'runz'",
            ),
            (
                {"schema_version": "1.0", "runs": True},
                "field 'runs' must be an integer",
            ),
            (
                {
                    "schema_version": "1.0",
                    "model_snapshot_status": "probably-pinned",
                },
                "field 'model_snapshot_status' must be one of",
            ),
            (
                {
                    "schema_version": "1.0",
                    "model_snapshot_status": "attested-immutable",
                    "agents": ["codex", "claude"],
                    "models": {"codex": "snapshot-a"},
                },
                "model snapshot attestation must cover exactly the selected agents",
            ),
        )
        for config, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, self.parse_error(config))

    def test_immutable_model_snapshot_attestation_is_explicit_and_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, _payload = self.attested_config(Path(temporary))
            args = matrix.parse_args(["--config", str(config)])
            self.assertEqual(
                args.config_metadata,
                {
                    "model_snapshot_status": "attested-immutable",
                    "model_snapshot_publication_eligible": True,
                    "selection_salt": "test-family",
                    "cache_regime": "cold-new-provider-session-per-arm",
                    "machine_regime": "dedicated-test-host",
                },
            )

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                matrix.parse_args(
                    [
                        "--config",
                        str(config),
                        "--model",
                        "codex=different-snapshot",
                    ]
                )
            self.assertIn("invalidate model_snapshot_status", stderr.getvalue())

    def test_attested_execution_requires_matching_frozen_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _payload = self.attested_config(root)
            args = matrix.parse_args(["--config", str(config)])
            with self.assertRaisesRegex(matrix.ConfigError, "requires --analysis-plan"):
                matrix.build_execution_context(args)

            protocol = matrix.frozen_execution_protocol(config)
            plan = root / "analysis-plan.lock.json"
            plan.write_text(
                json.dumps(
                    {
                        "publication_protocol": {
                            "matrix_config_sha256": "0" * 64,
                            "execution_protocol": protocol,
                            "execution_protocol_sha256": matrix.canonical_sha256(protocol),
                        }
                    }
                ),
                encoding="utf-8",
            )
            mixed = matrix.parse_args(
                ["--config", str(config), "--analysis-plan", str(plan)]
            )
            with self.assertRaisesRegex(matrix.ConfigError, "plan/config mismatch"):
                matrix.build_execution_context(mixed)

    def test_git_attestation_and_report_provenance_prevent_mixing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _payload = self.attested_config(root)
            protocol = matrix.frozen_execution_protocol(config)
            plan = root / "analysis-plan.lock.json"
            plan.write_text(
                json.dumps(
                    {
                        "publication_protocol": {
                            "matrix_config_sha256": matrix.sha256_file(config),
                            "execution_protocol": protocol,
                            "execution_protocol_sha256": matrix.canonical_sha256(protocol),
                        }
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            commands = (
                ["git", "init", "-q"],
                ["git", "config", "user.email", "benchmark@example.invalid"],
                ["git", "config", "user.name", "Benchmark Test"],
                ["git", "add", "publication.json", "analysis-plan.lock.json"],
                ["git", "commit", "-qm", "freeze preoutcome protocol"],
            )
            for command in commands:
                subprocess.run(command, cwd=root, check=True)

            args = matrix.parse_args(
                ["--config", str(config), "--analysis-plan", str(plan)]
            )
            actual_protocol, provenance = matrix.build_execution_context(args)
            self.assertEqual(actual_protocol, protocol)
            self.assertRegex(provenance["preoutcome_git_commit"], r"^[0-9a-f]{40,64}$")
            self.assertTrue(provenance["preoutcome_timestamp"])
            self.assertEqual(provenance["matrix_config_sha256"], matrix.sha256_file(config))
            self.assertEqual(provenance["analysis_plan_sha256"], matrix.sha256_file(plan))

            plan.write_text(plan.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            dirty_args = matrix.parse_args(
                ["--config", str(config), "--analysis-plan", str(plan)]
            )
            with self.assertRaisesRegex(matrix.ConfigError, "no index or worktree diff"):
                matrix.build_execution_context(dirty_args)

    def test_run_cell_injects_cell_bound_provenance_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            matrix_dir = Path(temporary) / "matrix"
            cell_dir = matrix_dir / "cells" / "001-test"
            cell = {
                "id": "001-test",
                "number": 1,
                "task": "release-evidence",
                "agent": "codex",
                "model": "snapshot",
                "effort": "medium",
                "topology": "single",
                "cell_dir": str(cell_dir),
                "command": ["python3", "run.py"],
                "cell_timeout_seconds": 17,
            }
            provenance = {
                "execution_id": "a" * 32,
                "matrix_config_sha256": "b" * 64,
                "analysis_plan_sha256": "c" * 64,
                "execution_protocol_sha256": "d" * 64,
                "preoutcome_git_commit": "e" * 40,
                "preoutcome_timestamp": "2026-07-14T00:00:00+00:00",
            }

            def completed(*_args: object, **_kwargs: object) -> SimpleNamespace:
                report_dir = cell_dir / "artifacts" / "run-1"
                report_dir.mkdir(parents=True)
                (report_dir / "report.json").write_text(
                    json.dumps({"benchmark": "wllm-agent-ab"}), encoding="utf-8"
                )
                (report_dir / "artifact-index.json").write_text(
                    json.dumps(
                        {"artifacts": [{"path": "report.json", "bytes": 1}]}
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(stdout="ok", stderr="", returncode=0)

            with mock.patch.object(
                matrix.run, "run_bounded_process_tree", side_effect=completed
            ):
                outcome = matrix.run_cell(cell, provenance)
            report_path = matrix_dir / str(outcome["report"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["matrix_provenance"]["cell_id"], "001-test")
            self.assertEqual(report["matrix_provenance"]["execution_id"], "a" * 32)
            self.assertEqual(outcome["report_sha256"], matrix.sha256_file(report_path))
            native_index = json.loads(
                (report_path.parent / "artifact-index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                native_index["artifacts"][0]["bytes"], report_path.stat().st_size
            )


if __name__ == "__main__":
    unittest.main()
