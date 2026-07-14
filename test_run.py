from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import run


class JsonlParserTests(unittest.TestCase):
    def test_usage_and_tool_items_are_counted_once(self) -> None:
        events = [
            {
                "type": "item.started",
                "item": {"id": "1", "type": "command_execution"},
            },
            {
                "type": "item.completed",
                "item": {"id": "1", "type": "command_execution"},
            },
            {
                "type": "item.completed",
                "item": {"id": "2", "type": "agent_message", "text": "done"},
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 60,
                    "output_tokens": 20,
                    "reasoning_output_tokens": 8,
                },
            },
        ]
        usage, calls, message, errors, completed = run.parse_jsonl(
            "\n".join(json.dumps(event) for event in events)
        )
        self.assertEqual(usage["total_tokens"], 120)
        self.assertEqual(usage["uncached_input_tokens"], 40)
        self.assertEqual(calls["command_execution"], 1)
        self.assertEqual(calls["total"], 1)
        self.assertEqual(message, "done")
        self.assertEqual(errors, [])
        self.assertTrue(completed)


class ManifestAndGradeValidationTests(unittest.TestCase):
    def test_task_id_cannot_escape_tasks_root(self) -> None:
        with self.assertRaisesRegex(SystemExit, "direct child"):
            run.load_task("../outside")

    def test_task_commands_must_be_non_empty_string_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tasks = Path(temporary) / "tasks"
            task = tasks / "bad"
            task.mkdir(parents=True)
            (task / "task.json").write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "title": "bad task",
                        "prompt": "repair",
                        "prepare": "python prepare.py",
                        "grade": ["python3", "grade.py"],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(run, "TASKS_ROOT", tasks):
                with self.assertRaisesRegex(SystemExit, "array of strings"):
                    run.load_task("bad")

    def test_grader_contract_rejects_exit_nan_range_and_incoherence(self) -> None:
        valid = {"exit_code": 0, "passed": 3, "total": 4, "score": 0.75}
        self.assertIsNone(run.grade_validation_error(valid))
        cases = (
            ({**valid, "exit_code": 1}, "exited with code"),
            ({**valid, "score": float("nan")}, "finite"),
            ({**valid, "score": 1.5}, "within"),
            ({**valid, "score": 0.5}, "inconsistent"),
            ({**valid, "passed": 5}, "cannot exceed"),
        )
        for grade, message in cases:
            with self.subTest(message=message):
                self.assertIn(message, run.grade_validation_error(grade) or "")

    def test_report_schema_uses_emitted_reasoning_and_outcome_fields(self) -> None:
        schema = json.loads(
            (run.BENCHMARK_ROOT / "schemas" / "report.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("reasoning", schema["required"])
        self.assertNotIn("effort", schema["properties"])
        statuses = schema["properties"]["records"]["items"]["properties"][
            "status"
        ]["enum"]
        self.assertIn("outcome_failure", statuses)

    def test_codex_preflight_timeout_is_bounded(self) -> None:
        with mock.patch.object(
            run.subprocess,
            "run",
            side_effect=run.subprocess.TimeoutExpired(["codex"], 30),
        ):
            with self.assertRaisesRegex(SystemExit, "timed out"):
                run.check_codex_auth("codex")


@unittest.skipUnless(os.name == "posix", "POSIX process-group behavior")
class ProcessTreeTests(unittest.TestCase):
    def test_bounded_process_starts_a_new_session(self) -> None:
        process = mock.Mock(pid=321, returncode=0)
        process.communicate.return_value = ("stdout", "stderr")
        with mock.patch.object(
            run.subprocess, "Popen", return_value=process
        ) as popen:
            result = run.run_bounded_process_tree(
                ["agent", "--run"], cwd=Path("/tmp"), timeout=12.0
            )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "stdout")
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertNotIn("creationflags", popen.call_args.kwargs)
        process.communicate.assert_called_once_with(timeout=12.0)

    def test_timeout_kills_process_group_and_preserves_streams(self) -> None:
        process = mock.Mock(pid=654, returncode=-run.signal.SIGKILL)
        first_timeout = run.subprocess.TimeoutExpired(
            ["agent"], 3.0, output="partial", stderr="partial-error"
        )
        process.communicate.side_effect = [
            first_timeout,
            ("complete stdout", "complete stderr"),
        ]
        with (
            mock.patch.object(run.subprocess, "Popen", return_value=process),
            mock.patch.object(run.os, "killpg") as killpg,
            self.assertRaises(run.subprocess.TimeoutExpired) as raised,
        ):
            run.run_bounded_process_tree(
                ["agent"], cwd=Path("/tmp"), timeout=3.0
            )
        killpg.assert_called_once_with(654, run.signal.SIGKILL)
        process.kill.assert_not_called()
        self.assertEqual(raised.exception.output, "complete stdout")
        self.assertEqual(raised.exception.stderr, "complete stderr")
        self.assertEqual(
            process.communicate.call_args_list,
            [mock.call(timeout=3.0), mock.call(timeout=10)],
        )

    def test_process_group_failure_falls_back_to_direct_kill(self) -> None:
        process = mock.Mock(pid=987)
        with mock.patch.object(
            run.os, "killpg", side_effect=PermissionError("denied")
        ):
            run.terminate_process_tree(process)
        process.kill.assert_called_once_with()

    def test_repeated_cleanup_timeout_never_communicates_unbounded(self) -> None:
        process = mock.Mock(pid=777)
        process.stdout = mock.Mock()
        process.stderr = mock.Mock()
        first_timeout = run.subprocess.TimeoutExpired(
            ["agent"], 3.0, output="first stdout", stderr="first stderr"
        )
        cleanup_timeout = run.subprocess.TimeoutExpired(
            ["agent"], 10.0, output="latest stdout", stderr="latest stderr"
        )
        process.communicate.side_effect = [first_timeout, cleanup_timeout]
        process.wait.side_effect = run.subprocess.TimeoutExpired(
            ["agent"], run.PROCESS_TREE_FINAL_WAIT_SECONDS
        )
        with (
            mock.patch.object(run.subprocess, "Popen", return_value=process),
            mock.patch.object(run, "terminate_process_tree") as terminate,
            self.assertRaises(run.subprocess.TimeoutExpired) as raised,
        ):
            run.run_bounded_process_tree(
                ["agent"], cwd=Path("/tmp"), timeout=3.0
            )
        self.assertEqual(
            process.communicate.call_args_list,
            [
                mock.call(timeout=3.0),
                mock.call(timeout=run.PROCESS_TREE_CLEANUP_TIMEOUT_SECONDS),
            ],
        )
        self.assertEqual(terminate.call_count, 2)
        process.stdout.close.assert_called_once_with()
        process.stderr.close.assert_called_once_with()
        process.wait.assert_called_once_with(
            timeout=run.PROCESS_TREE_FINAL_WAIT_SECONDS
        )
        self.assertEqual(raised.exception.output, "latest stdout")
        self.assertEqual(raised.exception.stderr, "latest stderr")

    def test_detached_descendant_holding_pipe_cannot_make_timeout_unbounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid_file = root / "detached.pid"
            parent = (
                "import pathlib, subprocess, sys, time; "
                "child = subprocess.Popen("
                "[sys.executable, '-c', 'import time; time.sleep(30)'], "
                "start_new_session=True); "
                f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); "
                "print('detached-ready', flush=True); "
                "time.sleep(30)"
            )
            detached_pid: int | None = None
            started = time.monotonic()
            try:
                with (
                    mock.patch.object(
                        run, "PROCESS_TREE_CLEANUP_TIMEOUT_SECONDS", 0.2
                    ),
                    mock.patch.object(run, "PROCESS_TREE_FINAL_WAIT_SECONDS", 0.2),
                    self.assertRaises(run.subprocess.TimeoutExpired) as raised,
                ):
                    run.run_bounded_process_tree(
                        [sys.executable, "-c", parent], cwd=root, timeout=0.25
                    )
                self.assertLess(time.monotonic() - started, 2.0)
                self.assertIn(
                    "detached-ready",
                    run.decode_timeout_stream(raised.exception.output),
                )
                self.assertTrue(pid_file.is_file())
                detached_pid = int(pid_file.read_text(encoding="utf-8"))
            finally:
                if detached_pid is None and pid_file.is_file():
                    detached_pid = int(pid_file.read_text(encoding="utf-8"))
                if detached_pid is not None:
                    try:
                        os.kill(detached_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


class WllmResolutionTests(unittest.TestCase):
    def make_executable(self, path: Path, text: str = "#!/bin/sh\nexit 0\n") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)
        return path.resolve()

    def args(self, *extra: str):
        return run.parse_args(["--runs", "1", *extra])

    def test_baseline_does_not_resolve_wllm(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {"WLLM_BIN": "/missing/env-wllm", "PATH": ""},
                clear=True,
            ),
            mock.patch.object(run, "detect_wllm_superproject") as detect,
            mock.patch.object(run.shutil, "which") as which,
            mock.patch.object(run.subprocess, "run") as subprocess_run,
        ):
            resolved = run.resolve_wllm(
                self.args("--arm", "baseline", "--wllm-bin", "/missing/cli-wllm")
            )
        self.assertIsNone(resolved)
        detect.assert_not_called()
        which.assert_not_called()
        subprocess_run.assert_not_called()

    def test_cli_path_has_highest_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cli_binary = self.make_executable(root / "cli-wllm")
            environment_binary = self.make_executable(root / "env-wllm")
            superproject = root / "source"
            self.make_executable(
                superproject / "target" / "release" / run.binary_name()
            )
            with (
                mock.patch.dict(
                    os.environ,
                    {"WLLM_BIN": str(environment_binary), "PATH": ""},
                    clear=True,
                ),
                mock.patch.object(
                    run, "detect_wllm_superproject", return_value=superproject
                ),
                mock.patch.object(run.subprocess, "run") as subprocess_run,
            ):
                resolved = run.resolve_wllm(
                    self.args("--wllm-bin", str(cli_binary))
                )
            self.assertEqual(resolved, cli_binary)
            subprocess_run.assert_not_called()

    def test_invalid_cli_path_does_not_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment_binary = self.make_executable(Path(temporary) / "env-wllm")
            with (
                mock.patch.dict(
                    os.environ,
                    {"WLLM_BIN": str(environment_binary), "PATH": ""},
                    clear=True,
                ),
                mock.patch.object(run, "detect_wllm_superproject") as detect,
            ):
                with self.assertRaisesRegex(SystemExit, "--wllm-bin"):
                    run.resolve_wllm(self.args("--wllm-bin", "/missing/cli-wllm"))
            detect.assert_not_called()

    def test_environment_path_precedes_superproject_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment_binary = self.make_executable(root / "env-wllm")
            superproject = root / "source"
            self.make_executable(
                superproject / "target" / "release" / run.binary_name()
            )
            with (
                mock.patch.dict(
                    os.environ,
                    {"WLLM_BIN": str(environment_binary), "PATH": ""},
                    clear=True,
                ),
                mock.patch.object(
                    run, "detect_wllm_superproject", return_value=superproject
                ),
                mock.patch.object(run.subprocess, "run") as subprocess_run,
            ):
                resolved = run.resolve_wllm(self.args())
            self.assertEqual(resolved, environment_binary)
            subprocess_run.assert_not_called()

    def test_invalid_environment_path_does_not_fall_back(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {"WLLM_BIN": "/missing/env-wllm", "PATH": ""},
                clear=True,
            ),
            mock.patch.object(run, "detect_wllm_superproject") as detect,
        ):
            with self.assertRaisesRegex(SystemExit, "WLLM_BIN"):
                run.resolve_wllm(self.args())
        detect.assert_not_called()

    def test_detected_superproject_precedes_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            superproject = root / "source"
            project_binary = self.make_executable(
                superproject / "target" / "release" / run.binary_name()
            )
            path_binary = self.make_executable(root / "bin" / run.binary_name())
            with (
                mock.patch.dict(
                    os.environ,
                    {"PATH": str(path_binary.parent)},
                    clear=True,
                ),
                mock.patch.object(
                    run, "detect_wllm_superproject", return_value=superproject
                ),
                mock.patch.object(run.subprocess, "run") as subprocess_run,
            ):
                resolved = run.resolve_wllm(self.args())
            self.assertEqual(resolved, project_binary)
            subprocess_run.assert_not_called()

    def test_detected_superproject_can_build_its_missing_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            superproject = Path(temporary) / "source"
            superproject.mkdir()
            project_binary = (
                superproject / "target" / "release" / run.binary_name()
            )

            def fake_cargo(
                command: list[str], *, cwd: Path, check: bool, timeout: int
            ) -> object:
                self.assertEqual(
                    command,
                    [
                        "cargo",
                        "build",
                        "--manifest-path",
                        str(superproject / "Cargo.toml"),
                        "--release",
                        "--locked",
                        "--bin",
                        "wllm",
                    ],
                )
                self.assertEqual(cwd, superproject)
                self.assertTrue(check)
                self.assertEqual(timeout, run.BUILD_TIMEOUT_SECONDS)
                self.make_executable(project_binary)
                return object()

            with (
                mock.patch.dict(os.environ, {"PATH": ""}, clear=True),
                mock.patch.object(
                    run, "detect_wllm_superproject", return_value=superproject
                ),
                mock.patch.object(run.subprocess, "run", side_effect=fake_cargo),
            ):
                resolved = run.resolve_wllm(self.args())
            self.assertEqual(resolved, project_binary.resolve())

    def test_path_prevents_build_when_superproject_binary_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            superproject = root / "source"
            superproject.mkdir()
            path_binary = self.make_executable(root / "bin" / run.binary_name())
            with (
                mock.patch.dict(
                    os.environ,
                    {"PATH": str(path_binary.parent)},
                    clear=True,
                ),
                mock.patch.object(
                    run, "detect_wllm_superproject", return_value=superproject
                ),
                mock.patch.object(run.subprocess, "run") as subprocess_run,
            ):
                resolved = run.resolve_wllm(self.args())
            self.assertEqual(resolved, path_binary)
            subprocess_run.assert_not_called()

    def test_standalone_uses_path_without_invoking_cargo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path_binary = self.make_executable(
                Path(temporary) / "bin" / run.binary_name()
            )
            with (
                mock.patch.dict(
                    os.environ,
                    {"PATH": str(path_binary.parent)},
                    clear=True,
                ),
                mock.patch.object(
                    run, "detect_wllm_superproject", return_value=None
                ),
                mock.patch.object(run.subprocess, "run") as subprocess_run,
            ):
                resolved = run.resolve_wllm(self.args())
            self.assertEqual(resolved, path_binary)
            subprocess_run.assert_not_called()

    def test_standalone_missing_binary_never_invokes_cargo(self) -> None:
        with (
            mock.patch.dict(os.environ, {"PATH": ""}, clear=True),
            mock.patch.object(run, "detect_wllm_superproject", return_value=None),
            mock.patch.object(run.subprocess, "run") as subprocess_run,
        ):
            with self.assertRaisesRegex(SystemExit, "Cargo was not invoked"):
                run.resolve_wllm(self.args())
        subprocess_run.assert_not_called()

    def test_no_build_falls_back_from_superproject_to_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            superproject = root / "source"
            superproject.mkdir()
            path_binary = self.make_executable(root / "bin" / run.binary_name())
            with (
                mock.patch.dict(
                    os.environ,
                    {"PATH": str(path_binary.parent)},
                    clear=True,
                ),
                mock.patch.object(
                    run, "detect_wllm_superproject", return_value=superproject
                ),
                mock.patch.object(run.subprocess, "run") as subprocess_run,
            ):
                resolved = run.resolve_wllm(self.args("--no-build"))
            self.assertEqual(resolved, path_binary)
            subprocess_run.assert_not_called()

    def test_no_build_prevents_superproject_cargo_without_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            superproject = Path(temporary) / "source"
            superproject.mkdir()
            with (
                mock.patch.dict(os.environ, {"PATH": ""}, clear=True),
                mock.patch.object(
                    run, "detect_wllm_superproject", return_value=superproject
                ),
                mock.patch.object(run.subprocess, "run") as subprocess_run,
            ):
                with self.assertRaisesRegex(SystemExit, "--no-build"):
                    run.resolve_wllm(self.args("--no-build"))
            subprocess_run.assert_not_called()

    def test_superproject_detection_uses_git_reported_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Cargo.toml").write_text(
                '[package]\nname = "wllm"\n', encoding="utf-8"
            )
            nested = root / "tools" / "wllm-benchmarks"
            nested.mkdir(parents=True)
            git_result = run.subprocess.CompletedProcess(
                args=[], returncode=0, stdout=f"{root}\n", stderr=""
            )
            with mock.patch.object(
                run.subprocess, "run", return_value=git_result
            ) as subprocess_run:
                detected = run.detect_wllm_superproject(nested)
            self.assertEqual(detected, root.resolve())
            subprocess_run.assert_called_once_with(
                [
                    "git",
                    "-C",
                    str(nested.resolve()),
                    "rev-parse",
                    "--show-superproject-working-tree",
                ],
                stdout=run.subprocess.PIPE,
                stderr=run.subprocess.PIPE,
                text=True,
                check=False,
                timeout=run.PREFLIGHT_TIMEOUT_SECONDS,
            )

    def test_standalone_below_cargo_project_is_not_a_superproject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Cargo.toml").write_text(
                '[package]\nname = "wllm"\n', encoding="utf-8"
            )
            nested = root / "wllm-benchmarks"
            nested.mkdir()
            git_result = run.subprocess.CompletedProcess(
                args=[], returncode=0, stdout="\n", stderr=""
            )
            with mock.patch.object(
                run.subprocess, "run", return_value=git_result
            ):
                self.assertIsNone(run.detect_wllm_superproject(nested))

    def test_unrelated_git_superproject_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Cargo.toml").write_text(
                '[package]\nname = "unrelated"\n', encoding="utf-8"
            )
            nested = root / "wllm-benchmarks"
            nested.mkdir()
            git_result = run.subprocess.CompletedProcess(
                args=[], returncode=0, stdout=f"{root}\n", stderr=""
            )
            with mock.patch.object(
                run.subprocess, "run", return_value=git_result
            ):
                self.assertIsNone(run.detect_wllm_superproject(nested))


class FixtureTests(unittest.TestCase):
    def test_repetition_prepares_once_and_verifies_full_clones(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = root / "suite"
            artifacts = root / "artifacts"
            suite.mkdir()
            artifacts.mkdir()

            def fake_prepare(
                task_dir: Path, manifest: dict[str, object], workspace: Path
            ) -> None:
                del task_dir, manifest
                (workspace / ".git").mkdir()
                (workspace / ".git" / "config").write_bytes(b"git metadata\n")
                (workspace / "src").mkdir()
                candidate = workspace / "src" / "tool.py"
                candidate.write_bytes(b"print('fixture')\n")
                candidate.chmod(0o755)
                (workspace / "tool-link").symlink_to("src/tool.py")

            with mock.patch.object(
                run, "prepare_workspace", side_effect=fake_prepare
            ) as prepare:
                verification = run.prepare_repetition_workspaces(
                    run_number=1,
                    arms=["baseline", "wllm"],
                    task_dir=root,
                    manifest={},
                    suite_dir=suite,
                    artifacts_dir=artifacts,
                )

            prepare.assert_called_once()
            self.assertTrue(verification["valid"])
            source_digest = verification["source"]["digest"]
            self.assertEqual(
                verification["workspaces"]["baseline"]["digest"], source_digest
            )
            self.assertEqual(
                verification["workspaces"]["wllm"]["digest"], source_digest
            )
            self.assertTrue((suite / "run-01-baseline/.git/config").is_file())
            self.assertTrue((suite / "run-01-wllm/tool-link").is_symlink())
            changed = suite / "run-01-baseline/src/tool.py"
            changed.chmod(0o644)
            self.assertNotEqual(run.workspace_digest(changed.parents[1])["digest"], source_digest)

    def test_retained_workspaces_do_not_follow_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = root / "suite"
            suite.mkdir()
            secret = root / "secret.txt"
            secret.write_text("do not copy", encoding="utf-8")
            (suite / "external-link").symlink_to(secret)
            destination = root / "kept"
            run.retain_workspaces(suite, destination)
            self.assertTrue((destination / "external-link").is_symlink())
            self.assertEqual((destination / "external-link").readlink(), secret)

    def test_fixture_is_deterministic_and_initial_bug_is_detected(self) -> None:
        task_dir, manifest = run.load_task("webhook-rotation")
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_path, second_path = Path(first), Path(second)
            run.prepare_workspace(task_dir, manifest, first_path)
            run.prepare_workspace(task_dir, manifest, second_path)
            for workspace in (first_path, second_path):
                self.assertEqual(
                    run.run_checked(
                        ["git", "config", "--get", "maintenance.auto"],
                        cwd=workspace,
                    ).stdout.strip(),
                    "false",
                )
                self.assertEqual(
                    run.run_checked(
                        ["git", "config", "--get", "gc.auto"], cwd=workspace
                    ).stdout.strip(),
                    "0",
                )
            self.assertEqual(
                (first_path / "src/webhook_auth.py").read_bytes(),
                (second_path / "src/webhook_auth.py").read_bytes(),
            )
            grade = run.grade_workspace(task_dir, manifest, first_path)
            self.assertGreater(grade["score"], 0)
            self.assertLess(grade["score"], 1)

            candidate = first_path / "src" / "webhook_auth.py"
            source = candidate.read_text(encoding="utf-8")
            source = source.replace(
                '''    # Kept during the rotation window so in-flight deliveries still validate.
    secret = environment.get("WEBHOOK_SECRET_PREVIOUS")
    if not secret:
        return False
    expected = _digest(payload, secret)
    return expected == presented
''',
                '''    current = environment.get("WEBHOOK_SECRET_CURRENT")
    if not current:
        return False
    candidates = (current, environment.get("WEBHOOK_SECRET_PREVIOUS"))
    return any(
        secret is not None
        and hmac.compare_digest(_digest(payload, secret), presented)
        for secret in candidates
    )
''',
            )
            candidate.write_text(source, encoding="utf-8")
            public = run.run_checked(
                [
                    "python3",
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-v",
                ],
                cwd=first_path,
            )
            self.assertIn("OK", public.stderr)
            fixed_grade = run.grade_workspace(task_dir, manifest, first_path)
            self.assertEqual(fixed_grade["score"], 1)

    def test_release_evidence_fixture_has_a_generalizable_full_solution(self) -> None:
        task_dir, manifest = run.load_task("release-evidence")
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            run.prepare_workspace(task_dir, manifest, workspace)
            public = run.run_checked(
                ["python3", "-m", "unittest", "discover", "-s", "tests", "-v"],
                cwd=workspace,
            )
            self.assertIn("OK", public.stderr)
            initial = run.grade_workspace(task_dir, manifest, workspace)
            self.assertEqual(initial["score"], 0.2)
            candidate = workspace / "src" / "release_evidence.py"
            candidate.write_text(
                '''from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_release_evidence(
    release: Mapping[str, Any],
    build: Mapping[str, Any],
    deployment: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    digest = str(build["artifact_sha256"])
    if digest.lower().startswith("sha256:"):
        digest = digest.split(":", 1)[1]
    digest = digest.lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid SHA-256 digest")
    return {
        "schema_version": policy["schema_version"],
        "release_id": release["release_id"],
        "git_commit": release["git_commit"],
        "environment": policy["environment"],
        "artifact": {
            "digest": f"sha256:{digest}",
            "size_bytes": int(build["artifact_size_bytes"]),
        },
        "deployment": {
            "rollout_id": deployment["rollout_id"],
            "region": str(deployment["region"]).lower(),
        },
    }
''',
                encoding="utf-8",
            )
            fixed = run.grade_workspace(task_dir, manifest, workspace)
            self.assertEqual(fixed["score"], 1)
            preflight = run.run_checked(
                ["python3", str(task_dir / "grade.py"), "--self-test"]
            )
            self.assertTrue(json.loads(preflight.stdout)["deterministic"])


class WllmBriefTests(unittest.TestCase):
    def test_brief_uses_bounded_process_tree(self) -> None:
        brief_output = "\n".join(
            (
                "wllm context",
                "schema: 1.0",
                "est: 42",
                "coverage: considered=1 selected=1 omitted=0",
                "[items]",
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            artifacts = root / "artifacts"
            workspace.mkdir()
            artifacts.mkdir()
            completed = run.subprocess.CompletedProcess(
                ["wllm"], 0, brief_output, ""
            )
            with mock.patch.object(
                run, "run_bounded_process_tree", return_value=completed
            ) as bounded:
                brief, estimate, _ = run.generate_wllm_brief(
                    wllm=root / "wllm",
                    workspace=workspace,
                    query="find the target",
                    budget=256,
                    artifacts_dir=artifacts,
                    stem="brief",
                    timeout=7,
                )
            self.assertEqual(brief, brief_output)
            self.assertEqual(estimate, 42)
            self.assertEqual(bounded.call_count, 1)
            self.assertEqual(bounded.call_args.kwargs["cwd"], workspace)
            self.assertGreater(bounded.call_args.kwargs["timeout"], 0)
            self.assertLessEqual(bounded.call_args.kwargs["timeout"], 7)

    def test_brief_is_exact_bounded_and_restores_workspace_state(self) -> None:
        fake_source = r'''#!/usr/bin/env python3
import sys
from pathlib import Path

if sys.argv[1:] == ["--version"]:
    print("wllm 1.2.3-test")
    raise SystemExit(0)
args = sys.argv[1:]
expected = {
    "--query": "visible task prompt",
    "--budget": "256",
    "--format": "compact",
}
if not args or args[0] != "context":
    raise SystemExit("missing context command")
for flag, value in expected.items():
    if flag not in args or args[args.index(flag) + 1] != value:
        raise SystemExit(f"unexpected {flag}")
for flag in ("--target", "--root"):
    if Path(args[args.index(flag) + 1]).resolve() != Path.cwd().resolve():
        raise SystemExit(f"unexpected {flag}")
Path(".wllm").mkdir(exist_ok=True)
Path(".wllm/generated-by-brief").write_text("remove me", encoding="utf-8")
print("wllm context")
print("schema: 1.0")
print("est: 42")
print("coverage: considered=3 selected=2 omitted=1")
print("[items]")
print('{"ref":"u1","path":"src/example.py","content":"target evidence"}')
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "wllm"
            fake.write_text(fake_source, encoding="utf-8")
            fake.chmod(0o755)
            workspace = root / "workspace"
            artifacts = root / "artifacts"
            (workspace / ".wllm" / "cache-v2").mkdir(parents=True)
            (workspace / ".wllm" / "user-state").write_text(
                "preserve me", encoding="utf-8"
            )
            (workspace / ".wllm" / "cache-v2" / "existing").write_text(
                "preserve cache", encoding="utf-8"
            )
            artifacts.mkdir()
            brief, estimate, duration = run.generate_wllm_brief(
                wllm=fake,
                workspace=workspace,
                query="visible task prompt",
                budget=256,
                artifacts_dir=artifacts,
                stem="run-01-wllm",
                timeout=5,
            )
            self.assertEqual(estimate, 42)
            self.assertGreaterEqual(duration, 0)
            self.assertIn("target evidence", brief)
            self.assertEqual(
                (artifacts / "run-01-wllm.brief.txt").read_text(encoding="utf-8"),
                brief,
            )
            self.assertEqual(
                (workspace / ".wllm" / "user-state").read_text(encoding="utf-8"),
                "preserve me",
            )
            self.assertEqual(
                (workspace / ".wllm" / "cache-v2" / "existing").read_text(
                    encoding="utf-8"
                ),
                "preserve cache",
            )
            self.assertFalse((workspace / ".wllm" / "generated-by-brief").exists())
            clean_workspace = root / "clean-workspace"
            clean_workspace.mkdir()
            run.generate_wllm_brief(
                wllm=fake,
                workspace=clean_workspace,
                query="visible task prompt",
                budget=256,
                artifacts_dir=artifacts,
                stem="run-02-wllm",
                timeout=5,
            )
            self.assertFalse((clean_workspace / ".wllm").exists())

    def test_post_brief_workspace_mutation_is_infrastructure_invalid(self) -> None:
        fake_source = r'''#!/usr/bin/env python3
from pathlib import Path
Path("unexpected.txt").write_text("mutation", encoding="utf-8")
print("wllm context")
print("schema: 1.0")
print("est: 12")
print("coverage: considered=1 selected=1 omitted=0")
print("[items]")
print('{"ref":"u1","path":"src/webhook_auth.py","content":"evidence"}')
'''
        task_dir, manifest = run.load_task("webhook-rotation")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = root / "suite"
            workspace = suite / "run-01-wllm"
            artifacts = root / "artifacts"
            workspace.mkdir(parents=True)
            artifacts.mkdir()
            run.prepare_workspace(task_dir, manifest, workspace)
            digest = run.workspace_digest(workspace)["digest"]
            fake = root / "wllm"
            fake.write_text(fake_source, encoding="utf-8")
            fake.chmod(0o755)
            record = run.execute_arm(
                run_number=1,
                arm="wllm",
                agent="codex",
                executable="codex-must-not-run",
                model="test-model",
                effort="medium",
                topology="single",
                timeout=5,
                task_dir=task_dir,
                manifest=manifest,
                suite_dir=suite,
                artifacts_dir=artifacts,
                wllm=fake,
                brief_budget=256,
                agent_info={"optional_flags": {}},
                fixture_digest=digest,
                fixture_artifact="fixture.json",
            )
            self.assertFalse(record["valid"])
            self.assertEqual(record["failure"]["category"], "fixture_mutation")

    def test_brief_timeout_is_a_censored_itt_outcome(self) -> None:
        task_dir, manifest = run.load_task("webhook-rotation")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = root / "suite"
            workspace = suite / "run-01-wllm"
            artifacts = root / "artifacts"
            workspace.mkdir(parents=True)
            artifacts.mkdir()
            run.prepare_workspace(task_dir, manifest, workspace)
            digest = run.workspace_digest(workspace)["digest"]
            with mock.patch.object(
                run,
                "generate_wllm_brief",
                side_effect=run.AgentRunError("wllm briefing timed out"),
            ):
                record = run.execute_arm(
                    run_number=1,
                    arm="wllm",
                    agent="codex",
                    executable="codex-must-not-run",
                    model="test-model",
                    effort="medium",
                    topology="single",
                    timeout=7,
                    task_dir=task_dir,
                    manifest=manifest,
                    suite_dir=suite,
                    artifacts_dir=artifacts,
                    wllm=root / "wllm",
                    brief_budget=256,
                    agent_info={"optional_flags": {}},
                    fixture_digest=digest,
                    fixture_artifact="fixture.json",
                )
            self.assertTrue(record["valid"])
            self.assertEqual(record["status"], "outcome_failure")
            self.assertEqual(record["duration_seconds"], 7.0)
            self.assertEqual(record["failure"]["category"], "wllm_brief_timeout")


class RunnerIntegrationTests(unittest.TestCase):
    def test_fake_ab_run_produces_a_graded_report(self) -> None:
        fake_source = r"""#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

if sys.argv[1:3] == ["login", "status"]:
    print("Logged in")
    raise SystemExit(0)

if sys.argv[1:] == ["--version"]:
    print("codex-cli 1.2.3-test")
    raise SystemExit(0)

if sys.argv[1:3] == ["exec", "--help"]:
    print("--json --sandbox --cd --model --config --ephemeral --ignore-user-config --ignore-rules --ask-for-approval")
    raise SystemExit(0)

workspace = Path(sys.argv[sys.argv.index("--cd") + 1])
prompt = sys.argv[-1]
if "Runtime wllm access is available" in prompt:
    subprocess.run(["wllm", "--version"], check=True, capture_output=True, text=True)
candidate = workspace / "src" / "webhook_auth.py"
source = candidate.read_text(encoding="utf-8")
source = source.replace(
    '''    # Kept during the rotation window so in-flight deliveries still validate.
    secret = environment.get("WEBHOOK_SECRET_PREVIOUS")
    if not secret:
        return False
    expected = _digest(payload, secret)
    return expected == presented
''',
    '''    current = environment.get("WEBHOOK_SECRET_CURRENT")
    if not current:
        return False
    candidates = (current, environment.get("WEBHOOK_SECRET_PREVIOUS"))
    return any(
        secret is not None
        and hmac.compare_digest(_digest(payload, secret), presented)
        for secret in candidates
    )
''',
)
candidate.write_text(source, encoding="utf-8")
events = [
    {"type": "item.completed", "item": {"id": "cmd", "type": "command_execution"}},
    {"type": "item.completed", "item": {"id": "msg", "type": "agent_message", "text": "fixed"}},
    {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 20, "output_tokens": 30, "reasoning_output_tokens": 10}},
]
for event in events:
    print(json.dumps(event))
"""
        fake_wllm_source = r'''#!/usr/bin/env python3
import sys

if sys.argv[1:] == ["--version"]:
    print("wllm 1.2.3-test")
    raise SystemExit(0)
print("wllm context")
print("schema: 1.0")
print("est: 42")
print("coverage: considered=3 selected=2 omitted=1")
print("[items]")
print('{"ref":"u1","path":"src/webhook_auth.py","content":"target evidence"}')
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "codex"
            fake.write_text(fake_source, encoding="utf-8")
            fake.chmod(0o755)
            fake_wllm = root / "wllm"
            fake_wllm.write_text(fake_wllm_source, encoding="utf-8")
            fake_wllm.chmod(0o755)
            results = root / "results"
            exit_code = run.main(
                [
                    "--task",
                    "webhook-rotation",
                    "--runs",
                    "1",
                    "--codex-bin",
                    str(fake),
                    "--wllm-bin",
                    str(fake_wllm),
                    "--output-dir",
                    str(results),
                ]
            )
            self.assertEqual(exit_code, 0)
            report_path = next(results.iterdir()) / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            schema = json.loads(
                (run.BENCHMARK_ROOT / "schemas" / "report.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(set(schema["required"]).issubset(report))
            self.assertEqual(len(report["records"]), 3)
            self.assertEqual(report["records"][0]["grade"]["score"], 1)
            self.assertEqual(report["records"][0]["usage"]["total_tokens"], 130)
            self.assertEqual(report["records"][0]["tool_calls"]["total"], 1)
            self.assertEqual(report["records"][1]["grade"]["score"], 1)
            self.assertEqual(report["records"][1]["arm"], "brief-only")
            self.assertEqual(report["records"][1]["wllm_brief_tokens"], 42)
            self.assertEqual(report["records"][1]["wllm_runtime_calls"], 0)
            self.assertEqual(report["records"][2]["grade"]["score"], 1)
            self.assertEqual(report["records"][2]["wllm_brief_tokens"], 42)
            self.assertEqual(report["records"][2]["wllm_runtime_calls"], 1)
            self.assertEqual(report["records"][2]["pipeline_actions"], 2)
            self.assertEqual(report["codex_version"], "codex-cli 1.2.3-test")
            self.assertEqual(report["wllm_version"], "wllm 1.2.3-test")
            self.assertEqual(report["treatment"], "brief_plus_runtime_cli")
            self.assertEqual(
                report["aggregate"]["paired"][
                    "geometric_mean_input_token_ratio"
                ],
                1.0,
            )

    def test_agent_exit_is_itt_outcome_and_later_arms_continue(self) -> None:
        fake_source = r"""#!/usr/bin/env python3
import json
import sys

if sys.argv[1:3] == ["login", "status"]:
    raise SystemExit(0)
if sys.argv[1:] == ["--version"]:
    print("codex-cli broken-test")
    raise SystemExit(0)
if sys.argv[1:3] == ["exec", "--help"]:
    print("--json --sandbox --cd --model --config --ignore-user-config --ignore-rules")
    raise SystemExit(0)
prompt = sys.argv[-1]
if "<<<BEGIN_WLLM_BRIEF_" not in prompt:
    print(json.dumps({"type": "error", "message": "model not available"}))
    raise SystemExit(1)
print(json.dumps({"type": "item.completed", "item": {"id": "msg", "type": "agent_message", "text": "done"}}))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 20, "output_tokens": 5}}))
"""
        fake_wllm_source = r'''#!/usr/bin/env python3
import sys
if sys.argv[1:] == ["--version"]:
    print("wllm invalid-continuation-test")
    raise SystemExit(0)
print("wllm context")
print("schema: 1.0")
print("est: 12")
print("coverage: considered=1 selected=1 omitted=0")
print("[items]")
print('{"ref":"u1","path":"src/webhook_auth.py","content":"evidence"}')
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "codex"
            fake.write_text(fake_source, encoding="utf-8")
            fake.chmod(0o755)
            fake_wllm = root / "wllm"
            fake_wllm.write_text(fake_wllm_source, encoding="utf-8")
            fake_wllm.chmod(0o755)
            results = root / "results"
            exit_code = run.main(
                [
                    "--task",
                    "webhook-rotation",
                    "--runs",
                    "2",
                    "--codex-bin",
                    str(fake),
                    "--wllm-bin",
                    str(fake_wllm),
                    "--output-dir",
                    str(results),
                ]
            )
            self.assertEqual(exit_code, 0)
            artifacts = next(results.iterdir())
            report = json.loads(
                (artifacts / "report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(report["records"]), 6)
            baseline = [
                record for record in report["records"] if record["arm"] == "baseline"
            ]
            treatment = [
                record for record in report["records"] if record["arm"] == "wllm"
            ]
            self.assertTrue(all(record["valid"] for record in baseline))
            self.assertTrue(
                all(record["status"] == "outcome_failure" for record in baseline)
            )
            self.assertTrue(all(record["valid"] for record in treatment))
            self.assertTrue(
                all(value is None for value in baseline[0]["usage"].values())
            )
            self.assertEqual(baseline[0]["grade"]["score"], 0)
            self.assertEqual(report["aggregate"]["failure_taxonomy"], {"agent_exit": 2})
            paired = report["aggregate"]["paired"]
            self.assertEqual(paired["complete_pairs"], 2)
            self.assertEqual(paired["valid_pairs"], 2)
            self.assertIsNotNone(paired["geometric_mean_duration_ratio"])
            self.assertIsNone(paired["geometric_mean_input_token_ratio"])
            self.assertTrue((artifacts / "artifact-index.json").is_file())
            self.assertTrue((artifacts / "run-01-baseline.failure.json").is_file())

    def test_agent_timeout_is_a_censored_itt_outcome(self) -> None:
        fake_source = r"""#!/usr/bin/env python3
import sys
import time
if sys.argv[1:3] == ["login", "status"]:
    raise SystemExit(0)
if sys.argv[1:] == ["--version"]:
    print("codex-cli timeout-test")
    raise SystemExit(0)
if sys.argv[1:3] == ["exec", "--help"]:
    print("--json --sandbox --cd --model --config --ignore-user-config --ignore-rules")
    raise SystemExit(0)
time.sleep(5)
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "codex"
            fake.write_text(fake_source, encoding="utf-8")
            fake.chmod(0o755)
            results = root / "results"
            exit_code = run.main(
                [
                    "--task",
                    "webhook-rotation",
                    "--arm",
                    "baseline",
                    "--timeout",
                    "1",
                    "--codex-bin",
                    str(fake),
                    "--output-dir",
                    str(results),
                ]
            )
            self.assertEqual(exit_code, 0)
            artifacts = next(results.iterdir())
            report = json.loads(
                (artifacts / "report.json").read_text(encoding="utf-8")
            )
            record = report["records"][0]
            self.assertTrue(record["valid"])
            self.assertEqual(record["status"], "outcome_failure")
            self.assertTrue(record["timed_out"])
            self.assertEqual(record["failure"]["category"], "agent_timeout")
            self.assertEqual(record["duration_seconds"], 1.0)
            self.assertTrue(record["failure"]["censored"])
            self.assertEqual(record["failure"]["censor_limit_seconds"], 1.0)
            self.assertTrue(all(value is None for value in record["usage"].values()))


class RuntimeToolIsolationTests(unittest.TestCase):
    def test_only_wllm_arm_can_reach_the_pinned_runtime_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            real = root / "wllm-real"
            real.write_text(
                "#!/bin/sh\nprintf 'pinned-runtime\\n'\n", encoding="utf-8"
            )
            real.chmod(0o755)
            tools = run.prepare_runtime_tool_shims(root / "tools", real)

            for arm in run.ARMS:
                environment, log = run.runtime_tool_environment(
                    arm=arm,
                    tools=tools,
                    artifacts_dir=artifacts,
                    stem=arm,
                )
                result = subprocess.run(
                    ["wllm", "--version"],
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                usage = run.runtime_tool_usage(log, arm)
                if arm == "wllm":
                    self.assertEqual(result.returncode, 0)
                    self.assertEqual(result.stdout.strip(), "pinned-runtime")
                    self.assertEqual(usage["calls"], 1)
                    self.assertEqual(usage["denied_attempts"], 0)
                else:
                    self.assertEqual(result.returncode, 126)
                    self.assertEqual(usage["calls"], 0)
                    self.assertEqual(usage["denied_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
