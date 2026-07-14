from __future__ import annotations

import json
import os
import tempfile
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
                command: list[str], *, cwd: Path, check: bool
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
    def test_fixture_is_deterministic_and_initial_bug_is_detected(self) -> None:
        task_dir, manifest = run.load_task("webhook-rotation")
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_path, second_path = Path(first), Path(second)
            run.prepare_workspace(task_dir, manifest, first_path)
            run.prepare_workspace(task_dir, manifest, second_path)
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


class RunnerIntegrationTests(unittest.TestCase):
    def test_fake_ab_run_produces_a_graded_report(self) -> None:
        fake_source = r"""#!/usr/bin/env python3
import json
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
            self.assertEqual(len(report["records"]), 2)
            self.assertEqual(report["records"][0]["grade"]["score"], 1)
            self.assertEqual(report["records"][0]["usage"]["total_tokens"], 130)
            self.assertEqual(report["records"][0]["tool_calls"]["total"], 1)
            self.assertEqual(report["records"][1]["grade"]["score"], 1)
            self.assertEqual(report["records"][1]["wllm_brief_tokens"], 42)
            self.assertEqual(report["records"][1]["pipeline_actions"], 2)
            self.assertEqual(report["codex_version"], "codex-cli 1.2.3-test")
            self.assertEqual(report["wllm_version"], "wllm 1.2.3-test")
            self.assertEqual(report["treatment"], "precomputed_context")
            self.assertEqual(
                report["aggregate"]["paired"][
                    "geometric_mean_input_token_ratio"
                ],
                1.0,
            )

    def test_invalid_codex_turn_aborts_instead_of_scoring_fixture(self) -> None:
        fake_source = r"""#!/usr/bin/env python3
import json
import sys

if sys.argv[1:3] == ["login", "status"]:
    raise SystemExit(0)
if sys.argv[1:] == ["--version"]:
    print("codex-cli broken-test")
    raise SystemExit(0)
if sys.argv[1:3] == ["exec", "--help"]:
    print("--json --sandbox --cd --model --config")
    raise SystemExit(0)
print(json.dumps({"type": "error", "message": "model not available"}))
raise SystemExit(1)
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "codex"
            fake.write_text(fake_source, encoding="utf-8")
            fake.chmod(0o755)
            exit_code = run.main(
                [
                    "--task",
                    "webhook-rotation",
                    "--arm",
                    "baseline",
                    "--codex-bin",
                    str(fake),
                    "--output-dir",
                    str(root / "results"),
                ]
            )
            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
