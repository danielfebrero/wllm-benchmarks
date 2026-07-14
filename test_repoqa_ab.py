from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import repoqa_ab


def dataset_fixture() -> dict[str, object]:
    languages: dict[str, object] = {}
    for language in ("python", "rust"):
        repositories = []
        for repository_number in range(2):
            repo = f"repo-{repository_number}"
            needles = []
            content = {}
            dependency = {}
            for needle_number in range(3):
                path = f"src/file_{needle_number}.txt"
                name = f"needle_{language}_{repository_number}_{needle_number}"
                source = f"function {name}() {{ return {needle_number}; }}\n"
                content[path] = source
                dependency[path] = []
                needles.append(
                    {
                        "name": name,
                        "description": f"returns the number {needle_number}",
                        "path": path,
                        "start_byte": 0,
                        "end_byte": len(source),
                        "start_line": 0,
                        "end_line": 1,
                    }
                )
            repositories.append(
                {
                    "repo": repo,
                    "content": content,
                    "dependency": dependency,
                    "needles": needles,
                }
            )
        languages[language] = repositories
    return languages


class RepoQAAdapterTests(unittest.TestCase):
    def write_dataset(self, root: Path) -> tuple[Path, str]:
        path = root / "repoqa.json"
        encoded = json.dumps(dataset_fixture(), sort_keys=True) + "\n"
        path.write_text(encoded, encoding="utf-8")
        return path, hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def test_plan_is_deterministic_balanced_and_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self.write_dataset(Path(temporary))
            dataset, observed = repoqa_ab.load_dataset(path, digest)
            instances = repoqa_ab.flatten_instances(dataset)
            first = repoqa_ab.balanced_select(instances, count=8, salt="fixed")
            second = repoqa_ab.balanced_select(instances, count=8, salt="fixed")
            self.assertEqual(first, second)
            plan = repoqa_ab.selection_document(
                first, dataset_sha256=observed, salt="fixed"
            )
            self.assertEqual(plan["dataset_sha256"], digest)
            self.assertEqual(plan["selected"], 8)
            self.assertEqual(set(plan["strata_counts"]["language"].values()), {4})
            self.assertEqual(len(plan["selection_sha256"]), 64)

    def test_plan_command_needs_no_repoqa_or_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self.write_dataset(Path(temporary))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = repoqa_ab.main(
                    [
                        "plan",
                        "--dataset",
                        str(path),
                        "--expect-dataset-sha256",
                        digest,
                        "--allow-unofficial-dataset",
                        "--count",
                        "4",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["selected"], 4)

    def test_materialization_contains_only_content_and_rejects_escape(self) -> None:
        repository = dataset_fixture()["python"][0]  # type: ignore[index]
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            repoqa_ab.materialize_repository(repository, workspace)  # type: ignore[arg-type]
            self.assertTrue((workspace / "src/file_0.txt").is_file())
            self.assertFalse((workspace / "needles.json").exists())
        bad = {"content": {"../escape": "bad"}}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(repoqa_ab.RepoQAError):
                repoqa_ab.materialize_repository(
                    bad, Path(temporary) / "workspace"
                )

    def test_prompt_has_same_instruction_and_no_hidden_metadata(self) -> None:
        prompt = repoqa_ab.build_prompt(
            "returns a number", "bounded context", arm="wllm"
        )
        self.assertIn("bounded context", prompt)
        self.assertIn("returns a number", prompt)
        self.assertNotIn("secret_needle_name", prompt)
        self.assertIn("Do not inspect the workspace", prompt)
        self.assertNotIn("official RepoQA", prompt)
        self.assertNotIn("wllm context", prompt)

    def test_agent_binary_default_follows_selected_agent(self) -> None:
        parser = repoqa_ab.build_parser()
        args = parser.parse_args(
            [
                "run",
                "--dataset",
                "/tmp/dataset.json",
                "--agent",
                "claude",
                "--wllm-bin",
                "wllm",
            ]
        )
        self.assertIsNone(args.agent_bin)
        self.assertEqual(
            repoqa_ab.agent_run.AGENT_DEFAULTS[args.agent]["binary"], "claude"
        )

    def test_aggregate_is_intention_to_treat(self) -> None:
        instance = {
            "id": "python::repo::needle",
            "language": "python",
            "repo": "repo",
            "position_ratio": 0.5,
        }
        baseline = repoqa_ab.outcome_record(
            pair_number=1,
            repetition=1,
            instance=instance,
            arm="baseline",
            status="completed",
            duration=10,
            brief_seconds=0,
            brief_tokens=None,
            fixture_digest="sha256:x",
            diagnostics=[],
        )
        baseline["grade"] = {"score": 1.0}
        treatment = repoqa_ab.outcome_record(
            pair_number=1,
            repetition=1,
            instance=instance,
            arm="wllm",
            status="agent_timeout",
            duration=20,
            brief_seconds=2,
            brief_tokens=100,
            fixture_digest="sha256:x",
            diagnostics=["timeout"],
        )
        result = repoqa_ab.aggregate([baseline, treatment])
        self.assertEqual(result["wllm"]["pass_rate_at_0_8"], 0)
        self.assertEqual(
            result["paired"]["geometric_mean_observed_cost_ratio_itt"], 2
        )
        self.assertIsNone(
            result["paired"]["geometric_mean_completed_duration_ratio"]
        )
        self.assertEqual(result["paired"]["median_score_delta"], -1)

    def test_aggregate_excludes_infrastructure_invalid_pair(self) -> None:
        instance = {
            "id": "python::repo::needle",
            "language": "python",
            "repo": "repo",
            "position_ratio": 0.5,
        }
        records = []
        for arm in ("baseline", "wllm"):
            records.append(
                repoqa_ab.outcome_record(
                    pair_number=1,
                    repetition=1,
                    instance=instance,
                    arm=arm,
                    status="fixture_verification_error",
                    duration=1,
                    brief_seconds=0,
                    brief_tokens=None,
                    fixture_digest="sha256:x",
                    diagnostics=["bad fixture"],
                    valid=False,
                    failure_phase="fixture",
                )
            )
        result = repoqa_ab.aggregate(records)
        self.assertEqual(result["paired"]["complete_pairs"], 1)
        self.assertEqual(result["paired"]["valid_pairs"], 0)
        self.assertEqual(result["paired"]["invalid_pairs"], 1)
        self.assertIsNone(
            result["paired"]["geometric_mean_observed_cost_ratio_itt"]
        )

    def test_run_requires_frozen_dataset_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.write_dataset(Path(temporary))
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                repoqa_ab.main(
                    ["run", "--dataset", str(path), "--wllm-bin", "wllm"]
                )
            self.assertIn("requires --expect-dataset-sha256", stderr.getvalue())

    def test_run_deletes_retrieval_source_before_both_agent_arms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path, digest = self.write_dataset(root)
            output = root / "results"
            tokenizer = {
                "model": repoqa_ab.TOKENIZER_MODEL,
                "requested_revision": repoqa_ab.TOKENIZER_REVISION,
                "resolved_revision": repoqa_ab.TOKENIZER_REVISION,
                "local_files_only": True,
                "loaded_file_sha256": {},
            }
            observed: list[tuple[str, set[str]]] = []

            def fake_agent_arm(**values: object) -> dict[str, object]:
                workspace = values["workspace"]
                self.assertIsInstance(workspace, Path)
                names = {path.name for path in workspace.iterdir()}  # type: ignore[union-attr]
                observed.append((str(values["arm"]), names))
                self.assertEqual(names, {".git"})
                retrieval = workspace.parents[1] / "retrieval"  # type: ignore[union-attr]
                self.assertFalse(any(retrieval.rglob("*")))
                return repoqa_ab.outcome_record(
                    pair_number=int(values["pair_number"]),
                    repetition=int(values["repetition"]),
                    instance=values["instance"],  # type: ignore[arg-type]
                    arm=str(values["arm"]),
                    status="completed",
                    duration=1,
                    brief_seconds=float(values["brief_seconds"]),
                    brief_tokens=values["brief_tokens"],  # type: ignore[arg-type]
                    fixture_digest=str(values["fixture_digest"]),
                    diagnostics=[],
                )

            official = {
                "code_context": "official context",
                "code_context_ntokens": 10,
                "tokenizer": tokenizer,
            }
            with (
                mock.patch.object(repoqa_ab, "repoqa_provenance", return_value={}),
                mock.patch.object(repoqa_ab, "repoqa_imports"),
                mock.patch.object(
                    repoqa_ab, "pinned_tokenizer", return_value=(object(), tokenizer)
                ),
                mock.patch.object(repoqa_ab, "official_context", return_value=official),
                mock.patch.object(repoqa_ab, "executable_path", return_value="/bin/true"),
                mock.patch.object(repoqa_ab.agent_run, "check_agent_auth"),
                mock.patch.object(
                    repoqa_ab.agent_run,
                    "inspect_agent",
                    return_value={"version": "test", "optional_flags": {}},
                ),
                mock.patch.object(
                    repoqa_ab,
                    "binary_provenance",
                    return_value={"path": "/bin/true", "sha256": "x", "version": "x"},
                ),
                mock.patch.object(
                    repoqa_ab.agent_run,
                    "generate_wllm_brief",
                    return_value=("brief context", 20, 0.1),
                ),
                mock.patch.object(repoqa_ab, "run_agent_arm", side_effect=fake_agent_arm),
            ):
                exit_code = repoqa_ab.main(
                    [
                        "run",
                        "--dataset",
                        str(dataset_path),
                        "--expect-dataset-sha256",
                        digest,
                        "--allow-unofficial-dataset",
                        "--count",
                        "1",
                        "--repetitions",
                        "1",
                        "--wllm-bin",
                        "wllm",
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual({arm for arm, _ in observed}, {"baseline", "wllm"})


if __name__ == "__main__":
    unittest.main()
