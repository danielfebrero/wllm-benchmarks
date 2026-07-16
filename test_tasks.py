from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import run


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


class NativeTaskTests(unittest.TestCase):
    def test_new_fixtures_are_deterministic_and_externally_graded(self) -> None:
        expected_initial = {
            "config-precedence": 3 / 9,
            "migration-lineage": 2 / 10,
            "single-file-control": 4 / 8,
            # Hard multi-hop tasks: broken fixtures must stay well below solve.
            "quota-settlement": 3 / 12,
            "authz-lattice": 2 / 12,
        }
        for task_id, expected_score in expected_initial.items():
            with self.subTest(task=task_id):
                task_dir, manifest = run.load_task(task_id)
                with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
                    first_path = Path(first)
                    second_path = Path(second)
                    run.prepare_workspace(task_dir, manifest, first_path)
                    run.prepare_workspace(task_dir, manifest, second_path)
                    self.assertEqual(snapshot(first_path), snapshot(second_path))
                    self.assertFalse((first_path / "grade.py").exists())
                    grade = run.grade_workspace(task_dir, manifest, first_path)
                    self.assertAlmostEqual(float(grade["score"]), expected_score)

                preflight = run.run_checked(
                    ["python3", str(task_dir / "grade.py"), "--self-test"]
                )
                self.assertTrue(json.loads(preflight.stdout)["deterministic"])

    def test_hard_tasks_are_not_trivial_for_public_suite(self) -> None:
        """Hard tasks expose only weak public tests and large archive noise."""
        for task_id in ("quota-settlement", "authz-lattice"):
            with self.subTest(task=task_id):
                task_dir, manifest = run.load_task(task_id)
                with tempfile.TemporaryDirectory() as temporary:
                    workspace = Path(temporary)
                    run.prepare_workspace(task_dir, manifest, workspace)
                    public = run.run_checked(
                        ["python3", "-m", "unittest", "discover", "-s", "tests", "-v"],
                        cwd=workspace,
                    )
                    self.assertIn("OK", public.stderr + public.stdout)
                    grade = run.grade_workspace(task_dir, manifest, workspace)
                    self.assertLess(
                        float(grade["score"]),
                        0.35,
                        msg=f"{task_id} initial grade too easy: {grade}",
                    )
                    archive_files = list((workspace / "archive").rglob("*"))
                    self.assertGreaterEqual(len([p for p in archive_files if p.is_file()]), 40)


if __name__ == "__main__":
    unittest.main()
