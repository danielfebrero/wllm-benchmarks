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


if __name__ == "__main__":
    unittest.main()
