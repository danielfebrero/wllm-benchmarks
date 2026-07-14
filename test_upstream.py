from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import upstream


class UpstreamTests(unittest.TestCase):
    def test_registered_suite_manifests_validate(self) -> None:
        suites = upstream.load_suites(upstream.DEFAULT_SUITES)
        self.assertIn("native-core", suites)
        for suite in suites.values():
            self.assertEqual(upstream.validate_suite(suite), [])

    def test_sha256_selection_is_stable_and_stratified(self) -> None:
        instances = [
            {"id": "a-1", "language": "a"},
            {"id": "a-2", "language": "a"},
            {"id": "b-1", "language": "b"},
            {"id": "b-2", "language": "b"},
        ]
        first = upstream.deterministic_select(
            instances, count=2, salt="fixed", strata=["language"]
        )
        second = upstream.deterministic_select(
            reversed(instances), count=2, salt="fixed", strata=["language"]
        )
        self.assertEqual(
            [upstream.instance_id(item) for item in first],
            [upstream.instance_id(item) for item in second],
        )
        self.assertEqual({item["language"] for item in first}, {"a", "b"})

    def test_select_command_can_lock_native_task_ids(self) -> None:
        suite = upstream.load_suites(upstream.DEFAULT_SUITES)["native-core"]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "selection.json"
            exit_code = upstream.select_command(
                suite,
                count=3,
                salt="fixed",
                strata=[],
                output=output,
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
