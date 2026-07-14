from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matrix


class MatrixTests(unittest.TestCase):
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
            for cell in plan["cells"]:
                self.assertIn("--agent", cell["command"])
                self.assertIn("--model", cell["command"])


if __name__ == "__main__":
    unittest.main()
