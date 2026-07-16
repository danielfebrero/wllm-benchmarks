from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import matrix_report


class MatrixReportTests(unittest.TestCase):
    def _write_cell_report(self, path: Path, *, solve: float, tokens: int, seconds: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "task": "release-evidence",
                    "agent": "codex",
                    "model": "test-model",
                    "reasoning": "low",
                    "topology": "single",
                    "aggregate": {
                        "baseline": {
                            "solve_rate": solve,
                            "median_score": solve,
                            "median_input_tokens": tokens,
                            "median_duration_seconds": seconds,
                            "valid_runs": 1,
                            "invalid_runs": 0,
                        },
                        "brief-only": {
                            "solve_rate": solve,
                            "median_score": solve,
                            "median_input_tokens": int(tokens * 0.9),
                            "median_duration_seconds": seconds * 0.9,
                            "valid_runs": 1,
                            "invalid_runs": 0,
                        },
                        "wllm": {
                            "solve_rate": solve,
                            "median_score": solve,
                            "median_input_tokens": int(tokens * 0.7),
                            "median_duration_seconds": seconds * 0.8,
                            "valid_runs": 1,
                            "invalid_runs": 0,
                        },
                        "contrasts": {
                            "wllm_over_baseline": {
                                "geometric_mean_input_token_ratio": 0.7,
                                "geometric_mean_duration_ratio": 0.8,
                                "median_score_delta": 0.0,
                                "valid_pairs": 1,
                            },
                            "brief_only_over_baseline": {
                                "geometric_mean_input_token_ratio": 0.9,
                                "geometric_mean_duration_ratio": 0.9,
                                "median_score_delta": 0.0,
                                "valid_pairs": 1,
                            },
                            "wllm_over_brief_only": {
                                "geometric_mean_input_token_ratio": 0.78,
                                "geometric_mean_duration_ratio": 0.89,
                                "median_score_delta": 0.0,
                                "valid_pairs": 1,
                            },
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_write_report_emits_html_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            matrix_dir = Path(temporary) / "matrix-test"
            matrix_dir.mkdir()
            report_rel = "cells/001/artifacts/run/report.json"
            self._write_cell_report(matrix_dir / report_rel, solve=1.0, tokens=10000, seconds=120.0)
            (matrix_dir / "artifact-index.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.1",
                        "generated_at": "2026-07-16T00:00:00+00:00",
                        "jobs": 1,
                        "timing_comparable": True,
                        "cells": [
                            {
                                "id": "001-release-evidence-codex-test-single",
                                "number": 1,
                                "task": "release-evidence",
                                "agent": "codex",
                                "model": "test-model",
                                "effort": "low",
                                "topology": "single",
                                "exit_code": 0,
                                "timed_out": False,
                                "duration_seconds": 400.0,
                                "report": report_rel,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (matrix_dir / "matrix-plan.json").write_text(
                json.dumps({"generated_at": "2026-07-16T00:00:00+00:00", "jobs": 1}) + "\n",
                encoding="utf-8",
            )

            html_path = matrix_report.write_report(matrix_dir)
            self.assertEqual(html_path, matrix_dir / "report.html")
            self.assertTrue(html_path.is_file())
            self.assertTrue((matrix_dir / "matrix-report.json").is_file())

            html = html_path.read_text(encoding="utf-8")
            # Charts are pure inline SVG (no Chart.js CDN / no empty canvas).
            self.assertIn("<svg class=\"chart\"", html)
            self.assertGreaterEqual(html.count("<svg class=\"chart\""), 4)
            self.assertNotIn("cdn.jsdelivr.net", html)
            self.assertNotIn("chart.umd", html)
            self.assertIn("How to read this", html)
            self.assertIn("release-evidence", html)
            self.assertIn("<rect ", html)

            payload = json.loads((matrix_dir / "matrix-report.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["cells_ok"], 1)
            self.assertAlmostEqual(payload["summary"]["geo_input_ratio_wllm_baseline"], 0.7)
            self.assertEqual(payload["matrix"]["cells"][0]["arms"]["wllm"]["median_input_tokens"], 7000)

    def test_missing_index_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                matrix_report.collect_matrix(Path(temporary))


if __name__ == "__main__":
    unittest.main()
