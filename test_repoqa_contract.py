from __future__ import annotations

import importlib.util
import os
import unittest
from unittest import mock

import repoqa_ab


class CharacterTokenizer:
    """Deterministic tokenizer stub; the contract under test is RepoQA's API."""

    @staticmethod
    def tokenize(value: str) -> list[str]:
        return list(value)


class RepoQAContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if importlib.util.find_spec("repoqa") is None:
            if os.environ.get("REPOQA_CONTRACT") == "1":
                raise AssertionError("REPOQA_CONTRACT=1 but repoqa is not importable")
            raise unittest.SkipTest(
                "pinned RepoQA dependency is exercised by the repoqa-contract CI job"
            )

    def test_pinned_provenance_context_and_official_grader(self) -> None:
        provenance = repoqa_ab.repoqa_provenance()
        self.assertEqual(
            provenance["revision"], repoqa_ab.REPOQA_HARNESS_REVISION
        )
        repoqa_ab.repoqa_imports()

        source = "def exact_answer(value):\n    return value * 2\n"
        needle = {
            "name": "exact_answer",
            "description": "Return the input multiplied by two.",
            "path": "answer.py",
            "start_byte": 0,
            "end_byte": len(source),
            "start_line": 0,
            "end_line": 2,
        }
        repository = {
            "repo": "contract-fixture",
            "content": {"answer.py": source},
            "dependency": {"answer.py": []},
            "needles": [needle],
        }

        with mock.patch(
            "repoqa.search_needle_function.AutoTokenizer.from_pretrained",
            return_value=CharacterTokenizer(),
        ) as tokenizer:
            context = repoqa_ab.official_context(
                repository,
                needle,
                language="python",
                position_ratio=0.5,
                code_context_size=1024,
                allow_model_download=False,
            )
        tokenizer.assert_called_once_with(
            repoqa_ab.TOKENIZER_MODEL,
            revision=repoqa_ab.TOKENIZER_REVISION,
            local_files_only=True,
        )
        self.assertIn(source.strip(), context["code_context"])
        self.assertGreater(context["code_context_ntokens"], 0)

        grade = repoqa_ab.score_answer(
            f"```python\n{source}```",
            language="python",
            repository=repository,
            needle=needle,
        )
        self.assertTrue(grade["passed"])
        self.assertEqual(grade["best_target"], "exact_answer")
        self.assertGreaterEqual(grade["best_similarity"], 0.8)

    def test_contract_job_cannot_silently_skip(self) -> None:
        if os.environ.get("REPOQA_CONTRACT") == "1":
            self.assertIsNotNone(importlib.util.find_spec("repoqa"))


if __name__ == "__main__":
    unittest.main()
