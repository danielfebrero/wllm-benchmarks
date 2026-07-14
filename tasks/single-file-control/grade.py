#!/usr/bin/env python3
"""Externally grade the single-file header sanitizer control."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable


SENSITIVE = {"authorization", "proxy-authorization", "x-api-key"}


def load_candidate(workspace: Path) -> Callable[..., dict[str, str]]:
    path = workspace / "src" / "header_sanitizer.py"
    spec = importlib.util.spec_from_file_location("candidate_header_sanitizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load src/header_sanitizer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sanitize_headers


def reference(headers: dict[str, str]) -> dict[str, str]:
    return {
        name: "[REDACTED]" if name.casefold() in SENSITIVE else value
        for name, value in headers.items()
    }


def preflight() -> dict[str, object]:
    sample = {
        "AUTHORIZATION": "secret-a",
        "x-api-key": "secret-b",
        "Accept": "application/json",
    }
    expected = {
        "AUTHORIZATION": "[REDACTED]",
        "x-api-key": "[REDACTED]",
        "Accept": "application/json",
    }
    if reference(sample) != expected or sample["AUTHORIZATION"] != "secret-a":
        raise RuntimeError("header reference failed")
    return {"deterministic": True, "sensitive_names": len(SENSITIVE)}


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        print(json.dumps(preflight(), sort_keys=True))
        return 0
    if len(sys.argv) != 2:
        raise SystemExit("usage: grade.py WORKSPACE | grade.py --self-test")

    preflight()
    total = 8
    try:
        candidate = load_candidate(Path(sys.argv[1]).resolve())
        lower = candidate({"authorization": "Bearer lower"})
        mixed = candidate({"AuThOrIzAtIoN": "Bearer mixed"})
        proxy = candidate({"Proxy-Authorization": "Basic hidden"})
        api = candidate({"X-API-KEY": "key-hidden"})
        combined_input = {
            "X-Trace": "trace-77",
            "AUTHORIZATION": "Bearer first",
            "authorization": "Bearer second",
            "Content-Type": "application/json",
            "x-Api-Key": "key-third",
        }
        before = dict(combined_input)
        combined = candidate(combined_input)
        expected_combined = reference(before)

        checks = [
            (
                "returns a plain new dict",
                type(combined) is dict and combined is not combined_input,
            ),
            (
                "redacts lowercase authorization",
                lower == {"authorization": "[REDACTED]"},
            ),
            (
                "redacts mixed-case authorization",
                mixed == {"AuThOrIzAtIoN": "[REDACTED]"},
            ),
            (
                "redacts proxy authorization case-insensitively",
                proxy == {"Proxy-Authorization": "[REDACTED]"},
            ),
            (
                "redacts API keys case-insensitively",
                api == {"X-API-KEY": "[REDACTED]"},
            ),
            ("redacts every sensitive spelling", combined == expected_combined),
            (
                "preserves ordinary fields, spelling, and order",
                list(combined) == list(before)
                and combined.get("X-Trace") == "trace-77"
                and combined.get("Content-Type") == "application/json",
            ),
            ("does not mutate the supplied mapping", combined_input == before),
        ]
    except Exception as error:
        print(
            json.dumps(
                {
                    "passed": 0,
                    "total": total,
                    "score": 0.0,
                    "failures": [
                        f"candidate execution failed: {type(error).__name__}: {error}"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0

    failures = [name for name, passed in checks if not passed]
    passed = len(checks) - len(failures)
    print(
        json.dumps(
            {
                "passed": passed,
                "total": len(checks),
                "score": passed / len(checks),
                "failures": failures,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
