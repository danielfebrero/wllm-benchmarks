#!/usr/bin/env python3
"""Create a deterministic, deliberately obvious single-file repair task."""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE = '''"""Sanitize HTTP headers before diagnostic logging."""

from __future__ import annotations

from collections.abc import Mapping


SENSITIVE_HEADER_NAMES = frozenset(
    {"authorization", "proxy-authorization", "x-api-key"}
)
REDACTED = "[REDACTED]"


def sanitize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a sanitized copy of *headers*.

    Header names are ASCII case-insensitive. Replace the value of every
    Authorization, Proxy-Authorization, and X-Api-Key field with ``REDACTED``.
    Preserve each original key spelling, input iteration order, and every
    non-sensitive value exactly. Return a plain new ``dict`` and never mutate
    the supplied mapping.
    """
    sanitized = dict(headers)
    for name in SENSITIVE_HEADER_NAMES:
        if name in sanitized:
            sanitized[name] = REDACTED
    return sanitized
'''


PUBLIC_TEST = '''from __future__ import annotations

import unittest

from src.header_sanitizer import sanitize_headers


class HeaderSanitizerTests(unittest.TestCase):
    def test_redacts_lowercase_authorization(self) -> None:
        self.assertEqual(
            sanitize_headers({"authorization": "Bearer secret"}),
            {"authorization": "[REDACTED]"},
        )

    def test_preserves_ordinary_headers(self) -> None:
        headers = {"Content-Type": "application/json", "X-Trace": "trace-17"}
        self.assertEqual(sanitize_headers(headers), headers)


if __name__ == "__main__":
    unittest.main()
'''


README = '''# Header logging repair

The requested behavior and supported fields are fully specified in the
docstring of the explicitly named source file. Run public validation with
`python3 -m unittest discover -s tests -v`.
'''


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: prepare.py WORKSPACE")
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    write(root / "src" / "__init__.py", "")
    write(root / "src" / "header_sanitizer.py", SOURCE)
    write(root / "tests" / "test_header_sanitizer.py", PUBLIC_TEST)
    write(root / "README.md", README)
    write(
        root / "wllm.toml",
        "[budgets]\ndefault_text = 1200\n\n"
        "[policy]\nexternal_extractors = false\nnetwork = false\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
