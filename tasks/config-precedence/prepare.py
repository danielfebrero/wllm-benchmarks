#!/usr/bin/env python3
"""Create a deterministic layered-configuration incident workspace."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SOURCE = '''"""Build the effective Orbit gateway configuration."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any


RUNTIME_PREFIX = "ORBIT__"


def _runtime_layer(environ: Mapping[str, str]) -> dict[str, Any]:
    layer: dict[str, Any] = {}
    for name, raw_value in environ.items():
        if not name.startswith(RUNTIME_PREFIX):
            continue
        path = [part.lower() for part in name[len(RUNTIME_PREFIX) :].split("__")]
        if not path or any(not part for part in path):
            continue
        try:
            value: Any = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        cursor = layer
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = value
    return layer


def build_effective_config(
    defaults: Mapping[str, Any],
    environment_config: Mapping[str, Any],
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """Combine supplied configuration layers into a new mapping."""
    effective = copy.deepcopy(dict(defaults))
    effective.update(_runtime_layer(environ))
    effective.update(copy.deepcopy(dict(environment_config)))
    return effective
'''


PUBLIC_TEST = '''from __future__ import annotations

import unittest

from src.effective_config import build_effective_config


class EffectiveConfigTests(unittest.TestCase):
    def test_returns_a_fresh_mapping(self) -> None:
        defaults = {"service": "orbit", "workers": 2}
        result = build_effective_config(defaults, {}, {})
        self.assertEqual(result, defaults)
        self.assertIsNot(result, defaults)

    def test_environment_can_replace_a_scalar(self) -> None:
        result = build_effective_config(
            {"log_level": "info"}, {"log_level": "warning"}, {}
        )
        self.assertEqual(result["log_level"], "warning")


if __name__ == "__main__":
    unittest.main()
'''


CONTRACT = '''# Effective configuration contract

Status: authoritative
Owner: Runtime Platform
Applies to: Orbit gateway configuration assembled by `src.effective_config`

The effective document is a recursive merge of three layers, from weakest to
strongest: shared defaults, the selected environment document, then runtime
variables. A mapping at a stronger layer merges recursively with a mapping at
the same path; every other value replaces the weaker value. Values absent from
a stronger layer remain present. No input mapping may be mutated.

Runtime variables use the `ORBIT__` prefix and double underscores between
lowercase output path components. For example, `ORBIT__HTTP__TIMEOUT_MS`
addresses `http.timeout_ms`. Decode each runtime value as JSON when it is valid
JSON, preserving booleans, numbers, nulls, arrays, and objects; otherwise keep
the original string. Ignore variables outside the Orbit namespace and names
with an empty path component.
'''


README = '''# Orbit gateway rollout CFG-2717

Run public validation with `python3 -m unittest discover -s tests -v`.

The live configuration contract is under `docs/contracts`, shared and
environment layers are under `config`, deployment-provided runtime variables
are recorded under `deployment/current`, and active observations are under
`evidence/live`. `archive/` contains retired rollout material and is never
authoritative for the active deployment.
'''


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def dump(path: Path, value: object) -> None:
    write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: prepare.py WORKSPACE")
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)

    write(root / "src" / "__init__.py", "")
    write(root / "src" / "effective_config.py", SOURCE)
    write(root / "tests" / "test_effective_config.py", PUBLIC_TEST)
    write(root / "docs" / "contracts" / "effective-configuration.md", CONTRACT)
    write(root / "README.md", README)
    write(
        root / "wllm.toml",
        "[budgets]\ndefault_text = 2500\n\n"
        "[policy]\nexternal_extractors = false\nnetwork = false\n",
    )

    dump(
        root / "config" / "defaults.json",
        {
            "service": "orbit-gateway",
            "http": {"timeout_ms": 2000, "retries": 2, "keepalive": True},
            "features": {"audit": False, "batch_size": 50},
            "logging": {"level": "info", "structured": True},
        },
    )
    dump(
        root / "config" / "environments" / "production.json",
        {
            "http": {"timeout_ms": 4500},
            "features": {"audit": True},
            "logging": {"level": "warning"},
        },
    )
    dump(
        root / "deployment" / "current" / "runtime-environment.json",
        {
            "rollout": "CFG-2717",
            "variables": {
                "ORBIT__HTTP__TIMEOUT_MS": "6500",
                "ORBIT__FEATURES__BATCH_SIZE": "125",
                "ORBIT__LOGGING__STRUCTURED": "false",
                "PATH": "/usr/local/bin:/usr/bin",
            },
        },
    )
    write(
        root / "evidence" / "live" / "CFG-2717.log",
        "2026-07-14T12:18:42Z level=ERROR rollout=CFG-2717 "
        "code=EFFECTIVE_CONFIG_MISMATCH missing=http.retries "
        "expected_http_timeout_ms=6500 actual_http_timeout_ms=4500 "
        "expected_batch_size=125 actual_batch_size=50\n",
    )

    # Closed rollouts create realistic lexical noise while remaining clearly
    # non-authoritative. Generation is byte-for-byte deterministic.
    for number in range(120):
        year = 2020 + number % 6
        rollout = f"CFG-{1100 + number}"
        retired_order = "runtime, defaults, environment" if number % 2 else "defaults only"
        write(
            root / "archive" / str(year) / rollout / "notes.md",
            f"# Archived rollout {rollout}\n\n"
            "Status: closed historical record\n\n"
            f"The retired loader used `{retired_order}` during {year}. "
            "This ordering must not be applied to current production.\n",
        )
        dump(
            root / "archive" / str(year) / rollout / "config.json",
            {
                "status": "retired",
                "rollout": rollout,
                "http_timeout_ms": 1000 + number,
                "runtime_prefix": f"LEGACY_{number:03d}_",
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
