#!/usr/bin/env python3
"""Externally grade effective configuration behavior."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable


def load_candidate(workspace: Path) -> Callable[..., dict[str, Any]]:
    path = workspace / "src" / "effective_config.py"
    spec = importlib.util.spec_from_file_location("candidate_effective_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load src/effective_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_effective_config


def merge(base: Any, overlay: Any) -> Any:
    if isinstance(base, Mapping) and isinstance(overlay, Mapping):
        result = copy.deepcopy(dict(base))
        for key, value in overlay.items():
            result[key] = merge(result[key], value) if key in result else copy.deepcopy(value)
        return result
    return copy.deepcopy(overlay)


def runtime_layer(environ: Mapping[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, raw_value in environ.items():
        if not name.startswith("ORBIT__"):
            continue
        path = [part.lower() for part in name[7:].split("__")]
        if not path or any(not part for part in path):
            continue
        try:
            value: Any = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        cursor = result
        for part in path[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        cursor[path[-1]] = value
    return result


def reference(
    defaults: Mapping[str, Any],
    environment_config: Mapping[str, Any],
    environ: Mapping[str, str],
) -> dict[str, Any]:
    return merge(merge(defaults, environment_config), runtime_layer(environ))


def invoke(
    candidate: Callable[..., dict[str, Any]],
    defaults: dict[str, Any],
    environment_config: dict[str, Any],
    environ: dict[str, str],
) -> tuple[Any, bool]:
    inputs = (defaults, environment_config, environ)
    before = copy.deepcopy(inputs)
    result = candidate(*inputs)
    return result, inputs == before


def preflight() -> dict[str, Any]:
    defaults = {"nested": {"left": 1, "right": 2}}
    environment = {"nested": {"left": 3}}
    environ = {"ORBIT__NESTED__RIGHT": "4"}
    first = reference(defaults, environment, environ)
    second = reference(copy.deepcopy(defaults), copy.deepcopy(environment), dict(environ))
    if first != {"nested": {"left": 3, "right": 4}} or first != second:
        raise RuntimeError("configuration reference is not deterministic")
    return {"deterministic": True, "reference_checks": 1}


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        print(json.dumps(preflight(), sort_keys=True))
        return 0
    if len(sys.argv) != 2:
        raise SystemExit("usage: grade.py WORKSPACE | grade.py --self-test")

    preflight()
    total = 9
    try:
        candidate = load_candidate(Path(sys.argv[1]).resolve())
        defaults = {
            "service": "orbit-gateway",
            "http": {"timeout_ms": 2000, "retries": 2, "keepalive": True},
            "features": {"audit": False, "batch_size": 50},
            "logging": {"level": "info", "structured": True},
        }
        environment = {
            "http": {"timeout_ms": 4500},
            "features": {"audit": True},
            "logging": {"level": "warning"},
        }
        environ = {
            "ORBIT__HTTP__TIMEOUT_MS": "6500",
            "ORBIT__FEATURES__BATCH_SIZE": "125",
            "ORBIT__LOGGING__STRUCTURED": "false",
            "ORBIT____BROKEN": "99",
            "PATH": "/not/configuration",
        }
        result, unchanged = invoke(
            candidate,
            copy.deepcopy(defaults),
            copy.deepcopy(environment),
            dict(environ),
        )
        expected = reference(defaults, environment, environ)

        alternate_defaults = {
            "network": {"ports": [80], "tls": {"enabled": False, "mode": "off"}},
            "workers": 2,
        }
        alternate_environment = {
            "network": {"ports": [8080], "tls": {"enabled": True}},
            "workers": 4,
        }
        alternate_environ = {
            "ORBIT__NETWORK__TLS__MODE": '"strict"',
            "ORBIT__NETWORK__PORTS": "[8443,9443]",
            "ORBIT__LABEL": "canary-blue",
            "OTHER__WORKERS": "900",
        }
        alternate, alternate_unchanged = invoke(
            candidate,
            copy.deepcopy(alternate_defaults),
            copy.deepcopy(alternate_environment),
            dict(alternate_environ),
        )
        alternate_expected = reference(
            alternate_defaults, alternate_environment, alternate_environ
        )

        checks = [
            ("returns a mapping", isinstance(result, dict)),
            (
                "runtime layer has highest precedence",
                isinstance(result, dict)
                and result.get("http", {}).get("timeout_ms") == 6500
                and result.get("features", {}).get("batch_size") == 125,
            ),
            (
                "environment overrides defaults",
                isinstance(result, dict)
                and result.get("features", {}).get("audit") is True
                and result.get("logging", {}).get("level") == "warning",
            ),
            (
                "recursive merge preserves weaker siblings",
                isinstance(result, dict)
                and result.get("http", {}).get("retries") == 2
                and result.get("http", {}).get("keepalive") is True,
            ),
            (
                "runtime values retain JSON types",
                isinstance(result, dict)
                and result.get("logging", {}).get("structured") is False
                and type(result.get("features", {}).get("batch_size")) is int,
            ),
            (
                "ignores unrelated and malformed names",
                isinstance(result, dict)
                and "path" not in result
                and "" not in result
                and result == expected,
            ),
            ("generalizes to alternate layers", alternate == alternate_expected),
            (
                "supports arrays and raw strings",
                isinstance(alternate, dict)
                and alternate.get("network", {}).get("ports") == [8443, 9443]
                and alternate.get("label") == "canary-blue",
            ),
            ("does not mutate inputs", unchanged and alternate_unchanged),
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
