#!/usr/bin/env python3
"""Grade a candidate without placing hidden checks in the agent workspace."""

from __future__ import annotations

import ast
import hashlib
import hmac
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable


def sign(payload: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"v1={digest}"


def load_verifier(workspace: Path) -> Callable[..., bool]:
    path = workspace / "src" / "webhook_auth.py"
    spec = importlib.util.spec_from_file_location("candidate_webhook_auth", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load src/webhook_auth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_signature


def uses_constant_time_compare(path: Path) -> bool:
    """Accept compare_digest through qualified or direct stdlib imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_aliases: set[str] = set()
    direct_aliases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"hmac", "secrets"}:
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module in {"hmac", "secrets"}:
            for alias in node.names:
                if alias.name == "compare_digest":
                    direct_aliases.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in direct_aliases:
            return True
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "compare_digest"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in module_aliases
        ):
            return True
    return False


def preflight() -> dict[str, object]:
    payload = b'{"id":"grader-self-test"}'
    first = sign(payload, "deterministic-secret")
    second = sign(payload, "deterministic-secret")
    if first != second or not first.startswith("v1=") or len(first) != 67:
        raise RuntimeError("webhook signing reference is not deterministic")
    return {"deterministic": True, "checks": 8}


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        print(json.dumps(preflight(), sort_keys=True))
        return 0
    if len(sys.argv) != 2:
        raise SystemExit("usage: grade.py WORKSPACE | grade.py --self-test")
    preflight()
    workspace = Path(sys.argv[1]).resolve()
    checks: list[tuple[str, Callable[[], bool]]] = []
    try:
        verify = load_verifier(workspace)
        payload = b'{"id":"evt_7429","amount":4200}'
        both = {
            "WEBHOOK_SECRET_CURRENT": "current-2026-07",
            "WEBHOOK_SECRET_PREVIOUS": "previous-2026-06",
        }
        checks = [
            (
                "accepts current secret",
                lambda: verify(
                    payload, sign(payload, both["WEBHOOK_SECRET_CURRENT"]), both
                )
                is True,
            ),
            (
                "accepts previous secret",
                lambda: verify(
                    payload, sign(payload, both["WEBHOOK_SECRET_PREVIOUS"]), both
                )
                is True,
            ),
            (
                "works without previous secret",
                lambda: verify(
                    payload,
                    sign(payload, both["WEBHOOK_SECRET_CURRENT"]),
                    {"WEBHOOK_SECRET_CURRENT": both["WEBHOOK_SECRET_CURRENT"]},
                )
                is True,
            ),
            (
                "rejects tampered payload",
                lambda: verify(
                    payload + b"!",
                    sign(payload, both["WEBHOOK_SECRET_CURRENT"]),
                    both,
                )
                is False,
            ),
            (
                "rejects malformed header",
                lambda: verify(payload, "not-a-signature", both) is False,
            ),
            (
                "rejects unknown version",
                lambda: verify(
                    payload,
                    sign(payload, both["WEBHOOK_SECRET_CURRENT"]).replace(
                        "v1=", "v2="
                    ),
                    both,
                )
                is False,
            ),
            (
                "rejects missing current configuration",
                lambda: verify(
                    payload,
                    sign(payload, both["WEBHOOK_SECRET_PREVIOUS"]),
                    {"WEBHOOK_SECRET_PREVIOUS": both["WEBHOOK_SECRET_PREVIOUS"]},
                )
                is False,
            ),
            (
                "uses constant-time comparison",
                lambda: uses_constant_time_compare(
                    workspace / "src" / "webhook_auth.py"
                ),
            ),
        ]
    except Exception as error:  # Broken imports are scored failures, not grader crashes.
        result = {
            "passed": 0,
            "total": 8,
            "score": 0.0,
            "failures": [f"candidate load failed: {type(error).__name__}: {error}"],
        }
        print(json.dumps(result, sort_keys=True))
        return 1

    failures: list[str] = []
    passed = 0
    for name, check in checks:
        try:
            if check():
                passed += 1
            else:
                failures.append(name)
        except Exception as error:
            failures.append(f"{name}: {type(error).__name__}: {error}")
    result = {
        "passed": passed,
        "total": len(checks),
        "score": passed / len(checks),
        "failures": failures,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
