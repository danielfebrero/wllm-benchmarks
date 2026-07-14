#!/usr/bin/env python3
"""Grade release evidence without exposing the complete contract as tests."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable


def load_builder(workspace: Path) -> Callable[..., dict[str, Any]]:
    path = workspace / "src" / "release_evidence.py"
    spec = importlib.util.spec_from_file_location("candidate_release_evidence", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load src/release_evidence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_release_evidence


def reference_output(
    release: dict[str, Any],
    build: dict[str, Any],
    deployment: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Small executable specification used only by the external grader."""
    digest = str(build["artifact_sha256"])
    if digest.lower().startswith("sha256:"):
        digest = digest.split(":", 1)[1]
    digest = digest.lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("artifact_sha256 must contain 64 hexadecimal characters")
    return {
        "schema_version": policy["schema_version"],
        "release_id": release["release_id"],
        "git_commit": release["git_commit"],
        "environment": policy["environment"],
        "artifact": {
            "digest": f"sha256:{digest}",
            "size_bytes": int(build["artifact_size_bytes"]),
        },
        "deployment": {
            "rollout_id": deployment["rollout_id"],
            "region": str(deployment["region"]).lower(),
        },
    }


def fixtures() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        {
            "release_id": "REL-2026.07.14.3",
            "git_commit": "4c6c9f7d4fe0d9d731bd41ae094b16f7083f37c2",
        },
        {
            "git_commit": "c1a0a5e29a99de91c41219f84f0f8c587b7cb418",
            "artifact_sha256": (
                "78E7C5E11A908B8F754883DF3E12DBE9"
                "E461CA12AD4C573FDE7E343BB921F206"
            ),
            "artifact_size_bytes": "18432",
        },
        {
            "rollout_id": "rollout-prod-771",
            "region": "EU-WEST-1",
            "environment": "staging",
        },
        {"schema_version": "2.1", "environment": "production"},
    )


def preflight() -> dict[str, Any]:
    """Prove that the reference has one stable, compact JSON representation."""
    args = fixtures()
    first = reference_output(*copy.deepcopy(args))
    second = reference_output(*copy.deepcopy(args))
    encoded_first = json.dumps(first, sort_keys=True, separators=(",", ":"))
    encoded_second = json.dumps(second, sort_keys=True, separators=(",", ":"))
    if first != second or encoded_first != encoded_second:
        raise RuntimeError("reference output is not deterministic")
    if len(encoded_first.encode("utf-8")) > 512:
        raise RuntimeError("reference output is unexpectedly large")
    return {"deterministic": True, "output_bytes": len(encoded_first.encode("utf-8"))}


def candidate_result(
    build_evidence: Callable[..., dict[str, Any]],
    args: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    before = copy.deepcopy(args)
    result = build_evidence(*args)
    return result, args == before


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        print(json.dumps(preflight(), sort_keys=True))
        return 0
    if len(sys.argv) != 2:
        raise SystemExit("usage: grade.py WORKSPACE | grade.py --self-test")

    preflight()
    workspace = Path(sys.argv[1]).resolve()
    total = 10
    try:
        build_evidence = load_builder(workspace)
        args = fixtures()
        result, inputs_unchanged = candidate_result(build_evidence, args)
        expected = reference_output(*fixtures())

        alternate = (
            {"release_id": "REL-2030.01.02.9", "git_commit": "f" * 40},
            {
                "git_commit": "0" * 40,
                "artifact_sha256": "sha256:" + "AB" * 32,
                "artifact_size_bytes": 7,
            },
            {
                "rollout_id": "rollout-canary-9",
                "region": "US-EAST-2",
                "environment": "development",
            },
            {"schema_version": "3.0", "environment": "canary"},
        )
        alternate_result, alternate_unchanged = candidate_result(
            build_evidence, copy.deepcopy(alternate)
        )
        alternate_expected = reference_output(*copy.deepcopy(alternate))

        repeated = build_evidence(*copy.deepcopy(args))
        json.dumps(result, sort_keys=True, separators=(",", ":"))
        malformed_rejected = False
        malformed = list(copy.deepcopy(args))
        malformed[1]["artifact_sha256"] = "not-a-sha256"
        try:
            build_evidence(*malformed)
        except ValueError:
            malformed_rejected = True

        checks: list[tuple[str, bool]] = [
            ("returns a JSON object", isinstance(result, dict)),
            (
                "uses effective policy schema and environment",
                result.get("schema_version") == expected["schema_version"]
                and result.get("environment") == expected["environment"],
            ),
            (
                "uses current catalog release identity",
                result.get("release_id") == expected["release_id"]
                and result.get("git_commit") == expected["git_commit"],
            ),
            (
                "canonicalizes artifact digest",
                isinstance(result.get("artifact"), dict)
                and result["artifact"].get("digest")
                == expected["artifact"]["digest"],
            ),
            (
                "emits artifact size as integer",
                isinstance(result.get("artifact"), dict)
                and result["artifact"].get("size_bytes") == 18432
                and type(result["artifact"].get("size_bytes")) is int,
            ),
            (
                "normalizes deployment fields",
                result.get("deployment") == expected["deployment"],
            ),
            (
                "emits exact deterministic document",
                result == expected and repeated == expected,
            ),
            (
                "generalizes to alternate inputs",
                alternate_result == alternate_expected,
            ),
            (
                "does not mutate inputs",
                inputs_unchanged and alternate_unchanged,
            ),
            ("rejects malformed digest", malformed_rejected),
        ]
    except Exception as error:
        failure = f"candidate execution failed: {type(error).__name__}: {error}"
        print(
            json.dumps(
                {"passed": 0, "total": total, "score": 0.0, "failures": [failure]},
                sort_keys=True,
            )
        )
        return 1

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
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
