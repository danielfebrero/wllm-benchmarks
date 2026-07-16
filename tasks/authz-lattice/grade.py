#!/usr/bin/env python3
"""Externally grade multi-hop authorization lattice decisions."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Callable


def load_candidate(workspace: Path) -> Callable[..., dict[str, Any]]:
    path = workspace / "src" / "authorize.py"
    spec = importlib.util.spec_from_file_location("candidate_authorize", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load src/authorize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.authorize


def resource_matches(pattern: str, resource: str) -> bool:
    if pattern == resource:
        return True
    if pattern.endswith("/*"):
        prefix = pattern[:-1]  # keep trailing slash
        if not resource.startswith(prefix):
            return False
        rest = resource[len(prefix) :]
        return bool(rest) and not rest.startswith("/")
    return False


def specificity(pattern: str) -> int:
    if pattern.endswith("/*"):
        body = pattern[:-2]
    else:
        body = pattern
    parts = [part for part in body.split("/") if part and part != "*"]
    return len(parts)


def role_closure(
    start_roles: Iterable[str], roles: Mapping[str, Mapping[str, Any]]
) -> set[str]:
    seen: set[str] = set()
    stack = list(start_roles)
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        if name in visiting:
            raise ValueError(f"role inheritance cycle at {name}")
        if name not in roles:
            raise ValueError(f"unknown role: {name}")
        visiting.add(name)
        for parent in roles[name].get("parents") or []:
            visit(str(parent))
        visiting.remove(name)
        seen.add(name)

    for role_name in stack:
        visit(str(role_name))
    return seen


def active_grant(grant: Mapping[str, Any], now: str) -> bool:
    return str(grant["not_before"]) <= now < str(grant["not_after"])


def reference(
    subject: str,
    action: str,
    resource: str,
    *,
    roles: Mapping[str, Mapping[str, Any]],
    grants: Iterable[Mapping[str, Any]],
    bindings: Mapping[str, Iterable[str]],
    now: str,
) -> dict[str, Any]:
    effective = role_closure(bindings.get(subject, []), roles)
    # (specificity, deny_first, label, effect) — sort key picks the winner.
    candidates: list[tuple[int, int, str, str]] = []

    for role_name in sorted(effective):
        for permission in roles[role_name].get("permissions") or []:
            if permission.get("action") != action:
                continue
            pattern = str(permission.get("resource"))
            if resource_matches(pattern, resource):
                candidates.append(
                    (specificity(pattern), 0, f"role:{role_name}", "allow")
                )

    for grant in grants:
        if grant.get("subject") != subject:
            continue
        if not active_grant(grant, now):
            continue
        if grant.get("action") != action:
            continue
        pattern = str(grant.get("resource"))
        if not resource_matches(pattern, resource):
            continue
        effect = str(grant.get("effect"))
        if effect not in ("allow", "deny"):
            continue
        label = f"grant:{grant.get('id')}"
        deny_first = 1 if effect == "deny" else 0
        candidates.append((specificity(pattern), deny_first, label, effect))

    if not candidates:
        return {
            "schema_version": "2.0",
            "allowed": False,
            "subject": subject,
            "action": action,
            "resource": resource,
            "matched": None,
            "reason": "default-deny",
        }

    # Highest specificity, then deny over allow, then smallest label.
    best_score = max(score for score, _, _, _ in candidates)
    at_score = [c for c in candidates if c[0] == best_score]
    best_deny = max(deny for _, deny, _, _ in at_score)
    at_effect = [c for c in at_score if c[1] == best_deny]
    winner = min(at_effect, key=lambda item: item[2])
    _score, _deny, label, effect = winner
    if effect == "deny":
        return {
            "schema_version": "2.0",
            "allowed": False,
            "subject": subject,
            "action": action,
            "resource": resource,
            "matched": label,
            "reason": "deny-override",
        }
    return {
        "schema_version": "2.0",
        "allowed": True,
        "subject": subject,
        "action": action,
        "resource": resource,
        "matched": label,
        "reason": "allow",
    }


def live_policy() -> tuple[
    dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, list[str]], str
]:
    roles = {
        "employee": {
            "parents": [],
            "permissions": [{"action": "read", "resource": "/v2/*"}],
        },
        "support": {
            "parents": ["employee"],
            "permissions": [
                {"action": "read", "resource": "/v2/tenants/*"},
                {"action": "write", "resource": "/v2/tenants/*"},
            ],
        },
        "support-lead": {
            "parents": ["support"],
            "permissions": [
                {"action": "read", "resource": "/v2/tenants/*"},
            ],
        },
        "contractor": {
            "parents": [],
            "permissions": [
                {"action": "read", "resource": "/v2/tenants/*"},
                {"action": "PATCH", "resource": "/v2/tenants/*"},
            ],
        },
        "billing-admin": {
            "parents": ["employee"],
            "permissions": [
                {"action": "PATCH", "resource": "/v2/tenants/*"},
            ],
        },
    }
    grants = [
        {
            "id": "g-deny-contractor-billing",
            "subject": "c-ortega",
            "action": "PATCH",
            "resource": "/v2/tenants/*",
            "effect": "deny",
            "not_before": "2026-01-01T00:00:00Z",
            "not_after": "2027-01-01T00:00:00Z",
        },
        {
            "id": "g-temp-support-patch",
            "subject": "s-nguyen",
            "action": "PATCH",
            "resource": "/v2/tenants/acme/billing",
            "effect": "allow",
            "not_before": "2026-07-14T00:00:00Z",
            "not_after": "2026-07-15T00:00:00Z",
        },
        {
            "id": "g-expired-intern",
            "subject": "intern-lee",
            "action": "read",
            "resource": "/v2/tenants/acme/billing",
            "effect": "allow",
            "not_before": "2025-01-01T00:00:00Z",
            "not_after": "2025-06-01T00:00:00Z",
        },
        {
            "id": "g-broad-deny-noise",
            "subject": "s-nguyen",
            "action": "PATCH",
            "resource": "/v2/*",
            "effect": "deny",
            "not_before": "2026-01-01T00:00:00Z",
            "not_after": "2027-01-01T00:00:00Z",
        },
    ]
    bindings = {
        "c-ortega": ["contractor"],
        "s-nguyen": ["support-lead"],
        "b-okonkwo": ["billing-admin"],
        "intern-lee": ["employee"],
    }
    now = "2026-07-14T12:00:00Z"
    return roles, grants, bindings, now


def invoke(
    candidate: Callable[..., dict[str, Any]],
    subject: str,
    action: str,
    resource: str,
    *,
    roles: dict[str, dict[str, Any]],
    grants: list[dict[str, Any]],
    bindings: dict[str, list[str]],
    now: str,
) -> tuple[Any, bool]:
    before = copy.deepcopy((roles, grants, bindings))
    result = candidate(
        subject,
        action,
        resource,
        roles=roles,
        grants=grants,
        bindings=bindings,
        now=now,
    )
    return result, before == (roles, grants, bindings)


def preflight() -> dict[str, Any]:
    roles, grants, bindings, now = live_policy()
    cases = [
        ("c-ortega", "PATCH", "/v2/tenants/acme/billing"),
        ("s-nguyen", "PATCH", "/v2/tenants/acme/billing"),
        ("intern-lee", "read", "/v2/tenants/acme/billing"),
        ("b-okonkwo", "PATCH", "/v2/tenants/acme/billing"),
    ]
    docs = [
        reference(s, a, r, roles=roles, grants=grants, bindings=bindings, now=now)
        for s, a, r in cases
    ]
    if docs[0]["allowed"] is not False or docs[0]["reason"] != "deny-override":
        raise RuntimeError(f"contractor case broken: {docs[0]}")
    if docs[1]["allowed"] is not True or docs[1]["matched"] != "grant:g-temp-support-patch":
        raise RuntimeError(f"support-lead case broken: {docs[1]}")
    if docs[2]["allowed"] is not True or docs[2]["reason"] != "allow":
        # intern has employee read /v2/* which matches billing path
        raise RuntimeError(f"intern case broken: {docs[2]}")
    if docs[3]["allowed"] is not True:
        raise RuntimeError(f"billing-admin case broken: {docs[3]}")
    encoded = json.dumps(docs, sort_keys=True, separators=(",", ":"))
    return {"deterministic": True, "output_bytes": len(encoded.encode("utf-8"))}


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        print(json.dumps(preflight(), sort_keys=True))
        return 0
    if len(sys.argv) != 2:
        raise SystemExit("usage: grade.py WORKSPACE | grade.py --self-test")

    preflight()
    workspace = Path(sys.argv[1]).resolve()
    total = 12
    try:
        candidate = load_candidate(workspace)
        roles, grants, bindings, now = live_policy()

        def decide(subject: str, action: str, resource: str) -> tuple[Any, bool]:
            return invoke(
                candidate,
                subject,
                action,
                resource,
                roles=copy.deepcopy(roles),
                grants=copy.deepcopy(grants),
                bindings=copy.deepcopy(bindings),
                now=now,
            )

        contractor, c_ok = decide("c-ortega", "PATCH", "/v2/tenants/acme/billing")
        support, s_ok = decide("s-nguyen", "PATCH", "/v2/tenants/acme/billing")
        intern, i_ok = decide("intern-lee", "read", "/v2/tenants/acme/billing")
        admin, a_ok = decide("b-okonkwo", "PATCH", "/v2/tenants/acme/billing")
        ghost, g_ok = decide("ghost", "read", "/v2/tenants/acme")

        expected_contractor = reference(
            "c-ortega",
            "PATCH",
            "/v2/tenants/acme/billing",
            roles=roles,
            grants=grants,
            bindings=bindings,
            now=now,
        )
        expected_support = reference(
            "s-nguyen",
            "PATCH",
            "/v2/tenants/acme/billing",
            roles=roles,
            grants=grants,
            bindings=bindings,
            now=now,
        )
        expected_admin = reference(
            "b-okonkwo",
            "PATCH",
            "/v2/tenants/acme/billing",
            roles=roles,
            grants=grants,
            bindings=bindings,
            now=now,
        )

        expired_ok = False
        expired_roles = {
            "temp": {
                "parents": [],
                "permissions": [],
            }
        }
        expired_grants = [
            {
                "id": "old",
                "subject": "x",
                "action": "read",
                "resource": "/a",
                "effect": "allow",
                "not_before": "2020-01-01T00:00:00Z",
                "not_after": "2020-02-01T00:00:00Z",
            }
        ]
        expired_decision, e_ok = invoke(
            candidate,
            "x",
            "read",
            "/a",
            roles=expired_roles,
            grants=expired_grants,
            bindings={"x": ["temp"]},
            now="2026-07-14T12:00:00Z",
        )
        expired_ok = (
            isinstance(expired_decision, dict)
            and expired_decision.get("allowed") is False
            and expired_decision.get("reason") == "default-deny"
        )

        cycle_rejected = False
        try:
            candidate(
                "loop",
                "read",
                "/x",
                roles={
                    "a": {"parents": ["b"], "permissions": []},
                    "b": {"parents": ["a"], "permissions": []},
                },
                grants=[],
                bindings={"loop": ["a"]},
                now=now,
            )
        except ValueError:
            cycle_rejected = True

        checks: list[tuple[str, bool]] = [
            ("returns mappings", all(isinstance(x, dict) for x in (contractor, support, admin))),
            (
                "uses authorization schema 2.0",
                isinstance(contractor, dict) and contractor.get("schema_version") == "2.0",
            ),
            (
                "deny overrides contractor role allow",
                contractor == expected_contractor,
            ),
            (
                "specific allow beats broader deny",
                support == expected_support,
            ),
            (
                "role inheritance supplies read allow",
                isinstance(intern, dict)
                and intern.get("allowed") is True
                and intern.get("reason") == "allow",
            ),
            (
                "billing admin inherited patch allow",
                admin == expected_admin,
            ),
            (
                "default deny for unknown subject",
                isinstance(ghost, dict)
                and ghost.get("allowed") is False
                and ghost.get("reason") == "default-deny",
            ),
            ("ignores expired grants", expired_ok),
            (
                "wildcard prefix matching",
                isinstance(admin, dict) and admin.get("allowed") is True,
            ),
            ("does not mutate inputs", all((c_ok, s_ok, i_ok, a_ok, g_ok, e_ok))),
            ("rejects role cycles", cycle_rejected),
            (
                "matched labels are stable",
                isinstance(contractor, dict)
                and contractor.get("matched") == "grant:g-deny-contractor-billing"
                and isinstance(support, dict)
                and support.get("matched") == "grant:g-temp-support-patch",
            ),
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
