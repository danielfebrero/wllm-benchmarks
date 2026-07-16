#!/usr/bin/env python3
"""Create a multi-hop authorization lattice workspace with realistic decoys."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SOURCE = '''"""Authorize subject actions against a resource lattice."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def authorize(
    subject: str,
    action: str,
    resource: str,
    *,
    roles: Mapping[str, Mapping[str, Any]],
    grants: Iterable[Mapping[str, Any]],
    bindings: Mapping[str, Iterable[str]],
    now: str,
) -> dict[str, Any]:
    """Return an authorization decision document."""
    # Broken production code: only checks direct role permissions, ignores
    # inheritance depth, deny overrides, grant specificity, and time bounds.
    subject_roles = list(bindings.get(subject, []))
    allowed = False
    matched = None
    for role_name in subject_roles:
        role = roles.get(role_name) or {}
        for permission in role.get("permissions", []):
            if permission.get("action") == action and permission.get("resource") == resource:
                allowed = True
                matched = f"role:{role_name}"
                break
        if allowed:
            break
    for grant in grants:
        if grant.get("subject") != subject:
            continue
        if grant.get("action") == action and grant.get("resource") == resource:
            allowed = grant.get("effect", "allow") == "allow"
            matched = f"grant:{grant.get('id')}"
            break
    return {
        "schema_version": "1.0",
        "allowed": allowed,
        "subject": subject,
        "action": action,
        "resource": resource,
        "matched": matched,
        "reason": "legacy-first-match",
    }
'''


PUBLIC_TEST = '''from __future__ import annotations

import unittest

from src.authorize import authorize


class AuthorizePublicTests(unittest.TestCase):
    def test_direct_role_allow(self) -> None:
        roles = {
            "reader": {
                "parents": [],
                "permissions": [{"action": "read", "resource": "/docs"}],
            }
        }
        decision = authorize(
            "alice",
            "read",
            "/docs",
            roles=roles,
            grants=[],
            bindings={"alice": ["reader"]},
            now="2026-07-14T12:00:00Z",
        )
        self.assertTrue(decision["allowed"])
        self.assertEqual(decision["subject"], "alice")

    def test_unknown_subject_denied(self) -> None:
        decision = authorize(
            "ghost",
            "read",
            "/docs",
            roles={},
            grants=[],
            bindings={},
            now="2026-07-14T12:00:00Z",
        )
        self.assertFalse(decision["allowed"])


if __name__ == "__main__":
    unittest.main()
'''


CONTRACT = '''# Authorization lattice contract v2

Status: authoritative
Owner: Identity Platform
Applies to: `src.authorize.authorize`

Return one JSON object with exactly:
`schema_version`, `allowed`, `subject`, `action`, `resource`, `matched`, `reason`.

`schema_version` is always the literal string `2.0`.
`subject`, `action`, and `resource` echo the request.

## Role closure

A subject's effective roles are the transitive closure of `bindings[subject]`
through each role's ordered `parents` list. Detect cycles with `ValueError`.
Permissions of every effective role are candidate allow rules.

## Resource matching

A permission or grant matches a request resource when its `resource` pattern
equals the request resource or is a prefix ending with `/*` that covers the
request path with at least one extra segment. Examples:
- `/v2/tenants/*` matches `/v2/tenants/acme` and `/v2/tenants/acme/billing`
- `/v2/tenants/acme` matches only that exact path
- `/v2/*` matches `/v2/tenants/acme` but is less specific than `/v2/tenants/*`

Specificity score = number of non-wildcard path segments in the pattern
(so `/v2/tenants/acme/billing` > `/v2/tenants/*` > `/v2/*`).

Action matching is exact string equality.

## Grants

Grants are evaluated in addition to role permissions. A grant is active only
when `not_before <= now < not_after` using the supplied ISO-8601 UTC timestamps
as opaque ordered strings (lexicographic compare is valid for the Z format used
in fixtures). Inactive grants are ignored.

Each grant has `effect` of `allow` or `deny`.

## Decision algorithm

1. Collect all matching candidates from effective role permissions (always
   allow-effect) and active grants (allow or deny), retaining each candidate's
   specificity, effect, and a stable match label (`role:<name>` or `grant:<id>`).
2. If no candidates match, deny with `matched=null` and `reason=default-deny`.
3. Otherwise choose the single winning candidate by:
   - highest specificity first;
   - on a specificity tie, deny beats allow;
   - on a further tie, lexicographically smallest match label.
4. Apply the winner's effect.

When a deny wins: `allowed=false`, `matched=<label>`, `reason=deny-override`.
When an allow wins: `allowed=true`, `matched=<label>`, `reason=allow`.
Do not mutate inputs. Missing role definitions referenced by bindings or parents
raise `ValueError`.
'''


README = '''# Security incident AZ-4404

Run public validation with `python3 -m unittest discover -s tests -v`.

Authoritative inputs:
- role catalog under `policy/roles/`
- subject bindings under `policy/bindings/`
- explicit grants under `policy/grants/`
- authorization contract under `docs/contracts/`
- live decision failures under `evidence/live/`

`archive/` holds retired ACL spreadsheets and must not authorize current traffic.
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
    write(root / "src" / "authorize.py", SOURCE)
    write(root / "tests" / "test_authorize.py", PUBLIC_TEST)
    write(root / "docs" / "contracts" / "authorization-lattice-v2.md", CONTRACT)
    write(root / "README.md", README)
    write(
        root / "wllm.toml",
        "[budgets]\ndefault_text = 2500\n\n"
        "[policy]\nexternal_extractors = false\nnetwork = false\n",
    )

    dump(
        root / "policy" / "roles" / "catalog.json",
        {
            "employee": {
                "parents": [],
                "permissions": [
                    {"action": "read", "resource": "/v2/*"},
                ],
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
        },
    )
    dump(
        root / "policy" / "bindings" / "subjects.json",
        {
            "c-ortega": ["contractor"],
            "s-nguyen": ["support-lead"],
            "b-okonkwo": ["billing-admin"],
            "intern-lee": ["employee"],
        },
    )
    dump(
        root / "policy" / "grants" / "active.json",
        [
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
        ],
    )
    dump(
        root / "evidence" / "live" / "decisions.json",
        {
            "now": "2026-07-14T12:00:00Z",
            "failures": [
                {
                    "subject": "c-ortega",
                    "action": "PATCH",
                    "resource": "/v2/tenants/acme/billing",
                    "expected_allowed": False,
                    "expected_reason": "deny-override",
                    "observed_allowed": True,
                },
                {
                    "subject": "s-nguyen",
                    "action": "PATCH",
                    "resource": "/v2/tenants/acme/billing",
                    "expected_allowed": True,
                    "expected_reason": "allow",
                    "observed_allowed": False,
                    "note": "time-bound exact grant must beat broader deny",
                },
            ],
        },
    )
    write(
        root / "evidence" / "live" / "AZ-4404.log",
        "2026-07-14T12:00:08Z level=ERROR incident=AZ-4404 "
        "code=AUTHZ_LATTICE_V2 "
        "detail=contractor_patch_allowed_despite_deny; "
        "support_lead_temp_grant_ignored; schema_version_stale\n",
    )

    for number in range(90):
        year = 2020 + number % 6
        write(
            root / "archive" / str(year) / f"acl-{number:03d}.md",
            f"# Retired ACL sheet {number}\n\n"
            "Status: closed historical record\n\n"
            "First-match spreadsheet ACLs without deny-overrides. "
            "Do not use for authorization lattice v2.\n",
        )
        dump(
            root / "archive" / str(year) / f"binding-{number:03d}.json",
            {
                "status": "retired",
                "subject": f"user-{number}",
                "roles": [f"legacy-role-{number % 7}"],
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
