#!/usr/bin/env python3
"""Create a deterministic, mixed-artifact incident workspace."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SOURCE = '''"""Webhook signature verification."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping


def _digest(payload: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def verify_signature(
    payload: bytes, signature_header: str, environment: Mapping[str, str]
) -> bool:
    """Return whether a v1 webhook signature is valid."""
    try:
        version, presented = signature_header.split("=", 1)
    except ValueError:
        return False
    if version != "v1":
        return False

    # Kept during the rotation window so in-flight deliveries still validate.
    secret = environment.get("WEBHOOK_SECRET_PREVIOUS")
    if not secret:
        return False
    expected = _digest(payload, secret)
    return expected == presented
'''


PUBLIC_TEST = '''from __future__ import annotations

import hashlib
import hmac
import unittest

from src.webhook_auth import verify_signature


def sign(payload: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"v1={digest}"


class WebhookAuthTests(unittest.TestCase):
    def test_previous_secret_remains_valid_during_overlap(self) -> None:
        payload = b'{"event":"invoice.paid"}'
        environment = {
            "WEBHOOK_SECRET_CURRENT": "new-secret",
            "WEBHOOK_SECRET_PREVIOUS": "old-secret",
        }
        self.assertTrue(verify_signature(payload, sign(payload, "old-secret"), environment))

    def test_tampered_payload_is_rejected(self) -> None:
        environment = {
            "WEBHOOK_SECRET_CURRENT": "new-secret",
            "WEBHOOK_SECRET_PREVIOUS": "old-secret",
        }
        self.assertFalse(verify_signature(b"tampered", sign(b"original", "old-secret"), environment))


if __name__ == "__main__":
    unittest.main()
'''


RUNBOOK = '''# Webhook signing-secret rotation

Status: authoritative
Owner: Payments Platform

The receiver validates the `X-Webhook-Signature` header as `v1=<64 lowercase
hex characters>`, using HMAC-SHA256 over the exact request bytes.

During the 24-hour rotation overlap, accept a signature produced by either
`WEBHOOK_SECRET_CURRENT` or `WEBHOOK_SECRET_PREVIOUS`. The current secret is
mandatory in production; the previous secret is optional and must be removed
after the overlap. Compare candidate digests in constant time. Unknown header
versions and malformed values are authentication failures, not exceptions.

Rotation order:

1. Move the old value to `WEBHOOK_SECRET_PREVIOUS`.
2. Install the new value as `WEBHOOK_SECRET_CURRENT`.
3. Deploy receivers, then rotate senders.
4. Remove the previous value after 24 hours.
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
    write(root / "src" / "webhook_auth.py", SOURCE)
    write(root / "tests" / "test_webhook_auth.py", PUBLIC_TEST)
    write(root / "docs" / "runbooks" / "webhook-secret-rotation.md", RUNBOOK)
    write(
        root / "evidence" / "live" / "receiver.log",
        "2026-07-14T09:31:04Z level=ERROR incident=INC-7429 route=/hooks/payments "
        "status=401 reason=signature_mismatch configured_key=WEBHOOK_SECRET_CURRENT\n",
    )
    write(
        root / "config" / "production.json",
        json.dumps(
            {
                "service": "webhook-receiver",
                "incident": "INC-7429",
                "secret_variables": [
                    "WEBHOOK_SECRET_CURRENT",
                    "WEBHOOK_SECRET_PREVIOUS",
                ],
                "rotation_overlap_hours": 24,
            },
            indent=2,
        )
        + "\n",
    )
    write(
        root / "README.md",
        "# Receiver incident workspace\n\nRun tests with "
        "`python3 -m unittest discover -s tests -v`. Operational truth lives "
        "under `docs/runbooks`; `archive/` is historical reference only.\n",
    )
    write(
        root / "wllm.toml",
        "[budgets]\ndefault_text = 2500\n\n"
        "[policy]\nexternal_extractors = false\nnetwork = false\n",
    )

    # Real workspaces contain years of superficially similar incidents. These
    # deterministic records add ranking/search noise without adding a large
    # binary fixture to the repository.
    for number in range(240):
        year = 2021 + number % 5
        digest = f"legacy-{number:04d}"
        content = (
            f"# Archived webhook incident {number:04d}\n\n"
            "Status: closed historical reference\n\n"
            f"In {year}, a webhook signature mismatch followed a rotation. "
            f"The retired deployment used `WEBHOOK_SECRET_{digest.upper().replace('-', '_')}`.\n"
            "This record is not an active runbook and its variable names must "
            "not be copied into production code.\n"
        )
        write(root / "archive" / str(year) / f"incident-{number:04d}.md", content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
