#!/usr/bin/env python3
"""TypeSafe/Jev client for THIS REPO'S RESEARCH TOOLING ONLY.

Loads `typesafe_credentials.py` and exports TYPESAFE_API_KEY to the environment
before `typesafe_sdk` is imported, mirroring how submit.py handles Kaggle
credentials. The key is never printed or logged.

NOT REACHABLE FROM THE SUBMISSION. The competition scores episodes in a sandbox with
no network and a 1.0s/turn actTimeout; main.py is stdlib-only and must stay that way.
tests/test_jev_isolation.py enforces that nothing main.py can reach imports this.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

_MISSING = """ERROR: typesafe_credentials.py not found.

  cp typesafe_credentials.example.py typesafe_credentials.py
  # then edit it and paste in your key from https://console.typesafe.ai/keys

It is already in .gitignore. Do not commit it and do not paste it into a
published notebook: it bills your TypeSafe account."""


def load_key() -> str:
    """Export the key to os.environ and return the pinned model id."""
    try:
        sys.path.insert(0, str(HERE))
        import typesafe_credentials as creds  # noqa: WPS433
    except ImportError as exc:
        print(_MISSING, file=sys.stderr)
        raise SystemExit(2) from exc

    key = (getattr(creds, "TYPESAFE_API_KEY", "") or "").strip()
    if not key:
        print("ERROR: typesafe_credentials.py has no TYPESAFE_API_KEY.", file=sys.stderr)
        raise SystemExit(2)
    os.environ["TYPESAFE_API_KEY"] = key

    base = (getattr(creds, "TYPESAFE_BASE_URL", "") or "").strip()
    if base:
        os.environ["TYPESAFE_BASE_URL"] = base
    return (getattr(creds, "TYPESAFE_DEFAULT_MODEL", "") or "jev-latest").strip()


def client(timeout: float = 120.0):
    """A configured TypeSafeClient plus the model id to send."""
    model = load_key()
    from typesafe_sdk import TypeSafeClient  # noqa: WPS433  (after the key is exported)

    return TypeSafeClient(model=model, timeout=timeout), model
