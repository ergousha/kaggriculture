"""The Jev research tooling must never be reachable from the submission.

Kaggle scores episodes in a sandbox with no network and a 1.0s/turn `actTimeout` over
719 steps, so a hosted API call cannot be part of an agent at all. `main.py` is
standard-library only and `submit.py` pre-flight already hard-fails on a disallowed
import in it; these tests extend the same guarantee to the TypeSafe tooling, so the
research side and the submission side cannot be confused later.

They also assert the credential file stays gitignored: it bills a real account.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = {"typesafe_sdk", "typesafe_credentials", "jev_client", "httpx", "httpx2",
             "requests", "urllib.request"}

# Everything the submission is allowed to reach: main.py plus the versioned snapshots
# the arena plays it against.
SUBMISSION_FILES = [ROOT / "main.py"] + sorted((ROOT / "opponents").glob("v*.py"))


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_submission_does_not_import_jev_tooling() -> None:
    for path in SUBMISSION_FILES:
        mods = _imports(path)
        leaked = {m for m in mods if m.split(".")[0] in FORBIDDEN or m in FORBIDDEN}
        assert not leaked, f"{path.name} imports research-only modules: {leaked}"


def test_main_has_no_network_capable_import() -> None:
    mods = _imports(ROOT / "main.py")
    net = {m for m in mods
           if m.split(".")[0] in {"socket", "http", "urllib", "ssl", "asyncio"}}
    assert not net, f"main.py imports network-capable modules: {net}"


def test_credentials_file_is_gitignored() -> None:
    r = subprocess.run(
        ["git", "check-ignore", "typesafe_credentials.py"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, (
        "typesafe_credentials.py is NOT gitignored — it bills a real TypeSafe account"
    )


def test_example_credentials_carry_no_real_key() -> None:
    """The committed template must never carry a real key.

    This test has already earned its place: the key was first pasted into the
    *example* file rather than the gitignored one, which would have committed a live
    credential. Parsed with ast, not string splitting, so a trailing comment on the
    assignment is not mistaken for the value.
    """
    path = ROOT / "typesafe_credentials.example.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if "TYPESAFE_API_KEY" not in names:
            continue
        assert isinstance(node.value, ast.Constant), "key must be a literal empty string"
        assert node.value.value == "", (
            "the committed example contains a real key — move it to "
            "typesafe_credentials.py (gitignored) and rotate the exposed one"
        )
