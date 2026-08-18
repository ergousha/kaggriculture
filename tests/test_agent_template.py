"""`scripts/build_route_agent.py` must emit exactly the agent we ship.

The template is the only implementation of the agent's runtime layers;
`encode_submission.py` reaches through `mining.common` to the same string. If it drifts
from `main.py`, the next route swap silently ships a *different* agent than the one that
was measured -- and the drift is invisible, because the repo's `ruff format --check` only
ever sees the hand-formatted `main.py` and never a freshly generated file.

PR #35 introduced exactly that: the template lost the two blank lines PEP8 wants before a
top-level `def`, so every emitted agent failed `ruff format --check` while CI stayed green.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
MAIN = os.path.join(PROJECT_ROOT, "main.py")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))


def _regenerate() -> str:
    """Render AGENT_TEMPLATE with main.py's own route, version and provenance."""
    from build_route_agent import AGENT_TEMPLATE

    with open(MAIN) as f:
        src = f.read()

    parts = re.search(r"_ROUTE_B85_PARTS = \[\n(.*?)\]\n", src, re.S)
    provenance = re.search(r"Route provenance:\n((?:  .*\n)+)", src)
    headline = re.search(r'^"""Route-replay agent(.*)$', src, re.M)
    version = re.search(r'^AGENT_VERSION = "(.*)"$', src, re.M)
    steps = re.search(r"^  steps: (\d+)$", src, re.M)
    assert parts and provenance and headline and version and steps

    return AGENT_TEMPLATE.format(
        provenance_line=headline.group(1),
        provenance_block="\n" + provenance.group(1).rstrip("\n"),
        n_steps=int(steps.group(1)),
        route_parts=parts.group(1),
        version=version.group(1),
    )


def test_template_regenerates_main_py_byte_for_byte() -> None:
    with open(MAIN) as f:
        assert _regenerate() == f.read()


def test_generated_agent_is_ruff_clean(tmp_path) -> None:
    generated = tmp_path / "route_agent.py"
    generated.write_text(_regenerate())
    for argv in (["format", "--check"], ["check"]):
        result = subprocess.run(
            [sys.executable, "-m", "ruff", *argv, str(generated)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
