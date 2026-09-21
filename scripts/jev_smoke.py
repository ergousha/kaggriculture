#!/usr/bin/env python3
"""One live Jev call. Confirms auth, records which model answered, prints usage.

    uv run python scripts/jev_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_client import client  # noqa: E402

cl, model = client()
from typesafe_sdk import Choice, Noul, NoulCriteria  # noqa: E402

# A claim from this repo's own notes, masked exactly as the corpus masks it.
STATE = {
    "what_this_is": "One masked claim from a Kaggle agent's research notes.",
    "claims": "C9999| The shipped route beat its one evaluation opponent <PCT:0> of "
              "the time offline and wins <PCT:1> live.",
}

with cl as c:
    r = c.system_one(
        STATE,
        {
            "measured_C9999": Noul(
                instructions="Claim C9999 asserts a specific measured result about "
                             "this agent's performance or about the game engine.",
                criteria=NoulCriteria(
                    true="States a result, quantity, rate or outcome that was measured, "
                         "including where the value is masked as a <TYPE:index> "
                         "placeholder",
                    false="States a plan, rationale, open question or tooling "
                          "description, with no measured result",
                ),
            ),
            "retired_C9999": Noul(
                instructions="The text of claim C9999 says the finding it describes has "
                             "been superseded, was wrong, or has been contradicted.",
            ),
            "artifact_C9999": Choice(
                instructions="Which kind of record would settle whether claim C9999 is "
                             "still accurate?",
                criteria={
                    "offline_match_results": None,
                    "live_leaderboard_results": None,
                    "engine_spec": None,
                    "not_checkable": None,
                },
            ),
        },
        model=model,
    )

print(f"requested model : {model}")
print(f"answered model  : {r.model}")
print(f"usage           : in={r.usage.input_tokens} out={r.usage.output_tokens}")
print()
print(f"measured_C9999  : noul={r.answers['measured_C9999'].noul:.3f}")
print(f"retired_C9999   : noul={r.answers['retired_C9999'].noul:.3f}")
a = r.answers["artifact_C9999"]
print(f"artifact_C9999  : choice={a.choice!r} confidence={a.confidence:.3f}")
for k, v in sorted(a.probabilities.items(), key=lambda kv: -kv[1]):
    print(f"                    {k:<26} {v:.3f}")
