#!/usr/bin/env python3
"""Group live Kaggle scores by RUNTIME HASH and report the spread within each group.

    uv run python scripts/audit_live_scores.py
    uv run python scripts/audit_live_scores.py --offline   # use the cached table

WHY THIS EXISTS. On 2026-09-21 this repo was about to pick an iteration baseline by
comparing `public_score` across versions. That comparison is unsound, and the proof is
in the repo's own submissions:

    opponents/v0_4_1.py and opponents/v0_4_2.py are BYTE-IDENTICAL runtimes
    (same sha256 over everything after AGENT_VERSION), and they scored
    2450.6 and 1816.9 -- 634 points apart.

v0.4.2 was submitted at all only because the release CI keys off AGENT_VERSION, not off
the code: bump the constant, no tag exists, CI submits. So the repo has accidentally run
the most valuable experiment available -- the same agent submitted three times
(v0.4.1 twice at 2300.1 and 2450.6, v0.4.2 once at 1816.9) -- and never scored it.

Until the within-identical-runtime spread is known, NO cross-version score difference
smaller than that spread means anything, and "v0.4.3 regressed by 640" is not a
supported claim. This script makes that spread a standing measurement instead of an
anecdote.

CONFOUND, stated because it is not resolved: `public_score` is a skill rating that
evolves while a submission plays and decays once it stops being the active one. Two
submissions read at different times are not guaranteed to be like-for-like even on
identical code. This script therefore reports the spread as an UPPER BOUND on
reproducibility, not as pure noise, and records the read time so the series can be
re-read later. Resolving decay from variance needs repeated reads of the same
submission over time -- which this script's --append mode accumulates.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CACHE = HERE / "logs" / "live_scores.json"


def runtime_hash(path: Path) -> str | None:
    """sha256 of everything from AGENT_VERSION on, excluding the constant itself.

    Mirrors how the v0.5.0 header verifies a byte-identical runtime: the docstring and
    the version constant are metadata, everything below is the agent.
    """
    if not path.exists():
        return None
    s = path.read_text(encoding="utf-8")
    i = s.find("AGENT_VERSION")
    if i < 0:
        return None
    body = "\n".join(line for line in s[i:].splitlines() if not line.startswith("AGENT_VERSION"))
    return hashlib.sha256(body.encode()).hexdigest()


def fetch_submissions() -> list[dict]:
    sys.path.insert(0, str(HERE))
    import kaggle_credentials as kc  # noqa: WPS433

    token = (getattr(kc, "KAGGLE_API_TOKEN", "") or "").strip()
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
    else:
        # getattr, not attribute access: kaggle_credentials.py is gitignored, so it
        # is absent in CI and its attributes cannot be resolved statically.
        os.environ["KAGGLE_USERNAME"] = str(getattr(kc, "KAGGLE_USERNAME", ""))
        os.environ["KAGGLE_KEY"] = str(getattr(kc, "KAGGLE_KEY", ""))
    from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: WPS433

    api = KaggleApi()
    api.authenticate()
    out = []
    for s in api.competition_submissions("kaggriculture"):
        out.append(
            {
                "date": str(s.date),
                "status": str(s.status).split(".")[-1],
                "score": float(s.public_score) if s.public_score else None,
                "description": s.description or "",
            }
        )
    return out


def version_of(desc: str) -> str | None:
    m = re.search(r"v(\d+\.\d+\.\d+)", desc)
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use logs/live_scores.json")
    ap.add_argument(
        "--append", action="store_true", help="append this read to the cache as a time series"
    )
    args = ap.parse_args()

    if args.offline:
        if not CACHE.exists():
            raise SystemExit("no cache; run without --offline once")
        subs = json.loads(CACHE.read_text())["reads"][-1]["submissions"]
        read_at = json.loads(CACHE.read_text())["reads"][-1]["read_at"]
    else:
        subs = fetch_submissions()
        read_at = dt.datetime.now(dt.UTC).isoformat()
        store = json.loads(CACHE.read_text()) if CACHE.exists() else {"reads": []}
        if args.append or not CACHE.exists():
            store["reads"].append({"read_at": read_at, "submissions": subs})
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(store, indent=2))

    # Map each submitted version to the runtime hash of its committed snapshot.
    groups: dict[str, list[tuple[str, float, str]]] = {}
    unmapped: list[tuple[str, float]] = []
    for s in subs:
        if s["score"] is None:
            continue
        ver = version_of(s["description"])
        if not ver:
            unmapped.append((s["description"][:40], s["score"]))
            continue
        snap = HERE / "opponents" / f"v{ver.replace('.', '_')}.py"
        h = runtime_hash(snap)
        if h is None:
            unmapped.append((f"v{ver} (no snapshot)", s["score"]))
            continue
        groups.setdefault(h, []).append((ver, s["score"], s["date"][:16]))

    print(f"read at {read_at}")
    print(f"scored submissions: {sum(1 for s in subs if s['score'] is not None)}\n")

    print("SCORES GROUPED BY RUNTIME HASH (identical hash = byte-identical agent)")
    print("=" * 74)
    worst_spread = 0.0
    for h, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        vers = sorted({v for v, _, _ in rows})
        scores = [sc for _, sc, _ in rows]
        spread = max(scores) - min(scores) if len(scores) > 1 else 0.0
        worst_spread = max(worst_spread, spread)
        print(f"\nruntime {h[:16]}  versions: {', '.join('v' + v for v in vers)}")
        for v, sc, d in sorted(rows, key=lambda r: r[2]):
            print(f"    {d}  v{v:<7} {sc:>8.1f}")
        if len(scores) > 1:
            print(
                f"    -> n={len(scores)}  mean {statistics.fmean(scores):.1f}  "
                f"SPREAD {spread:.1f} on IDENTICAL code"
            )

    print("\n" + "=" * 74)
    if worst_spread:
        print(f"Largest spread on byte-identical code: {worst_spread:.1f} points.")
        print(f"=> Treat any cross-version score gap below {worst_spread:.0f} as")
        print("   UNRESOLVED, not as a regression or an improvement.")
    else:
        print(
            "No runtime has more than one scored submission yet, so the "
            "reproducibility of `public_score` is still unmeasured."
        )
    if unmapped:
        print(f"\nunmapped submissions (no opponents/ snapshot): {len(unmapped)}")
        for d, sc in unmapped[:6]:
            print(f"    {sc:>8.1f}  {d}")


if __name__ == "__main__":
    main()
