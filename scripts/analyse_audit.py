#!/usr/bin/env python3
"""Analyse the cached Jev audit. No network, no spend -- reads logs/jev_audit_raw_*.json.

    uv run python scripts/analyse_audit.py

Answers the questions the proposal deferred to measurement:
  1. terse vs full criteria -- do they agree, and where do they diverge?
  2. `measured` noul vs the corpus's own code-derived numeric/conclusion label
  3. `retired` noul vs the corpus builder's regex+section stale flag -- the value-add
  4. artifact routing: distribution and the not_checkable share
  5. `self_supported` on conclusion claims
"""

from __future__ import annotations

import json
import statistics as stats
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
claims = json.loads((HERE / "logs" / "claims_sidecar.json").read_text())["claims"]


def load(variant: str) -> dict:
    raw = json.loads((HERE / "logs" / f"jev_audit_raw_{variant}.json").read_text())
    flat: dict[str, dict] = {}
    for req in raw["requests"].values():
        flat.update(req["answers"])
    return flat


def noul(ans: dict, key: str) -> float | None:
    """The noul probability, or None when the question was not asked for that claim."""
    a = ans.get(key)
    return a["noul"] if a else None


def nz(ans: dict, key: str) -> float:
    """Same, for call sites that have already established the question exists.

    Arithmetic on `float | None` is what made this file 26 of the 44 mypy errors that
    surfaced once scripts/ was added to the type check; keeping the two accessors
    separate is the fix rather than sprinkling asserts at each use.
    """
    v = noul(ans, key)
    if v is None:
        raise KeyError(key)
    return v


def rule(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


T, F = load("terse"), load("full")
ids = list(claims)

# ---------------------------------------------------------------- 1. terse vs full
rule("1. terse vs full criteria -- the question the proposal deferred")
for q in ("measured", "retired", "self_supported"):
    pairs: list[tuple[float, float]] = []
    for c in ids:
        t_val, f_val = noul(T, f"{q}_{c}"), noul(F, f"{q}_{c}")
        if t_val is not None and f_val is not None:
            pairs.append((t_val, f_val))
    if not pairs:
        continue
    ta = [p[0] for p in pairs]
    fa = [p[1] for p in pairs]
    mad = stats.mean(abs(a - b) for a, b in pairs)
    # agreement at the cookbook's own suggested band
    agree = sum((a >= 0.5) == (b >= 0.5) for a, b in pairs) / len(pairs)
    flip = [p for p in pairs if (p[0] >= 0.5) != (p[1] >= 0.5)]
    print(f"\n{q:<16} n={len(pairs)}")
    print(f"  mean noul       terse {stats.mean(ta):.3f}   full {stats.mean(fa):.3f}")
    print(f"  mean abs diff   {mad:.3f}")
    print(f"  agree at 0.5    {agree * 100:.1f}%   ({len(flip)} claims flip side)")

ca = [(T[f"artifact_{c}"]["choice"], F[f"artifact_{c}"]["choice"]) for c in ids]
print(f"\nartifact         n={len(ca)}")
print(f"  same top pick   {sum(a == b for a, b in ca) / len(ca) * 100:.1f}%")

# ------------------------------------------------- 2. measured vs the code-derived label
rule("2. `measured` noul vs the corpus's own numeric/conclusion label")
for kind in ("numeric", "conclusion"):
    measured = [nz(T, f"measured_{c}") for c in ids if claims[c]["kind"] == kind]
    print(
        f"  kind={kind:<11} n={len(measured):>3}  mean noul {stats.mean(measured):.3f}  "
        f"median {stats.median(measured):.3f}  >=0.5: "
        f"{sum(v >= 0.5 for v in measured) / len(measured) * 100:.0f}%"
    )
print("\n  numeric claims Jev says are NOT measured (noul < 0.2) -- masked values but")
print("  no measured result asserted:")
low = sorted(
    ((nz(T, f"measured_{c}"), c) for c in ids if claims[c]["kind"] == "numeric"),
    key=lambda x: x[0],
)[:5]
for v, c in low:
    print(f"    {c} {v:.2f}  {claims[c]['file']}:{claims[c]['line']}")
    print(f"           {claims[c]['text'][:105]}")

# --------------------------------------------------------- 3. retired: the value-add
rule("3. `retired` noul vs the corpus builder's regex/section stale flag")
flagged = [c for c in ids if claims[c]["self_marked_stale"]]
unflagged = [c for c in ids if not claims[c]["self_marked_stale"]]
print(
    f"  regex-flagged   n={len(flagged):>3}  mean noul "
    f"{stats.mean(nz(T, f'retired_{c}') for c in flagged):.3f}"
)
print(
    f"  not flagged     n={len(unflagged):>3}  mean noul "
    f"{stats.mean(nz(T, f'retired_{c}') for c in unflagged):.3f}"
)
for thr in (0.9, 0.8, 0.5):
    hits = [c for c in unflagged if nz(T, f"retired_{c}") >= thr]
    print(f"  unflagged claims with retired >= {thr}: {len(hits)}")
print("\n  TOP CANDIDATES the regex missed (both variants agree >= 0.8):")
cands = sorted(
    (c for c in unflagged if nz(T, f"retired_{c}") >= 0.8 and nz(F, f"retired_{c}") >= 0.8),
    key=lambda c: -nz(T, f"retired_{c}"),
)
for c in cands[:12]:
    k = claims[c]
    print(
        f"    {c} T={nz(T, f'retired_{c}'):.2f} F={nz(F, f'retired_{c}'):.2f}  "
        f"{k['file']}:{k['line']}"
    )
    print(f"           {k['text'][:110]}")
print(f"\n  total agreed candidates: {len(cands)}")

# ------------------------------------------------------------- 4. artifact routing
rule("4. artifact routing")
picks = Counter(T[f"artifact_{c}"]["choice"] for c in ids)
for name, n in picks.most_common():
    conf = stats.mean(
        T[f"artifact_{c}"]["confidence"] for c in ids if T[f"artifact_{c}"]["choice"] == name
    )
    print(f"  {name:<24} {n:>4}  ({n / len(ids) * 100:>4.1f}%)  mean confidence {conf:.2f}")
nc = [c for c in ids if T[f"artifact_{c}"]["choice"] == "not_checkable"]
print(
    "\n  not_checkable share by claim kind: "
    + ", ".join(
        f"{k}={sum(1 for c in nc if claims[c]['kind'] == k)}/"
        f"{sum(1 for c in ids if claims[c]['kind'] == k)}"
        for k in ("numeric", "conclusion")
    )
)

# ----------------------------------------------------------- 5. self_supported
rule("5. `self_supported` on conclusion claims")
sup: list[tuple[float, str]] = [
    (nz(T, f"self_supported_{c}"), c) for c in ids if noul(T, f"self_supported_{c}") is not None
]
print(
    f"  n={len(sup)}  mean {stats.mean(v for v, _ in sup):.3f}  "
    f"median {stats.median(v for v, _ in sup):.3f}"
)
print(f"  < 0.2 (conclusion with no evidence in the claim): {sum(v < 0.2 for v, _ in sup)}")
print("\n  weakest -- conclusions asserted without their evidence:")
for v, c in sorted(sup)[:6]:
    print(f"    {c} {v:.2f}  {claims[c]['file']}:{claims[c]['line']}")
    print(f"           {claims[c]['text'][:105]}")
