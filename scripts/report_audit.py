#!/usr/bin/env python3
"""Render the claim audit as a review-ordered report. No network, no spend.

    uv run python scripts/report_audit.py > logs/claim_audit.md

Confidence-gated: a claim is REPORTED only when both criteria variants agree, because
the terse/full comparison measured 61% agreement on `self_supported` -- a single
variant's verdict on the subtler questions is not stable enough to act on.
"""

from __future__ import annotations

import json
import statistics as stats
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
claims = json.loads((HERE / "logs" / "claims_sidecar.json").read_text())["claims"]


def load(v: str) -> dict:
    raw = json.loads((HERE / "logs" / f"jev_audit_raw_{v}.json").read_text())
    out: dict[str, dict] = {}
    for r in raw["requests"].values():
        out.update(r["answers"])
    return out, raw


T, rawT = load("terse")
F, rawF = load("full")
ids = list(claims)


def n(d: dict, q: str, c: str):
    a = d.get(f"{q}_{c}")
    return a["noul"] if a else None


def agreed(q: str, thr: float):
    """Claims where BOTH variants clear the threshold, strongest first."""
    out = []
    for c in ids:
        t, f = n(T, q, c), n(F, q, c)
        if t is not None and f is not None and t >= thr and f >= thr:
            out.append((min(t, f), t, f, c))
    return sorted(out, reverse=True)


def agreed_low(q: str, thr: float):
    out = []
    for c in ids:
        t, f = n(T, q, c), n(F, q, c)
        if t is not None and f is not None and t <= thr and f <= thr:
            out.append((max(t, f), t, f, c))
    return sorted(out)


def cite(c: str) -> str:
    k = claims[c]
    return f"[{k['file']}:{k['line']}]({k['file']}#L{k['line']})"


print("# Claim audit — research record\n")
tot_in = sum(r["input_tokens"] for r in list(rawT["requests"].values())
             + list(rawF["requests"].values()))
lat = [r["latency_s"] for r in list(rawT["requests"].values())
       + list(rawF["requests"].values())]
print(f"Model `{list(rawT['requests'].values())[0]['model']}`, "
      f"{len(claims)} claims, both criteria variants. "
      f"{tot_in:,} input tokens, **${tot_in / 1e9 * 42:.4f}** total, "
      f"{sum(lat):.1f}s of API time.\n")
print("Every finding below is agreed by BOTH variants. Single-variant verdicts are "
      "withheld: the terse/full comparison measured only 61% agreement on "
      "`self_supported`, so one variant alone is not stable enough to act on.\n")

# ---------------------------------------------------------------- retired
r = agreed("retired", 0.8)
missed = [x for x in r if not claims[x[3]]["self_marked_stale"]]
print(f"## Retired claims the section markers missed ({len(missed)})\n")
print("Claims that disown a finding in their own text but sit in a section carrying no "
      "`Superseded` / `wrong` marker. These are the tool's actual yield.\n")
for _lo, t, f, c in missed:
    print(f"- **{cite(c)}** — terse {t:.2f}, full {f:.2f}\n")
    print(f"  > {claims[c]['text'][:260]}\n")

# ---------------------------------------------------------------- unsupported
s = agreed_low("self_supported", 0.2)
print(f"\n## Conclusions asserted without their evidence ({len(s)})\n")
print("Low `self_supported` means the claim states a conclusion whose evidence sits "
      "elsewhere. Not an error — a pointer to prose that cannot be checked where it "
      "stands, which is the `README.md:1208` failure mode.\n")
for _hi, t, f, c in s[:15]:
    print(f"- {cite(c)} — terse {t:.2f}, full {f:.2f} — "
          f"{claims[c]['text'][:150]}")

# ---------------------------------------------------------------- routing
print("\n\n## Numeric claims routed to an artifact, for the Phase 3 comparator\n")
print("| artifact | claims | mean confidence |")
print("| --- | --- | --- |")
buckets: dict[str, list] = {}
for c in ids:
    a = T[f"artifact_{c}"]
    if F[f"artifact_{c}"]["choice"] != a["choice"]:
        continue  # variants disagree on where to look; withheld
    if n(T, "measured", c) < 0.5:
        continue
    buckets.setdefault(a["choice"], []).append(a["confidence"])
for name, confs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
    print(f"| `{name}` | {len(confs)} | {stats.mean(confs):.2f} |")
print(f"\n{sum(len(v) for k, v in buckets.items() if k != 'not_checkable')} claims "
      f"have an agreed artifact and an asserted measurement — these are what Phase 3's "
      f"numeric comparator would check.")
