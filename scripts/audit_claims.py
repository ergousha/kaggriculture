#!/usr/bin/env python3
"""Send the claim corpus to Jev and cache every raw answer. RESEARCH TOOLING ONLY.

    uv run python scripts/audit_claims.py --variant terse --only 2   # one request
    uv run python scripts/audit_claims.py --variant terse            # all requests
    uv run python scripts/audit_claims.py --variant full

Raw probabilities are persisted to logs/jev_audit_raw_<variant>.json and never
overwritten silently, so the analysis in scripts/analyse_audit.py is reproducible
without re-spending. Thresholding happens in the analysis, not here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_client import client  # noqa: E402
from jev_questions import (  # noqa: E402
    CORPUS,
    HERE,
    SIDECAR,
    pack,
    state_for,
    wire_questions,
)


def to_sdk(wire: dict) -> dict:
    """Convert the wire-format question dicts into SDK question objects."""
    from typesafe_sdk import Choice, Noul, NoulCriteria  # noqa: WPS433

    out: dict[str, object] = {}
    for key, q in wire.items():
        if q["type"] == "choice":
            out[key] = Choice(instructions=q["instructions"], criteria=q["criteria"])
        else:
            crit = q.get("criteria")
            out[key] = Noul(
                instructions=q["instructions"],
                criteria=NoulCriteria(true=crit["true"], false=crit["false"])
                if crit else None,
            )
    return out


def serialise(answers: dict) -> dict:
    out: dict[str, dict] = {}
    for key, a in answers.items():
        if hasattr(a, "noul"):
            out[key] = {"type": "noul", "noul": a.noul,
                        "confidence": getattr(a, "confidence", None)}
        else:
            out[key] = {"type": "choice", "choice": a.choice,
                        "confidence": a.confidence,
                        "probabilities": dict(a.probabilities)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["terse", "full"], default="terse")
    ap.add_argument("--only", type=int, help="send just this request index")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    corpus = json.loads(CORPUS.read_text())
    claims = json.loads(SIDECAR.read_text())["claims"]
    all_ids = [cid for ch in corpus["chunks"] for cid in ch["claim_ids"]]
    terse = args.variant == "terse"
    reqs = pack(all_ids, claims, terse)

    out_path = HERE / "logs" / f"jev_audit_raw_{args.variant}.json"
    store = json.loads(out_path.read_text()) if out_path.exists() else {
        "variant": args.variant, "requests": {}
    }

    targets = [args.only] if args.only is not None else range(len(reqs))
    cl, model = client(timeout=args.timeout)

    with cl as c:
        for i in targets:
            ids = reqs[i]
            if str(i) in store["requests"]:
                print(f"request {i}: cached, skipping")
                continue
            qs = to_sdk(wire_questions(ids, claims, terse))
            print(f"request {i}: {len(ids)} claims, {len(qs)} questions ... ", end="",
                  flush=True)
            t0 = time.time()
            r = c.system_one(state_for(ids, claims, terse), qs, model=model)
            dt = time.time() - t0
            print(f"{dt:.1f}s  in={r.usage.input_tokens} out={r.usage.output_tokens}")
            store["requests"][str(i)] = {
                "claim_ids": ids,
                "model": r.model,
                "latency_s": round(dt, 2),
                "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens,
                "answers": serialise(r.answers),
            }
            out_path.write_text(json.dumps(store, indent=2))

    done = store["requests"]
    tot_in = sum(v["input_tokens"] for v in done.values())
    print(f"\n{len(done)}/{len(reqs)} requests cached in "
          f"{out_path.relative_to(HERE)}")
    print(f"input tokens {tot_in}  ->  ${tot_in / 1e9 * 42:.6f} spent "
          f"(output free)")


if __name__ == "__main__":
    main()
