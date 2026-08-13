#!/usr/bin/env python3
"""Phase 4 -- encode the winning route into the v0.2.5 agent's `_ROUTE_B85_PARTS`.

    uv run python encode_submission.py                        # winner from the Phase 3 report
    uv run python encode_submission.py --hash 3f2a91ce...     # a specific candidate
    uv run python encode_submission.py --write-agent main_v0_2_5.py

Encoding is base85 of zlib of the JSON route, split into fixed-width chunks. The
chunking is not cosmetic: one multi-kilobyte string literal gets rewrapped by
`ruff format` on every run, so the route is stored as a list of 92-char chunks
(see main.py:37). The width and the compression settings come from
`mining.common`, which delegates the agent template to
`scripts/build_route_agent.py` -- so there is one implementation of the scheme,
not a reimplementation that can drift from the v0.2.4 artifact.

Three checks run before anything is emitted:

  1. ROUND TRIP: decode -> decompress -> compare to the source trace, element by
     element, and assert equality.
  2. SCHEME PARITY: decode the *existing* main.py `_ROUTE_B85_PARTS` with this
     module's own decoder and confirm it yields a 719-step route. If the scheme
     had drifted, this fails.
  3. CHUNK PARITY: confirm every chunk in main.py (except the last) is exactly the
     width we emit, so `ruff format` stays stable on the generated file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from mining import common
from mining.common import (
    CHUNK_WIDTH,
    PROJECT_ROOT,
    ROUTE_STEPS,
    chunk_b85,
    decode_route_b85,
    encode_route_b85,
)

MAIN_PARTS_RE = re.compile(r"_ROUTE_B85_PARTS\s*=\s*\[(.*?)\n\]", re.DOTALL)


def existing_parts(path: str) -> list[str]:
    """The `_ROUTE_B85_PARTS` chunks currently in an agent file."""
    with open(path) as f:
        src = f.read()
    m = MAIN_PARTS_RE.search(src)
    if not m:
        return []
    return re.findall(r'"([^"]*)"', m.group(1))


def check_scheme_parity(reference: str) -> tuple[bool, str]:
    """Decode the shipped v0.2.4 route with our decoder; confirm scheme identity."""
    parts = existing_parts(reference)
    if not parts:
        return False, f"could not find _ROUTE_B85_PARTS in {reference}"
    try:
        route = decode_route_b85("".join(parts))
    except Exception as exc:
        return False, f"decoding {os.path.basename(reference)} failed: {exc}"
    if not isinstance(route, list) or len(route) != ROUTE_STEPS:
        return False, f"{os.path.basename(reference)} decoded to {len(route)} steps"
    widths = {len(p) for p in parts[:-1]}
    if widths and widths != {CHUNK_WIDTH}:
        return (
            False,
            f"chunk width mismatch: {reference} uses {sorted(widths)}, we emit {CHUNK_WIDTH}",
        )
    if len(parts[-1]) > CHUNK_WIDTH:
        return False, f"final chunk is {len(parts[-1])} > {CHUNK_WIDTH}"
    return True, f"{len(parts)} chunks, {len(route)} steps, width {CHUNK_WIDTH}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 4: encode the winning route")
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--report", default="logs/cvar_report.json")
    ap.add_argument("--hash", help="candidate hash to encode (default: Phase 3's winner)")
    ap.add_argument("--reference", default="main.py", help="agent to check scheme parity against")
    ap.add_argument("--out", default="logs/route_b85_parts.txt", help="where to write the snippet")
    ap.add_argument("--write-agent", metavar="PATH", help="also emit a complete agent file")
    ap.add_argument("--version", default="0.2.5")
    ap.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="encode even if Phase 3 reported the winner does not beat v0.2.4",
    )
    args = ap.parse_args(argv)

    def abspath(p):
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    candidates = {c["hash"]: c for c in common.read_jsonl(abspath(args.candidates))}
    if not candidates:
        raise SystemExit("candidate pool is empty")

    target = args.hash
    holdout = None
    if not target:
        report_path = abspath(args.report)
        if not os.path.exists(report_path):
            raise SystemExit(f"no Phase 3 report at {report_path}; pass --hash explicitly")
        with open(report_path) as f:
            report = json.load(f)
        holdout = report.get("holdout")
        if not holdout:
            raise SystemExit("Phase 3 report has no holdout result; rerun rank_cvar.py")
        target = holdout["winner_hash"]
        if not holdout.get("winner_beats_baseline") and not args.allow_unvalidated:
            raise SystemExit(
                f"Phase 3 says the winner does NOT beat v0.2.4 on held-out CVaR_5 "
                f"(delta {holdout['cvar5_delta']:+,.0f}). Refusing to encode; "
                f"pass --allow-unvalidated to override."
            )

    # A hash prefix is enough, since these are printed truncated everywhere else.
    if target not in candidates:
        matches = [h for h in candidates if h.startswith(target)]
        if len(matches) != 1:
            raise SystemExit(f"hash {target!r} matched {len(matches)} candidates")
        target = matches[0]

    cand = candidates[target]
    route = decode_route_b85(cand["route_b85"])

    print(f"Phase 4: encoding candidate {target}")
    print(
        f"  provenance: {cand['team']} episode {cand['episode']} seat {cand['seat']}  "
        f"recorded ${cand['recorded_cash']:,.0f}"
    )
    if holdout:
        print(
            f"  held-out CVaR_5 ${holdout['winner']['cvar5']:,.0f} vs v0.2.4 "
            f"${holdout['baseline']['cvar5']:,.0f}  (delta {holdout['cvar5_delta']:+,.0f})"
        )
    print(f"  steps: {len(route)}")
    if len(route) != ROUTE_STEPS:
        raise SystemExit(f"route has {len(route)} steps, expected {ROUTE_STEPS}")

    # --- check 2: scheme parity against the shipped v0.2.4 agent
    ok, detail = check_scheme_parity(abspath(args.reference))
    print(f"  [{'PASS' if ok else 'FAIL'}] scheme parity vs {args.reference}: {detail}")
    if not ok:
        raise SystemExit("encoding scheme does not match the reference agent; refusing to emit")

    # --- encode
    blob = encode_route_b85(route)
    parts = chunk_b85(blob)
    payload = json.dumps(route, separators=(",", ":")).encode("utf-8")
    print(
        f"  encoded: {len(payload):,} B JSON -> {len(blob):,} B base85 "
        f"({len(parts)} chunks of {CHUNK_WIDTH})  ratio {len(blob) / len(payload):.3f}"
    )

    # --- check 1: round trip
    decoded = decode_route_b85("".join(parts))
    assert decoded == route, "round trip mismatch: decoded route differs from source"
    assert len(decoded) == len(route), "round trip mismatch: step count differs"
    for i, (a, b) in enumerate(zip(route, decoded, strict=True)):
        assert a == b, f"round trip mismatch at step {i}: {a} != {b}"
    print(f"  [PASS] round trip: decode -> decompress -> compare, {len(decoded)} steps identical")

    # Also confirm the canonical hash survives the trip, which is what ties the
    # shipped artifact back to the Phase 2/3 measurements.
    rehash = common.route_hash(common.normalize_route(decoded))
    match = rehash == target
    print(f"  [{'PASS' if match else 'FAIL'}] canonical hash round trip: {rehash[:16]}")
    if not match:
        raise SystemExit("hash mismatch after round trip; the encoded route is not the ranked one")

    snippet = "_ROUTE_B85_PARTS = [\n" + common.route_parts_source(blob) + "]\n"
    out_path = abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(snippet)
    print(f"\n  wrote paste-ready snippet -> {out_path}  ({len(snippet):,} B)")

    if args.write_agent:
        agent_path = abspath(args.write_agent)
        size = common.write_route_agent(
            route,
            agent_path,
            provenance={
                "episode": cand.get("episode"),
                "seat": cand.get("seat"),
                "team": cand.get("team", "?"),
                "recorded_cash": cand.get("recorded_cash"),
                "steps": len(route),
                "hash": target,
                "selected_by": "CVaR_5, held-out validated (see logs/cvar_report.json)",
            },
            version=args.version,
        )
        # The emitted file must itself decode to the same route.
        emitted = decode_route_b85("".join(existing_parts(agent_path)))
        assert emitted == route, f"{agent_path} does not decode to the winning route"
        print(f"  wrote agent -> {agent_path} ({size:,} B), verified decodes to the winning route")

    print("\n  first 2 and last 1 chunk:")
    for p in parts[:2]:
        print(f'    "{p}",')
    print("    ...")
    print(f'    "{parts[-1]}",')
    return 0


if __name__ == "__main__":
    sys.exit(main())
