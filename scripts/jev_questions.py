#!/usr/bin/env python3
"""The question set sent with each corpus chunk, plus a dry-run budget report.

    uv run python scripts/jev_questions.py              # budget only, no network
    uv run python scripts/jev_questions.py --show C0154 # the exact questions for one claim
    uv run python scripts/jev_questions.py --dump-request 0 > /tmp/req.json

NO NETWORK. This module only BUILDS requests; scripts/audit_claims.py sends them. Run
this first to see exactly what would be sent and what it would cost.

WHAT TO ASK, AND WHY THESE AND NOT OTHERS
=========================================
Derived from jev-1.13's documented behaviour (TypeSafe docs read 2026-09-21), not from
guesswork. Four properties of the model decide the whole question set:

1. "Choice question probabilities always add up to 1, so a line ranks first even when
   none answer the query" (cookbooks/semantic_find). A Choice is therefore a POINTER,
   never a truth test, and it always needs an escape hatch plus an independent Noul to
   say whether the pick means anything. Every Choice below carries a `not_checkable`
   option for exactly this reason.

2. A Noul probability "doesn't depend on the other options, so it can fall near zero"
   (ibid). Truth and existence questions are therefore always Nouls.

3. "Jev is not a calculator." No question here compares two numbers, counts anything, or
   asks how far off a figure is. The corpus has already masked every measurement value,
   so the model cannot attempt arithmetic even if a prompt invited it.

4. "Answers the question you wrote, not the one you meant" (literal reading). Every
   question states its exact condition and pins both boundaries in `criteria`.

Questions deliberately NOT asked:
  - "Is claim X still true?"  -- needs the number the model cannot see, and by design
    must not see. Code answers this, in Phase 3.
  - "How many claims in this section are stale?" -- "does not count reliably".
  - "How stale is this claim?" as a Score -- score levels are "weak in numerical
    calibration"; do not interpolate a magnitude between levels.
  - Anything about dates or ordering -- "reads dates as text, not ordered quantities".

BATCHING. All questions for a chunk go in ONE request: the state is sent once and
questions are evaluated in parallel. TypeSafe's own measurement of this pattern is
12.2x cheaper and 10.0x faster for 13 questions with no change in answers
(cookbooks/parallel_questions).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CORPUS = HERE / "logs" / "claims_corpus.json"
SIDECAR = HERE / "logs" / "claims_sidecar.json"

# The artifacts a claim could actually be settled against. Options are bare ids; the
# legend goes in the state ONCE rather than being repeated in 255 Choice questions.
# `not_checkable` is the mandatory escape hatch for a Choice that must sum to 1.
ARTIFACTS: dict[str, str] = {
    "daily_fingerprints": "logs/daily_fingerprints.csv — mined leaderboard episodes: "
    "per-team cash, op counts, strategy fingerprints",
    "draw_table": "logs/v043_draw_table.csv — per-draw route assignments with in-sample "
    "and held-out paired-diff evidence",
    "panel_winrate": "the panel win-rate table — win rate per opponent over paired seeds",
    "ladder_rank": "scripts/rank_ladder.py output — score against the tier 0-5 reference ladder",
    "candidates": "candidates.jsonl — mined 719-step routes and their recorded cash",
    "engine_spec": "the kaggle-environments 1.32.7 kaggriculture spec — prices, costs, "
    "caps, timings fixed by the engine",
    "arena_run": "local_arena.py output — self-play cash, win rate, crash/timeout counts",
    "not_checkable": "no data file settles this claim; it is narrative, a design "
    "rationale, or a judgement",
}


# ---------------------------------------------------------------------------
# Exact accounting. Builds the documented HTTP request body (state + questions)
# and measures it, so the budget is checked against the real payload rather than
# an estimate, and without needing the SDK or a key installed.
#
# The dry run exists because the first cut of this file put 255 claims in a
# request and came out at ~73k input tokens -- over the 64k cap -- with the
# repeated per-claim `criteria` outweighing the state 6:1. Packing is therefore
# driven by the MEASURED payload, not by the 255-option Choice cap.
# ---------------------------------------------------------------------------
TOK_PER_CHAR = 0.25  # ~4 chars/token; an estimate, and the only one left here
REQUEST_TOK_CAP = 64_000  # models: 64k for state plus all questions
STATE_TOK_CAP = 32_000  # models: 32k for state plus the longest question
PACK_HEADROOM = 0.80  # pack to 80% of the cap; placeholders vary in tokenisation


def wire_questions(claim_ids: list[str], claims: dict, terse: bool) -> dict:
    """The `questions` object as the HTTP API takes it (api.md request body).

    `terse` drops the per-claim `criteria` and relies on the definitions carried in
    the state instead. The docs are explicit that this is an empirical choice --
    "try your questions with and without criteria and keep whichever gives better
    answers on your documents" -- so Phase 5 decides it, not this file.
    """
    qs: dict[str, dict] = {}
    for cid in claim_ids:
        qs[f"artifact_{cid}"] = {
            "type": "choice",
            "instructions": f"Which data file listed in available_data_files would "
            f"settle whether claim {cid} is still accurate?",
            "criteria": dict.fromkeys(ARTIFACTS),
        }
        q_measured: dict[str, object] = {
            "type": "noul",
            "instructions": f"Claim {cid} asserts a specific measured result about "
            f"this agent's performance or about the game engine.",
        }
        q_retired: dict[str, object] = {
            "type": "noul",
            "instructions": f"The text of claim {cid} says the finding it describes "
            f"has been superseded, was wrong, or has been contradicted.",
        }
        if not terse:
            q_measured["criteria"] = {
                "true": "States a result, quantity, rate or outcome that was measured "
                "or is fixed by the engine, including where the value is "
                "masked as a <TYPE:index> placeholder",
                "false": "States a plan, an intention, a rationale, an open question, "
                "or a description of tooling, with no measured result",
            }
            q_retired["criteria"] = {
                "true": "The claim reports that an earlier belief, table, target or "
                "prescription turned out to be wrong or has been replaced",
                "false": "The claim asserts a finding in its own voice without disowning it",
            }
        qs[f"measured_{cid}"] = q_measured
        qs[f"retired_{cid}"] = q_retired

        if claims[cid]["kind"] == "conclusion":
            q_self: dict[str, object] = {
                "type": "noul",
                "instructions": f"The conclusion stated in claim {cid} is supported by "
                f"evidence given inside that same claim.",
            }
            if not terse:
                q_self["criteria"] = {
                    "true": "The claim states both a conclusion and the evidence or "
                    "mechanism it rests on",
                    "false": "The claim asserts a conclusion while the evidence for it "
                    "sits elsewhere or is not stated",
                }
            qs[f"self_supported_{cid}"] = q_self
    return qs


def state_for(claim_ids: list[str], claims: dict, terse: bool) -> dict:
    lines = "\n".join(f"{cid}| {claims[cid]['text']}" for cid in claim_ids)
    st = {
        "what_this_is": "Claims extracted from a Kaggle competition agent's research "
        "notes. Measurement values are masked as <TYPE:index> "
        "placeholders on purpose; judge what each claim MEANS, never "
        "the value of a masked number.",
        "available_data_files": ARTIFACTS,
        "claims": lines,
    }
    if terse:
        st["question_definitions"] = {
            "measured": "true when the claim states a result, quantity, rate or "
            "outcome that was measured or is fixed by the engine; false "
            "for plans, rationale, open questions or tooling description",
            "retired": "true when the claim reports that an earlier belief, table, "
            "target or prescription was wrong or has been replaced",
            "self_supported": "true when the claim states both a conclusion and the "
            "evidence or mechanism it rests on",
        }
    return st


def request_tokens(claim_ids: list[str], claims: dict, terse: bool) -> tuple[int, int]:
    st = json.dumps(state_for(claim_ids, claims, terse))
    qs = json.dumps(wire_questions(claim_ids, claims, terse))
    return int(len(st) * TOK_PER_CHAR), int(len(qs) * TOK_PER_CHAR)


def pack(all_ids: list[str], claims: dict, terse: bool) -> list[list[str]]:
    """Greedily fill requests up to the measured cap and the 255-option Choice cap."""
    out: list[list[str]] = []
    cur: list[str] = []
    for cid in all_ids:
        trial = cur + [cid]
        s_tok, q_tok = request_tokens(trial, claims, terse)
        over = (
            s_tok + q_tok > REQUEST_TOK_CAP * PACK_HEADROOM
            or s_tok > STATE_TOK_CAP * PACK_HEADROOM
            or len(trial) > CHOICE_OPTION_CAP
        )
        if cur and over:
            out.append(cur)
            cur = [cid]
        else:
            cur = trial
    if cur:
        out.append(cur)
    return out


CHOICE_OPTION_CAP = 255


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", metavar="CLAIM_ID")
    ap.add_argument(
        "--terse",
        action="store_true",
        help="drop per-claim criteria; definitions go in the state once",
    )
    ap.add_argument("--dump-request", type=int, metavar="REQUEST_INDEX")
    args = ap.parse_args()

    if not CORPUS.exists():
        raise SystemExit("run scripts/build_claims_corpus.py first")
    corpus = json.loads(CORPUS.read_text())
    claims = json.loads(SIDECAR.read_text())["claims"]
    all_ids = [cid for ch in corpus["chunks"] for cid in ch["claim_ids"]]

    if args.show:
        c = claims[args.show]
        print(f"{args.show}  {c['file']}:{c['line']}  [{c['kind']}]")
        print(f"section : {c['section']}")
        print(f"text    : {c['text']}")
        print(f"masked  : {c['values']}")
        qs = wire_questions([args.show], claims, args.terse)
        print(f"\nquestions sent for this claim ({'terse' if args.terse else 'full'}):")
        print(json.dumps(qs, indent=2))
        return

    for terse in (False, True):
        if args.terse and not terse:
            continue
        reqs = pack(all_ids, claims, terse)
        tot_tok = tot_q = 0
        label = "terse (criteria in state)" if terse else "full (criteria per claim)"
        print(f"\n=== {label} ===")
        print(
            f"{'req':>4} {'claims':>7} {'questions':>10} {'state':>8} "
            f"{'quest':>8} {'total':>8}  cap"
        )
        for i, ids in enumerate(reqs):
            s_tok, q_tok = request_tokens(ids, claims, terse)
            nq = len(wire_questions(ids, claims, terse))
            ok = "OK" if s_tok + q_tok <= REQUEST_TOK_CAP and s_tok <= STATE_TOK_CAP else "OVER"
            print(f"{i:>4} {len(ids):>7} {nq:>10} {s_tok:>8} {q_tok:>8} {s_tok + q_tok:>8}  {ok}")
            tot_tok += s_tok + q_tok
            tot_q += nq
        print(
            f"{len(reqs)} requests, {tot_q} questions, ~{tot_tok} input tokens, "
            f"${tot_tok / 1e9 * 42:.6f}/pass  (1,000 passes "
            f"${tot_tok / 1e9 * 42 * 1000:.2f})"
        )
        if not args.terse:
            continue

    print("\nrate limits are 1,200 req/min and 250k tok/s, so neither binds.")
    print("Output tokens are free, so only input size matters.")

    if args.dump_request is not None:
        reqs = pack(all_ids, claims, args.terse)
        ids = reqs[args.dump_request]
        body: dict[str, object] = {
            "model": "jev-1.13.0",
            "state": state_for(ids, claims, args.terse),
            "questions": wire_questions(ids, claims, args.terse),
        }
        shown_state = dict(state_for(ids, claims, args.terse))
        shown_state["claims"] = str(shown_state["claims"])[:500] + "\n...[truncated]"
        body["state"] = shown_state
        body["questions"] = dict(list(wire_questions(ids, claims, args.terse).items())[:4])
        print("\n--- request body (claims and questions truncated) ---")
        print(json.dumps(body, indent=2))


if __name__ == "__main__":
    main()
