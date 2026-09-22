# Finding: the day-6 routing table is not improvable on real-opponent data (2026-09-22)

The table is the agent's only real degree of freedom: 64 shop-pair draws, one route tape
each, out of 41 tapes. v0.4.3 fitted 19 of those cells against `opponents/v0_4_2.py` —
against ourselves — on in-sample cash. This is the refit against real rank 1-12 agents,
and it is a **negative result at every level of rigour**. Nothing ships.

Tooling: `scripts/v052_route_refit.py` (sweep + cross-validation),
`scripts/v052_holdout.py` (fresh-replay test of the survivors).

## Two engine facts the harness has to respect

**A replay is only coherent on its own board**, so every episode runs at the replay's own
seed, recovered from `info.seed`.

**The day-6 draw is not a function of the seed.** `scripts/enumerate_draws.py` recorded
that a pass-vs-pass draw table transfers to chassis-vs-chassis play on 0/12 seeds: agents
occupy tiles from step 0, weeds consume RNG only on still-empty tiles, and the streams
diverge. Measured here on 40 pairings, **only 10 reproduced the replay's own step-144
draw**. The draw must therefore be *observed* per (replay, seat) by running the pairing and
reading `observed_pair` back out of the arena.

This invalidates the draw labelling in `scripts/v052_replay_ab.py`
(`finding-2026-09-22-v052-routing-revert-rejected.md`), which read the draw from the
replay. Its paired deltas remain valid — a non-zero delta *proves* the two arms diverged —
but which draws it called "affected" was largely wrong, and one genuinely-affected episode
was discarded as a seat-asymmetric anchor non-tie.

The table only engages at step 144 (`_router` returns 0 before it), so steps 0-143 are
identical across every tape variant and the observed draw is invariant to the tape under
test. That is what makes the sweep valid.

## Coverage

115 kaggriculture replays with distinct seeds (`logs/leaderboard_replays/` +
`logs/september_spy/`), 230 pairings: **51 of 64 draws observed, 19 with >= 3 replays**.
Those 19 are the fittable set. `logs/v052_replay_draws.csv`.

## The sweep, and the winner's curse

19 draws x 41 tapes x every covering replay x both seats = **5,494 episodes**
(`logs/v052_route_sweep_table.csv`).

The criterion was **pre-registered before any result was inspected**: leave-one-replay-out
cross-validation of the *refit procedure*. For each draw and each held-out replay, pick the
best tape using only the other replays, then score (chosen − shipped) on the held-out one.

| | |
| --- | --- |
| in-sample gain, max of 41 tapes per draw | **+$5,650/draw** |
| out-of-sample, leave-one-replay-out (70 folds, 19 draws) | **−$151/draw** |
| | median +$326 · **37/70 folds positive** · t=−0.09 |

**15 of the 19 cells flip their chosen tape between folds**, which is noise by definition.
`FARMERS_MARKET/YARN_STORE` is the clearest: it picks tape 8, 10 or 12 depending on which
replay is held out, in-sample +$13,406, out-of-sample −$7,902.

That +$5,650 → −$151 collapse **is** the winner's curse, measured on the same kind of table
v0.4.3 was fitted on, with a better oracle than v0.4.3 had.

## The trap in leave-one-out, and the fresh-replay test

Four cells did not flip. For a **stable** pick LOO gives no independent check at all: each
replay is held out exactly once, so the LOO mean equals the in-sample gain *by
construction*. Those four were still selection-on-the-same-data — the exact evidence class
that produced v0.4.3 — so they were frozen as pre-specified hypotheses in
`scripts/v052_holdout.py` and tested against **116 replays downloaded after the sweep and
never used in selection**.

| draw | shipped → candidate | in-sample | fresh holdout | pairings won |
| --- | --- | --- | --- | --- |
| `PET_CAFE/YARN_STORE` | 11 → 12 | +$19,634 | **−$4,196** | 3/5 |
| `PIZZA_SHOP/SMOOTHIE_SHOP` | 124 → 2 | +$8,873 | **−$1,837** | 0/2 |
| `BRUNCH_SPOT/BAKERY` | 103 → 110 | +$1,438 | **−$8,466** | 0/2 |
| `PIZZA_SHOP/YARN_STORE` | 128 → 115 | +$8,706 | no fresh coverage | — |

**0 of the 3 testable cells beat the shipped tape.** All four outcomes are reported, not
only the ones that failed: with four independent tests a nominal p<0.05 winner arises by
chance ~19% of the time, which is why the pass/fail was fixed in advance.

## Verdict

The shipped v0.4.3/v0.5.1 table survives the sweep, the cross-validation and the fresh
holdout. Combined with `finding-2026-09-22-v052-routing-revert-rejected.md`, which found
reverting it costs $4,143/draw against real opponents, the table is neither improvable nor
removable on the evidence available. **The routing table is done.**

## Limitations, stated because they bound the conclusion

* 19 of 64 cells were fittable; the other 45 have < 3 replays and are untested.
* The fresh holdout is thin — 9 pairings across 3 cells — and is new episodes of the *same*
  top-team population, so it tests generalisation to new boards, not to new opponents.
* Replay opponents are open-loop: they do not react to us. They play at leaderboard tempo,
  which is the property that matters here, but they are not adaptive play.
* Margins against replay opponents span ±$150k, far wider than live play's ~$5.9k sd, so
  per-cell power is low even at 3-5 replays.

## What this implies for the next attempt

The binding constraint is **corpus size, not ideas**. Every dead end this session — the
routing table, sell metering, the v0.4.1 revert — died from thin evidence or a wrong
oracle, not from an exhausted search space. `scripts/mine_daily.py --dataset` can take the
corpus from ~230 pairings to thousands, which is the prerequisite for any per-cell claim
about the remaining 45 draws.

Do not spend a submission slot in the meantime. Only 2 count, a version bump auto-submits
from `main`, and a byte-identical runtime would evict v0.5.0 at 1772.9 — a submission still
climbing at a 54.4% win rate — and replace it with a duplicate of v0.5.1 starting from
Kaggle's default rating.
