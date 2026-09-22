# Finding: v0.5.2 proposed, tested, REJECTED — and the ~1700 was never a regression (2026-09-22)

Two results. The first retires the premise that v0.5.0/v0.5.1 regressed. The second is a
hypothesis this note raised, built, gated and **threw away**, plus the analysis error that
produced it.

No agent change ships from this work. `main.py` is byte-identical to v0.5.1.

## 1. `public_score` was reporting maturity, not strength

| version | episodes | overall win rate | paired margin | score |
| --- | --- | --- | --- | --- |
| v0.4.0 | 133 | 49.6% | +$4,245 | 2591.5 |
| v0.4.3 | 602 | 38.5% | +$538 | 1785.8 |
| v0.5.0 | 103 | **54.4%** | +$5,030 | 1772.9 |
| v0.5.1 | 104 | **51.0%** | +$5,162 | 1745.7 |

A submission starts at Kaggle's default rating and climbs while it out-performs, so a win
rate above 50% is the signature of an agent that is **under-rated and still climbing**.
v0.5.0 and v0.5.1 were read ~15 hours old.

`v0.5.0`'s runtime is byte-identical to v0.4.3, so it cannot be a regression against it —
yet the two differ by 419.6 points of score and 15.9 points of win rate. All of that is
maturity. The converged number is v0.4.3's: 602 episodes, margin decayed to +$538
(t=1.38). **This chassis is an ~1800-class agent**, and v0.4.0's 2591.5 was a high-water
mark after 133 episodes — that same submission went 39.8% at +$407 against 2000+ opponents.

Corollary worth keeping: only 2 submissions count toward the leaderboard, so a slot spent
on a change whose predicted effect is below the noise floor (v0.5.1 self-reported
+$11/game) evicts a submission that had not finished converging.

## 2. The rejected hypothesis: revert the v0.4.3 routing table

`scripts/audit_opponent_bands.py` pairs every live episode's two rewards and buckets them
by the opponent's leaderboard rating. Pooling submissions by runtime hash, it appeared to
show that the v0.4.3 per-draw table cost ~$3,690/episode against 2000+ opposition
(Welch t=+4.48), with the EXP240 graft carrying the gain. A v0.5.2 was built on that: the
router restored to v0.4.1's dual-expert form, `_V043_TABLE` deleted, runtime verified
byte-identical to v0.4.1/v0.4.2.

**That result was an artifact of pooling.** Per submission, in the 2000+ band:

| runtime | version | submission | n | win rate | margin | t |
| --- | --- | --- | --- | --- | --- | --- |
| `65a15fa2` no table | 0.4.2 | 56257221 | 396 | 34.8% | −$832 | −2.36 |
| `65a15fa2` no table | 0.4.1 | 56253417 | 155 | 36.8% | −$162 | −0.30 |
| `65a15fa2` no table | 0.4.1 | 56248498 | 38 | 89.5% | **+$15,470** | +6.64 |
| `2c8306b6` +table | 0.4.3 | 56287565 | 349 | 33.2% | −$668 | −2.49 |
| `2c8306b6` +table | 0.5.0 | 56439402 | 18 | 33.3% | −$1,917 | −0.77 |
| `2c8306b6` +table | 0.5.0 | 56429111 | 7 | 28.6% | +$3,669 | +0.77 |

**Spread on byte-identical code: $16,303 for the no-table runtime, $5,586 for the table
runtime.** The claimed effect was $3,690. It was inside the noise floor of its own
measurement, and the pooled mean was carried by one 38-episode submission.

The cause is the confound the script's own docstring flags: the opponent rating is
**today's** leaderboard snapshot. A submission's first few dozen episodes are played
against teams whose rating *then* bears little relation to their rating *now*, so for
young or small-n submissions the bucket label is close to meaningless. The method is only
usable between submissions that played contemporaneously, at comparable n.

This is the same trap as `docs/finding-2026-09-21-live-score-not-comparable.md`, one level
down: that note established `public_score` is not comparable across submissions, and here
the *paired margin* turned out not to be either, when bucketed on a stale rating proxy.

### Two independent tests then rejected the revert outright

**Self-play** — `local_arena.py --agent main.py --opponent opponents/v0_5_1.py`, 12 seeds:
**0W-7T-5L**. Discounted at the time, and correctly so: self-play against our own snapshot
is the criterion that promoted the table, so it cannot arbitrate its removal.

**Real leaderboard agents** — `scripts/v052_replay_ab.py`. The recorded action streams of
the rank 1-12 teams in `logs/september_spy/`, each replayed at its **own** seed recovered
from `info.seed` (a replay is only coherent on the board it was recorded against, and the
seed also pins the day-6 draw). Both seats, 196 episodes; the 25 unaffected draws are an
exact-tie anchor and produced 1 seat-asymmetric non-tie, which was dropped.

On the draws this change actually touches, the revert **loses 11 of 13 at −$4,143/draw.**

This is the one un-confounded test of the three — fixed seeds, fixed strong opponents,
both seats — and it says the v0.4.3 table is *better* than v0.4.1's routing against real
top-band play, which is the opposite of the claim that motivated v0.5.2.

### A bug in the harness, found by its own anchor

The first run marked 24 draws as affected, and 11 of them came back with a delta of
exactly $0. `changed_cells()` had compared the shipped table against `_EXP_SHOP_ROUTES`
alone, but v0.4.1's router is **dual-expert**: `_EXP_SHOP_ROUTES` for non-YARN pairs and a
separate inline V39 table for any pair containing `YARN_STORE`. Fourteen YARN draws are
identical between the two arms. Fixed; the true count is **20 changed draws**, not 27.

## 3. Also falsified: the "realised $/unit" thesis

Fingerprinting 50 of our live episodes against the 49 spied top-12 episodes showed we
bank the same revenue as the leaders ($82.4k vs $81.9k) on 31% worse realised prices
($53.1/unit vs $69.3/unit), concentrated in the products with convex above-baseline price
penalties. That looked like a sell-metering opportunity worth ~$5,100/episode.

**It is not a policy gap.** Paired within our own episodes, our realised $/unit is
statistically indistinguishable from our *same-episode opponent's* — strawberry $80.1 vs
$81.2, wool $107.4 vs $108.9, fertilizer $38.8 vs $38.6. Price realisation is a property
of the episode's shared market, not of how either seat sells into it. The top teams' better
prices come with their episodes.

Do not re-chase, either: `sell_orders_wasted_pct` reads 98.2% for us against the leaders'
44.6%, which is an artifact of ordering qty >> on-hand. Counting sell *orders*, zero-fill
is 66.2% ours vs 62.8% theirs, and we hit the 10-slot-per-turn cap on 3.9% of turns vs
their 5.0%. Not a defect.

## 4. A real measurement bug, fixed

`scripts/audit_live_scores.py:runtime_hash` anchored on `s.find("AGENT_VERSION")` — the
first *textual* occurrence. From v0.5.0 on the module docstring discusses `AGENT_VERSION`,
so the anchor landed inside the header and hashed prose as runtime. That is why v0.4.3 and
v0.5.0 were being reported as different runtimes when they are the same agent. Now
anchored on the assignment line; they group correctly and show a 419.6-point score spread
on identical code.

## What to keep from this

* **`scripts/v052_replay_ab.py` is the artifact worth keeping.** It is the repo's first
  offline gate that plays real top-band opponents rather than our own snapshots, and it is
  the test that rejected this change. Run it before any future route or tape edit.
* `scripts/audit_opponent_bands.py` is useful for *ranking* contemporaneous submissions at
  comparable n, and misleading otherwise. Its docstring says so.
* The routing table is **not** a known defect. `kaggriculture-agent-is-at-a-local-optimum`
  still holds: 9 of 11 structural changes lost paired seeds, and this is attempt 12.
* The next real candidate needs a mechanism that is *not* route-table tuning and *not*
  sell timing, since both are now measured as dead ends.
