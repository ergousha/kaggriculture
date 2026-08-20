# Kaggriculture agent

A two-tier agent for the Kaggle [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
competition, plus a local arena for measuring it, a macro-strategy search, and a
submission pipeline.

Verified against **kaggle-environments 1.32.6** (kaggriculture spec version `0.1.0`), which
is what the competition runs. The engine version is load-bearing: the same game on the
same seed pays out completely differently across releases, so a measurement taken on
1.32.3 is not comparable to one taken on 1.32.6.
Python 3.12+.

**v0.2.0 changed architecture.** The submission is now a *route-replay* agent: one
recorded 719-step action route plus bounded WEED repair, SELL-slot ordering by price
impact, and hands alignment. It scores **60/60 against the reference ladder** where the
hand-written agent scored 36/60 and could not beat tier 6 at all. The reason is in
[docs/experiments.md](docs/experiments.md): above ~$56k this is not a strategy
competition — every ladder rung above 5, and the current #1, is a fixed plan with a thin
repair layer. The hand-written two-layer agent is still here as `opponents/v0_1_1.py`
and `search/` still operates on it; everything below about planners and schedulers
describes that line, not the current submission.

```
main.py                       # THE submission — v0.2.0 route replay, stdlib only
scripts/build_route_agent.py   # bakes a downloaded episode into a route-replay agent
local_arena.py                # match runner, metrics, decision logs, A/B + sweep rig
submit.py                     # pre-flight, submit, poll, history
search/space.py               # the ~23-number macro vector that is actually searched
search/objective.py           # P(win) + CVaR scoring (NOT mean cash)
search/cem.py                 # cross-entropy-method search + loss-tail diagnosis
search/evolution.py           # population-based search over the same vector
search/harness.py             # paired-seed match harness used by both searches
search/smoke_test.py          # unit tests for the search stack
search/route_search.py        # issue #26: mutation-and-accept over the raw route (blocks #27-#30)
opponents/adaptive.py         # sparring partner (see "Opponents" for provenance)
opponents/vX_Y_Z.py           # every submitted version, kept as a sparring partner
scripts/mine_daily.py         # mines Kaggle's daily episode dumps into strategy fingerprints
scripts/rank_ladder.py        # ranks the agent against the reference ladder — THE gate
opponents/ladder/             # Rayk Kretzschmar's tier 0-5 reference agents (MIT)
scripts/probe_agent.py        # schema probe; writes logs/probe_schema.json
scripts/sync_opponent.py      # pre-commit hook that versions main.py into opponents/
scripts/download_yesterday.sh # bash script to download the latest episode dumps automatically
scripts/analyse_routes.py     # pandas analysis script to classify static vs dynamic routes
scripts/compare_routes.py     # compares unit-op distribution between two replay JSONs or agents
kaggle_credentials.example.py # copy to kaggle_credentials.py (gitignored)
logs/                         # replays, decision logs, submission history, search output
```

## Setup

```bash
uv sync
```

The submission itself needs nothing installed: `main.py` is standard library only.
`kaggle-environments` and `kaggle` are for the arena and the submit pipeline; `numpy`
is used by the tooling, not by the agent.

## Quick start

```bash
uv run python scripts/probe_agent.py                                    # confirm the schema
uv run python scripts/rank_ladder.py --fetch                            # get the ladder
uv run python scripts/rank_ladder.py --episodes 6                       # which rung are we on
uv run python scripts/mine_daily.py                                     # what the frontier does
uv run python local_arena.py --agent main.py --opponent baseline --episodes 30
uv run python local_arena.py --agent main.py --opponent opponents/adaptive.py --episodes 30
uv run python -m search.cem --diagnose 40                                # loss-tail report
uv run python -m search.cem --iterations 12 --pop 16 --episodes 8       # macro search
uv run python submit.py --dry-run
```

> **Read "What the leaderboard data says" before trusting anything else in this file,
> and [docs/experiments.md](docs/experiments.md) before acting on it.** Mining the daily
> episode dumps falsified several of the economic claims the agent was built on — but it
> also produced targets that then lost 0/30 paired seeds, because ranking seats by final
> cash mostly ranks them by a random shop draw. Sections that are known-wrong are marked
> as such rather than deleted, and `docs/experiments.md` records which derived changes
> actually survived an A/B.

---

## Implementation Notes

The following mechanics are useful to keep in mind when designing the agent:

- **16 hands cost $2,583/day** (hire cost is `farmHandCostMult * fib(n)`).
- **The opponent's whole farm is public** — money, tiles, hands, quadrants. Only shed/seeds/carried inventories are hidden.
- **Shed cap 100 applies to mid-day `PLACE` too**, so you cannot stockpile in carried inventories.
- **Compute budget:** `actTimeout: 1.0`s/turn, plus a **60s overage bank** per episode.
- **The env loads the LAST callable defined in your file**, not the one named `agent`
  (`agent.py: get_last_callable` → `[v for v in env.values() if callable(v)][-1]`).
  A helper defined below `agent` silently becomes the agent and every turn errors.
  This cost the first working build 1,436 ERROR statuses. `submit.py` asserts against it.
- **`obs.private.shed` predates the turn's unit actions.** The interpreter applies
  `_apply_unit_action` for the farmer and every hand (`kaggriculture.py` L922-926,
  kaggle-environments 1.32.6), **then** `_process_market` (L928), then `_town_consume`
  (L929). The shed you observe is therefore the shed *before* this turn's `PLACE` /
  `DROP` / end-of-day deposits land in it — a **lower bound**, not the sellable quantity.
  Sizing an order with `qty = min(route_qty, observed_shed)` truncates every sale, every
  turn, for every product, and the collapse to near-$0 that follows looks economic and is
  not. `main.py` is safe only because it replays recorded quantities verbatim.
  [tests/test_observation_ordering.py](tests/test_observation_ordering.py) pins the
  ordering; the measurement rounds it cost are in
  [docs/experiments.md](docs/experiments.md#market-layer-measured-and-exhausted-issue-23).

### The economics that decide the strategy

**The market does not start empty.** Inventory begins at **10,000 units per product**
and town demand drains it every interval, so price is not a pot you deplete — it is a
level that *recovers*, and over 30 days it **rises**. Median trajectory over 69 real
Kaggle episodes (`scripts/mine_daily.py`), start → peak → end:

| Product | start | peak | end | |
| --- | --- | --- | --- | --- |
| MILK | $160 | **$329** | $329 | +106%, never retraces |
| STRAWBERRY | $120 | **$294** | $294 | +145%, never retraces |
| WOOL | $200 | $247 | $247 | +24% |
| EGG | $50 | $69 | $69 | +38%, and it is the flattest curve in the game |
| WHEAT | $25 | $52 | $52 | +108% — buying wheat gets *worse*, selling it gets better |
| CARROT | $35 | $42 | $42 | +20% |
| MELON | $250 | $281 | **$269** | the **only** product that ends below its peak |

**That table is from the 2026-08-05 field, and it does not generalise.** Re-mining the
**2026-08-10** dump (150 episodes / 300 seats) shows prices rise only while the field
*under*-produces. The 08-10 field has converged and crushes everything it touches:

| product | 08-05 start→peak→end | 08-10 start→peak→end |
| --- | --- | --- |
| MILK | 160 → 329 → 329 | 169 → 222 → **3** |
| STRAWBERRY | 120 → 294 → 294 | 128 → 206 → **110** |
| WHEAT | 25 → 52 → 52 | 28 → 46 → 36 |
| WOOL | 200 → 247 → 247 | 206 → 218 → 181 |
| MELON | 250 → 281 → 269 | 256 → 272 → **25** |
| EGG | 50 → 69 → 69 | 50 → 62 → 62 |

Only EGG and CARROT hold their price, and only because nobody produces them. So the rule
is **not** "produce late" — it is **"sell before the field does."** Town demand sets a
drain rate and revenue is a race to fill it.

What survives both days:

1. **The curve shapes are real and worth knowing.** `MELON` and `WOOL` are `sq` above
   target (amp 0.01 and 0.058), so they collapse on volume — ~150 melon past I0 takes it
   from $250 to $31. `MILK` and `STRAWBERRY` are `linear` above target and tolerate
   volume far better.
2. **But "melon is a trap" does not follow, and was measured false.** Cutting
   `MELON_TILE_TARGET` from 9 to 3 lost **0W-30L** against v0.0.9. Melon's 1.3% revenue
   share in the field is low because everything else is bigger, not because it loses
   money — the field still sells 30.6 melon for $1,410. A low share is not a verdict.
3. **Egg is worthless, and this one holds.** 0.0% of top-decile revenue, the entire field
   runs **zero coops and zero geese**, and turning our own egg engine back on measures
   **−62.3%** on 30 paired seeds (better on 0/30, p~0.0). This is the one product claim
   that is both unanimous in the field and confirmed by A/B.

The full 08-10 comparison, the mechanics behind each number, and the change plan derived
from them are in [docs/experiments.md](docs/experiments.md).

**Fertilizer is sellable, and not applying any is the mistake.** Selling it is fine and
the frontier does it too (11.5% of top-decile revenue) — supply is not the constraint,
because `fertilizer_available` resets on **every animal tile every day**, so 14 pastures
yield ~420 units an episode. The field applies ~75 of those and sells the rest. Applying
is worth far more at the margin: `FERTILIZE` sets `fertilized_until_day = day + 2`
(3 days inclusive), and a watered+fertilized strawberry production event adds **2 units
instead of 1**, so with strawberry's interval of 2 a single application covers two
production events and two applications take a tile from 4 units to 8 — roughly $400 of
strawberry for 2 fertilizer that would have sold for ~$140.

`FERTILIZE` is already in `main.py`'s legal `UNIT_OPS` set; `build_tasks` never emits it.
This agent has issued **zero** `FERTILIZE` ops across every episode measured, and takes
39.0% of its revenue from selling the raw fertilizer instead.

#### Superseded: the "capped pot" table

The table below was the basis for the melon-plus-egg plan, and it is wrong. It was
produced by walking the price curve down from **inventory 0**, so it models a virgin
market being flooded once. The real market starts at 10,000 and is continuously
drained, so it never traverses that part of the curve.

<details>
<summary>The original cumulative-revenue table (do not plan from this)</summary>

| Product | N=50 | N=100 | N=200 | N=400 | N=1600 |
| --- | --- | --- | --- | --- | --- |
| **EGG** | $2,244 | $4,371 | $8,510 | $16,559 | **$62,421** |
| MELON | $12,098 | $21,721 | $26,527 | $26,727 | $27,927 |
| WHEAT | $1,127 | $2,193 | $4,293 | $8,313 | $31,443 |
| WOOL | $7,655 | $7,969 | $8,069 | $8,269 | $9,469 |
| MILK | $5,430 | $6,205 | $6,305 | $6,505 | $7,705 |
| CARROT | $1,482 | $2,738 | $4,832 | $7,853 | $11,438 |
| TOMATO | $2,411 | $4,318 | $7,221 | $10,453 | $12,199 |
| STRAWBERRY | $3,648 | $3,847 | $3,947 | $4,147 | $5,347 |

It concluded that egg is the only product that scales, that melon is a one-shot
$21.7k opening, that wool (~$8k) and milk (~$6.5k) are capped side pots saturated by
2 sheep and 2 cows, and that growing wheat beats buying it at a 1:1 goose ratio.
Every one of those is contradicted by the measured data above. The instructive part
is *why* it was wrong: the curve was replicated exactly and then evaluated at a
starting inventory the game never visits.

</details>

---

## Agent architecture

`main.py` is the whole agent: hand-written, standard library only, no learned weights.

**Layer A — `StrategicPlanner`** (macro, once per turn)
- `plan_roles` assigns every unlocked non-shed tile a role (`COOP`, `PASTURE_SHEEP`,
  `PASTURE_COW`, `MELON`, `STRAWBERRY`, `WHEAT`), nearest-the-shed first since animals
  need wheat carried out daily. Coops are built **just-in-time against cash**.
- `market_orders` walks a **capital priority ladder**. Cash owed to the egg engine
  (vacant coops + feed for birds already owned) is ring-fenced as `engine_claim`;
  land and cash-crop seed can only draw on what is left. **This is now known to be
  backwards** — egg is 0.6% of frontier revenue, so the ladder's top claim funds the
  game's weakest income line and starves land and livestock, which are its strongest.
  See "What the leaderboard data says". Not yet changed, because changing it is a
  measured strategy change and not a docs fix.
- `target_hands` hires to `MAX_HANDS`, throttled by `HIRE_CASH_FRACTION`. Note that
  `hires_today` resets every day, so hands are re-hired daily — pausing hiring is not
  a saving, it is a shutdown (measured: −65.9%).
- Everything sellable is sold on sight. Withholding to defend a price measured worse
  and unsold stock scores nothing.

**Layer B — `SpatialScheduler`** (micro, every turn)
- `build_tasks` enumerates every useful (tile, op) with a priority. Survival outranks
  yield: a plant at `consecutive_unwatered >= 1` dies tonight, an animal at
  `consecutive_unfed >= 1` escapes tonight and is unrecoverable.
- Two pre-passes handle work the generic loop structurally cannot:
  `_ferry_animals` (PICKUP at shed → walk → PLACE is a two-tile chain, and a `PLACE`
  task is only assignable to a unit already carrying the animal) and `_provision_feed`
  (`FEED` consumes wheat from the acting unit's own inventory).
- `assign` is **greedy nearest-capable unit**, one task per unit, one unit per tile.
  **Deliberately not MAPF** — units may legally share tiles, so there are no collisions
  to resolve. An exactly optimal assignment ships alongside it and is switched **off**;
  see "Optimal assignment loses to greedy".

**`MarketAnalyzer`** replicates the price curve exactly, and derives
`units_sellable_above(item, floor)` by walking the curve the way the interpreter does.

**`OpponentTracker`** reads their public farm to classify them
(`EARLY_RUSH` / `ANIMAL_LONGTERM` / `MELON_MAXXER` / `BALANCED`) and to forecast
imminent supply from visible on-tile yield and maturities. Their shed is hidden, so
every forecast is a **lower bound** — stated in the code. It feeds the decision log
only; four separate attempts to let it drive behaviour all measured worse.

**Safety guard** — blanket `try/except` returning a legal no-op, a time check against
`actTimeout`, and `_validate_unit_op` dropping anything the interpreter would not
recognise. Across every run reported here: **0 crashes, 0 timeouts, 0 invalid statuses.**

---

## What the leaderboard data says

> **The confound that governs every number here.** Cash rank in an episode is mostly an
> episode-level dice roll shared by both players. Shops are drawn at random with
> replacement every 3 days (`town["unlocked_shops"].append(rng.choice(sorted(SHOPS)))`)
> up to 8 instances, and four of the eight shop types buy strawberry — so how much demand
> an episode has is a coin-flip sequence. Within an episode the two seats' realised
> strawberry price differs by a mean of **$7.20/unit**; between episodes the stdev is
> **$56.80** (range $25–$227), and the number of strawberry-buying shops explains it
> (1 shop → $37/unit, 6 shops → $190/unit).
>
> So **sorting seats by final cash sorts them mostly by luck.** Over 300 seats,
> `corr(cash, sold_STRAWBERRY)` is +0.088, `corr(cash, ops_productive)` is +0.095, and
> `corr(cash, owned_tiles)` is **−0.139**. Volume, throughput and land are not what
> separate the cohorts. Any "the top decile does X, so do X" inference from this section
> is unsafe unless X is something the *whole* field does.
>
> Because the draw is shared by both seats it cancels pairwise, and the competition ranks
> pairwise — so the gate that matters is the within-episode margin against the frozen
> previous version. Full working in [docs/experiments.md](docs/experiments.md).

Kaggle publishes the previous day's episodes as a daily dataset (~21 GB, ~700
episodes). `scripts/mine_daily.py` streams it and keeps a ~1 KB fingerprint per
player-seat, so a day compresses to well under a megabyte. Everything below comes
from **69 real episodes / 138 player-seats** already in `logs/leaderboard_replays/`.

Revenue is attributed exactly, not estimated: every `SELL` order is multiplied by the
market price **at the step it was issued**.

| | mean cash | MILK | STRAWBERRY | WOOL | MELON | WHEAT | FERT | **EGG** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Top decile (n=13) | **$129,657** | 32.5% | 18.7% | 16.2% | 13.9% | 9.9% | 7.5% | **0.6%** |
| Top quartile (n=34) | $93,670 | 29.1% | 15.6% | 17.1% | 16.1% | 11.8% | 9.0% | 0.8% |
| **Ours (n=73)** | **$44,152** | 11.8% | 22.2% | 29.4% | 19.2% | 3.4% | 13.2% | 0.8% |
| Bottom quartile (n=34) | $18,958 | 10.0% | 20.3% | 18.1% | 26.8% | 8.6% | 8.8% | 2.1% |

Distribution over all 138 seats: min $1,110, p25 $33,114, median $39,962, p75 $56,902,
max **$187,844**.

**The egg engine earns 0.6–0.8% of revenue at every skill level, including ours.** It is
not a back-half income source, it is a rounding error, and `main.py` gives it the top
claim on capital and builds 22 coops per episode to serve it.

### The gap is throughput, not cleverness

Per farm, ours versus the top decile:

| | ours | top decile | ratio |
| --- | --- | --- | --- |
| final cash | 44,152 | 129,657 | **2.94×** |
| productive unit-ops | 940 | 2,125 | **2.26×** |
| `WATER` ops | 319 | 750 | 2.35× |
| `HARVEST` ops | 130 | 262 | 2.01× |
| `FERTILIZE` ops | **0** | 44 | ∞ |
| wheat bought | 75 | 592 | **7.93×** |
| cows bought | 2.7 | 9.5 | 3.49× |
| land plots bought | 1.6 | 6.1 | 3.79× |
| hires issued | 239 | 293 | 1.23× |
| hands at end | 4.0 | 9.8 | 2.45× |
| `PASS` unit-turns | **1,294** | 705 | 0.55× |
| coops at end | **22.4** | 2.5 | 0.11× |

Cash tracks productive ops almost linearly (2.94× versus 2.26×), and the two cohorts
spend a comparable number of unit-turns on logistics (3,700 versus 4,282). So this is
not a routing problem and not a strategy-subtlety problem: **the frontier converts
roughly twice as many unit-turns into work, on a farm with 2.5× the labour.** Our agent
idles 1,294 unit-turns while holding 4 hands, having built 22 coops for a 0.8% revenue
line. The known "worker PASS rate 23–33%" gap is the same finding measured from inside.

**Wheat: buy it, don't grow it.** The top decile buys 592 wheat per episode and keeps
~1 wheat tile, then sells surplus for 9.9% of revenue. Feeding ~16 animals off owned
tiles costs the land and the unit-turns that the animals themselves need.

**The frontier runs at zero cash.** The top scorer's balance was **$8 on day 4** — every
dollar is converted to capacity immediately. `LAND_CASH_BUFFER = 1961.9` keeps ~$2k idle
against a game where the winning line is fully invested.

### Reading the daily manifest

Alongside the dumps, the manifest tracks the field: median rating climbed 670 → 3,068
between 2026-07-30 and 2026-08-09, but the last three days read 3,028 / 3,068 / 3,068,
and the top-to-median gap narrowed from 483 to 150 while daily episode counts fell
864 → 687. The field has converged. A knob tweak will not move a rank from here; the
2.9× throughput gap will.

### Caveats on the above

- The 73 seats in the table above are **v0.0.5/v0.0.6**, from the 2026-08-05 field.
  v0.0.9 *is* live (submission 55426703, 2026-08-11) and has its own numbers: **10 seats,
  5W-3L, mean $50,957** against a 2026-08-10 field median of $83,606 and a top decile of
  $132,689 — a **2.60×** gap. That comparison, not the one above, is the current
  baseline; see [docs/experiments.md](docs/experiments.md).
- The market findings are day-dependent and were re-measured on 08-10. Price *drift*
  did not survive; the melon/wool collapse, the egg result and the throughput gap did.
- v0.0.9's `MAX_COWS = 9` / `MAX_SHEEP = 6` bracket the field's 9.3 / 4.1, so the CEM
  search found roughly the right livestock. The problem is upstream of the ceilings:
  22 pastures **plus 14.7 coops** consume 36.7 of our 50 tiles, leaving ~13 for crops
  against the field's ~60, and `engine_claim` is what funds the coops first.

---

## Results

v0.0.9, seats alternated each episode, 720 steps, seeded and reproducible.

**These numbers do not predict leaderboard placement.** The same agent family that
reports $73k here averaged $44k across 73 real Kaggle seats, against a frontier of
$188k. `starter` and `adaptive` barely produce, so they neither contest the shared pots
nor force tempo; beating them measures survival, not throughput. Treat this table as a
reliability smoke test.

| Opponent | Eps | Mean cash | Median | Min | Win rate | Crashes | p95 turn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` (`starter`) | 30 | **$73,032** | $72,776 | $46,107 | **100%** (30W 0L) | 0 | 0.30 ms |
| `adaptive` | 30 | **$68,872** | $72,361 | $46,362 | **100%** (30W 0L) | 0 | 0.28 ms |
| `v0.0.6` (previous agent) | 30 | $57,172 | $57,354 | $23,258 | **70.0%** (21W 9L) | 0 | 0.29 ms |
| `mirror` (self-play) | 30 | $54,093 | $50,947 | $25,382 | 0W **27T** 3L | 0 | 0.31 ms |

All acceptance criteria pass on every non-mirror opponent: zero crashes, zero
timeouts, zero invalid statuses, p95 turn compute under 0.1% of the 1s budget (the
gate is 50%), and it beats every opponent on mean cash over 30 episodes.

Self-play resolving into 27 exact ties is the expected result for two identical
deterministic agents on a shared seed. The three losses come from the interpreter
processing atomic orders (`HIRE`, `BUY_LAND`) in player-index order. Self-play cash
is ~25% lower than either scripted matchup, because both sides race for the same
finite melon pot — **that is the more realistic guide to leaderboard scoring** than
the numbers against `baseline`/`adaptive`.

### v0.0.8 replaced this agent with a neural network and lost 26×

Worth stating up front because it is the most expensive lesson in the repo. v0.0.8
dropped `StrategicPlanner`/`SpatialScheduler` entirely and shipped a behaviour-cloned
policy emitting atomic actions, distilled from PyTorch into base64 numpy weights.

| Agent | vs `baseline`, 4 paired seeds | Record |
| --- | --- | --- |
| v0.0.8 (behaviour-cloned atomic actions) | **$2,910** | 0W 4L |
| v0.0.6 (this architecture) | **$75,668** | 4W 0L |

It lost to the environment's own one-farmer starter bot. The cause was structural,
not a training budget problem, and it is worth spelling out because it generalises:

1. **The action vocabulary had no `FEED`, `CARE` or `PICKUP` token.** `BASE_ACTIONS`
   was `PASS/N/S/E/W/WATER/HARVEST/DEMOLISH` plus `PLACE_*`, and the encoder mapped
   anything unknown to class 0 = `PASS`. So every expert `FEED` in the training data
   was labelled `PASS`, and the egg engine — the only unbounded income in the game —
   was **inexpressible** by construction.
2. **The observation was not Markov.** The global vector was
   `[day, hour, money, hands_count]`. No market prices, no glut counters, no shed,
   no animals, no maturity, no `consecutive_unfed`. A market policy cannot be learned
   from a state that does not contain prices.
3. **Labels collided.** Hand actions were written into a 10×10 grid keyed by position,
   but units may share a tile, so co-located hands overwrote each other; and ~84 of
   100 cells were `PASS`, so cross-entropy converged on "PASS everywhere".
4. **The online loop was not a policy gradient.** Rollouts were taken with `argmax`,
   so the policy was deterministic and nothing was sampled from it. `∇log π(a|s)·A`
   requires `a ~ π`; weighting `argmax` actions by an episode-level advantage over 20
   episodes measures seed luck, not policy quality — with no critic, no GAE and no
   baseline over 720 steps of credit.

The whole atomic-action path has been deleted: `agent_torch.py`,
`rl/architecture.py`, `rl/action_space.py`, `rl/dataset_builder.py`,
`rl/train_offline.py`, `rl/train_online.py`, `rl/offline_bc.py`,
`rl/export_to_numpy.py`, `rl/numpy_inference.py`, `rl/distill_to_main.py`,
`scripts/build_submission.py`, `scripts/train_pipeline.py`, the `torch`
dependency, and `opponents/v0_0_7.py` / `opponents/v0_0_8.py` — the two agent
snapshots that carried base64 network weights. The behaviour-cloning datasets,
`.pt` checkpoints and elite-trajectory store under `logs/` went with them
(~600 MB). Everything is recoverable from git history at tags `v0.0.7`/`v0.0.8`.

### Optimal assignment loses to greedy

Layer B is a weighted bipartite matching, and matching has an exact
polynomial-time solution, so "replace greedy with Hungarian" looks like free
money. It is not. `main.py` ships a complete Jonker-Volgenant solver
(`_hungarian_min_cost`, brute-force verified on 300 random matrices) and a
priority-tiered optimal assignment (`SpatialScheduler._assign_optimal`) behind
`FLAGS["HUNGARIAN_ASSIGN"]`, and the flag is **off**:

| Objective | Result |
| --- | --- |
| flat max-value, `priority / (1 + travel)` | **−37.6%** (8 seeds, 0/8) |
| flat max-value, `priority − 8·travel` | **−38.0%** (8 seeds, 0/8) |
| priority-tiered, min total travel inside each tier | **−31.3%** vs `baseline` (p~0.0, 3/30), **−40.4%** vs `adaptive` (p~0.0, 3/30) |

Three things came out of chasing this, and they are the transferable part:

1. **Priorities are deadlines, not utilities.** Any objective that trades
   priority against travel will let a cheap nearby chore outrank a plant that
   dies tonight. Only a lexicographic objective (tier first, travel only inside
   a tier) is even defensible here.
2. **Do not collapse tasks to one per tile.** An animal tile carries `FEED`,
   `HARVEST` and `CARE`. Keeping only the top-priority one drops the fallback:
   when nobody is carrying wheat the `FEED` is unassignable, and the tile must
   stay eligible for its `HARVEST`.
3. **Greedy has an emergent property the matching destroys.** Greedy leaves its
   *leftover* units clustered near the shed, and every animal and every sack of
   feed enters the farm through a shed `PICKUP`. Minimising total travel scatters
   the leftovers to the perimeter, where `_logistics` can only return `PASS`.
   Patching that (staging idle units at the shed) recovered nearly half the loss
   — −44.3% → −20.2% — which confirms the mechanism but does not close the gap.

The lesson is not "optimisation is bad". It is that the objective the greedy rule
implicitly optimises is not the one that was written down, and the written one is
worse.

### Tuning status

Every constant in `main.py` is annotated `TUNED`, `UNPROVEN`, `PLACEHOLDER` or `DERIVED`.
`TUNED` means a paired-seed A/B measured it. What graduated:

| Knob | Result | Evidence |
| --- | --- | --- |
| `HIRE_HANDS` | **+517.7%** | p~0.0, better on 20/20 seeds |
| `EXPAND_LAND` | **+199.1%** | p~0.0, better on 19/20 |
| `PREMIUM_LIVESTOCK` | **+17.8%** | p~0.020, 14/20 |
| `ANIMAL_CARE` | **+31.0%** in self-play; n.s. vs scripted opponents | p~0.021, 22/30 (mirror) |
| the 23-number macro vector | **21W–9L** head-to-head vs v0.0.6 | 30 paired seeds, p~0.021 |

The macro vector is no longer hand-swept one constant at a time; see
"Where learning belongs". The v0.0.9 values came out of a CEM search and were then
validated head-to-head against the version they were derived from.

### Every PvP heuristic tried has lost, and all four were deleted

This is the most useful negative result here, so it is worth stating plainly. The
opponent's **entire farm is public** — money, tiles, hands, quadrants — so the
adversarial features in the environment specification are all *implementable*. Four
have been implemented and A/B'd, and all four measured worse or inert:

| Heuristic | Result | Verdict |
| --- | --- | --- |
| `PRICE_FLOOR_SELLING` — withhold capped goods to defend their price | **−1.8%** in self-play (p~0.00014, better on 6/30) | deleted |
| `PREDATORY_TIMING` — dump ahead of their forecast harvest | **exactly 0 delta** over 80 paired seeds | deleted |
| `OPPONENT_ADAPTIVE` — cede melon land when they contest melon | **−26.9%** vs adaptive (p~0.0, better on **0/30**) | deleted |
| `ENDGAME_POSTURE` — lock in / gamble based on the terminal cash gap | **−10.7%** vs baseline (p~1e-05, 1/30), **−8.2%** vs adaptive (p~0.001, 2/30) | deleted |

The pattern is consistent, and the right explanation is **latency**, not "tempo" in
the hand-waving sense. Every production decision in this game pays out 8–12 days
after it is taken, and the opponent's board is only informative about decisions they
have *already* made. Information whose reaction lag exceeds the horizon over which
it is actionable has value zero, and acting on it costs the tempo spent switching.
Formally: the observation lag is roughly equal to the action's payoff lag, so the
open-loop plan **is** the closed-loop equilibrium.

That predicts exactly where opponent-awareness *could* pay: decisions with no lag.
There are two, and both are now measured. Selling is instantaneous — and
`PREDATORY_TIMING` returned exactly zero delta over 80 seeds, because the agent
already sells everything on sight and there is no timing left to optimise. The
terminal cash comparison is also lag-free in the last week — and `ENDGAME_POSTURE`,
which locked down capital spend when ahead and extended the livestock windows when
behind, lost 10.7%. So the lag argument survives its own strongest test.

Withholding stock is worst of all, because unsold inventory scores $0 and melon's
price never recovers enough for the held units to clear.

`OpponentTracker` is retained (composition, cash, labour, profile label) but it feeds
**the decision log only** — it does not change what the agent does. That is stated in
its docstring so nobody mistakes it for a live input.

### Known gaps

Ordered by measured cost. The first five are all from the leaderboard mining and none
of them is fixed yet.

- **The egg engine is funded first and earns 0.6% of revenue.** `engine_claim` in
  `market_orders` ring-fences cash for coops and goose feed ahead of land and livestock,
  and the agent builds 22.4 coops per episode. The frontier builds 2.5 and buys 9.5 cows
  to our 2.7. This is the single largest misallocation in the file.
- **Zero `FERTILIZE` ops, ever.** The op is legal and present in `UNIT_OPS`;
  `build_tasks` never emits it. Meanwhile the agent *sells* ~190 fertilizer per episode
  for 13.2% of revenue, at $100, instead of applying it to strawberry at $294.
- **Wheat is grown, not bought.** 75 bought per episode against the frontier's 592. The
  owned wheat tiles cost land and unit-turns that the animals need.
- **Land and labour are under-bought.** 1.6 plots and 4.0 end-of-episode hands against
  6.1 and 9.8. `LAND_CASH_BUFFER = 1961.9` holds ~$2k idle in a game whose winning line
  sits at $8 on day 4.
- **Melon is still 19.2% of our revenue.** It is the bottom quartile's signature (26.8%)
  and the only product whose price ends below its peak.
- **Worker PASS rate 23–33%**, which the mining measures as 1,294 idle unit-turns per
  episode against the frontier's 705. Two attempts to close it are already recorded as
  failures (see "Optimal assignment loses to greedy", and `IDLE_PREPOSITION` below), so
  routing is not the lever — but the cohort data says the frontier spends *more*
  unit-turns on logistics than we do, not fewer, so the leak is not travel. It has 2.5×
  the labour and enough productive work to give it.
- **No frontier sparring partner exists.** `starter` and `adaptive` do not contest pots,
  self-play only measures the agent against its own blind spots, and a leaderboard
  replay goes bankrupt by day 12 (see "Opponents"). Every A/B in this file was measured
  against opponents that produce a fraction of what the leaderboard produces.
- `IDLE_PREPOSITION` — walking otherwise-idle units toward the shed instead of
  `PASS` — looked promising at 8 seeds (+3.2%, better on 7/8) and then measured
  **−16.4%** vs baseline (p~4e-05, 4/30) and **−8.7%** vs adaptive (p~0.015, 11/30)
  at 30. Deleted. Filed here mostly as a reminder that 8 seeds is not a measurement.
- **Tail.** The $2,244 collapse reported for v0.0.5 is gone: over 24 paired
  seat/seed evaluations the minimum is $36,840 and CVaR@25% is $45,354
  (`python -m search.cem --diagnose 40`). Self-play min is $25,382, which is
  the number to watch.
- `FERRY_MAX_UNITS`, `ANIMAL_BACKLOG_CAP` and `WHEAT_CARRY_PER_UNIT` are still
  `PLACEHOLDER` — reasoned, not swept, and not in the searched vector.

---

## Robust route mining (v0.2.7)

The route-replay agent scores well and varies wildly. A fixed 719-step route banks $156k
on a favourable seed and collapses on an unfavourable one, because shops respawn every 3
days and production decisions pay off 8–12 days later. By the time the market tells you
which route was right, it is far too late to change it. Reacting mid-game would cost the
throughput that makes the route-replay approach work at all.

So the stochasticity is handled offline instead of at runtime: mine every route the
leaderboard has already played, stress-test each one against a panel of strong opponents
across thousands of simulated markets, and ship the route that wins most reliably.

Five scripts, run in order. **The full runbook, including the daily replay download, is
under "Re-running the pipeline" below.**

```bash
uv run python scripts/fetch_team_ranks.py --refresh  # -> logs/team_ranks.json
uv run python mine_replays.py --workers 12           # -> candidates.jsonl
uv run python simulate_candidates.py --workers 12 \
    --panel-rank-min 185 --panel-team-top 469        # -> logs/simulation_results.jsonl
uv run python rank_cvar.py --workers 12              # -> logs/cvar_report.json
uv run python encode_submission.py --write-agent main.py --version 0.2.7
```

`encode_submission.py` refuses to emit unless Phase 3 recorded that the winner beat the
incumbent held out (override with `--allow-unvalidated`). `simulate_candidates.py`
supports `--resume`, keyed per `(candidate, opponent, seed)`.

> **Selection metric changed in v0.2.6.** v0.2.5 was selected on own-cash CVaR₅ against a
> single opponent, and that choice did not survive contact with the ladder — see
> "Live outcome" below. Selection is now **mean win rate across an opponent panel**, with
> the worst-opponent win rate as the robustness tiebreak. Cash CVaR₅ and margin CVaR₅ are
> still computed and reported, as diagnostics.

### The metric: panel win rate, with CVaR₅ as a diagnostic

Selection is **mean win rate across the opponent panel**, tie-broken by the win rate
against the single panel member that counters the candidate best. The competition scores a
skill rating driven by wins, and cash turns out to be a poor proxy for them — see "Live
outcome".

Where CVaR₅ still appears, it means **the mean of the worst 5% of outcomes**, not the score
*at* the 5th percentile. The percentile tells you where the boundary is; CVaR tells you how
bad things are past it — it separates "bad seeds earn $80k" from "bad seeds earn $20k".
Two are reported: `cash CVaR₅` (own cash, the v0.2.5 selection metric) and `margin CVaR₅`
(the head-to-head margin, which removes the common-mode market component). Margin CVaR₅ is
typically slightly negative even for a strong route: on the worst 5% of seeds it loses.

### Method

- **Threshold $85k, deliberately modest.** A route that banked $110k did so partly on a
  lucky seed, so a high threshold selects *for* luck-dependence — the opposite of the
  goal. See the finding below; this was not a hedge, it decided the outcome.
- **Fidelity gate.** Every candidate's original episode is reconstructed closed-loop on
  its recovered seed (see [docs/replay_schema.md](docs/replay_schema.md)) with both seats
  replaying verbatim, and must reproduce the recorded rewards **exactly**. This catches
  extraction and format bugs before they poison everything downstream.
- **Common random numbers, mandatory.** Every candidate is scored on the *identical* seed
  list. Without paired seeds a CVaR comparison between two routes is mostly a comparison
  of their seed luck. The sets are nested — screen (12) ⊂ mid (50) ⊂ final (500) — so each
  stage reuses the episodes its predecessor paid for; the holdout (500) is disjoint.
- **Three-stage sieve.** Deduplication recovers only ~1.4% of this corpus, so the pool
  stays near its full size and a flat screen at final resolution would cost days. All
  candidates are screened on 12 seeds against the anchor, then the top 150 on 30 seeds
  against the panel, then the top 12 on 100 seeds. Sample size is seeds x panel, so the
  final stage is 600 games per candidate (win-rate standard error ~2%).
- **Opponent panel, not one opponent.** Mid and final stages score against 6 structurally
  different strong routes chosen by greedy max-min action distance and biased toward
  distinct teams (`mining/panel.py`), anchored on the incumbent. Self-matches are excluded
  from a candidate's own aggregate, since a panel member can only draw itself.
- **Panel drawn from a rating band, not from the whole field** (`--panel-source`, new in
  v0.2.7). See "Aiming the panel at the ladder" below; `screen-top` reproduces v0.2.6.
- **What is scored is what ships:** the route baked into the real agent template, so the
  three runtime layers (WEED repair, SELL-slot ordering, hands alignment) are in play.

### Results — v0.2.7 (leaderboard-band panel, 7-day corpus)

Same corpus as v0.2.6, re-mined so every candidate carries its team's ladder rank:
4,790 replays (140 GB, 2026-08-08 → 08-14) → 2,541 files with a qualifying seat →
**4,315 unique candidates, all passing the fidelity gate exactly, zero exclusions, zero
rejects**. 4,157 of them joined a team on the 2026-08-17 leaderboard (245 teams). Then
79,980 sieve episodes plus 1,000 held out, with **zero** crashes, timeouts, invalid
statuses or harness errors.

Panel: ranks 185–469 (2300.2–2599.1 rating), 635 eligible candidates from 42 teams, five
mined members spanning five distinct teams at `min-dist-to-earlier` 0.99–1.00. No
saturation warning on the mid or final stage.

Winner: **`044a7741e9`, team "Ueddy", episode 93065370 seat 1**, recorded $89,408.

| | mean win | worst opp | margin CVaR₅ | cash CVaR₅ | cash mean |
| --- | --- | --- | --- | --- | --- |
| **Winner, vs 6-opponent panel** | **93.8%** | **77.0%** | −$8,881 | $46,291 | $93,286 |
| Winner, held out vs 5 common opponents | 93.2% | 79.0% | −$9,533 | — | $90,702 |
| v0.2.6 incumbent, same held-out grid | 83.6% | 51.0% | −$9,849 | — | $90,013 |
| **Winner, head-to-head vs v0.2.6, held out** | **96.0%** | — | — | — | $87,019 |

Winner's-curse shrinkage was again negligible: selection 93.8% → held-out 93.2%.

**The incumbent's panel score is now a real number, and that is the headline.** In the
v0.2.6 run Phase 3 reported the incumbent at **0.2%** against its own panel — an artifact,
because every member had been selected for beating it. On a panel drawn from the ladder
instead, the incumbent scores **83.6%**. The measurement that was rigged is no longer
rigged, which is what this version was for.

**But 92% of the improvement is one opponent.** The +9.6% mean delta decomposes as:

| panel member | team (rank) | winner | incumbent | delta |
| --- | --- | --- | --- | --- |
| `ebfc911eaa` | lllleeeo (435) | 95.0% | 51.0% | **+44** |
| `800dc80f5c` | Eishkaran Singh (222) | 93.0% | 91.0% | +2 |
| `62b81aa8a3` | Tom3 (195) | 99.0% | 98.0% | +1 |
| `b8b9267d1c` | HealthStone (204) | 100.0% | 100.0% | 0 |
| `8f7dd57d5f` | researchstudio.site (466) | 79.0% | 78.0% | +1 |

44 of 48 points come from `ebfc911eaa`; against the other four the winner is within two
points of the incumbent. The runbook's rule is "a winner strong against five members and
weak against one is an exploit" — this is the mirror image and carries the same risk. The
honest expectation is *a wash plus one favourable matchup*, not a 9.6% broad gain.

**Every finalist shares a worst opponent.** All 12 finalists lose most often to
`8f7dd57d5f` (researchstudio.site, rank 466), at 77–78%. That is not one candidate's
exploit — it is a systematic weakness of the strategy class the whole corpus contains, and
no route in 4,315 fixes it. This is the production gap [#31](https://github.com/ergousha/kaggriculture/issues/31)
describes, visible from the other side.

**`correlation(recorded cash, mean win rate) = −0.54`** across the finalists — the same
finding as below, but far stronger on this panel than the −0.05 measured against the old
one. The winner banked $89,408, well below the pool's $103,991 median.

<details>
<summary>Previous run — v0.2.6, panel selected against the incumbent (superseded)</summary>


4,791 replays (140 GB, 2026-08-08 → 08-14) → 2,541 files with a qualifying seat →
**4,315 unique candidates, all passing the fidelity gate exactly, zero exclusions, zero
rejects**. Then 82,020 sieve episodes plus 800 held-out, with zero crashes, timeouts,
invalid statuses or harness errors.

Winner: **`18057e3167`, team "somewhere after", episode 93034871 seat 0**, recorded
$154,009.

| | mean win | worst opp | margin CVaR₅ | cash CVaR₅ | cash mean |
| --- | --- | --- | --- | --- | --- |
| **Winner, vs 6-opponent panel** | **92.6%** | **84.0%** | −$1,328 | $51,376 | $92,766 |
| Winner, held out vs 4 common opponents | 93.0% | 83.0% | −$1,094 | — | $94,204 |
| **Winner, head-to-head vs v0.2.5, held out** | **96.0%** | — | −$923 | $46,776 | $94,105 |
| v0.2.5 incumbent, same head-to-head | 4.0% | — | — | $40,887 | $84,805 |

Winner's-curse shrinkage was negligible this time: selection 92.6% → held-out 93.0%.

**Read the panel comparison with care.** Phase 3 reports the incumbent at 0.2% against the
panel, which is an artifact, not a measurement: the panel is drawn from the top screen
performers, and the screen ranks by win rate against the incumbent — so every panel member
beats it ~100% by construction. The honest number is the direct head-to-head, **96/100 on
held-out seeds with a +$9,300 mean margin**. Always run that separately; the runbook
includes it as a step.

The worst-opponent column earns its place. Finalist #4 averages 85.7% but wins only
**50.0%** against one panel member, and #8 averages 78.5% with a **10.0%** worst case —
both single-opponent exploits that a mean-only ranking would have promoted.

</details>

<details>
<summary>Previous run — v0.2.5, cash-CVaR₅ selection (superseded)</summary>

3,413 replays (100 GB) → 3,089 candidates, all passing the fidelity gate. 61,268
paired-seed episodes.

| | CVaR₅ | mean | median | p5 | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| Winner (Hamed Seyed-allaei, ep 92449798 seat 0) | $44,655 | $87,997 | $85,209 | $50,881 | $34,522 | $154,606 |
| v0.2.4 baseline | $43,045 | $84,914 | $81,406 | $48,563 | $31,938 | $148,251 |

Held out on 500 fresh seeds; delta +$1,610 CVaR₅, ahead on 275/500 paired seeds, p≈0.014.
This is the run whose metric failed to transfer — see "Live outcome" below.

</details>

### Aiming the panel at the ladder (v0.2.7)

The v0.2.6 panel had two selection biases stacked on top of each other, and the second was
the larger.

**One:** members were drawn from the top `--panel-from-top` *screen* performers, and the
screen ranks by win rate against the incumbent anchor. So every member beat the incumbent
~100% by construction, and the panel measured "routes that counter *us*".

**Two, and worse:** the pool those members came from is whoever happened to appear in the
daily replay dumps — a sample of the whole field weighted by episode volume, not by
strength. We are matched by rating (~2290 at the time), so the pipeline was optimising win
rate against the *median of the field* while the ladder pays for beating the band above
us. The transfer gap had been measured twice by then and both times pointed the same way:

| run | offline | live |
| --- | --- | --- |
| v0.2.5, single opponent | 97.0% | ~47% |
| v0.2.6, 6-opponent panel | 92.6% mean / 84.0% worst | plateau at ~2290 |

Widening the panel narrowed the gap; aiming it should narrow it further.

**The join.** `scripts/fetch_team_ranks.py` writes `logs/team_ranks.json` from the public
leaderboard CSV, Phase 1 stamps `team_rank` onto every candidate, and Phase 2's new
`--panel-source leaderboard-top` draws members only from candidates whose team sits inside
`--panel-rank-min .. --panel-team-top`. Selection *within* that window is unchanged: the
same greedy max-min action-distance diversity and distinct-team bias, still anchored on the
incumbent so results stay comparable. Team names are unique on the leaderboard and the
replay's `info.TeamNames` carries the same string, so name is a sound key — Kaggle does not
put team ids in a replay.

Nothing new had to be mined. The existing corpus already spans 255 teams, and every band
worth aiming at is deep enough to build a panel from:

| band | ranks | candidates | teams |
| --- | --- | --- | --- |
| ≥2846 | 1–30 | 479 | 12 |
| 2600–2900 | 20–184 | ~1,000 | ~50 |
| **2300–2600** | **185–469** | **635** | **42** |

**Why a band and not the top.** The obvious reading — aim at the leaders — is the one the
data rejects. [#22](https://github.com/ergousha/kaggriculture/issues/22) scored every
opponent the top 5 teams played and found rank-5 `peikopon`, at 2987, matches *nobody*
above 3000: isolation into the >3000 pool is a **consequence** of crossing 3000, not the
route to it. Selecting against that meta would optimise for games we are not matched into
while discarding the opponents that actually set our score. So the panel is drawn from
**2300–2600** — ranks 185–469, i.e. from our own rating (2294, rank 476) up to ~300 points
above it. The window is meant to advance as we climb; `--panel-rank-min` is the knob.

**Two limits, both real.** The rank is a snapshot of *today*, while a mined replay was
played days earlier by whatever agent that team was running then — a team that has climbed
since gets credit its mined route did not earn. The snapshot filename is recorded in
`logs/team_ranks.json` and echoed by Phase 1 so any claim can be dated. And the mid-stage
*pool* (top `--k-mid` by screen) is still cut by win rate against the incumbent, so a route
that beats the band but loses to the incumbent never reaches the panel at all. That cut is
what makes a 4,315-route pool affordable; fixing it means screening against the panel,
which costs six times the compute.

### Finding: recorded cash does not predict robustness

Across the 20 finalists, `correlation(recorded_cash, CVaR₅) = **-0.05**` — essentially
zero, and slightly negative. The most robust route in the entire 100 GB corpus banked
**$89,802** in its own episode. Finalist #4 banked $85,542, barely clearing the threshold.
**Filtering the pool at an "elite" $110k would have discarded the winner.**

The mechanism is visible directly in the corpus: one byte-identical 719-step trace appears
twice, banking **$96,946** on one seed and **$131,597** on another — a 36% spread from
seed alone. Ranking mined routes by the cash they happened to bank is ranking their luck.

A corollary, stated because it is a real limit: the $85k threshold is *itself* seed-noisy,
so it drops robust routes that drew a bad seed. The pool is a lower bound on what the
corpus contains. Fixing that would mean simulating everything.

### Finding: the winner's curse is bigger than the win

The selected route's CVaR₅ was $48,295 on the evaluation seeds and **$44,655** on the
held-out set — a **$3,640 shrinkage, larger than its entire $1,610 margin** over v0.2.4.
Picking the maximum of ~3,100 noisy estimates overfits the seed set it was chosen on.
Without the held-out re-run this section would be claiming a ~$5k gain that does not
exist. Any future route selection must re-validate on fresh seeds; the ranking table is
not the result.

### Live outcome: the metric did not transfer, and why

v0.2.5 ran 177 live episodes between 2026-08-13 and 2026-08-15. The caveat above —
"the gain is modest and may not transfer" — is now measured rather than hypothetical.

| | v0.2.5 (CVaR-selected) | v0.2.4 | |
| --- | --- | --- | --- |
| Mean cash | $93,257 | $89,157 | +4.6% |
| **Live CVaR₅** | **$46,755** | **$48,208** | **−$1,453** |
| Win rate, overall | 67.2% | 42.5% | |
| Win rate, last 25% of games | **46.7%** | **16.0%** | |
| Crashes / errors | 0 | 0 | |

**The route is better; the metric we selected it on is not.** Offline we predicted
+$1,610 CVaR₅ and delivered **−$1,453**. The improvement showed up entirely in win
rate — which the pipeline never optimised. Note also that raw leaderboard score is a
trap here: both submissions climb then settle as matchmaking finds their level, and
v0.2.4's headline 2220 is a *falling* number attached to an agent losing 84% of its
recent games. Judge a submission on its converged win rate, not its score at an
arbitrary age.

Two measured causes:

This repo already knew. "[The objective is P(win), not E[cash]](#the-objective-is-pwin-not-ecash)"
has been in this README since before the pipeline was written, and the pipeline optimised
E-tail[cash] anyway. The measurement below is what that cost.

**Cash is 86% common-mode.** On live episodes `corr(our cash, opponent cash) = +0.86`,
because both seats draw from one shared market. Own-cash CVaR therefore mostly measures
*was this a good seed* — a factor that moves both players together and cancels in the
head-to-head that sets the rating. Our own cash barely predicts winning: correlation
with margin is only +0.31, and having above-median cash raises the win rate from 61.4%
to just 73.0%. The competition scores P(win); we optimised E-tail[cash].

**Single-opponent overfitting.** The shipped route beat its one evaluation opponent
**97.0%** of the time offline and wins ~47% live. v0.2.4 was not a weak opponent — the
*median* mined candidate beats it 0% of the time, and only 8.4% of the pool beats it
≥90%. Selecting from that tail selects for routes that exploit one opponent's specific
market timing on a shared order book. At 97% the metric is also saturated: it cannot
rank the finalists at all, so among the leaders the pick was effectively arbitrary.

Corpus staleness was *not* a factor — median replay cash drifted only +4.6% across the
five days mined.

**What changed as a result.** Phase 2 now scores every candidate against a diverse
opponent **panel** (`mining/panel.py`) instead of one route, and Phase 3 selects on
**mean win rate across the panel** with the worst-opponent win rate as the robustness
tiebreak. Cash CVaR₅ and margin CVaR₅ are still reported, as diagnostics. Both phases
print a saturation warning if the leaders exceed 95%, because that is the condition that
produced this result. Self-matches are excluded from a candidate's own aggregate, and
the held-out winner-vs-incumbent comparison is run on the panel members common to both.

The panel's effect is immediate and visible: against one opponent the leaders sat at
100.0% (unrankable); against a 4-opponent panel the leader fell to 80.8% with a
worst-case of 23.3%, and one candidate averaging 75.8% turned out to win just **3.3%**
against a single member — an exploit the old metric would have shipped.

### Caveats

- **Offline win rates are still optimistic.** v0.2.5 read 97% offline and delivered ~47%
  live; v0.2.6 read 92.6%/84.0% and plateaued at ~2290. v0.2.7 reads 93.8% mean / 77.0%
  worst against a panel drawn from our own rating band, which is the best-conditioned
  measurement this pipeline has produced — but it is still six opponents standing in for a
  live field of hundreds. Expect the live figure below the offline one.
- **The panel used to be selected against the incumbent — fixed in v0.2.7, and measured.**
  Under `--panel-source screen-top` members come from the top screen performers, and the
  screen ranks by win rate against the incumbent anchor, so all six beat it ~100% by
  construction: v0.2.6's Phase 3 scored the incumbent at **0.2%** against its own panel.
  On the v0.2.7 ladder-band panel the incumbent scores **83.6%**. Run the direct
  head-to-head anyway — it is the cleanest read and the runbook has it as a step.
- **A panel of six cannot tell you which of its members the live field contains.** v0.2.7's
  entire +9.6% edge over the incumbent sits on one of the five (`ebfc911eaa`, +44 points;
  the rest are within two). A panel makes single-opponent exploits *visible*; it does not
  stop the selected route from having one.
- **It does not fix the tail, and the tail against our own band is deep.** Margin CVaR₅ is
  **−$9,533** held out — on the worst 5% of seeds the winner loses by ~$9.5k. That is far
  worse than v0.2.6's −$923, and it is not a regression: the old figure was measured
  against a panel of routes selected for losing to the incumbent. −$9.5k is what the tail
  costs against the band we are actually matched into. Cash CVaR₅ (~$46k against a ~$91k
  mean) is still a wide distribution. Mining finds more robust routes, not robust ones.
- **Seed distribution is an assumption.** Local seeds are sequential integers; Kaggle
  assigns each episode a seed we cannot observe or reproduce. All market randomness comes
  from `random.Random((seed * 1_000_003) ^ day)`, so these seeds do span the shop-draw
  space, but nothing here can verify the distributions match. `--seed-mode random31`
  samples the same 31-bit range the engine's own fallback uses, as a second read.
- **Finalist concentration got worse, and the failure mode is now visible.** v0.2.7's 12
  finalists span **2 teams** (Kostiantyn Isaienkov 8, Ueddy 4), against 5 teams in v0.2.6
  and 18-of-20 from one team in v0.2.5, and their win rates sit within a 2.5% spread — the
  selection is choosing between near-identical routes. The shared failure mode is no longer
  hypothetical: all 12 lose most often to the *same* panel member, at 77–78%.
- **Local results have not historically predicted the ladder.** Four consecutive versions
  won their local paired-seed gates and moved the live score by nothing. Treat every number
  above as a veto, not a forecast.

### Seat symmetry

A trace mined from seat 1 replays correctly from seat 0. Verified twice: swapping both
seats' traces swapped their scores exactly (`[54528, 52963]` → `[52963, 54528]`), and
episode 91605633 was a natural mirror in which both seats played a byte-identical trace
and both scored exactly $155,241. So mining need not preserve seat assignment — though
`candidates.jsonl` records it anyway, since you need it to pair a trace with its reward.

This does **not** mean the seats are independent: `town.unlocked_shops` and
`market["inventory"]` are shared state, so the opponent genuinely perturbs the economy and
must be held constant across any CVaR comparison.

### Re-running the pipeline

Full cycle: pull the new daily replay dumps, re-mine, re-sieve, re-validate, ship. Budget
**~8 h wall on 12 workers**, almost all of it Phase 2. Every step is resumable and every
gate is a hard stop — if one fails, do not proceed to the next.

**0. Pull the new days.** Each daily dump is a ~450 MB zip that unpacks to ~20 GB of JSON
(~690 episodes). Datasets are named `kaggle/kaggriculture-episodes-YYYY-MM-DD` and appear
a day in arrears. Unpack each into its own `replays/<date>/` directory — `mine_replays.py`
searches recursively and does not care how the days are split.

```bash
# one day; repeat per date, or loop. Skips itself if the directory already looks full.
uv run python - <<'PY'
import os
from submit import load_credentials; load_credentials()
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
for day in ("2026-08-15", "2026-08-16"):          # <- edit
    dest = f"replays/{day}"; os.makedirs(dest, exist_ok=True)
    if len([f for f in os.listdir(dest) if f.endswith(".json")]) > 500:
        print(f"{day}: already present, skipping"); continue
    api.dataset_download_files(f"kaggle/kaggriculture-episodes-{day}", path=dest, unzip=True)
PY
du -sh replays/*        # sanity: ~20 GB and ~690 files per day
```

Watch disk: seven days is ~140 GB. Old days can be deleted once mined — `candidates.jsonl`
carries the compressed traces, so the pool survives without the raw replays.

**0b. Refresh the ladder ranks** (~10 s). Phase 1 stamps each candidate with its team's
rank, and Phase 2 builds the opponent panel from it, so a stale snapshot aims the panel at
last week's field:

```bash
uv run python scripts/fetch_team_ranks.py --refresh     # -> logs/team_ranks.json
```

**1. Mine** (~12 min for 7 days / 4,800 replays):

```bash
uv run python mine_replays.py --workers 12 2>&1 | tee logs/phase1.log
```

*Gate:* the fidelity line must read `N admitted, 0 excluded`, and `logs/mine_rejects.jsonl`
must be empty. Anything else means extraction broke on the new days — stop and inspect.
Also read the `ladder join` block: it prints how many candidates the top-10/30/100 bands
contain, and the panel can only be as good as that.

**2. Sieve** (~7 h; screen is ~85% of it):

```bash
nohup uv run python simulate_candidates.py --workers 12 > logs/phase2.log 2>&1 &
tail -f logs/phase2.log
```

The v0.2.7 run aimed the panel at the 2300–2600 band, which on that day's snapshot was
ranks 185–469 — re-derive the bounds from the current leaderboard rather than reusing
these, and advance the window as we climb:

```bash
nohup uv run python simulate_candidates.py --workers 12 \
    --panel-rank-min 185 --panel-team-top 469 > logs/phase2.log 2>&1 &
```

Pass `--panel-source screen-top` to reproduce the v0.2.6 selection instead. Everything
that cannot build a panel is checked *before* the six-hour screen, not after it, and the
screen is panel-independent — so changing the band mid-run costs nothing if you `--resume`.

*Watch two things.* The panel printout after the screen stage — members should span
distinct teams with `min-dist-to-earlier` ≳ 0.3, which the run now warns about itself
(`grep '!!' logs/phase2.log`). And `!! SATURATED` on the **mid or final** stage, meaning
the leaders all exceed 95% and the metric cannot rank them. (Saturation on the *screen*
stage is expected and harmless — it runs against the single anchor by design.) If it fires
later, widen and resume:

```bash
uv run python simulate_candidates.py --workers 12 --resume --panel-size 10 --panel-team-top 100
```

**3. Rank and validate** (~5 min):

```bash
uv run python rank_cvar.py --workers 12 2>&1 | tee logs/phase3.log
```

*Gate:* `VERDICT: PASS`, exit code 0. Read the **per-opponent** breakdown, not the mean —
a winner strong against most of the panel and weak against one member is an exploit, and
the live field contains that member.

*Also run the direct head-to-head against the incumbent.* The panel is built from routes
that beat the anchor, so the incumbent's panel score is rigged against it and its headline
delta is inflated. The honest number is one-on-one on held-out seeds:

```bash
uv run python local_arena.py --agent logs/_mined_agents/<winner-hash>.py \
    --opponent opponents/<incumbent>.py --episodes 100 --seed 1000500 --workers 12
```

**4. Encode and ship:**

```bash
uv run python encode_submission.py --write-agent main.py --version 0.2.7
uv run python -m ruff format main.py && uv run python -m ruff check main.py
uv run python scripts/rank_ladder.py --episodes 1 --require-perfect    # gate: 10/10
uv run python scripts/sync_opponent.py                                 # freeze opponents/v0_2_7.py
uv run python submit.py --dry-run                                      # then without --dry-run
```

**5. Judge the result honestly, two days later.** Do *not* read the raw leaderboard score:
a submission climbs and then settles as matchmaking finds its level, so a fresh score is
meaningless and a stale one can be a falling number attached to a collapsing agent. Compare
converged win rates instead — the last quartile of each submission's episodes:

```bash
uv run python scripts/examine_agent.py "Automated release v0.2.7" --limit 10
```

`.gitignore` already excludes `replays/`, `logs/` and `candidates.jsonl`; nothing from a
run needs committing except `main.py`, the new `opponents/vX_Y_Z.py`, and whatever you
learn.

---

## Route synthesis (issue #26): searching in action space

Selection over the mined pool is exhausted — every candidate is the same strategy sampled
4,315 times. `search/route_search.py` is the harness for the only direction left:
**mutating a route directly**, seeded from the shipped incumbent. It is what issues
#27–#30 block on. It ships no agent; the gate is that with zero mutations it changes
nothing.

```bash
uv run python -m search.route_search --self-test     # the four no-change gates
uv run python -m search.route_search --iterations 8 --workers 12   # a real pass
```

How it works: load the incumbent route (hash-verified against the candidate pool), apply
one of six individually toggleable mutation operators, bake the mutant into the **exact
deployable artifact** (`mining.common.write_route_agent`, so WEED repair / SELL-slot
ordering / hands alignment are in play), evaluate it through the Phase 2 engine
(`simulate_candidates.run_stage` — common random numbers, the v0.2.7 leaderboard-band
panel anchored on the incumbent, resumable per `(hash, opponent, seed)`), and accept on
**mean panel win rate with worst-opponent win rate as tiebreak** — the same metric
`rank_cvar.py` selects on. A hash already evaluated is never re-run.

The six operators, each `--no-<name>` away from off, matching the issue's list:

| operator | what it does | safe by construction? |
| --- | --- | --- |
| `shift_task_block` | shift a unit's movement run by ±k steps, re-aligning the tail | **yes** — only moves a movement burst into surrounding PASSes; preserves every non-move op |
| `retarget_plant` | retarget a `PLANT` to another crop (#28) | half — rewrites the matching `BUY_SEED` so buy/plant stay consistent |
| `swap_herd` | convert a `BUY_ANIMAL COW` to `SHEEP` (#27) | no — cadence repair (interval 3→2) is #27's operator, deliberately not approximated here |
| `assign_idle` | give a PASS turn a productive task (#28) | half — gated to units already adjacent to farmed ground, so no unit travels |
| `repath` | re-path a movement run between fixed endpoints (#29) | placeholder — returns `None` until #29's shortest-path operator lands |
| `move_sell_and_buy` | move a `SELL` and the `BUY` it funds together (#30) | placeholder — returns `None` until #30's joint operator lands |

Measured budget on 12 workers: one candidate evaluation is **30 seeds × 6 panel = 180
episodes ≈ 35 min** at ~2.4 s/episode (Phase 2 screen measured 0.8 ep/s on 8 workers).
That is the number #27–#30 scope against. A mutation that produces an invalid action is
rejected and counted (`rejected_invalid`); empirically the seed's movement-shift at the
final step replays **bit-identical cash on all six panel opponents**, confirming that
shift is a true no-op where it claims to be.

The gates (`--self-test`, also pinned in `tests/test_route_search.py`):

1. **Zero mutations → byte-identical route.** Baking the normalized seed round-trips to
   the incumbent's hash (`044a7741e9`).
2. **A mutation's no-op property holds.** The shift operator preserves the full
   non-movement op signature (every non-move unit op and every market order survives in
   place), so moving a walk cannot disturb the schedule it walks between.
3. **The identity artifact replays clean** through `local_arena` — zero invalid, zero
   crashes, zero timeouts (skipped loudly if `kaggle_environments` is absent).
4. **The budget is printed**, so the dependent issues can be scoped.

**Live validation: none.** This ships no agent. The carried-forward warning applies to
everything it will ever emit: **local results have not historically predicted the ladder**
— four consecutive selection passes won their local gates and moved the live score by
nothing. Every accept from this loop is a veto, not a forecast; hold out fresh seeds
(the v0.2.5 run measured a $3,640 winner's-curse shrinkage against a $1,610 margin).

---

## What game theory actually says about this game

Worth writing down, because the obvious textbook models give the right advice for
the wrong reasons, and the wrong reasons predict the wrong next experiment.

> **Correction from the leaderboard data.** The premise underneath this whole section —
> that the glut counter is cumulative and monotone — is false for every product except
> melon. Inventory starts at 10,000 and town demand drains it, so price *recovers* and
> rises across the episode. The conclusions below mostly survive, but for different
> reasons than the ones given, and the differences change what to try next. Corrections
> are inline.

**It is not Cournot.** Cournot has firms choosing quantities each period against a
price that depends on *current* total supply, and its equilibrium involves restraint.
Here the glut counter is **cumulative and monotone** — price is a stock you deplete,
not a flow you influence. The correct model is common-pool extraction (Hotelling with
rivalry): the pot goes to whoever draws it down first, restraint is strictly
dominated because the rival simply takes what you leave, and the finite horizon adds
a *second*, independent reason not to withhold (unsold stock scores $0). That matches
`PRICE_FLOOR_SELLING` measuring −1.8%.

> **Wrong for the right answer.** The glut counter is *not* monotone: town demand
> replenishes it, so the market is much closer to Cournot-with-recovery than to
> common-pool extraction. Restraint is still dominated, but only because unsold stock
> scores $0 — the "rival takes what you leave" half does not hold, since what you leave
> is largely restored. Melon is the one product where the extraction model is accurate,
> because the field's dump rate exceeds the drain rate. The practical difference: since
> prices *rise*, production should be back-half weighted, which the "deplete it first"
> model actively argues against.

**It is not Chicken.** Chicken is anti-coordination: mutual aggression is the
catastrophe. Here, if both players plant melon nobody crashes — the ~$26.5k pot is
simply split. The right model is a **Tullock contest**: your share is roughly your
share of production, so over a wide region the best response to more opponent effort
is *more* effort, not less. That is why `OPPONENT_ADAPTIVE` — ceding melon land when
contested — measured −26.9% on 0/30 seeds.

**Eggs are not a "dominant strategy".** Dominance is a property of strategies, not of
products. The precise statement is stronger and more useful: because EGG's glut curve
is `log` with target 0.20, its marginal revenue is nearly independent of *total*
supply, so the two players' payoffs are **separable** in the egg dimension. The egg
sub-game has no strategic interaction at all — it is a single-agent MDP wearing a
Markov-game costume, which is exactly why it can be optimised without modelling the
opponent.

> **True and irrelevant.** The separability argument is correct and it is why the egg
> sub-game looked so attractive: a clean single-agent MDP is a much nicer object than a
> contested pot. But the same flat curve that makes egg strategically inert also makes
> it *poor* — it drifts $50 → $69 while milk goes $160 → $329. Egg is 0.6% of frontier
> revenue. Tractability was mistaken for value, and the agent's capital ladder was built
> around the most analysable line rather than the most profitable one.

**Why playing deaf is correct** is a latency argument, not a tempo slogan: see the
PvP table above. Information whose reaction lag exceeds the horizon over which it is
actionable has value zero, so the open-loop plan *is* the closed-loop equilibrium.
The useful part of that framing is that it makes a falsifiable prediction — that
opponent-awareness can only pay on zero-lag decisions — and both zero-lag decisions
in this game (sell timing, endgame posture) have now been tested and both failed.

> **The same latency argument applies to market adaptation.** Live replays of `v0.2.4`
> (the 719-step route-replay agent) across multiple losses (e.g. against `gisgisgis`,
> `tyz123456`, `Raiden.B`) showed the route executing flawlessly every time—even correctly
> triggering the WEED repair logic without desyncing—but still losing. The gap is
> entirely due to the random shop draw: the route was recorded on a seed where it earned
> $156k, but under adverse live shop spawns, those mathematically identical actions only
> yielded ~$73k–$79k. Adapting to the market mid-game is practically unfeasible because
> the 8–12 day production lag exceeds the 3-day shop spawn rate. Reverting to a dynamic
> planner to chase shops would just resurrect the ~2.5x throughput gap the static route
> closed. The optimal strategy remains an open-loop plan, but ideally one less sensitive
> to shop draws (like selling fertilizer) or with even higher baseline throughput.


**Where the textbook framing was actually load-bearing** is none of the above; it is
the scoring rule. Pairwise ranking means the objective is P(win), not E[cash], and
that changed a real decision: it is why `search/objective.py` scores CVaR of the margin,
and why the parameter set that ships is the one that won a head-to-head rather than
the one with the best mean.

---

## Where learning belongs

The game decomposes cleanly, and the two halves want completely different tools:

| | Layer A (macro) | Layer B (micro) |
| --- | --- | --- |
| Decision | what to produce, when to buy, how much labour | which unit does which task |
| Size | ~23 numbers, changes on a daily timescale | 17 units × ~50 tasks, every turn |
| Strategic content | all of it | none |
| Known algorithm | none | weighted bipartite matching, exact, polynomial |

So: **learning goes where there is no closed form.** Cloning atomic actions puts a
noisy approximator on top of a problem that has an exact solution and throws away
the economics; that is v0.0.8, and it cost 26×. *AlphaStar Unplugged* works because
StarCraft's micro layer has no exact solution, the observation is complete, and the
data is ~10⁶ games. None of those three hold here.

### The objective is P(win), not E[cash]

The competition ranks agents pairwise, so a $1 win and a $50,000 win score the same.
`search/objective.py` therefore scores a smooth P(win) surrogate plus a CVaR term — the
mean of the worst 25% of seeds — so that fixing a collapsing seed is worth more than
improving an already-won one.

**What the tail is measured on turned out to matter more than the optimiser.** The
first version took CVaR of *own cash*. Against a weak scripted opponent every seed is
won by a mile, so own-cash variance is noise; the search duly bought tail safety with
production, raised the worst seed by 6.5% — and the resulting agent lost **8W–22L**
head-to-head against the very strategy it was derived from. Scoring CVaR of the
*margin* `(me − opp)` instead, and adding the frozen incumbent to the opponent pool,
produced the v0.0.9 vector, which wins that head-to-head **21W–9L**.

### Macro search

```bash
# loss-tail report for the agent exactly as it stands
uv run python -m search.cem --diagnose 40

# cross-entropy-method search; always spar against the frozen incumbent
uv run python -m search.cem --iterations 12 --pop 16 --episodes 8 \
    --opponents baseline,opponents/v0_0_6.py

# population-based alternative over the same vector and objective
uv run python -m search.evolution --generations 15 --pop-size 12 --episodes 8
```

Both searches evaluate every candidate in a generation on the **identical seed set**
and on **both seats** (common random numbers), so a difference between candidates is
strategy and not luck. `search/space.py` reads the agent's live constants, so a
search starts from the file as it actually stands rather than from range midpoints,
and writes variants by rewriting a copy — no tuning hooks inside `main.py`.

**Always validate a search result head-to-head before adopting it.** A CEM candidate
that looked better on every scripted opponent lost 8W–22L against the incumbent; the
one that shipped was checked at 30 paired seeds against `opponents/v0_0_6.py` first.

```bash
uv run python -m unittest search.smoke_test
```

---
## Local arena

```bash
# metrics + decision logs
uv run python local_arena.py --agent main.py --opponent baseline --episodes 30 --log-decisions

# graduate a PLACEHOLDER: paired-seed A/B with a significance check
uv run python local_arena.py --agent main.py --opponent baseline --episodes 30 --ablate EXPAND_LAND

# sweep a numeric constant
uv run python local_arena.py --agent main.py --opponent baseline --episodes 30 --sweep MAX_HANDS=10,12,16

# self-play, and head-to-head against a frozen previous version
uv run python local_arena.py --agent main.py --opponent mirror --episodes 30
uv run python local_arena.py --agent main.py --opponent opponents/v0_0_6.py --episodes 30

# save and inspect replays
uv run python local_arena.py --agent main.py --opponent baseline --episodes 3 --save-replays 3
uv run python local_arena.py --replay logs/match_run_0042.json
```

Reported per run: mean/median/min/max/sd final cash, win rate, crashes, timeouts,
invalid statuses, per-turn compute (p50/p95/max), action no-op rate with a per-op
breakdown, shed overflow losses, worker idle rate, market orders dropped to the
10/turn cap, and which heuristics fired.

A 30-episode A/B takes under a minute. **Run 30, not 8** — `IDLE_PREPOSITION` read
+3.2% on 8 seeds and −16.4% on 30.

Three details worth knowing:

- **Seats alternate** every episode (`swap`), so neither the player-index advantage nor
  the market's player-order tie-breaking biases a result.
- **Variants are generated by rewriting a copy of the agent file**, so `--ablate` and
  `--sweep` need no tuning hooks inside `main.py`.
- Shed overflow and no-op counts are invisible in the observation, so the arena wraps
  the interpreter's own `_drop_inventories_to_shed` / `_apply_unit_action` and
  attributes them to one player by object identity.

### Opponents

`baseline` is the env's built-in `starter` (one farmer, one carrot tile). `random` and
`pass` are also built in.

`opponents/adaptive.py` is **not** a port of the public
"adaptive-farming-strategy-for-kaggriculture" notebook. That source could not be
retrieved: kaggle.com renders competition and notebook pages in JS and returns no
usable HTML to a plain fetch. It implements the same *idea*, and it exists because
`starter` is trivially beaten.

`opponents/vX_Y_Z.py` is every previously submitted agent, written automatically by
the `sync-opponent` pre-commit hook. These are the most useful sparring partners in
the repo: they are the only ones that contest the same pots at the same tempo, and
**a change that does not beat the previous version head-to-head is not an
improvement**, whatever it does to `baseline`. `v0_0_7` and `v0_0_8` are absent on
purpose — both embedded network weights, and `v0_0_8` loses to `starter`.

`opponents/leaderboard_replay.py` replays a downloaded 720-step Kaggle episode
turn-by-turn. **It had never worked, and even fixed it is not a cash-comparable
sparring partner.** Both halves of that are worth recording.

Three independent bugs, each of which failed silently as "the opponent finished on
exactly its $3,000 starting money" — which reads as a weak opponent, not a broken one:

1. **`__file__` at module level.** The env loads an agent with
   `exec(compile(src, path), {})`, and that `{}` has no `__file__`, so the import raised
   `NameError` and the env rejected the agent as `InvalidArgument` before turn 1. The
   path is now recovered from the calling frame's `co_filename`, which `compile()` does
   set.
2. **`obs.get("step", 0)`.** `step` lives in the *shared* observation, so only the seat
   at index 0 receives it — and `local_arena` alternates seats, so on half of every run
   the replay read step 0 on all 720 turns. Now derived from the per-seat `day`/`hour`
   (`step == day * turnsPerDay + hour`).
3. **Newest `.json` by mtime, no schema check.** `logs/leaderboard_replays/` holds 116
   *Halite 4* replays alongside the 70 kaggriculture ones, so the newest file was
   usually Halite: every step index missed, every turn returned `PASS`. Candidates are
   now sniffed for `"name": "kaggriculture"` from a 64 KB head, the default pick is the
   highest-scoring episode, the default seat is its winner, and selection is announced
   on stderr.

**And then it still does not work as an opponent, for a reason no fix addresses.** A
replay is open-loop: it re-issues an action stream that was only meaningful against the
state it was recorded in. Replayed against our agent on its own seed, the $187,844
winner scores **$28**. The money trajectory is identical to the recording through day 4,
diverges by $96 on day 5, and is bankrupt by day 12:

| day | recorded | replayed |
| --- | --- | --- |
| 4 | $8 | $8 |
| 5 | $326 | $422 |
| 12 | $552 | $22 |
| 18 | $8,182 | $0 |
| 24 | $79,553 | $0 |

Its 872 `WATER` and 364 `FEED` ops all execute — the op histogram matches the recording
exactly — but it ordered 422 `SELL STRAWBERRY` against a shed holding 0, because the
purchases those harvests depended on failed. **The frontier strategy is maximally
capital-invested** (balance $8 on day 4), which is precisely why it cannot absorb a $96
perturbation. That fragility is itself the most useful thing the replay taught us.

So: useful for reproducing a frontier *action stream* and for the offline statistics in
`scripts/mine_daily.py`, useless for comparing cash. The right frontier sparring partner
is a **scripted reimplementation** of the mined strategy (≈15 pastures at 9 cows / 6
sheep, wheat bought not grown, strawberry fertilized, full land buyout, 12 hands), which
is closed-loop and does not fall over. That is not written yet.

```bash
uv run python local_arena.py --agent main.py --opponent leaderboard --episodes 10
uv run python local_arena.py --agent main.py \
    --opponent logs/leaderboard_replays/episode-90158870-replay.json --episodes 10
```

---

### Daily episode mining (`scripts/mine_daily.py`)

Kaggle publishes the previous day's episodes as a dataset (~21 GB, ~700 episodes). None
of it is useful raw and none of it is worth keeping. The script streams each replay,
keeps a ~1 KB fingerprint per player-seat, and discards the episode — a day compresses
to well under a megabyte, so `--append` turns the CSV into a time series of what the
field is doing.

```bash
# mine what is already downloaded, write logs/daily_fingerprints.csv, print the report
uv run python scripts/mine_daily.py

# fetch a daily dump first (WARNING: ~21 GB), mine it, append to the running CSV
uv run python scripts/mine_daily.py --dataset kaggriculture-episodes-2026-08-09 --append

# re-print the cohort report without re-parsing anything
uv run python scripts/mine_daily.py --report-only
```

Per seat it records final cash, realised revenue per product (`SELL` volume × the price
at the step it was issued, not a curve estimate), `BUY_*`/`HIRE`/`BUY_LAND` volumes, the
full unit-op histogram split into productive versus logistics, end-of-episode
composition (owned tiles, hands, crop tiles, animals, fertilized tiles), unsold shed
stock, and the episode's per-product price trajectory.

The report prints cohort comparisons (top decile / top quartile / ours / bottom
quartile), the exogenous price drift table, and a gap table of ours versus the frontier.
`--me` selects our seats by team-name substring. Non-kaggriculture files are skipped by
a 64 KB head check, so pointing it at a directory containing Halite replays is safe.

This is the input the macro search should be seeded from: the frontier composition it
reports maps directly onto the parameters in `search/space.py`.

---

### Agent Failure Analysis (`scripts/examine_agent.py`)

Fetches your active Kaggle matches for a given submission description, isolates the
losses, and downloads those replays into `logs/failures_<version>/` for debugging.

```bash
uv run python scripts/examine_agent.py v0.0.9
uv run python scripts/examine_agent.py v0.0.9 --limit 5
```

---

### Versioning and Opponent Sync

When you update the `AGENT_VERSION` in `main.py` and commit, a `pre-commit` hook automatically runs `scripts/sync_opponent.py`. This copies your `main.py` into the `opponents/` directory as `vX_Y_Z.py` and stages it. This ensures that every submitted agent version remains available as a sparring partner in `local_arena.py`.

---

## Submitting

```bash
cp kaggle_credentials.example.py kaggle_credentials.py          # then edit
uv run python submit.py --dry-run                               # all checks, no submission
uv run python submit.py                                         # check, submit, poll
```

Pre-flight hard-fails on: a disallowed import in `main.py`; a missing/wrong-arity
`agent`; `agent` not being the last callable defined; the agent failing to load the way
the env loads it; or any crash, timeout or invalid status in the smoke test. Credentials
are exported to the environment before `kaggle` is imported, so no `~/.kaggle/kaggle.json`
is needed, and the key is never printed or written to the history log.

> **Security.** `kaggle_credentials.py` grants full API access to your Kaggle account —
> it can submit, download and delete on your behalf. It is in `.gitignore`; keep it out
> of version control and out of any notebook you publish.

After submitting:

```bash
kaggle competitions submissions kaggriculture
kaggle competitions episodes <SUBMISSION_ID>
kaggle competitions replay <EPISODE_ID>
kaggle competitions logs <EPISODE_ID> 0
kaggle competitions leaderboard kaggriculture -s
```

Note: you must accept the rules at
https://www.kaggle.com/competitions/kaggriculture ("Join Competition") before any
submission will be accepted.
