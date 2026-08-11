# Experiment log

A running record of what was changed, why, how it was measured, and what happened.
Kept in `docs/` and not `logs/` because `logs/` is gitignored — this file is meant to
survive for the next version.

Data source for every "field" number below: `scripts/mine_daily.py` over sampled Kaggle
daily episode dumps. Fingerprint CSVs live in `logs/` and are regenerable.

---

## Measurement: what counts as evidence

**`--opponent baseline` is not a gate.** v0.0.9 reports $64,351 against `starter` and
averaged **$50,957** across its 10 real Kaggle seats, against a field median of $83,606
and a top decile of $132,689. `starter` and `adaptive` do not contest pots or force
tempo, so beating them measures survival. Keep it only as a reliability smoke test
(crashes / timeouts / invalid statuses / p95 compute).

**A leaderboard replay is not a gate either.** `opponents/leaderboard_replay.py` had
three bugs and is now fixed, but even fixed it is open-loop: replayed against a different
opponent the $187,844 winner scores $28, because the frontier strategy runs at $8 cash on
day 4 and cannot absorb a $96 perturbation. See README "Opponents".

**Two gates that do work:**

1. **Profile distance.** Run local matches with `--save-replays`, mine them with
   `scripts/mine_daily.py`, and compare our fingerprint to the field's top decile. This
   is dense, per-episode, and independent of how strong the opponent is. The table in
   "Diagnosis" is exactly this comparison, and every row is a target.
2. **Head-to-head against the frozen previous version**, 30 paired seeds, seats
   alternated. Both sides contest the same pots, so this is the closest local proxy for
   P(win). A change that does not beat the previous version head-to-head is not an
   improvement.

Kaggle itself is the third gate, with a budget of **5 submissions/day**. Track a live
submission with `api.competition_list_episodes(sub_id)`; ~9-12 episodes accumulate within
a few hours, which is enough to see a 2× move and nowhere near enough to see 5%.

### The confound that invalidates cash-sorted cohort analysis

**Read this before deriving any target from a "top decile does X" table, including the
ones further down this file.**

Cash rank in a Kaggriculture episode is mostly an episode-level dice roll, shared by both
players. Measured on 60 episodes of the 2026-08-10 dump:

- Within an episode, the two seats' realised strawberry price differs by a mean of
  **$7.20/unit**.
- Between episodes, the stdev of episode-mean strawberry price is **$56.80**, range
  **$25 – $227**.

The driver is `town["unlocked_shops"].append(rng.choice(sorted(SHOPS)))` — shops are
drawn **at random with replacement** every `townShopUnlockInterval` (3) days, up to
`MAX_SHOP_INSTANCES` (8). Four of the eight shop types buy strawberry, so how much
strawberry demand an episode has is a coin-flip sequence:

| strawberry-buying shops unlocked | strawberry $/unit | mean cash |
| --- | --- | --- |
| 1 | $37 | $67,902 |
| 2 | $64 | $68,128 |
| 3 | $82 | $73,361 |
| 4 | $98 | $82,790 |
| 5 | $144 | $100,001 |
| 6 | $190 | $107,717 |

So sorting 300 seats by final cash sorts them mostly **by luck**, and every cohort ends
up with the same mix of strategies. That is exactly what the composition tables showed —
top decile and bottom quartile identical on productive ops, pastures, cows, coops, hands,
tiles and FERTILIZE ops — and it was misread here as "the field has converged on one
strategy". It had not; the cohorts are the same strategies under different dice.

Consistent with that, correlations against final cash over 300 seats:

| variable | corr with cash |
| --- | --- |
| attributed gross revenue | **+0.822** |
| `rev_STRAWBERRY` | **+0.666** |
| `rev_MILK` | +0.634 |
| hands at end | +0.130 |
| `ops_productive` | +0.095 |
| **`sold_STRAWBERRY`** (volume) | **+0.088** |
| `op_WATER` | +0.079 |
| `op_PASS` | −0.094 |
| **`owned_tiles`** | **−0.139** |
| **`land_buys`** | **−0.139** |

Revenue correlates with cash because it *is* cash. Volume does not. Throughput barely
does. **Land is mildly negative.** Every "close the 2.6× gap by matching the top decile's
75 tiles and 2,600 productive ops" target in the plan below was derived from a luck-sorted
ranking and does not survive this.

**What this means for measurement.** Because the shop draw is shared by both seats, it
cancels in a pairwise comparison. The competition ranks pairwise. So the only gate worth
trusting is the **within-episode margin** — paired seeds against the frozen previous
version — and that gate was correctly rejecting these changes while this file was arguing
with it. Trust the arena.

**What survives the confound.** Compositional facts of the form "nobody in the field does
this" are not cash-sorted and still hold: all 300 seats ran **zero coops and zero geese**,
and every cohort ran ~73 `FERTILIZE` ops while we run 0. Those are statements about what
the field believes, and they are independently confirmed below by A/B.

### Correction to an earlier reading of the data

An earlier pass over 69 episodes from **2026-08-05** concluded that prices *rise* over an
episode (MILK 160 → 329, STRAWBERRY 120 → 294) and that production should therefore be
back-half weighted. On the **2026-08-10** field that is false:

| product | 08-05 start→peak→end | 08-10 start→peak→end |
| --- | --- | --- |
| MILK | 160 → 329 → 329 | 169 → 222 → **3** |
| STRAWBERRY | 120 → 294 → 294 | 128 → 206 → **110** |
| WHEAT | 25 → 52 → 52 | 28 → 46 → 36 |
| WOOL | 200 → 247 → 247 | 206 → 218 → 181 |
| MELON | 250 → 281 → 269 | 256 → 272 → **25** |
| EGG | 50 → 69 → 69 | 50 → 62 → 62 |

Prices rise only while the field *under*-produces. The 08-10 field has converged and now
crushes every product it touches; only EGG and CARROT hold their price, because nobody
produces them. **So the rule is not "sell late" — it is "sell before the field does."**
Town demand sets a drain rate; revenue is a race to fill it.

This is also why the tooling had to be fixed before any of it could be trusted: see
"Attribution bugs found while building the gate" below.

---

## Diagnosis: v0.0.9 versus the 2026-08-10 top decile

v0.0.9 = 10 real Kaggle seats (submission 55426703, live 2026-08-11).
Field = 150 sampled episodes / 300 seats from `kaggriculture-episodes-2026-08-10`,
top decile n=30.

| metric | v0.0.9 | top decile | ratio |
| --- | --- | --- | --- |
| **final cash** | **$50,957** | **$132,689** | **2.60×** |
| productive unit-ops | 1,170 | 2,601 | 2.22× |
| `PASS` unit-turns | 2,201 | 996 | 0.45× |
| `WATER` | 187 | 988 | **5.28×** |
| `PLANT` | 36 | 195 | **5.45×** |
| `HARVEST` | 133 | 386 | 2.90× |
| `FERTILIZE` | **0** | 75 | ∞ |
| `CARE` / `FEED` | 264 / 244 | 293 / 298 | 1.11× / 1.22× |
| wheat seed bought | 22 | 143 | **6.61×** |
| strawberry seed bought | 11 | 37 | 3.51× |
| **strawberry sold** | **13** | **229** | **17.6×** |
| milk sold | 58 | 113 | 1.95× |
| fertilizer sold | 250 | 222 | 0.89× |
| coops at end | **14.7** | **0.0** | — |
| geese alive | 0.3 | **0.0** | — |
| pastures at end | 22.0 | 14.2 | 0.64× |
| cows / sheep alive | 8.6 / 6.0 | 9.3 / 4.1 | 1.09× / 0.69× |
| owned tiles | 50 | 75 | 1.50× |
| hands at end | 4.1 | 9.9 | 2.42× |
| wasted SELL order volume | **73.2%** | 28.6% | — |

Revenue mix, top decile: STRAWBERRY 39.5%, MILK 25.5%, WHEAT 13.6%, FERTILIZER 11.5%,
WOOL 8.5%, MELON 1.3%, **EGG 0.0%**.
Ours: FERTILIZER 39.0%, MILK 27.3%, WOOL 10.6%, MELON 10.4%, STRAWBERRY 7.5%, WHEAT 4.0%.

### The one finding that explains most of the gap

**Land allocation.** We hold 50 tiles, of which 22 pastures + 14.7 coops = **36.7 are
animal structures**, leaving ~13 for crops. The top decile holds 75 tiles with **14.2
pastures and zero coops**, leaving ~60 for crops. Per *crop tile* our watering rate is
comparable (14 vs 16 waters/tile); we simply have 4.6× fewer crop tiles. Everything
downstream — `PLANT` 5.5×, `WATER` 5.3×, strawberry sold 17.6× — follows from that one
allocation decision, and the allocation is driven by `engine_claim` funding coops first
for a revenue line worth 0.0%.

Notable: cohorts inside the 08-10 field are nearly *identical* in composition (all ~2,600
productive ops, 14 pastures, 9.3 cows, 0 coops, 9.9 hands, 75 tiles, 73 FERTILIZE). The
top decile separates from the bottom quartile almost entirely on **strawberry revenue
share** (39.5% vs 25.0%). The field has converged on the composition; execution on
strawberry is what is still being contested.

### Mechanics that justify the changes

Read out of `kaggle_environments/envs/kaggriculture/kaggriculture.py`:

- `STRAWBERRY`: seed $100, `first_yield_day` 10, `interval` 2, `max_yield` 4,
  `ongoing` True. It produces 4 times (planting +10, +12, +14, +16), each event adding
  **1 unit, or 2 if the tile was watered and fertilized that day**, and `HARVEST` resets
  `yield_units` without killing an ongoing crop. So fertilizing every production event
  takes a tile from 4 units to 8.
- `FERTILIZE` sets `fertilized_until_day = day + 2`, active for 3 days inclusive. With
  strawberry's interval of 2, **one FERTILIZE covers two production events** if applied
  on a production day. Two applications cover a tile's whole life: +4 strawberry for 2
  fertilizer.
- Fertilizer supply is not scarce: `fertilizer_available` is reset to True on **every
  animal tile every day**, so 14 pastures yield ~420 units over an episode. Applying ~75
  and selling the rest is not a trade-off. Our agent applies 0.
- `FERTILIZER` is excluded from `TOWN_CENTER_PRODUCTS`, so the town never drains it and
  its price only falls: linear at $0.20/unit from $100, hitting the $1 floor at ~500 units
  sold. It is a fixed ~$25k pot shared with the opponent — real, but capped, and we
  currently take 39% of our revenue from it.
- `_commit_unit` fills a SELL **only from `private["shed"]`**, one unit at a time,
  re-quoting after each unit, and aborts the order when the shed runs dry. `main.py:697`
  sizes its orders as `shed + _carried_adjacent_to_shed(...)`, which the interpreter never
  honours — hence 73.2% wasted SELL volume against a 10-orders/turn cap.
- `GOOSE` $300 → ~25 eggs at $50-62 with 1 wheat/day feed; `COW` $400 → ~11-22 milk at
  $169-222. A cow is 3-5× a goose per structure *and* per unit-turn, and a coop occupies a
  tile that could carry a crop.
- `MELON` above-target curve is `sq` (amp 0.01): dumping ~150 units past I0 takes it from
  $250 to $31. `WOOL` is also `sq` (amp 0.058) — even harsher. `MILK` and `STRAWBERRY` are
  `linear` above target, so they tolerate volume far better. This is why melon and wool
  are the two products to under-produce and milk/strawberry the two to lean on.

### Attribution bugs found while building the gate

Both were in `scripts/mine_daily.py` and both produced confidently wrong numbers, so
they are worth recording as a warning about this replay format:

1. **Ordered volume is not filled volume.** Counting `qty × quoted price` attributed
   $76,652 of fertilizer revenue to an episode whose final cash was $50,957. Fills must
   be clamped to the seat's shed and the price integrated unit by unit.
2. **`steps[t]["action"]` is the action that *produced* `steps[t]`, not the one taken
   from it.** `steps[0]`'s action is empty for every seat and a step's money already
   includes that step's own sales. Pricing an action against its own step's market
   under-counted revenue by 7× (attributed $18,670 against $132,689 of cash). Verified
   on both a locally-generated and a downloaded replay. `leaderboard_replay.py` had the
   same off-by-one and is fixed.

After both fixes, attributed gross revenue lands within ~4-7% of final cash plus costs;
the residual is the opponent's interleaved orders moving the price mid-fill.

---

## v0.1.0 — what was tried and what actually held

The plan started as a "land allocation" package aimed at the top decile's 75 tiles and
2,600 productive ops. Four of its five parts lost, and the shop-draw confound above is
why the targets were wrong. What shipped is the intersection of what the field
unanimously does and what wins paired seeds.

All A/Bs: 30 paired seeds against `opponents/v0_0_9.py`, seats alternated.

| change | result | verdict |
| --- | --- | --- |
| `EGG_ENGINE` off — no geese, no coops, no `engine_claim` | ON is **−62.3%**, better on **0/30**, p~0.0 | **kept off** |
| Pin animal roles to tiles that already hold a structure | 23.8 pastures for 13 animals → churn removed | **kept** |
| SELL sized from `shed` only, not shed + carried | wasted SELL volume **68% → 0%** | **kept** |
| `LAND_CASH_BUFFER` 1961.9 → 400, `LAND_LAST_DAY` 16 → 22 | $45,238 vs $54,773, 5W-25L | **reverted** |
| `MELON_TILE_TARGET` 9 → 3 | $43,059 vs $58,635, **0W-30L** | **reverted** |
| `MAX_SHEEP` 6 → 4 | 15W-15L vs 23W-7L at 6 | **reverted** |
| Sale-rate limiting, flat floor at 1.0 × base | $13,760, 0W-30L, 305 units lost to shed overflow | flag off |
| Sale-rate limiting, 10% per-order slippage cap | $45,144 vs $58,635, 5W-25L | flag off |

Four lessons, all of them about the measurement rather than the game:

1. **Freeing land amplifies whatever the allocator does with it.** Dropping the land
   buffer took us from 50 to 96.9 owned tiles — the whole board, ~$7k — and there was no
   labour to work it. It also pushed melon to its 9-tile cap, so we sold 55 units into a
   `sq` collapse. The land package looked like it was failing on land; it was failing on
   everything downstream of land.
2. **A low revenue *share* is not evidence a product loses money.** Melon is 1.3% of
   top-decile revenue, which read as "cut it". The field still sells 30.6 melon for
   $1,410; the share is low because everything else is bigger. Cutting it cost 0W-30L.
3. **`--sweep` reports its best by mean cash, and the objective is P(win).** `MAX_SHEEP=4`
   wins on mean ($56,531 vs $54,383) and loses on win rate (15W-15L vs 23W-7L). Read the
   win rate.
4. **The empty-structure bug relocates rather than disappears.** 14.4 empty coops became
   ~11 empty pastures the moment land expanded, because roles are recomputed every turn
   from a shed-distance sort and new land re-sorts ahead of built structures. Neither is
   visible without fingerprinting the finished farm.

### v0.1.0 measured result

| opponent | eps | mean cash | win rate | crashes | p95 turn |
| --- | --- | --- | --- | --- | --- |
| `opponents/v0_0_9.py` | 30 | **$54,383** (opp $51,608) | **76.7%** (23W 7L) | 0 | 0.33 ms |
| `baseline` | 30 | $72,568 | 100% (30W) | 0 | 0.30 ms |
| `opponents/adaptive.py` | 30 | $70,554 | 100% (30W) | 0 | 0.27 ms |
| `mirror` | 30 | $53,866 | 11W 10T 9L | 0 | 0.29 ms |

Every acceptance criterion passes. For reference v0.0.9 measured $73,032 vs `baseline`
and $68,872 vs `adaptive`, so the scripted opponents are flat-to-better while the
head-to-head is decisively better — which is the ordering to want, since the scripted
numbers are not predictive.

---

## Plan for what remains

The throughput targets from the cash-sorted cohorts are withdrawn. What is left is
supported either by an unanimous compositional fact about the field or by mechanics read
out of the interpreter.

### v0.1.1 — fertilize strawberry (the last unanimous field fact)

Every cohort in the 08-10 field runs ~73 `FERTILIZE` ops; we run **0**. This is not
cash-sorted, so the confound does not apply. The mechanics support it independently:
`FERTILIZE` sets `fertilized_until_day = day + 2` (3 days inclusive) and a
watered+fertilized production event adds 2 units instead of 1, so with strawberry's
interval of 2 one application covers two events and two applications take a tile from 4
units to 8. Supply is free — `fertilizer_available` resets on every animal tile every
day, ~420 units an episode against ~75 needed.

Emit `FERTILIZE` from `build_tasks` on strawberry tiles that are watered and inside a
production window. **Gate:** paired seeds, plus `sold_STRAWBERRY` per strawberry tile
roughly doubling.

### v0.1.2 — labour, diagnosed before tuned

We end on 4.1 hands against a `MAX_HANDS` of 12, so the ceiling is not the throttle;
`HIRE_CASH_FRACTION = 0.0513` interacting with low cash is. Note `hires_today` resets
daily, so hands are re-hired every day and pausing hiring is a shutdown, not a saving
(measured −65.9%). Diagnose where the hire loop actually stops before changing anything.

### Not worth pursuing

- **Matching the field's 75 tiles / 2,600 productive ops.** `corr(cash, owned_tiles)` and
  `corr(cash, land_buys)` are both −0.139; `corr(cash, ops_productive)` is +0.095. These
  were luck-sorted artefacts.
- **Sell timing or restraint.** `corr(mean strawberry sale step, $/unit)` is −0.241 and
  the buckets are flat; the price spread is the shop draw. Two implementations lost.
- **Reducing variance to chase the top decile.** The top decile is the lucky tail of a
  field whose median is $83,606. The reachable target is the median, and the way to beat
  a specific opponent is the within-episode margin.

### Submission budget

5/day. One submission per version that clears its paired-seed gate, held for approval
first, leaving spares for a revert. Do not spend a submission to measure a change the
paired-seed gate already rejected — that was tried eight times above and the gate was
right every time.

---

## Results

| version | submitted | change | local gate | live episodes | live mean cash | live W-L | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0.0.8 | 2026-08-10 | behaviour-cloned atomic actions | — | 12 | $2,908 | 3W-8L | reverted, see README |
| v0.0.9 | 2026-08-11 | CEM macro vector, heuristic restored | $64,351 vs baseline | 10 | $50,957 | 5W-3L | baseline for the work below |
| v0.1.0 | pending approval | egg engine deleted, pasture role churn fixed, SELL sized from shed | **23W-7L vs v0.0.9** | — | — | — | awaiting submit |
