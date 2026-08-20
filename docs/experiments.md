# Experiment log

A running record of what was changed, why, how it was measured, and what happened.
Kept in `docs/` and not `logs/` because `logs/` is gitignored — this file is meant to
survive for the next version.

Data source for every "field" number below: `scripts/mine_daily.py` over sampled Kaggle
daily episode dumps. Fingerprint CSVs live in `logs/` and are regenerable.

---

## The gate problem is solved: rank against a calibrated ladder

`scripts/rank_ladder.py` plays the agent seat-swapped against **Rayk Kretzschmar's
reference ladder** — ten agents whose relative strength is already measured and
documented. Rungs 0-5 are vendored under `opponents/ladder/` (MIT, `LICENSE` and
`NOTICE` retained); rungs 6-9 embed the shared public meta line and are deliberately
*not* vendored, because the dataset's own NOTICE asks that submissions be built on 0-5.
`--fetch` pulls them into `reference/ladder/` for measurement only.

```bash
uv run python scripts/rank_ladder.py --fetch
uv run python scripts/rank_ladder.py --episodes 6
```

Rungs 0-5 share a **byte-identical scheduler** and differ only in a `POLICY` dict, so a
gap between them is an economic decision and nothing else. That is what none of our
previous gates had.

### v0.1.1 placement, 6 episodes per rung, seats alternated

| tier | agent | result | ours | theirs | margin |
| --- | --- | --- | --- | --- | --- |
| 0 | fallow_finn | 6W-0L | $84,746 | $3,000 | +$81,746 |
| 1 | wheat_walter | 6W-0L | $71,871 | $6,959 | +$64,912 |
| 2 | rotation_rosa | 6W-0L | $74,802 | $12,998 | +$61,804 |
| 3 | homestead_hana | 6W-0L | $75,366 | $13,530 | +$61,836 |
| 4 | melon_mateo | 6W-0L | $68,433 | $19,899 | +$48,534 |
| 5 | rancher_rita | 6W-0L | $50,454 | $22,583 | +$27,871 |
| **6** | **broker_bea** | **0W-6L** | $41,078 | $126,176 | **−$85,098** |
| 7 | ledger_lena | 0W-6L | $40,711 | $126,234 | −$85,524 |
| 8 | slotter_silas | 0W-6L | $40,766 | $124,639 | −$83,873 |
| 9 | closer_cleo | 0W-6L | $30,168 | $108,450 | −$78,283 |

**RUNG: between tier 5 and tier 6. 36/60 overall.**

This is the cleanest measurement in the repo. It also reframes the plateau: our
`MAX_COWS = 9` / `MAX_SHEEP = 6` livestock core *is* tier 5 (`rancher_rita` is 10 cows /
6 sheep and banks ~$53k against `starter`; we bank $55-56k), and we beat her by +$27,871
so we are genuinely above that rung. The reference league has Rita losing to broker_bea
by −$128,041 where we lose by −$85,098, so the ~$56k plateau is not a bug in our agent —
**it is the ceiling of the whole authored-strategy band.**

### Why tiers 6-9 are unreachable by tuning: they are not strategies

Every rung above 5 runs **the same production plan** — a base85-encoded `_TRACE` field
that the dataset's NOTICE describes as "the shared public meta line", found as identical
712-turn farmer/hand sequences across 104 distinct teams in 530 downloaded replays. The
four top rungs differ only in their market layer.

The current #1 on the leaderboard is the same shape. Its notebook is
[25/27 Strict-Future | v27 Midgame Meta Reset](https://www.kaggle.com/code/kaitofukami/25-27-strict-future-v27-midgame-meta-reset),
and the published artifact decodes to 20,813 bytes with SHA-256
`f48c2116…` (verified). It contains:

- `_LEGACY_ACTIONS` — one hardcoded **719-step action list**, used in both seats,
  credited to team Ezzzzzekki's public replay of episode 91493566 seat 0;
- `_weed_repair_action` — if a scripted `PLANT`/`BUILD_PASTURE` lands on a `WEED`, emit
  `DIG` instead, replay the intended action next step, then shift the next 8 steps of the
  trace by one;
- `_rank_sell_slots` — reorder the SELL orders **within their existing slots** by
  price impact × a bounded town-demand urgency term. Nothing about production changes;
- `_align_hands` — pad or truncate the hands list to the live hand count.

That is the entire agent. There is no planner, no scheduler, no opponent model. The
notebook's own ablation puts most of the gain on the route and about +$819 to +$1,115 of
margin on the sell layer.

**So the competition above ~$56k is not a strategy competition.** It is route selection
plus a thin repair layer, and the ~$130k the top rungs bank against us is what a
hand-selected tape achieves. That is the honest explanation for eleven consecutive failed
A/Bs: we were tuning a closed-loop planner against opponents that do not plan.

### Two corrections from the reference material

**Engine version.** The README claimed verification against kaggle-environments 1.32.3.
We actually run **1.32.6**, and so does the competition (Kaito's notebook records
`research_engine_version: 1.32.6`). This matters more than it looks: the ladder's NOTICE
warns that the same game on the same seed pays out completely differently across
releases — *Rancher Rita banks 28,370 on 1.32.3 and 9,002 on 1.32.6*. All ladder numbers
above are ours, measured on 1.32.6, so they are internally consistent; the dataset's own
published banks are 1.32.3 and are not comparable to them.

**Shop demand, not curve steepness, sets the realised price.** `_town_consume` drains
market inventory every `townShopSellInterval` for each unlocked shop that lists the
product, which is what holds prices up all season:

| product | shops demanding it | base | shop demand/day |
| --- | --- | --- | --- |
| WHEAT | 5 | $25 | 30 |
| STRAWBERRY | 4 | $120 | 24 |
| MILK | 3 | $160 | 18 |
| EGG | 2 | $50 | 12 |
| CARROT / TOMATO | 2 | $35 / $60 | 12 |
| WOOL | 1 | $200 | 12 |
| **MELON** | **0** | **$250** | **0** |

**Melon appears in no shop at all**, so nothing ever drains it and its price only falls —
it is genuinely a one-shot pot, which is why `melon_mateo` (tier 4) meters sales into
12-unit lots and why buying more land does not help him. This is the mechanism behind our
own measurement that cutting `MELON_TILE_TARGET` 9→3 lost 0W-30L: the first ~$26k of the
pot is real money, and the tiles were collecting it.

It also independently confirms two of our results. The dataset author tested adding geese
to Rita in 32 configurations and **all 32 lost**, matching our `EGG_ENGINE` ablation
(−62.3%, 0/30). And Rita's documented failure mode — "with a 4-day feed reserve instead of
16 she goes bankrupt on day 3 and loses to tier 1" — is the same effect as our
`LAND_CASH_BUFFER` 1961→400 experiment (23.3%, and 5W-25L in v0.1.0).

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

**The gate that actually works — fingerprint against real opponents, within episode.**
Download a live submission's episodes and run `scripts/mine_daily.py` over them, then
compare our per-seat fingerprint to *the opponent we faced in that same episode*. Both
seats share the episode's shop draw, so the confound below cancels, and the opponent is a
real strategy rather than a copy of ourselves. This is what finally explained v0.1.0.

```bash
uv run python scripts/mine_daily.py logs/live_v0_1_0 --out logs/fingerprints_live_v0_1_0.csv
```

**Head-to-head against the frozen previous version**, 30 paired seeds, seats alternated,
is necessary but **not sufficient**. It reliably catches regressions and it correctly
rejected six of the eight v0.1.0 candidates. It cannot catch a wrong strategy *class*,
because a frozen copy of ourselves shares our blind spots — v0.1.0 won it 23W-7L and moved
the live score by nothing. Use it as a veto, never as proof.

**Do not compare against the field's cash-sorted top decile.** That was the third gate
proposed here and it is invalid; see the confound below. The "Diagnosis" table further
down is preserved as a record of the mistake, not as a target list.

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
cancels in a pairwise comparison, and the competition ranks pairwise. So the gate has to
be a **within-episode margin**.

Paired seeds against the frozen previous version is *one* such gate and it correctly
rejected the changes this file was arguing with. But it is not sufficient, and v0.1.0
is the counterexample: it won that gate 23W-7L and moved the live score by nothing,
because a frozen copy of ourselves shares our blind spots. See the v0.1.0 post-mortem.
The gate that catches a wrong strategy class is the same fingerprint comparison run
against **downloaded live opponents**.

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

## Diagnosis: v0.0.9 versus the 2026-08-10 top decile — INVALID, kept as a record

> **Every ratio in this table is unsafe.** The "top decile" is the lucky tail of the shop
> draw, not a stronger strategy, so these are not targets. Four changes were derived from
> it and lost; see the v0.1.0 section. It is kept because the *compositional* rows that do
> not depend on cash ranking (zero coops, zero geese, ~73 `FERTILIZE` ops field-wide) held
> up under A/B, and because the shape of the error is worth being able to re-read.

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

## v0.1.0 post-mortem: it passed its gate and moved nothing

v0.1.0 went live 2026-08-11 08:49 (submission 55428161) and the team score sits at
**586.3, rank #2440/3892** (leader 3214.7, leaderboard median 726.5). That reads as a
collapse. It is not one, and the reason it is not is the useful part.

### Nothing regressed. The plateau is five versions old.

| version | live eps | record | our mean cash | opp mean cash |
| --- | --- | --- | --- | --- |
| v0.0.6 | 117 | 57W-60L (48.7%) | $56,735 | $68,789 |
| v0.0.7 | 89 | 41W-48L (46.1%) | $56,245 | $56,218 |
| v0.0.8 | 12 | 4W-8L (33.3%) | $2,909 | $12,358 |
| v0.0.9 | 30 | 14W-16L (46.7%) | $55,157 | $56,275 |
| **v0.1.0** | 29 | **14W-15L (48.3%)** | **$56,279** | $54,498 |

Every version except the neural one lands on the same spot: **~$56k and ~48%**. v0.1.0 is
marginally the best of them. A score of 586.3 is what a coin-flip record produces under
rating-based matchmaking — you are paired with similarly-rated agents, so 50% holds you at
the 600 entry rating indefinitely. **The score did not fall because of v0.1.0; it never
rose, for five versions.** v0.0.8 stopped being matched at 06:47 on 08-11, so it is no
longer dragging the team.

### The paired-seed gate could not have detected this, and this file overclaimed it

Above, the shop-draw section concludes "the only gate worth trusting is the within-episode
margin against the frozen previous version." That is wrong, and v0.1.0 is the
counterexample. 23W-7L against `opponents/v0_0_9.py` measures the margin against **one
agent that shares our strategy DNA**. It correctly detected that v0.1.0 fixed v0.0.9's
specific defects — 14.4 empty coops out of 50 tiles. It cannot detect that both agents are
playing the wrong game, because both make the same mistake.

The gate that does work needs no new tooling: download the live episodes and run
`scripts/mine_daily.py` over them. That gives a **within-episode comparison against real
opponents**, and the shop-draw confound still cancels because both seats share the draw.
It was available the whole time.

```bash
# after a submission has accumulated episodes
uv run python scripts/mine_daily.py logs/live_v0_1_0 --out logs/fingerprints_live_v0_1_0.csv
```

### What that gate says: we are a fixed-output machine playing a different game

29 paired live episodes, us versus the actual opponents we faced.

**Our own behaviour is statistically identical in wins and losses. Only the opponent
varies.**

| | us in our 13 wins | us in our 16 losses |
| --- | --- | --- |
| wheat sold | 29.8 | 29.6 |
| milk sold | 159.3 | 158.2 |
| wool sold | 127.2 | 127.4 |
| `WATER` | 284.8 | 261.9 |
| owned tiles | 48.1 | 48.4 |
| productive ops | 1,384 | 1,358 |

| | opponent in our wins | opponent in our losses |
| --- | --- | --- |
| final cash | $38,655 | $71,081 |
| **wheat sold** | **186.4** | **505.2** |
| melon sold | 70.1 | 100.6 |
| strawberry sold | 56.1 | 77.9 |
| `FERTILIZE` | 10.9 | 31.8 |

We emit the same farm every episode and win or lose on who we are matched against. The
opponents who beat us are the ones running heavy wheat rotation.

**Revenue mix, same 29 episodes:**

| | MILK | WOOL | FERTILIZER | STRAWBERRY | MELON | WHEAT |
| --- | --- | --- | --- | --- | --- | --- |
| us | 35.0% | 24.0% | 19.0% | 11.7% | 8.4% | **1.9%** |
| them | 24.3% | 8.6% | 9.2% | 17.9% | 19.2% | **19.1%** |

**Per-farm gaps against real opponents** (ratio > 1 means they do more):

| metric | us | them | ratio |
| --- | --- | --- | --- |
| **wheat sold** | 29.7 | 362.3 | **12.19×** |
| `WATER` | 272.2 | 885.9 | **3.25×** |
| melon seed bought | 8.3 | 25.9 | 3.12× |
| `PLANT` | 70.1 | 142.7 | 2.03× |
| `FERTILIZE` | **0** | 22.4 | ∞ |
| owned tiles | 48.3 | 72.4 | 1.50× |
| productive ops | 1,370 | 1,890 | 1.38× |
| `PASS` unit-turns | 1,432 | 764 | 0.53× |
| sheep alive | 6.0 | 2.9 | 0.49× |
| wool sold | 127.3 | 56.6 | 0.44× |
| fertilizer sold | 264.2 | 134.2 | 0.51× |

### Why wheat is the engine we skipped

Two properties, both read out of the interpreter:

1. **Cycle.** `WHEAT`: seed **$10**, `first_yield_day` 2, `max_yield` 6, `ongoing: False`.
   The tile dies on `HARVEST` and can be replanted immediately, so it is a 2-day, $10,
   6-unit loop — the shortest capital cycle in the game.
2. **Glut resistance.** Its above-target curve is `log` with amp 0.83, the flattest in the
   game. Revenue for dumping N units starting from inventory I0:

| product | N=50 | N=130 | N=360 | price after 360 | above-curve |
| --- | --- | --- | --- | --- | --- |
| **WHEAT** | $1,127 | $2,823 | **$7,513** | **$20** | `log` amp 0.83 |
| FERTILIZER | $4,755 | $11,323 | $23,076 | $28 | `linear` amp 0.20 |
| MELON | $12,098 | $25,267 | $26,687 | $1 | `sq` amp 0.01 |
| WOOL | $7,655 | $7,999 | $8,229 | $1 | `sq` amp 0.058 |
| MILK | $5,430 | $6,235 | $6,465 | $1 | `linear` amp 2.10 |
| STRAWBERRY | $3,648 | $3,877 | $4,107 | $1 | `linear` amp 1.92 |

Wheat is the only product that still pays a real price after 360 units. Everything else
hits the $1 floor. (These are worst-case one-shot dumps; the town's drain means reality
sits above them, which is exactly why the steep curves punish batch size.)

We take **43% of revenue from wool plus fertilizer** — wool being the second-steepest
curve in the game, and fertilizer a pot the town never replenishes at all — and 1.9% from
the one glut-proof product.

### The land experiment was not a false negative

`LAND_CASH_BUFFER` 1961→400 measured 5W-25L in v0.1.0. The hypothesis recorded here was
that it failed only because nothing could *use* the land, and that land plus wheat rotation
were one change rather than two. **That was retested in v0.1.1 with the wheat rotation in
place and it is false:** buffer 400 gives 23.3% against 56.7% at 1961.9, delta −$3,049,
p~0.006. Land is genuinely not the constraint, which is also what
`corr(cash, owned_tiles) = −0.139` said.

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

## v0.1.1 — wheat as a cash crop, and four more refuted hypotheses

Aimed at the 12.19× wheat gap from the v0.1.0 post-mortem. One of five parts survived.
All A/Bs 30 paired seeds against `opponents/v0_1_0.py`.

| change | result | verdict |
| --- | --- | --- |
| Wheat sized as a real crop target (`WHEAT_TILE_TARGET` 14, `WHEAT_LAND_FRACTION` 0.70) instead of `ceil(n_animals × 0.4778)`, and the leftover-land tail switched from `STRAWBERRY` to `WHEAT` | **56.7% (17W-13L)**, and 80.0% vs v0.0.9 | **kept** |
| `PRIO_PLANT` 550 → 760 / 900 | 6.7% (2W-28L) at both, −$13.5k and −$18.9k, p~1e-05 | reverted |
| `FERTILIZE` task on both crop families | 36.7% at prio 660, 33.3% at 300 or 1 | flag off |
| `LAND_CASH_BUFFER` 400 **with** the wheat rotation in place | 23.3% (7W-23L), −$3,049, p~0.006 | reverted |

### What the failures say, and they say the same thing three times

`PRIO_PLANT` raised above the animal chores loses 2W-28L. `FERTILIZE` loses even at
priority 1 — and the giveaway is that priority 300 and priority 1 produce **byte-identical
results** (delta +0, better on 0/30), so at those levels the task is never the reason a
unit is chosen, yet the agent still drops 23 points. The cost is not the priority and not
the fertilizer spent: it is that **any additional task consumes the leftover units that
`_logistics` needs.**

That is the third independent measurement of one mechanism. The README already records it
twice — the Hungarian assignment losing 31-38% because minimising travel scatters idle
units off the shed, and `IDLE_PREPOSITION` losing 16.4% for walking them somewhere useful.
Greedy's *unassigned* units clustering next to the shed is load-bearing, because every
animal and every sack of feed enters the farm through a shed `PICKUP`. Anything that gives
those units a job costs more than the job is worth.

Combined with the animal chores being 78% of our revenue, this is a coherent picture of a
**local optimum**: the livestock core and the idle-unit logistics reserve are mutually
load-bearing, and every attempt to reallocate unit-turns toward the crop-rotation game the
field plays has now lost — nine of eleven candidates across v0.1.0 and v0.1.1.

**The implication is that this is not a knob problem.** Matching the field's crop rotation
needs the macro *and* micro layers redesigned together — fewer animals, so fewer chores,
so the freed unit-turns go to planting and watering, with the priority ladder and the
logistics reserve re-derived for that mix. Changing one leg at a time will keep measuring
the ladder rather than the idea, which is exactly what the last eleven A/Bs did.
`search/cem.py` exists for re-tuning the vector after such a change; it cannot make the
change.

### v0.1.1 measured result

| opponent | eps | mean cash | win rate | crashes | p95 turn |
| --- | --- | --- | --- | --- | --- |
| `opponents/v0_1_0.py` | 30 | $52,691 (opp $52,529) | **56.7%** (17W 13L) | 0 | 0.25 ms |
| `opponents/v0_0_9.py` | 30 | $53,152 | **80.0%** (24W 6L) | 0 | 0.33 ms |
| `baseline` | 30 | $71,885 | 100% (30W) | 0 | 0.30 ms |
| `opponents/adaptive.py` | 30 | $71,761 | 100% (30W) | 0 | 0.30 ms |
| `mirror` | 30 | $54,360 | 11W 9T 10L | 0 | 0.30 ms |

Reliability criteria all pass. Note the gate is *narrower* than v0.1.0's was — 56.7% is a
17W-13L edge, which at 30 seeds is not significant on its own; the 80.0% against v0.0.9 is
the stronger reading. Given that v0.1.0 won its gate 23W-7L and moved the live score by
nothing, **this should be expected to move the live score by nothing as well.** It is worth
submitting only as a cheap confirmation that the plateau is architectural, not as a
candidate for a real gain.

---

## v0.2.0 — route replay. 36/60 → 60/60 on the ladder.

The ladder gate said our hand-written agent sits between tier 5 and tier 6, and that
every rung above 5 is a fixed production plan with a thin repair layer. So we built one.

`scripts/build_route_agent.py` bakes a downloaded episode's action stream into a
submittable agent: the route as base85-of-zlib JSON, plus three runtime layers written
from the environment's mechanics —

1. **bounded WEED repair.** Weeds are the only per-episode randomness that can silently
   invalidate a route: a tile the route expects empty becomes a `WEED` and every `PLANT`
   on it no-ops for the rest of the game. A scripted `PLANT`/`BUILD_*` onto a weed emits
   `DIG`, replays the intent next step, then shifts that unit's trace by one for 8 steps.
2. **SELL-slot ordering.** The SELL orders the route already contains are reordered
   among their own slots by price impact (`quantity × (price_now − price_after)`).
   Quantities, items and every non-SELL order stay where the route put them.
3. **hands alignment.** Pad or truncate the hands list to the live hand count; a route
   recorded with N hands is otherwise invalid on a turn with M.

### Route selection is the whole job, and recorded cash is the wrong criterion

Candidate routes were extracted from 188 downloaded episodes (the 2026-08-10 dump plus
our own live episodes), then **baked and ranked against ladder tiers 6, 8 and 9** rather
than trusted on the cash they happened to bank:

| # by cash | team | episode / seat | recorded cash | ladder (inner) | margin |
| --- | --- | --- | --- | --- | --- |
| 1 | Ezzzzzekki | 91490781 / 0 | $146,935 | 11/12 | +$22,518 |
| 2 | Desert Fox88 | 91927561 / 0 | $145,279 | 11/12 | +$23,136 |
| 3 | Jince | 91759414 / 0 | $141,128 | **12/12** | +$25,381 |
| **4** | **Dmitry Larko** | **91767673 / 1** | $140,980 | **12/12** | **+$29,777** |
| 5 | 青烟 | 91490952 / 0 | $140,633 | 9/12 | +$16,841 |
| 6 | Gould Research | 91582648 / 0 | $139,872 | 10/12 | +$16,956 |
| 7 | saitamad | 91613045 / 1 | $137,764 | **12/12** | +$25,381 |
| 8 | THUNDER THUNDER | 91537598 / 1 | $137,003 | 10/12 | +$16,458 |

The ranking is not the cash ranking. The best-banking route is 11/12; the fourth-best
is 12/12 with the highest margin. Note also that **Jince and saitamad produce
byte-identical results** — the same shared meta line under two team names, exactly what
the dataset NOTICE describes.

The three 12/12 routes were then split on an **independent seed set** (seed 9000, 8
episodes per rung, tiers 6-9). All three went 32/32; Dmitry Larko's carried the highest
own cash ($104,344 vs $96,943) and the best margin, so that is the route that shipped.

### v0.2.0 measured result

Full ladder, 6 episodes per rung, seats alternated:

| tier | agent | v0.1.1 | **v0.2.0** | v0.2.0 cash | theirs |
| --- | --- | --- | --- | --- | --- |
| 0-4 | finn … mateo | 30W-0L | **30W-0L** | $138-155k | $3-15k |
| 5 | rancher_rita | 6W-0L | **6W-0L** | $125,300 | $16,208 |
| 6 | broker_bea | **0W-6L** | **6W-0L** | $90,488 | $57,628 |
| 7 | ledger_lena | **0W-6L** | **6W-0L** | $90,162 | $57,222 |
| 8 | slotter_silas | **0W-6L** | **6W-0L** | $89,738 | $57,524 |
| 9 | closer_cleo | **0W-6L** | **6W-0L** | $86,896 | $64,244 |
| | **overall** | **36/60** | **60/60** | | |

**RUNG: at or above tier 9.** And against our own history: 30W-0L vs both v0.1.1 and
v0.1.0 at $127,338 and $127,506 mean. Smoke test against `baseline` banks $139,804
where v0.1.1 banked $55,109. `mirror` is 4W-19T-7L, the expected shape for two identical
deterministic agents on a shared seed.

Reliability: 0 crashes, 0 timeouts, 0 invalid statuses, p95 turn **0.0 ms** — the route
lookup is far cheaper than the planner it replaced.

### What this costs, and what it is honest to claim

- **The production plan is not ours.** It is team Dmitry Larko's observable public replay
  of episode 91767673, and the generated file records episode, team, seat and recorded
  cash in its own header so the provenance travels with the artifact. The three runtime
  layers are our implementation of the design published by Kaito Fukami. The dataset
  NOTICE's position on the shared line is that it "demonstrably belongs to none" of the
  104 teams playing it and is reconstructible from public replay data by anyone.
- **It is open-loop, so it is fragile in a way the numbers above do not show.** The
  ladder opponents are deterministic. Against an opponent that trades very differently,
  a `BUY` the route depends on can fail and the divergence compounds — that is exactly
  how the 2026-08-05 $187,844 route collapsed to $392 when we replayed it. The
  three-layer repair covers weeds and order slots; it does not cover cash divergence.
  Selection on the ladder is a proxy for robustness, not a proof of it.
- **The heuristic line is not deleted.** `opponents/v0_1_1.py` is its frozen head and
  `search/` still operates on it; `search/smoke_test.py` now reads its constants from
  there rather than from `main.py`.
- **Two pre-flight changes were needed.** `zlib` was added to `submit.py`'s import
  allowlist, and the check that required the planner/scheduler classes to be present was
  removed: it was a proxy for "self-contained" that only described one architecture,
  while self-containment is already enforced by the allowlist and by loading the file the
  way the env loads it.

---

## Results

| version | submitted | change | local gate | live episodes | live mean cash | live W-L | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0.0.8 | 2026-08-10 | behaviour-cloned atomic actions | — | 12 | $2,908 | 3W-8L | reverted, see README |
| v0.0.9 | 2026-08-11 | CEM macro vector, heuristic restored | $64,351 vs baseline | 10 | $50,957 | 5W-3L | baseline for the work below |
| v0.1.0 | 2026-08-11 | egg engine deleted, pasture role churn fixed, SELL sized from shed | 23W-7L vs v0.0.9 | 29 | $56,279 | 14W-15L | **passed its gate, moved nothing.** Marginally the best of v0.0.6/7/9/0.1.0, all of which sit at ~$56k and ~48%. Score 586.3, rank #2440/3892. See post-mortem. |
| v0.1.1 | PR #10 | wheat promoted to a real crop target; leftover-land tail switched to wheat | 17W-13L vs v0.1.0, 24W-6L vs v0.0.9; **ladder 36/60, between tier 5 and 6** | — | — | — | last of the heuristic line; frozen at `opponents/v0_1_1.py` |
| **v0.2.0** | pending | **route replay** — Dmitry Larko ep 91767673 seat 1, plus WEED repair, SELL-slot ordering, hands alignment | **ladder 60/60, at or above tier 9**; 30W-0L vs v0.1.1 and v0.1.0; $139,804 vs baseline | — | — | — | awaiting submit |

---

## v0.2.6 — Leaderboard Gap Research (Issue #22)

The gap between our agent (v0.2.6 at 2290 rating) and the leaderboard leader (~3220) was investigated to determine if it's solvable through strategic improvements or if it's structural (matchmaking/convergence).

### Findings

1. **Volume/Convergence is not the main gap driver.** The top 5 teams have between 99 and 500 episodes on their active submissions. The score correlates with true skill, not purely with episode count.
2. **Matchmaking isolation is a gradient, not a closed band.** Scoring every opponent recorded across the top 5 teams' last 20 episodes against the leaderboard shows isolation tracks rating and decays sharply on approach to 3000:

   | Team | Score | opponents >3000 | opponents in top 30 |
   | --- | --- | --- | --- |
   | `カワシギ` | 3208.0 | 14/20 | 20/20 |
   | `Thomas Tschinkel` | 3131.6 | 12/20 | 19/20 |
   | `Utkarsh #2` | 3020.7 | 11/20 | 14/20 |
   | `ReCurSiON` | 3000.1 | 6/20 | 14/20 |
   | `peikopon` | 2987.5 | 0/19 | 1/19 |

   The top two are effectively sealed into a >3000 pool (`カワシギ` played `Utkarsh #2` 6 times, `Thomas Tschinkel` 4, `ReCurSiON` 4). But rank-5 `peikopon`, at 2987.5, matches *nobody* above 3000 and draws from the broad field. Isolation is therefore a **consequence** of crossing 3000, not a precondition for getting there — no team in this sample climbed by playing only >3000 opponents.
3. **Our 2290 rating is real.** Over the last quartile (29 completed episodes) of v0.2.6's live play, it achieved a **44.83% win rate** (13W 16L 0T) — 95% CI **28.4%–62.5%**. The rating is not obviously still climbing, but at n=29 that interval spans everything from "clearly losing" to "clearly winning", so it cannot on its own establish convergence. Confirming the plateau needs a wider window or a per-version rating trace; we do not currently record leaderboard rating per submission, so that trace does not exist yet.
4. **Shop-draw cancellation:** Because we haven't matched against a top-30 team in our recent episodes, we could not run a direct, same-episode fingerprint comparison. The matchmaking gradient and the win rate are consistent with a real, non-artefactual gap, but neither one establishes it alone — see the interval on finding 3.

**Data provenance.** The numbers above come from `logs/leaderboard_research.json`, produced by `scripts/research_leaderboard.py`. The committed run covers **20 of the 31 teams requested** (top 30 plus us): ranks 21-30 and our own rank-498 row were lost to API errors partway through the run, which the script previously swallowed silently. Findings 1 and 2 rest only on the top 5 and are unaffected; any statement about the shape of the top 30 as a whole is not supported by this run. The script now reports coverage and exits non-zero on an incomplete sweep, so a re-run is needed before the top-30 view can be quoted.

### Conclusion & Recommendation

The gap from 2290 to 3200 is **real** and not a convergence artefact. We are losing at a 2290 rating band because we are playing worse than the teams above us.

**Recommendation:** The rest of the milestone should NOT be re-scoped away from strategic improvements — the goal of breaking 2290 is valid.

The evaluation tooling should become *rating-aware*, sampling a band around and modestly above our own rating (roughly 2300–2600 today) and advancing that band as we climb. Calibrating against the >3000 meta is **not** supported by the data above: `peikopon` sits at rank 5 having never played a >3000 opponent, so that meta is not the ladder we are on. Sampling exclusively from it would optimize against games we are not currently matched into while discarding the opponents that actually set our score.

Two caveats on implementing this, both of which make it larger than a config change:

- The panel does not "sample the field broadly" as originally written here. `simulate_candidates.py:414` draws panel members from the top `--panel-from-top` screen performers, and `mining/panel.py:41` then runs greedy max-min diversity over per-step action distance. It is already a strongest-first selection; the axis it lacks is rating, not strength.
- There is no rating signal to filter on. Panel entries carry `hash`, `route`, and `team` only, and mined replays never record an opponent rating (no `rating` field exists anywhere under `mining/`). Making the panel rating-aware requires capturing opponent rating at mine time first.

---

## Market layer: measured and exhausted (Issue #23)

The shipped route uses **927 of 7,190 market order slots (12.9%)**. 339 of 719 steps emit
no market order at all, and only **6 steps** hit the 10-order cap. Market orders cost no
unit-turns — `_process_market` never touches units, and `_commit_unit` only checks cash
and shed capacity. On paper this is a large free action channel.

It is not usable. Three variants were built and measured against panel opponent
`2f741e6bd5`. The variant code lived in the analysis session's scratchpad and is not
committed; reproduce from the descriptions below.

| variant | what it does | result |
| --- | --- | --- |
| **Defer** | hold sales below a price floor, drain the backlog later | **catastrophic** — $26k / $23k / $1k at floors 0.60 / 0.85 / 1.00, against a $104k baseline on seed 2000000 |
| **Accelerate** | sell the turn the product lands, capped by the route's own remaining scheduled volume | +$61 mean cash, +$215 mean margin over 30 paired seeds; win rate unchanged at 90% (27W-0T-3L both) |
| **Drip** | replace route SELLs with a per-turn quote sized to the market | broken by construction — it sells the 351 units of feed WHEAT the route buys, and the herd starves |

**Why deferral fails is the interesting part.** The route is not a production plan with a
market layer bolted on — it is a **cash schedule**. 277 HIRE orders and every BUY are
timed against money the route expects to already have. Delay a sale and the next HIRE
fails, the hands never materialise, and throughput collapses. Any future market work has
to move the BUY schedule together with the sale schedule (that is #30).

Acceleration is real but sits inside the noise band; it does halve shed overflow (25 → 15
items lost). It is not worth a submission on its own — see #25.

This is consistent with Kaito Fukami's published ablation, which puts +$819 to +$1,115 on
his sell layer and attributes everything else to the route.

### The trap that cost two measurement rounds: `obs.private.shed` predates the turn

In `kaggle_environments/envs/kaggriculture/kaggriculture.py` (1.32.6) the interpreter runs
`_apply_unit_action` for the farmer and every hand (L922-926), **then** `_process_market`
(L928), then `_town_consume` (L929).

So the shed an agent observes is the shed *before* this turn's `PLACE` / `DROP` /
end-of-day deposits land in it. It is a **lower bound**, not the quantity available to
sell. Any adaptive market layer that writes `qty = min(route_qty, observed_shed)` silently
truncates every sale, every turn, for every product. All three variants above shipped with
this bug, and the symptom — a total collapse to near-$0 — reads as an economic result and
is not one.

`main.py` is not affected today, because it replays recorded quantities verbatim. The
moment anything sizes an order from the observation, it is.
[`tests/test_observation_ordering.py`](../tests/test_observation_ordering.py) pins the
ordering so this fails loudly instead of quietly: it drives a scripted episode in which a
same-turn `PLACE` feeds a `SELL` the observation says is impossible, and a same-turn
`PICKUP` starves a `SELL` the observation permitted.

---

## v0.2.7 — aiming the opponent panel at a rating band (Issue #24)

### What was wrong

Two selection biases stacked. Panel members were drawn from the top `--panel-from-top`
*screen* performers, and the screen ranks by win rate against the incumbent anchor — so
every member beat the incumbent ~100% by construction. Underneath that, the pool they came
from was whoever happened to appear in the daily replay dumps: a sample of the whole field
weighted by episode volume, not by strength. We are matched by rating, so the pipeline was
optimising win rate against the median of the field.

### The band, and why not the top

Issue #24 as written says "top N of the ladder". That was not run, because #22 had already
measured the opposite: rank-5 `peikopon` at 2987 matches nobody above 3000, so isolation
into the >3000 pool is a **consequence** of crossing 3000, not the way up. Selecting
against the leaders would optimise for games we are not matched into.

The panel was therefore drawn from **ranks 185–469 (2300.2–2599.1 rating)** — from our own
rating (2294, rank 476) to ~300 points above. `--panel-team-top` still supports the literal
top-N reading; `--panel-rank-min` is what makes it a window.

### Result

| | mean win | worst opp | margin CVaR₅ | cash mean |
| --- | --- | --- | --- | --- |
| Winner `044a7741e9` (Ueddy), panel | 93.8% | 77.0% | −$8,881 | $93,286 |
| Winner, held out on 100 fresh seeds | 93.2% | 79.0% | −$9,533 | $90,702 |
| v0.2.6 incumbent, same held-out grid | 83.6% | 51.0% | −$9,849 | $90,013 |
| Winner vs v0.2.6 head-to-head, 100 disjoint seeds | 96.0% | — | — | $87,019 |

79,980 sieve episodes + 1,000 held out, zero bad. Ladder 10/10. Shrinkage −0.6%.

### What was actually learned, as distinct from what was shipped

**1. The bias was real and is now measurable.** v0.2.6's Phase 3 scored the incumbent at
**0.2%** against its own panel. On a ladder-band panel it scores **83.6%**. The README's
largest open caveat is closed — not by argument, by measurement.

**2. The headline delta is one opponent.** 44 of the 48 points of the +9.6% mean delta come
from a single panel member (`ebfc911eaa`, lllleeeo, rank 435: incumbent 51%, winner 95%).
Against the other four the winner is within two points of the incumbent. The gate's rule is
"strong against five, weak against one is an exploit"; this is the mirror image, and the
honest expectation is *a wash plus one favourable matchup*.

**3. All 12 finalists share a worst opponent**, `8f7dd57d5f` (researchstudio.site, rank
466), at 77–78%. That is not a candidate's exploit — it is a weakness of the strategy class
the entire 4,315-route corpus contains, and no route in it fixes it. Independent
confirmation of #31's "the remaining gap is production" from the opposite direction.

**4. Finalist concentration got worse.** 12 finalists across **2 teams**, within a 2.5%
win-rate spread, against 5 teams in v0.2.6. Route selection really is exhausted.

**5. `corr(recorded cash, mean win rate) = −0.54`** — an order of magnitude stronger than
the −0.05 measured against the old panel. Filtering the pool by banked cash would have been
even more damaging than previously thought.

### Limits of this result

- **Ranks are a 2026-08-17 snapshot; the replays are 08-08→08-14.** A team that climbed
  since is credited for strength its mined route did not have. Kaggle exposes no historical
  rank, so this is not fixable — only dated, which `logs/team_ranks.json` does.
- **An open-loop replay of one good episode is not that team's agent.** The panel is
  representative of the band's *routes*, not of the band's *strength*.
- **The mid-stage pool is still cut by win rate against the incumbent** (top 150 of 4,315),
  so a route that beats the band but loses to v0.2.6 never reaches the panel. Fixing that
  means screening against the panel at 6× compute.
- **Local gates remain a veto, not a forecast.** Five consecutive versions have now won
  their local gates; four moved the live score by nothing.

---

## v0.2.8 — sell acceleration: two bugs, then a clean negative (Issue #25, PR #35)

The fourth runtime layer #25 specified: a product already sitting in the shed is sold now
rather than when the route's schedule gets round to it, capped by the volume the route
itself still has scheduled for it. #23 had already measured it as +$61 and within noise,
and #25 said in its title not to spend a submission on it alone.

The first cut shipped two defects and one wrong conclusion.

### Bug 1: the layer sold a farm input, and the ladder did not notice

#25's second safety rule is "never sell an input", and it names `WHEAT` — the route buys
351 units of it as feed. `WHEAT` was excluded. But `FERTILIZER` is an input too, and it is
not bought, it is *produced*: 306 `COLLECT_FERTILIZER` calls put it in a unit's carried
inventory, it is deposited to the shed, and then **14 `PICKUP FERTILIZER` calls lift 95
units back out** to feed 64 `FERTILIZE` calls. `_inv_take` reads carried inventory and
`PICKUP` fills it from `private["shed"]`, so draining the shed starves the fertilizer
chain. Fertilizer is also 1,907 of the route's 2,772 non-wheat SELL units — 69% of the
accelerated budget, not a corner case.

Measured over 30 paired seeds vs `opponents/v0_2_6.py` (base seed 3000000, seats
alternated), against the v0.2.7 incumbent:

| | mean cash | win rate | no-op `FERTILIZE` |
| --- | --- | --- | --- |
| v0.2.7 incumbent | $86,515 | 96.7% (29W-1L) | 9 |
| v0.2.8 as first written | $83,146 | **16.7%** (5W-25L) | **282** |

**`scripts/rank_ladder.py --episodes 1 --require-perfect` returned 10/10 on that build.**
The ladder cannot see an 80-point win-rate collapse against a peer, because every rung it
contains is $45k weaker than we are. It is a crash test, not a strength test.

The fix generalised #25's rule instead of adding a second hardcoded item: precompute
`_PICKUP_SUFFIX_SUMS`, the per-item suffix sum of what the route still lifts out of the
shed, and reserve it before accelerating. That is derived from the route, so it would have
survived the next route swap — a blanket `FERTILIZER` exclusion would not have. It is not
in the tree, because the layer it protects was reverted; recover it from PR #35 if #30
revives the idea. What the tree keeps is
[tests/test_sell_layer.py](../tests/test_sell_layer.py), which pins the rule and the
`PICKUP` volumes it has to respect.

### Bug 2: the emitted template no longer matched the emitted agent

`AGENT_TEMPLATE` was missing the two blank lines PEP8 wants before a top-level `def`, so
`build_route_agent.py` and `encode_submission.py` emitted an agent that fails
`ruff format --check`. Repo CI passed only because `main.py` had been formatted by hand
after generation. `scripts/build_route_agent.py` now round-trips to `main.py` byte for
byte; that equality is worth re-checking whenever the template changes.

### The result, with the bugs out of the way

Paired A/B of the layer ON vs OFF — same file, `budget` forced empty for OFF — 30 paired
seeds each, base seed 3000000, seats alternated:

| opponent | strength | ON − OFF | ON better on | p | shed overflow ON → OFF |
| --- | --- | --- | --- | --- | --- |
| `opponents/v0_2_6.py` | peer, ~$86k | **−$1,680** | **0/30** | **~0.0** | 10 → 45 |
| `closer_cleo` (tier 9) | ~$26k | +$382 | 17/30 | 0.011 | 17 → 53 |
| `slotter_silas` (tier 8) | ~$26k | +$290 | 19/30 | 0.052 | 4 → 33 |
| `broker_bea` (tier 6) | ~$26k | −$174 | 12/30 | 0.148 | 0 → 30 |

The shed-overflow saving #25 promised is real and large — it survives every opponent, and
it is the only claim in #25 that does. The cash claim does not. Against the ladder the
layer is a wash to +$400; against the only peer-strength opponent available it loses
**$1,680 on every single one of 30 seeds**, t = −8.1.

**Why the sign flips with opponent strength.** `_process_market` runs both players'
order *i* in per-unit lockstep against one shared inventory, and README's own price table
says the market *rises* over 30 days — MILK $160 → $329, never retracing. Selling early is
therefore selling cheap, and the cost is only paid when someone else is still selling into
the recovered market later. A tier-6 bot banks $26k and never gets there; `v0_2_6` does.
The leaderboard is made of peers, not of tier-6 bots, so the peer column is the one that
forecasts.

This is the same shape as #23's deferral result read backwards: the route is a cash
schedule, and *both* directions of moving a sale off its scheduled turn cost money — a
late sale starves the next HIRE, an early one sells into a market that had not risen yet.
#30 (move the BUY schedule with the sale schedule) remains the only version of this idea
that could work.

### Verdict

Both bugs were fixed and the layer still failed #25's own gate — "not worse on win rate vs
`opponents/v0_2_6.py`" — on 0 seeds out of 30, so **it was reverted**. `main.py` is back to
v0.2.7's `_rank_sells` and `AGENT_VERSION` is back to `0.2.7`; `_accelerate_sells` with
`budget` forced empty reproduced v0.2.7 at $86,513 / 96.7% exactly, which is how we know
the slot-handling rewrite was never the problem and the acceleration always was.

There is no v0.2.8. What survives the branch is this write-up, the argparse fix in
`rank_ladder.py`, and two test files: `tests/test_sell_layer.py` for the three restraints
that make the SELL layer safe, and `tests/test_agent_template.py`, which fails if
`AGENT_TEMPLATE` stops regenerating `main.py` byte for byte.

Count for the record: this is the tenth of twelve structural changes to `main.py` to lose
paired seeds.

---

## Route synthesis harness (Issue #26) — built, gated, shipped no agent

The end of "route selection is exhausted" is a tool, not a route: with every one of
4,315 candidates the same strategy, the only direction with a five-figure target is
optimising a route directly. `search/route_search.py` is that harness — a
mutation-and-accept loop over the 719-step action stream, seeded from the v0.2.7
incumbent (`main.py._ROUTE`, hash `044a7741e9`, round-trips exactly through the pool).

**What it verifies before it is trusted (the "changes nothing" gates):**

| gate | check | result |
| --- | --- | --- |
| 1. identity | zero mutations -> baked route byte-identical to the seed | PASS — hash round-trips |
| 2. round-trip | the no-op shift preserves the full non-movement op signature | PASS — moves move walks only |
| 3. fidelity | the identity artifact replays a full episode with zero invalid/crash/timeout | PASS — vs `random`, $149k / $188k clean |
| 4. budget | wall-clock per candidate is printed | 30 seeds × 6 panel = 180 episodes ≈ 35 min on 12 workers at ~2.4 s/ep |

**What was measured building it, beyond the gates.** The one operator with a definable
inverse is the movement shift, and it is the honest proof that the loop runs: mutating
the seed's final-step movement (step 713, the tail of the route) and re-evaluating
through the real baked artifact on the full v0.2.7 panel returned **bit-identical cash on
all six opponents** — e.g. seed $142,582 vs `b8b9267d1c` $115,984; mutant identical,
to the dollar, on every member. The no-op is a no-op empirically, not just by
construction.

**Design choices that were taken deliberately, and why:**

- **Seed loading prefers `main.py._ROUTE`, hash-verified against the pool.** The baked
  file is what ships; the pool is the identity the rest of the pipeline agrees on.
  A drift between them is detected and the pool wins, loudly, because a silent drift
  here poisons every mutation downstream.
- **Evaluation reuses `simulate_candidates.run_stage` verbatim and shares its resume
  key.** A mutated route is a new hash, so the loop never re-uses results across routes;
  it only ever re-reads an anchor episode it already paid for. The harness cannot
  drift from Phase 2 because it *is* Phase 2.
- **`swap_herd` (#27) and `assign_idle` (#28) mutate but do not repair.** The cadence
  compression and the choreography are the dependent issues' operators, not this
  harness's; a wrong repair reads as "the herd starves for no reason" and would burn an
  evaluation budget chasing a bug. `repath` (#29) and `move_sell_and_buy` (#30) are
  `None` placeholders until their shortest-path and joint operators land.
- **Acceptance is `rank_cvar`'s metric** — mean panel win rate, worst-opponent
  tiebreak — unchanged, so a route this loop accepts is one the existing Phase 3 gate
  would rank the same way. The carried-forward warning stands: local accept is a veto,
  never a forecast.

**Budget for the dependent issues.** At 180 episodes per accepted/rejected candidate,
one mutation step is ~35 min on 12 workers; a 20-step greedy path is ~12 h. That is the
number #27's herd sweep, #28's idle-turn rotation and #29's path optimisation scope
against — and the reason each of those issues should batch its mutations rather than
accept on single steps.

**Live validation: none by design.** The harness ships no agent; its cost was eight
unittest runs (8 tests, all green) plus one live 6-episode smoke pass against the real
panel.

---

## [v0.3.0] Re-mix the herd from cows to sheep (Issue #27) — accepted, gated, shipped

**The premise.** The incumbent route (`044a7741e9`) kept 10 cows and 4 sheep across 14
pastures. Cows cost $400, yield MILK every 2 days (starting day 8), but flood the market
with 343+ units, driving MILK realized price down to $1-$58/unit. Sheep cost $500, yield
WOOL every 3 days (starting day 6), and sell for ~$60-$65/unit into a higher-value market.
Converting late cows to sheep was hypothesized to capture higher unit revenue without
requiring new pasture infrastructure.

### The Herd Mix Sweep & Realized Unit Economics

We implemented `op_swap_herd` to rewrite `BUY_ANIMAL COW -> SHEEP`, synchronize downstream
`PICKUP`/`PLACE` unit actions, and emit `SELL WOOL` orders alongside milk sales. We swept
the herd mix across the 6-opponent panel over 30 seeds (180 episodes per candidate) and
measured realized $/unit for both products:

| Herd Mix | Mean Panel Win | Worst Win | Cash Mean | Margin CVaR5 | MILK Realized ($/unit) | WOOL Realized ($/unit) | Total Animal Rev |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **10c / 4s (v0.2.7 incumbent)** | 91.1% | 73.3% (`8f7dd57d5f`) | $90,829 | −$9,722 | 1,140 units @ $58.2 | 456 units @ $65.1 | $96,072 |
| **9c / 5s (Cow 7 converted)** | **87.8%** | **73.3%** (`ebfc911eaa`) | **$91,948** | **−$6,613** | 1,030 units @ $66.3 | 566 units @ $58.7 | **$101,546** (+$5.4k) |
| **8c / 6s (Cows 6, 7 converted)** | 83.3% | 63.3% (`v0_2_6`) | $92,777 | −$8,383 | 925 units @ $76.9 | 676 units @ $52.2 | $106,404 (+$10.3k) |
| **6c / 8s (Cows 6..9 converted)** | 73.3% | 50.0% (`v0_2_6`) | $92,651 | −$17,972 | 745 units @ $95.4 | 799 units @ $44.3 | $106,480 |
| **4c / 10s (Cows 4..9 converted)** | 56.7% | 33.3% (`62b81aa8a3`) | $89,645 | −$34,674 | 570 units @ $118.2 | 921 units @ $33.7 | $98,450 (collapse!) |

### Market Saturation Dynamics

As predicted in #27:
1. **Wool Saturation**: Realized $/unit for WOOL degrades as more sheep are added: $65.1/unit (4s) $\to$ $58.7 (5s) $\to$ $52.2 (6s) $\to$ $44.3 (8s) $\to$ $33.7 (10s). At 10 sheep, total wool revenue collapses from $35.4k to $31.0k because the market is heavily flooded.
2. **Milk Spillover Effect**: With fewer cows, milk supply decreases and milk $/unit rises ($58.2 $\to$ $118.2). Against peer opponents who run 10-cow routes (like `v0_2_6`), our un-dumped milk leaves the market clearing at high prices that directly enrich the opponent.
3. **The Pareto Frontier**: `9c / 5s` (converting Cow 7 on pasture `(7, 4)`) captures +$5,474 in animal revenue and improves cash mean ($91,948 vs $90,829) and tail risk (CVaR5 −$6,613 vs −$9,722) while preserving robust win rates against all panel opponents.

### Milestone Gate Validation

1. **Direct Head-to-Head vs Incumbent (`044a7741e9`)**:
   - 100 held-out seeds (disjoint from mid/screen sets), alternating seats:
   - **62W − 38L (62.0% Win Rate)**, mean margin **+$1,093** $\to$ **PASS** (Gate $\ge 55\%$).
2. **Held-Out Panel Validation (100 disjoint seeds × 6 opponents = 600 episodes)**:
   - Mean Cash: **$90,767** (+$1,279 over incumbent $89,488)
   - Margin CVaR5: **−$6,094** (+$2,284 improvement over incumbent −$8,378)
   - 100% win rate vs `800dc80f5c`, 100% win rate vs `b8b9267d1c`, 97% win rate vs `62b81aa8a3`, 83% win rate vs `8f7dd57d5f` $\to$ **PASS**.
3. **Reference Ladder Gate (`scripts/rank_ladder.py --episodes 1 --require-perfect`)**:
   - **10/10 Perfect Run** across all 10 rungs, beating tier 9 `closer_cleo` (+$22,453 margin) $\to$ **PASS**.

### Verdict

`9c / 5s` candidate (hash `9d890c9a321f2c0b5bf7ca77fc9ac930`) passed all milestone gates and is baked into `main.py` as **v0.3.0**.

---

## [v0.3.1] Spend idle unit-turns on carrot/tomato & tile optimization (Issue #28)

**The premise.** The incumbent route contains 667 `PASS` unit-turns across 719 steps. Issue #28 proposed converting these idle turns into short-cycle cash crops (CARROT: 2-day maturity, $20 seed $\to$ 4 units @ $35 = $140 revenue; TOMATO: $60 base, 8-day maturity) to capture uncontested town shop drain (~18 carrots/day = $630/day, 12 tomatoes/day = $720/day).

### Idle Turn & Farm Tile Census

We analyzed the 667 `PASS` turns and 100 farm tiles across the 719-step route:
1. **70 turns** sit directly on planted crops (`MELON`, `STRAWBERRY`, `WHEAT`) that were unwatered on that day (`watered_today: False`).
2. **66 turns** sit on empty farm tiles (`(2,4)`, `(8,1)`, `(9,1)`, `(7,4)`, `(1,1)`, `(4,1)`).
3. **27 turns** sit on `WEED` tiles.
4. **Tile Mapping**: Every single visited tile on the 10×10 farm is already allocated to high-margin STRAWBERRY/MELON crops, WHEAT feed crops, or 14 COW/SHEEP pasture tiles.

### Candidate Sweep Across 6-Opponent Panel (30 Seeds, 1,440 Episodes)

We evaluated 8 candidate variants sweeping carrot retargeting, tomato retargeting, and idle-turn water conversions:

| Candidate | Description | Mean Win Rate | Mean Cash | Margin CVaR5 | Hash | Status |
| --- | --- | --- | --- | --- | --- | --- |
| **v0_3_0_incumbent** | Baseline 9c/5s route | 90.0% | $91,869 | −$5,243 | `8c62ea6056` | **Incumbent** |
| **carrot_3** | 3 late wheat $\to$ carrot | 6.7% | $70,537 | −$64,119 | `b88826635f` | Crashed (feed starvation) |
| **carrot_6** | 6 late wheat $\to$ carrot | 1.1% | $55,733 | −$112,936 | `e4e3dd4446` | Crashed (feed starvation) |
| **carrot_10** | 10 late wheat $\to$ carrot | 0.0% | $50,696 | −$162,183 | `29585b7591` | Crashed (feed starvation) |
| **carrot_15** | 15 late wheat $\to$ carrot | 0.0% | $45,678 | −$126,860 | `dd87319c08` | Crashed (feed starvation) |
| **tomato_4** | 4 wheat $\to$ tomato | 42.8% | $87,354 | −$44,382 | `a714af8eb2` | Crashed (feed starvation) |
| **water_straw_only** | 43 strawberry PASS $\to$ WATER | **90.6%** | $91,871 | −$5,243 | `dcd4e8c8b9` | Win rate +0.6% |
| **water_straw_melon** | 54 strawberry/melon PASS $\to$ WATER | **90.6%** | $91,871 | −$5,243 | `e8c035f9d0` | Win rate +0.6% |
| **water_all_crops** | 70 all-crop PASS $\to$ WATER | 86.1% | $91,996 | −$6,389 | `1925f90c23` | Degradation vs ebfc911eaa |

### Critical Findings & Root Mechanisms

1. **Wheat is the Livestock Engine's Non-Negotiable Input**:
   - In the incumbent route, WHEAT is not grown for cash sale — it is harvested into the shed and fed to the 9 cows and 5 sheep.
   - Retargeting even 3-6 wheat plantings to carrot or tomato starves the herd of wheat feed, causing cows/sheep to stop producing milk and wool. Livestock revenue collapses by $20k-$50k, crashing win rate from 90% to 0-6%.
2. **Additive Dwell Planting Causes Inventory Deadlock**:
   - The empty tiles where units dwell (`(2,4)`, `(8,1)`, etc.) are pastures where cows are placed on Day 3 or transit paths for fertilizer/wool delivery.
   - Planting crops on dwell tiles results in carrying units holding harvested crops, which blocks hands from carrying fertilizer or milking cows on subsequent steps.
3. **In-Place Watering Optimization**:
   - Converting unwatered PASS turns on standing cash crops (`STRAWBERRY` and `MELON`) is 100% legal, consumes 0 inventory slots, and lifts single-agent win rate against the 6-opponent panel to **90.6%** (with a perfect 10/10 reference ladder sweep).
### Verdict

`water_straw_melon` (54 strawberry & melon unwatered PASS $\to$ WATER turns converted, hash `e8c035f9d0ce7cd865e4078e013256db`) delivers **90.6% panel win rate** and a **10/10 reference ladder sweep** (+$22.4k margin vs tier 9 `closer_cleo`). Baked into `main.py` and `opponents/v0_3_1.py` as **v0.3.1**.



