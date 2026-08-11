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

## Results

| version | submitted | change | local gate | live episodes | live mean cash | live W-L | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0.0.8 | 2026-08-10 | behaviour-cloned atomic actions | — | 12 | $2,908 | 3W-8L | reverted, see README |
| v0.0.9 | 2026-08-11 | CEM macro vector, heuristic restored | $64,351 vs baseline | 10 | $50,957 | 5W-3L | baseline for the work below |
| v0.1.0 | 2026-08-11 | egg engine deleted, pasture role churn fixed, SELL sized from shed | 23W-7L vs v0.0.9 | 29 | $56,279 | 14W-15L | **passed its gate, moved nothing.** Marginally the best of v0.0.6/7/9/0.1.0, all of which sit at ~$56k and ~48%. Score 586.3, rank #2440/3892. See post-mortem. |
| v0.1.1 | not submitted | wheat promoted to a real crop target; leftover-land tail switched to wheat | 17W-13L vs v0.1.0, 24W-6L vs v0.0.9 | — | — | — | passes a narrow gate; expected to move the live score by nothing, for the reasons in the v0.1.1 section |
