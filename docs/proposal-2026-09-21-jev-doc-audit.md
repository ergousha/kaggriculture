# Proposal: a Jev-backed claim auditor for the research record (2026-09-21)

Research tooling only. Nothing here touches `main.py`, the route, or any selection
decision. API facts below were read from TypeSafe's docs on 2026-09-21 and are cited;
cost and corpus figures were computed against this repo; everything else is labelled as
a hypothesis.

## The problem this solves

This repo carries **3,182 lines of load-bearing prose** (`README.md` 1,543 +
`docs/experiments.md` 1,639) whose factual claims keep being falsified by later
measurement. That is documented policy, not speculation — `README.md:80` says
"Sections that are known-wrong are marked as such rather than deleted", so the corpus
accumulates stale claims by design:

- `README.md:176` — "Superseded: the 'capped pot' table", the basis for the
  melon-plus-egg plan, "and it is wrong".
- `README.md:200` — "Every one of those is contradicted by the measured data above."
- `docs/replay_schema.md` — still asserts replays cannot be re-simulated; they can, from
  `info.seed`.
- `docs/experiments.md:342` — two mining bugs "both produced confidently wrong numbers".

The cost is recorded in the repo's own words at `README.md:1208`: "the wrong reasons
predict the wrong next experiment". Finding these is currently manual, and the corpus
grows every panel run.

**Why a model at all.** Everywhere else in this project a simulator is the oracle —
replays re-simulate exactly from `info.seed`, so "is this route better" is answerable to
the exact dollar and no model should be asked. "Does this paragraph still match the
measured data" has no simulator. That asymmetry is the entire justification for this
tool, and it is the boundary the design must not cross.

## What Jev is, concretely

- `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <key>`; Python SDK
  `typesafe-sdk` (requires >=3.10; this repo is >=3.12).
- Three question types: `Choice` (pick one of an enumerated set), `Score` (ordered
  levels), `Noul` (probability that a yes/no statement is true). Every question in a
  request is evaluated against one `state` in parallel — "adding Nouls barely changes the
  response time" ([primitives/noul](https://docs.typesafe.ai/primitives/noul)).
- Model `jev-1.13.0` (`jev-latest` alias). **Price $42/Btok input, output free.**
  Rate limits 250k tok/s, 1,200 req/min. Context 64k/request, 32k for `state` plus the
  longest question ([models](https://docs.typesafe.ai/models)).

**Cost of this tool is not a consideration.** The full corpus is ~42,200 tokens
(computed: 31,277 words x ~1.35). One complete audit pass costs **$0.0018**; a thousand
passes cost **$1.77**. It can run on every commit without thought.

## The design constraint that shapes everything: Jev cannot do the arithmetic

TypeSafe publishes a jaggedness page for this model
([jev-1.13](https://docs.typesafe.ai/model-jaggedness/jev-1.13), reviewed 2026-09-17),
and its failure modes land directly on this repo's content:

| Documented failure mode | Why it bites here |
| --- | --- |
| "Jev is not a calculator... implement any mathematical logic in code" | Nearly every claim here is numeric: `$2,583/day`, `corr = -0.05`, `+$1,610` predicted vs `-$1,453` delivered, `60/64 assignments` |
| "does not count reliably... error grows with the size of the thing being counted" | Rules out "how many claims in this section are stale" |
| "score levels are weak in numerical calibration" — do not interpolate a magnitude | Rules out using `Score` to estimate how far off a number is |
| "Large state full of irrelevant detail: filter first" | Rules out sending the whole 42k corpus as one `state` |
| "Literal reading — answers the question you wrote, not the one you meant" | Every instruction needs the exact condition plus boundary cases in `criteria` |

So the split is hard and non-negotiable:

- **Code owns** locating candidate claims, parsing the numbers out of them, loading the
  current measured value from the repo's artifacts, and doing every comparison.
- **Jev owns** only the semantic half that has no oracle: does this sentence assert a
  checkable fact, which measured artifact would settle it, and does the stated conclusion
  still follow from the stated evidence.

A design in which Jev is handed a claim and a table and asked "is this still true" is the
wrong design. It walks into failure mode #2 and would launder a bad signal into a
confident-looking one — the exact mistake `rank_cvar.py` already documents costing a
predicted `+$1,610` and a delivered `-$1,453`.

## Phases

### Phase 0 — plumbing

1. `uv add typesafe-sdk`.
2. `typesafe_credentials.example.py` / `typesafe_credentials.py` — **done**, mirroring the
   `kaggle_credentials` pattern; the real file is gitignored (`.gitignore:5`) and verified
   invisible to `git status`.
3. `scripts/jev_client.py` — loads the credentials module, exports `TYPESAFE_API_KEY`
   before importing `typesafe_sdk`, pins `jev-1.13.0`, never logs the key. Fails with the
   same copy-the-example message `submit.py:83` uses.
4. `scripts/jev_smoke.py` — one Noul against a fixed state, asserts a response and prints
   `response.model` and `usage`. Confirms auth and records which version answered.

**Guardrail, enforced not just documented:** a test asserting no module reachable from
`main.py` imports `typesafe_sdk` or `typesafe_credentials`. `submit.py` pre-flight already
hard-fails on a disallowed import in `main.py` (`README.md:1521`); this extends that to the
research tooling so the two can never be confused.

### Phase 1 — the claim corpus (code only, no Jev) — **BUILT, measured 2026-09-21**

`scripts/build_claims_corpus.py`. Do not send the markdown. Build a derived corpus whose
shape is dictated by three documented properties of `jev-1.13`:

**Mask the measurements, keep the meaning.** Because Jev "is not a calculator", every
measurement-grade numeral is replaced by a typed placeholder (`<MONEY:0>`, `<PCT:1>`,
`<CORR:2>`, `<RATIO:3>`, `<BIGNUM:4>`, `<MULT:5>`) before anything is sent; the raw values
live in `logs/claims_sidecar.json`, which the model never sees. **A model that cannot see
`$2,583` cannot be wrong about `$2,583`.** Versions (`1.32.7`), dates and small
identifiers (`tier 6`, `day 40`, `v0.4.3`) are deliberately left intact — they are
semantic content, they carry no arithmetic, and masking them destroys the very meaning the
model is there to judge.

**Filter first.** Code fences, ASCII trees, collapsed `<details>` run history, badges and
table rows are dropped, and only sentences asserting a quantity or drawing a conclusion
are kept. Markdown here is hard-wrapped, so lines are reflowed into paragraphs *before*
sentence splitting — splitting per line yields fragments ("It scores 60/60 against the
reference ladder where the") that no auditor can evaluate.

**Measured on this repo:**

| | value |
| --- | --- |
| claims extracted | **563** (276 numeric, 287 conclusion) |
| numerals masked | 557 |
| table rows skipped | 443 |
| reflow fragments dropped | 62 |
| self-marked stale (Phase 5 positives) | 5 |
| raw prose | 31,277 words, ~42,223 tok |
| corpus | 14,771 words, ~19,940 tok |
| reduction | **52.8% fewer input tokens (2.12x)** |

Validated invariants, all passing: no nested placeholders, every placeholder index
resolves to a sidecar value of the same type, no unmasked money or percent leaks, no
placeholder glued to an adjacent word, every chunk inside both caps.

**Cost is not the reason to do this.** A raw pass is $0.0018 and a corpus pass $0.0008 —
the saving is a tenth of a cent. The reason is accuracy: "large state full of irrelevant
detail" is a named failure mode, and 443 table rows plus the collapsed history of
superseded runs is exactly that. Treat the token reduction as a proxy for noise removed,
not as a budget win.

#### Two bugs this phase already caught

Both were silent, and both would have poisoned every downstream comparison:

1. **Nested placeholders.** Running the number patterns as successive passes re-masked
   digits inside placeholders already emitted — `1.32.7` became `<VER:<NUM:2>>`. Fixed by
   a single-pass alternation in which version and date groups match *first* and are
   emitted unchanged, so nothing downstream can chew on their digits.
2. **A leading-list-marker regex that ate numerals.** `^\s*(?:[>#]+|[-*+]|\d+\.)\s*`
   stripped the `97.` from a line beginning `**97.0%**`, leaving **`0%`** — and the
   `[-*+]` branch stripped the minus from a leading `-0.05`, **flipping the sign on a
   correlation**. **46 source lines were corrupted**, including `97.0% → 0%`,
   `100.0% → 0%`, `50.0% → 0%` and `1.32.3 → 32.3`. Every alternative now requires
   trailing whitespace. An audit run on the unfixed corpus would have reported drift
   almost everywhere, and the numbers it cited would have been fabrications of the
   extractor.

The second one is the argument for Phase 5 in miniature: the pipeline looked fine, the
output looked plausible, and it was wrong in the direction that produces false alarms.

### Phase 2 — the question set — **BUILT (dry-run only), 2026-09-21**

`scripts/jev_questions.py`. Builds the exact HTTP request body and reports its measured
size; it sends nothing. What to ask follows from how the primitives behave:

- **`Choice` is a pointer, never a truth test.** Its "probabilities always add up to 1, so
  a line ranks first even when none answer the query". One Choice per claim points at the
  artifact that would settle it, over 8 bare option ids whose legend sits in the state
  once rather than being repeated per question (the `criteria: None` trick from
  `cookbooks/semantic_find`). A mandatory `not_checkable` option absorbs claims no file
  can settle — without it the sum-to-1 constraint invents a match for narrative prose.
- **`Noul` is independent**, "so it can fall near zero", which is what a truth or
  existence question needs. Three per claim: `measured` (is a measured result asserted at
  all — gates Phase 3), `retired` (does the claim disown itself, cross-checking the corpus
  builder's regex), and `self_supported` on conclusion claims only (does the stated
  conclusion rest on evidence stated inside the same claim). That last one is the only
  genuinely oracle-free judgement in the set, and it is `README.md:1208` — "the wrong
  reasons predict the wrong next experiment" — turned into a testable proposition.
- **Every question states its exact condition** and pins both boundaries in `criteria`,
  because the model "answers the question you wrote, not the one you meant".

**Deliberately not asked**, each ruled out by a named failure mode: *"is claim X still
true?"* (needs the number the model must not see — code answers it in Phase 3); *"how many
claims here are stale?"* ("does not count reliably"); *"how stale is this claim?"* as a
`Score` (levels are "weak in numerical calibration", do not interpolate a magnitude);
anything comparing dates or orderings ("reads dates as text, not ordered quantities").

**Batching.** All questions for a chunk go in one request — the state is sent once and
questions run in parallel. TypeSafe's own measurement of this pattern is 12.2x cheaper and
10.0x faster for 13 questions with no change in answers.

**Measured request budget** (1,976 questions over 563 claims):

| variant | requests | input tokens | cost/pass |
| --- | --- | --- | --- |
| full (`criteria` per claim) | 5 | ~218,620 | $0.0092 |
| terse (`criteria` once in state) | 3 | ~130,220 | $0.0055 |

The dry run earned its place immediately: the first cut packed 255 claims per request and
came out at **~73k tokens, over the 64k cap**, because the repeated per-claim `criteria`
outweighed the state 6:1. Packing is now driven by the measured payload, not by the
255-option `Choice` cap.

Two findings worth carrying forward:

1. **The 255-option `Choice` cap is not the binding constraint** — the question payload is.
   Even terse, questions outweigh the state roughly 4:1, because per-claim fan-out repeats
   an instruction and 8 option ids 563 times.
2. **`--terse` is an empirical question, not a preference.** The docs say to "try your
   questions with and without `criteria` and keep whichever gives better answers on your
   documents", so both variants are built and **Phase 5 decides**, on the labelled set.

### Phase 3 — numeric reconciliation, in code

For each claim Jev routed to an artifact, code loads the current value and compares it to
the parsed number with a tolerance. Output per claim: `AGREES`, `DRIFTED` (with both
numbers and the delta), `UNRESOLVED` (artifact lacks the field), or `STALE_MARKED`.
No model involvement in this phase at all.

### Phase 4 — report and gate

`scripts/audit_claims.py` writes `logs/claim_audit.md`: drifted claims first with
`file:line` anchors and both numbers, then low-`conclusion_follows` passages, then
unresolved. Confidence-gated per TypeSafe's routing pattern — middling probabilities go to
a review list, not to an automatic verdict.

Wired as a **non-blocking** pre-commit report at first, alongside
`scripts/sync_opponent.py`. It becomes advisory-blocking only if Phase 5 earns it.

### Phases 0, 3, 4 and 5 — **EXECUTED 2026-09-21**

Key in place, `typesafe-sdk` 0.7.0, model `jev-1.13.0` (pinned, and the response's
`model` field confirms it answered). `scripts/jev_client.py` exports the key before the
SDK import; `scripts/jev_smoke.py` made the first call; `scripts/audit_claims.py` sent
the corpus and cached every raw probability to `logs/jev_audit_raw_{terse,full}.json`;
`scripts/analyse_audit.py` and `scripts/report_audit.py` read the cache, so re-analysis
costs nothing. Report: `logs/claim_audit.md`.

**Total spend for the whole study: $0.0145** (344,621 input tokens, both variants,
1,976 questions each). **9.3 seconds of API time in total.** The parallel-question claim
holds: 413 questions in one request returned in **1.6s**, 788 in **1.2s**. The dry run's
token estimate was accurate to **0.1%** (27,929 predicted vs 27,958 actual), so the
chars/4 heuristic can be trusted for budgeting.

#### Result 1: the tool works, at precision 5/5

Five claims disown a finding in their own text while sitting in a section carrying no
`Superseded` marker — invisible to the corpus builder's regex, found by Jev, **both
variants agreeing ≥0.85**. All five verified by hand against the source:

| claim | what it retires |
| --- | --- |
| `README.md:216` | "**This is now known to be backwards** — egg is 0.6% of frontier revenue" |
| `README.md:659` | the v0.2.6 incumbent panel score, "an artifact, because every member had been selected for beating it" |
| `README.md:1185` | "Previously reported shed-overflow magnitudes... are not evidence" |
| `docs/experiments.md:1593` | "invalidates previously reported shed-overflow figures", naming #25's and #23's tables |
| `docs/experiments.md:208` | every "close the 2.6× gap" target, "derived from a luck-sorted ranking and does not survive this" |

**Precision 5/5 = 1.00 on the regex-missed set**, against a kill criterion of 0.7. The
separation is clean: regex-flagged claims average `retired` 0.61, unflagged ones 0.16.

The `README.md:216` case is worth recording, because I first scored it a **false
positive** and was wrong. The claim's citation pointed at `README.md:212`, whose source
text is about `plan_roles` assigning tile roles and retires nothing — so the hand-check
read the wrong lines and rejected a correct finding. The line-attribution bug (below) had
been corrupting the verification, not just the report.

#### Result 2: a third bug, caught only because a finding was verified by hand

Every sentence in a reflowed block inherited the **block's** first line, and a block
routinely spans a whole bullet list. Citations landed 4-5 lines off target —
`README.md:216` reported as `212`, `README.md:1185` as `1180`. For a tool whose entire
output is `file:line` citations, that is disqualifying, and it is invisible in aggregate
statistics: the corpus still had 563 claims and every invariant still passed.

Fixed by carrying a `(char_offset, source_line)` map through the reflow and resolving each
sentence to its own line. After the fix, **501 of 563 claims anchor to the exact line**
and the mean offset is **0.02 lines**. The claim text sent to Jev is byte-identical, so
the cached answers stayed valid and nothing had to be re-sent.

Running total: **three silent bugs**, all of which produced plausible output —
nested placeholders, numerals eaten by a list-marker regex (46 lines, `97.0% → 0%`), and
mis-attributed citations. None would have been caught by a passing test suite.

#### Result 3: `criteria` matter, and exactly where the docs say they would

The question the proposal deferred, now measured — agreement between the two variants at
a 0.5 threshold:

| question | agreement | mean abs diff | verdict |
| --- | --- | --- | --- |
| `retired` | **97.0%** | 0.042 | stable either way; use terse |
| `measured` | 87.2% | 0.129 | usable; full runs 0.11 higher |
| `artifact` (top pick) | 75.7% | — | routing is soft; needs agreement gating |
| `self_supported` | **61.0%** | 0.222 | **not stable enough to act on alone** |

The pattern matches the documented literal-reading behaviour: the sharper the boundary,
the less `criteria` change the answer. `retired` has a crisp boundary and is insensitive
to them; `self_supported` asks whether evidence is "inside the same claim", which is a
judgement about scope, and there the two phrasings disagree on **112 of 287 claims**.

Consequence, implemented in `report_audit.py`: **a finding is reported only when both
variants agree.** That is the confidence-gating pattern applied to prompt phrasing rather
than to probability, and on this corpus it is the difference between a usable report and
a noisy one. Answering open question 2 from the first draft: `self_supported` **stays, but
demoted** — as a pointer to prose that cannot be checked where it stands, never as a
verdict, and never on one variant.

#### Result 4: Jev's `measured` router beats the code label that trained it

The corpus labels a claim `numeric` when it carries a masked measurement. Jev's `measured`
noul agrees loosely — 75% of `numeric` claims score ≥0.5, 32% of `conclusion` claims do —
and **the disagreements are Jev being right**. The `numeric` claims it scores below 0.2
are recommendations and plans that merely contain a number:

- `docs/experiments.md:782` (0.04) — "Recommendation: The rest of the milestone should NOT
  be re-scoped away from strategic improvements"
- `README.md:601` (0.06) — "Where CVaR₅ still appears, it means the mean of the worst 5% of
  outcomes" — a definition, not a measurement

This is the genuine division of labour working as designed: "has a numeral" is a regex's
judgement, "asserts a measured result" is not, and Phase 3's comparator should be gated on
Jev's answer rather than on the corpus label. **122 claims** have both an agreed artifact
and an asserted measurement; those are the comparator's real input.

#### Result 5: routing is the weakest link

`not_checkable` absorbed **21.5%** of claims, and sensibly — 95 of 287 conclusion claims
versus 26 of 276 numeric ones, so the escape hatch is doing its job rather than swallowing
everything. But mean confidence on the routing Choice is **0.48-0.53** for the
high-volume artifacts, against 0.69 for `engine_spec`, and the variants pick the same
artifact only 75.7% of the time. Routing is the part of this design least ready to act on,
and gating it on cross-variant agreement drops the actionable set to 122 claims out of 563.

## What this is not

It will not move the leaderboard. The submission is unchanged and unchangeable by this
work: no network in the episode sandbox, `actTimeout: 1.0`s/turn over 719 steps
(`README.md:93`). The agent's real blocker is unaffected — a local optimum where 9 of 11
structural changes lost paired seeds, and a local arena that does not predict the live
field. No hosted decision API moves that; better selection does.

What it buys is narrower and real: fewer cycles spent reasoning from a premise the repo
already disproved somewhere in 3,182 lines nobody re-reads.

## Open questions for the next session

1. Should `docs/*.ipynb` and the proposal docs be in scope, or only `README.md` +
   `docs/experiments.md`? Proposals are dated snapshots and arguably *should* go stale.
2. Is `conclusion_follows` worth keeping if the numeric comparator carries the precision?
   Phase 5's labelled set answers this, and it is the question I would most like measured.
