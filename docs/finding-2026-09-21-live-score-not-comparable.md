# Finding: `public_score` is not comparable across submissions (2026-09-21)

Measured today with `scripts/audit_live_scores.py`, which groups every scored Kaggle
submission by the sha256 of its committed runtime. Numbers below are reads from the
Kaggle API, not estimates.

## The evidence

`opponents/v0_4_1.py` and `opponents/v0_4_2.py` are **byte-identical runtimes** — the
same sha256 over everything after `AGENT_VERSION`, same length. They were submitted
three times between them:

| read | version | score |
| --- | --- | --- |
| 2026-09-15 07:10 | v0.4.1 | 2300.1 |
| 2026-09-15 11:40 | v0.4.1 | 2450.6 |
| 2026-09-15 14:57 | v0.4.2 | 1816.9 |

**n=3, mean 2189.2, spread 633.7 points on byte-identical code.**

v0.4.2 exists at all only because `.github/workflows/release-and-publish.yml` keys the
release off `AGENT_VERSION`, not off the code: bump the constant, find no matching tag,
submit. The repo therefore ran the most valuable experiment available — the same agent
submitted three times — and never scored it.

## Why the spread is probably not noise

A fourth data point, collected 20 minutes after submission today:

| read | version | score |
| --- | --- | --- |
| 2026-09-21 12:36 (t+20min) | v0.5.0 | **673.6** |

v0.5.0's runtime is byte-identical to v0.4.3, whose score reads 1812.9. A fresh
submission therefore **starts far below its eventual level and climbs as it plays
episodes**, and the endgame proposal already recorded the other half of the effect —
"max-active = v0.4.0's decayed 2632.3" — so a submission that stops being active decays.

`public_score` is a skill rating that is a function of *when you read it* relative to the
submission's own lifecycle. Two submissions read at different maturities are not
like-for-like even when the code is identical. The 633.7 spread is an **upper bound on
reproducibility**, most of it likely convergence and decay rather than variance in play.

## What this retracts

Three claims made earlier today, in this repo's own working notes and in conversation,
are **not supported** and should not be built on:

1. **"v0.4.3 is a ~640-point live regression."** v0.4.3 reads 1812.9 and v0.4.0 reads
   2591.5, but those reads sit at different points in each submission's lifecycle. The
   gap is smaller than the spread measured on identical code. Unresolved, not a
   regression.
2. **"v0.4.2/v0.4.3 lost ~640 rating that the offline gate missed."** The offline gate
   may well overstate — this repo has measured that before — but *this* number is not
   evidence of it. v0.4.2 is the same code as v0.4.1.
3. **"v0.4.0 at 2591.5 is the best live configuration."** It is the highest single read,
   but v0.2.1 reads 2607.4 on a completely different (pre-chassis) runtime, which should
   make anyone suspicious of ranking on single reads at all.

## The rule this sets

**Treat any cross-version score gap below ~634 points as unresolved.** A version
comparison needs either repeated reads of both submissions over time, or a paired
offline measurement — never two single reads taken at different maturities.

`scripts/audit_live_scores.py --append` accumulates a time series of reads so the
convergence and decay curves can be separated from genuine skill differences. That
separation is the prerequisite for any future claim that one version beat another live,
and it costs no submission slots — only elapsed time.

## Why this was found now

The Jev claim audit (`docs/proposal-2026-09-21-jev-doc-audit.md`) flagged five places
where this repo reported a measurement as evidence and later disowned it, including
`README.md:659` ("an artifact, because every member had been selected for beating it")
and `README.md:1185` ("Previously reported shed-overflow magnitudes... are not
evidence"). Looking for the sixth instance is what prompted checking whether the live
scores were comparable. The audit produced no change to the agent and cannot — it reads
prose, and the simulator is the oracle for anything about routes — but its subject
matter is exactly this failure mode, and the failure mode was live.
