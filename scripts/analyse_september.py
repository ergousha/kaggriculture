#!/usr/bin/env python3
"""Analyze the September field cohort against our v0.4.0 chassis profile.

Reads the September (2026-09-13 daily batch, 654 episodes / 1308 seats)
fingerprints from logs/daily_fingerprints.csv and answers the Phase-1
questions from docs/proposal-2026-09-14-endgame.md:

  1. What does the cash distribution look like, and where do leaders sit?
  2. What separates top-decile seats from the chassis-class mean?
  3. Route/portfolio signals: TOMATO/CARROT/EGG pot products, herd mix,
     fertilizer use, wheat handling, shed waste.
"""

from __future__ import annotations

import sys

import pandas as pd

CSV = "logs/daily_fingerprints.csv"


def main() -> None:
    df = pd.read_csv(CSV)
    # The batch rows were appended with the kagglehub cache path as source.
    sep = df[df["source"].str.contains("episodes-2026-09-13", na=False)]
    if len(sep) == 0:
        sep = df.tail(1308)
    print(f"September-09-13 seats: {len(sep)} from {sep['episode_id'].nunique()} episodes")

    print("\n--- cash distribution ---")
    print(sep["cash"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(0).to_string())

    top = sep[sep["cash"] >= sep["cash"].quantile(0.9)]
    mid = sep[(sep["cash"] >= sep["cash"].quantile(0.4)) & (sep["cash"] < sep["cash"].quantile(0.6))]

    cols = [
        "hires", "land_buys", "owned_tiles", "hands", "ops_productive",
        "rev_WHEAT", "rev_STRAWBERRY", "rev_MELON", "rev_MILK", "rev_WOOL",
        "rev_TOMATO", "rev_EGG", "rev_CARROT", "rev_FERTILIZER",
        "sold_WHEAT", "sold_STRAWBERRY", "sold_TOMATO", "sold_EGG", "sold_MELON",
        "animal_SHEEP", "animal_COW", "animal_GOOSE",
        "fertilized_tiles", "unsold_shed_units",
        "price_STRAWBERRY_end", "price_WOOL_end", "price_MILK_end", "price_MELON_end",
    ]
    have = [c for c in cols if c in sep.columns]
    cmp = pd.DataFrame({"top10%": top[have].mean(), "mid": mid[have].mean()})
    cmp["delta"] = cmp["top10%"] - cmp["mid"]
    print("\n--- top decile vs middle band (means per seat) ---")
    print(cmp.round(1).to_string())

    # how many seats farm the pot products at all
    print("\n--- pot-product participation ---")
    for prod in ("TOMATO", "EGG", "CARROT"):
        sellers = (sep[f"rev_{prod}"] > 0).sum()
        big = (sep[f"rev_{prod}"] >= 2000).sum()
        print(
            f"{prod}: {sellers}/{len(sep)} seats sell any ({sellers / len(sep):.0%}), "
            f"{big} realize >= $2k, mean rev ${sep[f'rev_{prod}'].mean():.0f}"
        )

    # wheat: do leaders dump early?
    print("\n--- wheat & fertilizer handling ---")
    for label, grp in (("top10%", top), ("mid", mid)):
        print(
            f"{label}: rev_WHEAT ${grp['rev_WHEAT'].mean():.0f} "
            f"sold_WHEAT {grp['sold_WHEAT'].mean():.0f}u "
            f"bought_WHEAT {grp['bought_WHEAT'].mean():.0f}u "
            f"fert rev ${grp['rev_FERTILIZER'].mean():.0f} "
            f"fert sold {grp['sold_FERTILIZER'].mean():.0f}u"
        )

    # shed waste and liquidation
    print("\n--- endgame waste ---")
    for label, grp in (("top10%", top), ("mid", mid)):
        print(f"{label}: unsold_shed_units {grp['unsold_shed_units'].mean():.1f}")

    # team-level: which named teams appear (from the spy runs, if any matched)
    if "team" in sep.columns:
        counts = sep["team"].value_counts()
        print("\n--- most-sampled teams (top 15) ---")
        print(counts.head(15).to_string())


if __name__ == "__main__":
    sys.exit(main())