#!/bin/bash
# Download yesterday's replays using mine_daily.py
YESTERDAY=$(date -v-1d +%Y-%m-%d)
echo "Downloading dataset kaggriculture-episodes-${YESTERDAY}..."
uv run python scripts/mine_daily.py --dataset kaggriculture-episodes-${YESTERDAY}
