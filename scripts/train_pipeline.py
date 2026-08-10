import glob
import os
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
REPLAYS_DIR = os.path.join(PROJECT_ROOT, "replays")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")


def run_cmd(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=PROJECT_ROOT)


def main():
    start_date = datetime(2026, 8, 7)
    end_date = datetime(2026, 8, 9)

    current_date = start_date
    dates = []
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    os.makedirs(REPLAYS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print(f"Pipeline will process {len(dates)} days of Kaggle matches.")

    # 1. Download and parse day by day
    for d in dates:
        chunk_file = os.path.join(LOGS_DIR, f"dataset_{d}.npz")
        if os.path.exists(chunk_file):
            print(f"Chunk {chunk_file} already exists, skipping download.")
            continue

        slug = f"kaggle/kaggriculture-episodes-{d}"
        print(f"\n--- Processing {slug} ---")

        # Clean replays dir before downloading
        run_cmd(f"rm -rf {REPLAYS_DIR}/*")

        # Download
        print("Downloading from Kaggle...")
        try:
            # We use --unzip directly
            run_cmd(f"uv run kaggle datasets download -d {slug} -p {REPLAYS_DIR} --unzip")
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to download {slug}. It may not exist. Skipping.")
            continue

        # Parse
        print("Parsing JSON replays into ML dataset...")
        run_cmd(
            f"uv run python -m rl.dataset_builder --replay-dir {REPLAYS_DIR} --output {chunk_file}"
        )

        # Cleanup
        print("Cleaning up raw JSON files...")
        run_cmd(f"rm -rf {REPLAYS_DIR}/*")

    # 2. Merge chunks
    print("\n--- Merging all dataset chunks ---")
    chunk_files = glob.glob(os.path.join(LOGS_DIR, "dataset_2026-*.npz"))
    if not chunk_files:
        print("No dataset chunks found! Aborting.")
        sys.exit(1)

    all_data = {"planes": [], "globals": [], "farmer_acts": [], "hands_acts": [], "market_acts": []}

    for f in sorted(chunk_files):
        print(f"Loading {f}...")
        try:
            with np.load(f) as data:
                for k in all_data:
                    if k in data:
                        all_data[k].append(data[k])
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    print("Concatenating arrays...")
    merged = {}
    for k, v_list in all_data.items():
        if v_list:
            merged[k] = np.concatenate(v_list, axis=0)

    master_path = os.path.join(LOGS_DIR, "offline_rl_dataset.npz")
    print(f"Saving master dataset to {master_path}...")
    np.savez_compressed(master_path, **merged)
    print(f"Master dataset size: {len(merged.get('planes', []))} samples.")

    # 3. Train
    print("\n--- Training Model ---")
    run_cmd("uv run python -m rl.train_offline --epochs 10")

    # 4. Export and build submission
    print("\n--- Building Submission ---")
    run_cmd("uv run python scripts/build_submission.py")

    print("\nPipeline Complete!")


if __name__ == "__main__":
    main()
