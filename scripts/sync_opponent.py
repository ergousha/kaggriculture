#!/usr/bin/env python3
import os
import re
import shutil
import subprocess


def main():
    try:
        with open("main.py") as f:
            content = f.read()
    except FileNotFoundError:
        print("main.py not found. Skipping sync.")
        return

    match = re.search(r'^AGENT_VERSION\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if not match:
        print("AGENT_VERSION not found in main.py. Skipping sync.")
        return

    version = match.group(1)
    version_str = version.replace(".", "_")
    target_file = f"opponents/v{version_str}.py"

    os.makedirs("opponents", exist_ok=True)

    if not os.path.exists(target_file):
        print(f"Sync Opponent: Creating new opponent version {target_file}")
        shutil.copy("main.py", target_file)
        subprocess.run(["git", "add", target_file], check=True)
    else:
        with open(target_file) as f:
            target_content = f.read()

        if content != target_content:
            print(f"Sync Opponent: Updating existing opponent file {target_file}")
            shutil.copy("main.py", target_file)
            subprocess.run(["git", "add", target_file], check=True)


if __name__ == "__main__":
    main()
