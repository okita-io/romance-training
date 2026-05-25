#!/usr/bin/env python3
"""Run all data preparation pipelines to normalize sources to YouTube format."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PREP_DIR = Path(__file__).resolve().parent


def run_script(script_path: Path, description: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print("=" * 60)
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(PREP_DIR), check=False)
    if result.returncode != 0:
        print(f"❌ {description} failed with code {result.returncode}")
        return False
    print(f"✅ {description} completed")
    return True


def main() -> None:
    scripts = [
        (PREP_DIR / "prepare_project_gutenberg.py", "Project Gutenberg normalization"),
        (PREP_DIR / "prepare_fiction1b.py", "Fiction-1B normalization"),
    ]

    success = True
    for script, desc in scripts:
        if not run_script(script, desc):
            success = False
            break

    if success:
        print(f"\n{'=' * 60}")
        print("✅ ALL PREPARATION COMPLETE")
        print("=" * 60)
        print("\nNext: Combine all normalized datasets:")
        print("  python tools/data_preparation/combine_all_datasets.py")
    else:
        print("\n❌ Some preparation steps failed. Check logs.")


if __name__ == "__main__":
    main()
