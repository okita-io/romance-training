#!/usr/bin/env python3
"""Download Hugging Face datasets into source-data/hf/."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from unified_corpus import repo_dir_name, slug_from_repo_id

ROOT = Path(__file__).resolve().parents[2]
HF_ROOT = ROOT / "source-data" / "hf"
MANIFESTS = ROOT / "source-data" / "manifests"


def _run_hf_download(repo_id: str, local_dir: Path, include: str | None) -> None:
    cmd = [
        "hf",
        "download",
        repo_id,
        "--type",
        "dataset",
        "--local-dir",
        str(local_dir),
    ]
    if include:
        cmd.extend(["--include", include])
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def download_dataset(
    repo_id: str,
    *,
    include: str | None = None,
    force: bool = False,
) -> Path:
    local_dir = HF_ROOT / repo_dir_name(repo_id)
    if local_dir.exists() and not force:
        print(f"Already present: {local_dir}")
        return local_dir

    HF_ROOT.mkdir(parents=True, exist_ok=True)
    _run_hf_download(repo_id, local_dir, include)

    slug = slug_from_repo_id(repo_id)
    manifest = MANIFESTS / f"{slug}.json"
    if not manifest.exists():
        print(
            f"Note: no manifest at {manifest.relative_to(ROOT)} — "
            "convert_hf_sources.py will auto-detect fields, or add a manifest for custom mapping."
        )
    return local_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="HF dataset id, e.g. TristanBehrens/lovecraftcorpus")
    parser.add_argument(
        "--include",
        default=None,
        help="Glob of files to download (default: entire repo)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the local directory exists",
    )
    args = parser.parse_args()

    try:
        path = download_dataset(args.repo_id, include=args.include, force=args.force)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except FileNotFoundError:
        raise SystemExit(
            "hf CLI not found. Install: curl -LsSf https://hf.co/cli/install.sh | bash -s"
        ) from None

    print(f"Downloaded to {path}")


if __name__ == "__main__":
    main()
