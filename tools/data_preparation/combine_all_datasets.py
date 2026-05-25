#!/usr/bin/env python3
"""Combine all normalized training datasets into final unified corpus."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from paths import PROCESSED, TRAINING


def load_dataset(path: Path, source_name: str) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"  Skipping {source_name}: {path} not found")
        return []
    samples: list[dict[str, Any]] = []
    for split_path in (path / "train.jsonl", path / "validation.jsonl"):
        if not split_path.exists():
            continue
        with open(split_path, encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample.setdefault("metadata", {})
                sample["metadata"].setdefault("source", source_name)
                samples.append(sample)
    print(f"  Loaded {len(samples)} samples from {source_name}")
    return samples


def main() -> None:
    datasets = {
        "youtube": TRAINING["youtube_combined"],
        "project_gutenberg": TRAINING["project_gutenberg_normalized"],
        "fiction1b": TRAINING["fiction1b_normalized"],
    }

    print("=== COMBINING ALL NORMALIZED DATASETS ===\n")
    print(f"Corpus root: {PROCESSED.parent.parent}\n")

    all_samples: list[dict[str, Any]] = []
    total_words = 0
    for name, path in datasets.items():
        samples = load_dataset(path, name)
        all_samples.extend(samples)
        if samples:
            words = sum(len(sample["text"].split()) for sample in samples)
            total_words += words
            print(f"    {name}: {len(samples)} samples, ~{words:,} words")

    print(f"\nTotal before deduplication: {len(all_samples)} samples, ~{total_words:,} words")

    random.seed(42)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.9)
    train = all_samples[:split_idx]
    val = all_samples[split_idx:]

    output_dir = TRAINING["final_combined"]
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for sample in val:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    train_words = sum(len(sample["text"].split()) for sample in train)
    val_words = sum(len(sample["text"].split()) for sample in val)
    cats = Counter(sample["metadata"].get("category", "unknown") for sample in all_samples)
    heats = Counter(sample["metadata"].get("heat_level", "unknown") for sample in all_samples)
    sources = Counter(sample["metadata"].get("source", "unknown") for sample in all_samples)

    print("\n✅ FINAL COMBINED DATASET")
    print(f"   Train: {len(train)} samples ({train_words:,} words)")
    print(f"   Val: {len(val)} samples ({val_words:,} words)")
    print(f"   Total: {len(all_samples)} samples ({train_words + val_words:,} words)")
    print(f"\n   By source: {dict(sources)}")
    print(f"   By category: {dict(cats)}")
    print(f"   By heat level: {dict(heats)}")
    print(f"\n   Location: {output_dir}")


if __name__ == "__main__":
    main()
