#!/usr/bin/env python3
"""Normalize Fiction-1B enhanced dataset to YouTube format."""

from __future__ import annotations

import json
import random
from pathlib import Path

from paths import TRAINING, fiction1b_source


def normalize_metadata(meta: dict) -> dict:
    return {
        "source": meta.get("source", "fiction1b"),
        "category": meta.get("category", "contemporary"),
        "heat_level": meta.get("heat_label", meta.get("original_heat", "moderate")),
        "confidence": meta.get("confidence", 1.0),
        "video_id": meta.get("sample_index", "unknown"),
        "title": f"Fiction1B Sample {meta.get('sample_index', '')}",
        "tags": [],
        "word_count": meta.get("text_length", 0) // 5,
        "chunk_size": 500,
        "chunk_overlap": 50,
        "chunk_index": 0,
        "total_chunks": 1,
    }


def should_include_sample(meta: dict, min_confidence: float = 0.6) -> bool:
    return meta.get("confidence", 1.0) >= min_confidence


def main() -> None:
    f1b_root = fiction1b_source()
    output_dir = TRAINING["fiction1b_normalized"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if not f1b_root.exists():
        raise SystemExit(f"Fiction-1B source not found: {f1b_root}")

    print(f"Scanning Fiction-1B dataset at {f1b_root}")

    jsonl_files = list(f1b_root.rglob("*.jsonl"))
    jsonl_files = [p for p in jsonl_files if "training_by_heat" not in str(p) and "editorial_evals" not in str(p)]

    print(f"Found {len(jsonl_files)} JSONL files to process")

    all_samples = []
    rejected = 0

    for file_idx, filepath in enumerate(jsonl_files):
        try:
            with open(filepath, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as exc:
            print(f"  Skipping {filepath.name}: {exc}")
            continue

        relative = filepath.relative_to(f1b_root)

        for line_idx, line in enumerate(lines):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = entry.get("text", "")
            if not text or len(text.split()) < 100:
                continue

            meta_raw = entry.get("metadata", {})
            if not should_include_sample(meta_raw):
                rejected += 1
                continue

            meta = normalize_metadata(meta_raw)
            meta["source_file"] = str(relative)
            meta["file_line"] = line_idx
            all_samples.append({"text": text, "metadata": meta})

        if (file_idx + 1) % 20 == 0:
            print(f"  Processed {file_idx + 1}/{len(jsonl_files)} files → {len(all_samples)} samples")

    print(f"\nTotal samples collected: {len(all_samples)}")
    print(f"Rejected (low confidence): {rejected}")

    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    cats: dict[str, int] = {}
    heats: dict[str, int] = {}
    for sample in all_samples:
        cat = sample["metadata"]["category"]
        heat = sample["metadata"]["heat_level"]
        cats[cat] = cats.get(cat, 0) + 1
        heats[heat] = heats.get(heat, 0) + 1

    print("\n✅ Normalized Fiction-1B corpus:")
    print(f"   Train: {len(train_samples)} samples")
    print(f"   Val: {len(val_samples)} samples")
    print(f"   Output: {output_dir}")
    print(f"\nCategories: {cats}")
    print(f"Heat levels: {heats}")


if __name__ == "__main__":
    main()
