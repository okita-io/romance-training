#!/usr/bin/env python3
"""Normalize Project Gutenberg romance corpus to YouTube format."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from paths import TRAINING, project_gutenberg_source
from tools.data_preparation.gutenberg_corpus import clean_gutenberg_prose


def infer_category(text: str, title: str = "") -> str:
    text_lower = (title + " " + text).lower()
    if any(w in text_lower for w in ["vampire", "werewolf", "shifter", "supernatural", "witch"]):
        return "paranormal"
    if any(w in text_lower for w in ["duke", "earl", "lord", "lady", "regency", "victorian", "historical"]):
        return "classic"
    if any(w in text_lower for w in ["billionaire", "millionaire", "ceo", "tycoon", "mogul"]):
        return "billionaire"
    if any(w in text_lower for w in ["gothic", "haunted", "mansion", "mystery"]):
        return "gothic"
    return "contemporary"


def infer_heat_level(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["explicit", "graphic", "hardcore", "bdsm", "kink", "penetration"]):
        return "explicit"
    if any(w in text_lower for w in ["steamy", "sensual", "erotic", "pleasure", "intimate", "body", "naked"]):
        return "steamy"
    if any(w in text_lower for w in ["sweet", "clean", "innocent", "chaste", "kiss"]):
        return "sweet"
    if any(w in text_lower for w in ["mild", "tasteful", "suggestion"]):
        return "mild"
    return "moderate"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def clean_text(text: str) -> str:
    return clean_gutenberg_prose(text)


def main() -> None:
    source_path = project_gutenberg_source()
    output_dir = TRAINING["project_gutenberg_normalized"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise SystemExit(f"Project Gutenberg source not found: {source_path}")

    print(f"Loading Project Gutenberg corpus from {source_path}")
    with open(source_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(entries)} raw entries")

    all_samples = []
    for i, entry in enumerate(entries):
        raw_text = entry.get("text", "")
        if not raw_text or len(raw_text.split()) < 100:
            continue

        text = clean_text(raw_text)
        if len(text.split()) < 100:
            continue

        category = infer_category(text, entry.get("title", ""))
        heat_level = infer_heat_level(text)
        chunks = chunk_text(text, chunk_size=500, overlap=50)

        for chunk_idx, chunk in enumerate(chunks):
            all_samples.append(
                {
                    "text": chunk,
                    "metadata": {
                        "source": "project_gutenberg",
                        "book_id": i,
                        "category": category,
                        "heat_level": heat_level,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "chunk_size": 500,
                        "chunk_overlap": 50,
                        "word_count": len(chunk.split()),
                    },
                }
            )

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(entries)} entries → {len(all_samples)} chunks")

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

    print("\n✅ Normalized Project Gutenberg corpus:")
    print(f"   Train: {len(train_samples)} samples")
    print(f"   Val: {len(val_samples)} samples")
    print(f"   Output: {output_dir}")


if __name__ == "__main__":
    main()
