"""Enrich romance corpus with detailed LLM-generated metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Requires editable install of romance-factory: pip install -e "../romance-factory"
from romance_factory.annotator import LMStudioBackend, RomanceMetadataAnnotator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument("--output", type=Path, help="Output JSONL file")
    parser.add_argument("--limit", type=int, help="Limit samples")
    parser.add_argument("--lmstudio-url", default="http://127.0.0.1:1234", help="LM Studio URL")
    args = parser.parse_args()

    if not args.output:
        args.output = args.input.with_name(f"{args.input.stem}_enriched{args.input.suffix}")

    print(f"Enriching: {args.input} → {args.output}")

    backend = LMStudioBackend(base_url=args.lmstudio_url)
    annotator = RomanceMetadataAnnotator(backend)
    annotator.annotate_jsonl(args.input, args.output, limit=args.limit)

    print(f"✅ Done. Cache: {annotator.cache_hits} hits, {annotator.cache_misses} misses")


if __name__ == "__main__":
    main()
