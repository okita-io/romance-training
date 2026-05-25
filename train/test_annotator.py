#!/usr/bin/env python3
"""Test the metadata annotator on a single sample."""

import json
import sys
from pathlib import Path

sys.path.insert(0, '/Users/alexokita/romance-factory/src')

from romance_factory.annotator import RomanceMetadataAnnotator, LMStudioBackend

pg_train = Path("/Users/alexokita/romance-corpus/sources/project_gutenberg/train.jsonl")
if not pg_train.exists():
    print(f"Error: {pg_train} not found")
    sys.exit(1)

with open(pg_train) as f:
    first = json.loads(f.readline())

text = first.get('text', '')[:800]
title = first.get('metadata', {}).get('title', 'Untitled')

print("=== TESTING ANNOTATOR ===")
print(f"Sample (8000 chars): {text[:8000]}...")
print(f"Title: {title}")
print("\nAnalyzing with LM Studio...")

backend = LMStudioBackend()
annotator = RomanceMetadataAnnotator(backend)

try:
    result = annotator.analyze_text(text, title)
    print("\n✅ Analysis successful!")
    print(f"  Genres: {[g.value for g in result.genre]}")
    print(f"  Heat: {result.heat_level.value}")
    print(f"  Plot types: {[pt.value for pt in result.plot_types]}")
    print(f"  Tropes: {[t.value for t in result.tropes][:5]}")
    print(f"  Cliffhanger: {result.cliffhanger} (strength: {result.cliffhanger_strength})")
    print(f"  Plot twist: {result.plot_twist} (strength: {result.plot_twist_strength})")
    print(f"  POV: {result.pov}")
    print(f"  Emotional tone: {result.emotional_tone}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
