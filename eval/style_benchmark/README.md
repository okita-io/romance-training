# Style benchmark — Ashenmere

A **fixed, documented prompt suite** for comparing prose style fidelity across training runs, base models, and fine-tuned adapters.

Every generation uses the same fantasy world, character roster, seven plots, and Leech & Short style targets so you can measure **deltas** between runs instead of one-off prompts.

## Fixture (`fixture.json`)

| Section | Contents |
|---------|----------|
| **World** | Ashenmere — fractured archipelago, salt-magic, Ash Court |
| **Female leads** | 7 names (Aeliana Thornweave … Isolde Ravencrest) |
| **Male leads** | 7 names (Cassian Vale … Lucien Merrow) |
| **Plots** | 7 romance/fantasy setups, each with a distinct **style_target** |
| **Scene types** | `opening`, `climax_reveal`, `romantic_encounter` |

### Style mixes (one per plot)

| Plot | Style label | Emphasis |
|------|-------------|----------|
| plot_01 | Courtly formal omniscient | Lexical + hypotaxis + distant omniscient |
| plot_02 | Intimate first-person FID | Viewpoint + melancholic tone |
| plot_03 | Colloquial paratactic thriller | Simple lexis + tense + minimal segmentation |
| plot_04 | Lyrical figurative third-limited | High figurative density + end-focus |
| plot_05 | Sardonic distant social satire | Tone + narrative distance |
| plot_06 | Archaic epic hypotactic | Register + subordination salience |
| plot_07 | Mixed POV comedic romantic | Mixed POV + comedic tone |

Each `style_target` sets all 16 LLM-scored dimensions from `source/style_rubric.json` (lexical, grammatical, figurative, cohesion, viewpoint, context, textual principles).

**Total benchmark size:** 7 plots × 3 scenes = **21 generations** per run.

## Two-pass workflow

1. **Pass 1 (optional)** — world/roster primer  
   `--setup-only` writes one continuity document from the shared fixture.

2. **Pass 2 (default)** — scene generation + classification  
   For each plot × scene, the runner:
   - builds a style-conditioned generation prompt
   - calls the **generator** model
   - classifies output with `classify_passage` (default `--classify-pass both`)
   - records **delta** (target vs classified profile)

## Result record schema

Each line in `results/*.jsonl`:

```json
{
  "run_id": "20260629T120000Z_baseline",
  "label": "baseline_llama10b",
  "plot_id": "plot_04",
  "scene_type": "climax_reveal",
  "style_label": "Lyrical figurative third-limited",
  "style_target": { "tone": "lyrical", "figurative_density": "high", "..." : "..." },
  "model": "llama3.2-moe-ultra-instruct-10b",
  "classifier_model": "llama3.2-moe-ultra-instruct-10b",
  "prompt_system": "...",
  "prompt_user": "...",
  "generated_text": "...",
  "classification": { "register": "formal_literary", "tone": "lyrical", "..." : "..." },
  "delta": {
    "match_score": 0.6875,
    "match_count": 11,
    "compared_count": 16,
    "fields": { "tone": { "target": "lyrical", "actual": "lyrical", "match": true } }
  }
}
```

Use **`label`** to tag training iterations: `baseline`, `batch_001_pretrained`, `batch_001_finetuned`, etc.

## Commands

```bash
# Inspect prompts without LLM calls
python tools/style_evaluation/run_benchmark.py --dry-run --limit 3

# Full run (generation + classification)
LLM_MODEL=llama3.2-moe-ultra-instruct-10b
python tools/style_evaluation/run_benchmark.py \
  --label baseline_llama10b \
  --output eval/style_benchmark/results/baseline_llama10b.jsonl

# Single plot smoke test
python tools/style_evaluation/run_benchmark.py \
  --plot plot_03 --scene opening --label smoke

# Generate only (classify later with a fixed classifier)
python tools/style_evaluation/run_benchmark.py \
  --label finetuned_gen --no-classify

python tools/style_evaluation/run_benchmark.py reclassify \
  eval/style_benchmark/results/finetuned_gen.jsonl \
  --classifier-model llama3.2-moe-ultra-instruct-10b

# Summarize one run
python tools/style_evaluation/run_benchmark.py summarize \
  eval/style_benchmark/results/baseline_llama10b.jsonl

# Compare baseline vs after fine-tune
python tools/style_evaluation/run_benchmark.py compare \
  eval/style_benchmark/results/baseline_llama10b.jsonl \
  eval/style_benchmark/results/batch001_finetuned.jsonl \
  --output eval/style_benchmark/results/batch001_comparison.json
```

## Interpreting deltas

- **`match_score`** — fraction of target LLM dimensions exactly matched by the classifier (0–1).
- Compare **`mean_match_score`** across runs; positive **`delta`** in `compare` means the candidate run classified closer to target.
- Use **`field_hit_rate`** in the summary to see which rubric dimensions transfer (e.g. `register`, `pov`) vs which stay noisy (`mind_style`, `free_indirect_discourse`).

The classifier is part of the measurement loop — for fair before/after comparisons, use the **same classifier model and `--classify-pass`** across runs.

## Related files

- `source/style_rubric.json` — dimension definitions
- `tools/style_classification/classify_passage.py` — classification
- `tools/style_evaluation/benchmark.py` — prompt + delta helpers
