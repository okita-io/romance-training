# Style benchmark — Ashenmere

A **fixed, documented prompt suite** for comparing prose style fidelity across training runs, base models, and fine-tuned adapters.

Every generation uses the same fantasy world, seven plots, and Leech & Short style targets so you can measure **deltas** between runs instead of one-off prompts. **Character names are assigned fresh per run** (seeded) so models cannot memorize fixed lead names across baseline vs fine-tune comparisons.

**Prompt shape** mirrors romance-factory Phase 7 act prose (`PromptBuilder.build_act_generation_prompt`): prose-engine system voice, rough-draft length budget, verbosity / narrative-purpose contracts, and a `format_style_targets`-style block — not the older flat Leech-writer system prompt.

## Fixture (`fixture.json`)

| Section | Contents |
|---------|----------|
| **World** | Ashenmere — fractured archipelago, salt-magic, Ash Court |
| **Plots** | 7 romance/fantasy setups with `summary_template` + distinct **style_target** |
| **Scene types** | `opening`, `climax_reveal`, `romantic_encounter` (each maps to chapter/act + romance-focus level) |
| **Names** | Assigned at run time — default **LLM namer** (RF `character_namer` + overused-name gate, up to 100 retries/lead). Use `--name-mode syllable` for offline. |

### Naming telemetry (training metric)

Each run records a `naming` block on every jsonl line and in the `.summary.json`:

```json
{
  "naming": {
    "mode": "llm",
    "model": "llama3.2-moe-ultra-instruct-10b",
    "max_retries": 100,
    "total_characters": 14,
    "total_attempts": 23,
    "mean_attempts": 1.64,
    "max_attempts_used": 4,
    "characters": [
      {
        "plot_id": "plot_01",
        "role": "female_lead",
        "name": "Phaedra Saltwick",
        "attempts": 2,
        "rejections": [
          {"attempt": 1, "name": "Elara Vance", "reason": "overused first name: elara", "code": "overused_first"}
        ]
      }
    ]
  }
}
```

Track **`mean_attempts`** and **`total_attempts`** across training sessions — the goal is for fine-tuned models to escape Elara / Isolde / etc. in fewer tries.

Overused lists: romance-factory `prompt_engineering/overused_llm_names.json` when the submodule is nested, merged with `eval/style_benchmark/overused_llm_names.json` (adds Isolde, Aeliana, …). Override: `STYLE_BENCHMARK_OVERUSED_NAMES_PATH`.

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
# Inspect prompts without LLM calls (names seeded from run_id)
python tools/style_evaluation/run_benchmark.py --dry-run --limit 3

# Reproducible names across two runs (same seed → same roster; syllable mode during dry-run)
python tools/style_evaluation/run_benchmark.py --dry-run --name-seed my-baseline-v1 --limit 1

# Live run with LLM naming (default) — watch naming attempts in summary
python tools/style_evaluation/run_benchmark.py \\
  --label baseline_llama10b \\
  --name-seed baseline-v1 \\
  --max-name-retries 100 \\
  --output eval/style_benchmark/results/baseline_llama10b.jsonl

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

# Conformity trend across training sessions (ordered: baseline first)
python tools/style_evaluation/run_benchmark.py compare-sessions \
  baseline:eval/style_benchmark/results/baseline_llama10b.jsonl \
  batch_001:eval/style_benchmark/results/batch001_finetuned.jsonl \
  batch_002:eval/style_benchmark/results/batch002_finetuned.jsonl \
  --output eval/style_benchmark/results/training_trend.json
```

## Interpreting deltas

- **`match_score`** — fraction of target LLM dimensions exactly matched by the classifier (0–1).
- Compare **`mean_match_score`** across runs; positive **`delta`** in `compare` means the candidate run classified closer to target.
- Use **`compare-sessions`** with ordered result files to track conformity across training iterations: step deltas, per-field hit-rate trends, plot/scene breakdowns, and naming `mean_attempts` when present.
- Use **`field_hit_rate`** in the summary to see which rubric dimensions transfer (e.g. `register`, `pov`) vs which stay noisy (`mind_style`, `free_indirect_discourse`).

The classifier is part of the measurement loop — for fair before/after comparisons, use the **same classifier model and `--classify-pass`** across runs.

## Related files

- `source/style_rubric.json` — dimension definitions
- `tools/style_classification/classify_passage.py` — classification
- `tools/style_evaluation/benchmark.py` — prompt + delta helpers (RF-aligned Phase 7 shape)
- `romance-factory` — `src/romance_factory/generate/prompt_builder.py`, `src/romance_factory/style/targets.py` (reference prompts)
