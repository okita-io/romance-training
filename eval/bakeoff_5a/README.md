# Phase 5A — Evaluator bake-off

Held-out agree/reject study for the **Gemma 4 style GGUF** judge (step 3000).

Gate for promoting the GGUF to the default Phase 2 classifier and unlocking **5B / 6A**. See `docs/PHASE5_STYLE_STEERING.md`.

## Setup (RTX 3090 + LM Studio)

1. Download / copy `gemma4-style-step3000-q4_k_m.gguf` (or Q5) from Hugging Face / Spark.
2. Load in LM Studio. Recommended settings: temp `1.0`, top_p `0.95`, top_k `64`, **thinking off**.
3. Enable the local server (`http://localhost:1234/v1`).
4. Point the bake-off at that model:

```bash
export LLM_BASE_URL=http://localhost:1234/v1
export LLM_MODEL=gemma4-style-step3000-q4_k_m   # exact LM Studio id
export LLM_DISABLE_THINKING=1
```

From Spark, use the 3090 host instead of localhost (see repo `.env` `LLM_BASE_URL`).

## Commands

```bash
# Build fixed eval set (30 gold + 10 traps) — can run on Spark, no GPU needed
python tools/style_evaluation/bakeoff_5a.py build-set \
  --output eval/bakeoff_5a/eval_set.jsonl

# Run judge (needs LM Studio with Gemma GGUF loaded)
python tools/style_evaluation/bakeoff_5a.py run \
  --eval-set eval/bakeoff_5a/eval_set.jsonl \
  --output eval/bakeoff_5a/results/gemma4_step3000_q4.jsonl

# Smoke test first 3 passages
python tools/style_evaluation/bakeoff_5a.py run --limit 3

# Re-summarize an existing results file
python tools/style_evaluation/bakeoff_5a.py summarize \
  eval/bakeoff_5a/results/gemma4_step3000_q4.jsonl
```

Outputs beside the results JSONL:

| File | Contents |
|------|----------|
| `*.agree.jsonl` | Passages where all gate fields exact-match + JSON parsed |
| `*.reject.jsonl` | Mismatches / parse failures |
| `*.summary.json` | Hit rates + go / no-go |

## Gate fields

`tone`, `pov`, `register`, `free_indirect_discourse`, `figurative_density`

Default thresholds (see `DEFAULT_THRESHOLDS` in `bakeoff_5a.py`): JSON parse ≥ 90%, full-gate exact ≥ 60%, plus per-field floors.

## Trap set

Auto-sampled rare / conflicting profiles: mixed POV, heavy FID, archaic register, first person, comedic tone, low figurative density.
