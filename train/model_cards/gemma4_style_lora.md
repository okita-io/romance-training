---
library_name: peft
base_model: google/gemma-4-26B-A4B-it
tags:
- lora
- sft
- style-classification
- prose-evaluator
- unsloth
- trl
pipeline_tag: text-generation
---

# Gemma 4 Style LoRA — prose style evaluator

**Purpose:** This adapter is a **style evaluator / judge**, not a fiction writer.

It classifies and scores prose against Leech & Short–style dimensions (register, POV, tone, FID, figurative density, dialogue density, and related fields) and emits structured JSON style profiles. Those judgments feed a **Phase 5A bake-off**: if agreement with held-out labels is strong enough, the evaluator unlocks steering cards and later training of a **fiction-stylish writer LLM** (and long-form novel merge).

| Role | This model | Not this model |
|---|---|---|
| **Judge** | Yes — classify / score passages | — |
| **Steer** | Provides the signal for steering cards | Does not write novels itself |
| **Writer** | Used later to train / score a writer | Do not treat this LoRA as the stylish fiction generator |

Roadmap: `docs/PHASE5_STYLE_STEERING.md` (Phases 5–6).

## Intended use

- Produce full `style_profile` JSON for a passage
- Score whether a draft matches a target **steering card** (voice / style / tone)
- Power the **5A agreement bake-off** on quantized GGUF (LM Studio / 3090)
- After 5A pass: bulk re-label corpora and drive writer ↔ evaluator loops

## Out of scope

- Open-ended story generation or “write a novel”
- Plot / character consistency (separate stack)
- Literary quality preference ranking without a human agreement study

## Base model

Fine-tuned from Gemma 4 26B-A4B QAT instruct weights via Unsloth LoRA SFT on DGX Spark (`spark-4f07`). Chat template: non-thinking Gemma-4 (JSON-friendly classification).

## Training

- Task mix: style-profile classification + single-dimension judgments with short rationales
- Config: `train/train_config.gemma4_spark.toml`
- Target: 3000 steps; export Q4/Q5 GGUF for inference on the 3090

## Quick eval prompt shape

```text
Provide a complete style profile for this passage as JSON.
Passage:
...
```

Cite TRL / Unsloth as appropriate for the training stack; see repo `README.md` for machine split and data flow.
