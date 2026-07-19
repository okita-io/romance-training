---
library_name: peft
base_model: google/gemma-4-26B-A4B-it
base_model_relation: adapter
tags:
- lora
- sft
- style-classification
- prose-evaluator
- stylistics
- unsloth
- trl
- gemma4
license: gemma
language:
- en
pipeline_tag: text-generation
---

# Gemma 4 Style LoRA (step 3000) — prose style evaluator

**Purpose:** This adapter is a **style evaluator / judge**, not a fiction writer.

It classifies and scores English prose against a **14-metric Leech & Short–derived rubric** and emits structured JSON `style_profile`s (plus short natural-language judgments). Those judgments power a bake-off and later **steering cards** used to train / score a stylish fiction writer and assemble long-form novels that stay on voice, style, and tone.

| Role | This model | Not this model |
|---|---|---|
| **Judge** | Yes — classify / score passages | — |
| **Steer** | Provides signal for steering cards | Does not write novels itself |
| **Writer** | Used later to train / score a writer | Do not treat this LoRA as a story generator |

Companion GGUFs (merged + quantized for LM Studio / llama.cpp): [`alexokita/gemma4-style-step3000-GGUF`](https://huggingface.co/alexokita/gemma4-style-step3000-GGUF).

## Intended use

- Produce a full `style_profile` JSON for a prose passage
- Answer single-dimension questions (tone, POV, figurative density, FID, …) with a short rationale
- Score whether a draft matches a target **steering card** (voice / style / tone)
- Run agreement bake-offs vs held-out labeled prose (and Style-in-Fiction exemplars)
- After a successful bake-off: bulk re-label corpora and drive writer ↔ evaluator loops

**Recommended inference settings (LM Studio / llama.cpp):** temperature `1.0`, top_p `0.95`, top_k `64`, **thinking off** for JSON / classification tasks. Prefer the Q4_K_M or Q5_K_M GGUF for day-to-day use on a 24 GB GPU.

## Out of scope

- Open-ended story generation or “write a novel”
- Plot / character consistency (separate stack)
- Overall literary quality preference ranking without a human agreement study

## How it was trained

| Item | Value |
|---|---|
| **Base** | Gemma 4 26B-A4B instruct QAT weights (`unsloth/gemma-4-26B-A4B-it-qat-q4_0-unquantized`, HF parent of the QAT GGUF family) |
| **Method** | LoRA SFT via Unsloth + TRL on **DGX Spark** (`spark-4f07`, ~128 GB unified Blackwell + CUDA) |
| **Config** | `train/train_config.gemma4_spark.toml` |
| **LoRA** | rank 16, alpha 16, text-only MoE targets (`experts.gate_up_proj`, `experts.down_proj` + attention/MLP projections) |
| **Sequence** | 2048 tokens |
| **Batch** | microbatch 8 × grad accum 2 (effective 16) |
| **Steps** | **3000** (final checkpoint; best eval loss ≈ **2.378**) |
| **LR** | 2e-4 with warmup |
| **Chat template** | Non-thinking `gemma-4` (JSON-friendly) |
| **Host** | Spark only — RTX 3090 is for quantized inference, not training |

Training mix (Phase 3 instruction pairs from classified prose):

- **Classification (~25%)** — “Provide a complete style profile…” → full JSON of the 14 metrics (+ related computables in the profile)
- **Judgment (~75%)** — “What tone / POV / figurative density…?” → bold label + short explanation

Approx. **430k** train / **48k** validation pairs under `train/style_training/`.

## Training data

Styled fiction chunks labeled with `metadata.style_profile`, then converted to instruction pairs. Corpora include (HF sources):

- Literotica stories
- Erotic / fantasy / general fiction book corpora (e.g. Korshuk BookRix sets)
- Project Gutenberg fiction
- Horror novel chunks

Labels were produced by the Phase 2 pipeline (`tools/style_classification/run_pipeline.py`): **computable** metrics (spaCy / textstat) plus **LLM** semantic metrics grounded in a Style-in-Fiction rubric and RAG knowledge base (`source/style_rubric.json`, `source/extracted/style_knowledge.jsonl`).

## The 14 Style-in-Fiction rubric metrics

Derived from Leech & Short, *Style in Fiction* (2nd ed.), operationalized in `source/style_rubric.json` (`stats.dimensions = 14`).

| # | Metric ID | Category | How generated | Values / type |
|---|---|---|---|---|
| 1 | `lexical_complexity` | Lexical | LLM (Pass 1) | `simple_colloquial` / `neutral` / `complex_literary` |
| 2 | `lexical_density` | Lexical | Computable | continuous 0–1 (content-word ratio) |
| 3 | `register` | Lexical | LLM (Pass 1) | `formal_literary`, `formal_technical`, `neutral_narrative`, `colloquial`, `dialect`, `archaic` |
| 4 | `sentence_complexity` | Grammatical | LLM (Pass 1) | `simple_paratactic` / `moderate` / `complex_hypotactic` |
| 5 | `sentence_length_mean` | Grammatical | Computable | mean words/sentence |
| 6 | `subordination_ratio` | Grammatical | Computable | hypotaxis proxy |
| 7 | `figurative_density` | Figurative | LLM (Pass 1) | `low` / `moderate` / `high` |
| 8 | `cohesion` | Cohesion | LLM (Pass 1) | `loose` / `standard` / `tight` |
| 9 | `dialogue_ratio` | Grammatical | Computable | continuous 0–1 (direct speech share) |
| 10 | `pov` | Viewpoint | LLM (Pass 1) | `first_person`, `second_person`, `third_limited`, `third_omniscient`, `mixed` |
| 11 | `mind_style` | Viewpoint | LLM (Pass 2) | `standard` / `distinct` / `deviant` |
| 12 | `narrative_distance` | Viewpoint | LLM (Pass 2) | `intimate` / `moderate` / `distant` |
| 13 | `free_indirect_discourse` | Viewpoint | LLM (Pass 2) | `none` / `sparse` / `moderate` / `heavy` |
| 14 | `tone` | Context | LLM (Pass 2) | `neutral`, `lyrical`, `sardonic`, `melancholic`, `comedic`, `tense`, `contemplative` |

**How those labels were generated:**

1. **Phase 1 — Rubric from Style in Fiction** — Vision-transcribe *Style in Fiction* PDF → markdown; chunk into RAG `style_knowledge.jsonl`; distill checklist + 14 operational `dimensions` into `style_rubric.json`.
2. **Phase 2 — Classify prose** — Chunk fiction (~sentence-aware); compute spaCy/textstat fields for the four **computable** metrics; run multi-pass LLM classification with rubric + retrieved SiF excerpts (`--pass both`: Pass 1 lexical/syntax/discourse + POV; Pass 2 tone / mind style / narrative distance / FID).
3. **Phase 3 — SFT pairs** — Turn each styled row into classification + judgment instruction pairs for this LoRA.

Additional textual principles in the rubric (`end_focus`, `segmentation`, `prose_rhythm`, `climax`, …) may appear in full profiles from the classifier pipeline; the **14 rows above** are the core training/eval dimensions for this adapter.

## Example prompts

**Full profile:**

```text
Provide a complete style profile for this passage as JSON.

Passage:
...
```

**Single dimension:**

```text
What tone does the author strike in this passage?

Passage:
...
```

**Steering-card check (post–Phase 5B):**

```text
Does this passage match the following steering card?
List mismatches only (field: expected vs observed).

Card: { "pov": "first_person", "register": "colloquial", "tone": "tense", ... }
Passage:
...
```

## Files in this release

| Artifact | Description |
|---|---|
| `adapter_model.safetensors` + `adapter_config.json` | PEFT LoRA @ step 3000 (~1.9 GB) |
| Tokenizer / chat template | Gemma 4 non-thinking template |
| GGUFs | See [`alexokita/gemma4-style-step3000-GGUF`](https://huggingface.co/alexokita/gemma4-style-step3000-GGUF) (Q5_K_M ~18 GB, Q4_K_M ~16 GB) |

## Limitations

- Optimized for **analytic** JSON / short judgments, not creative generation
- Label noise follows the Phase 2 teacher models; bake-off against held-out gold (and SiF exemplars) before trusting bulk re-label
- Multimodal / vision paths of base Gemma 4 are unused (text-only LoRA)

## Citation / acknowledgements

- Leech, G. & Short, M. *Style in Fiction* (2nd ed.) — theoretical source of the rubric
- [Unsloth](https://github.com/unslothai/unsloth), TRL, Hugging Face PEFT
- Base: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) / Unsloth QAT training weights

```bibtex
@software{gemma4_style_lora_step3000,
  title  = {Gemma 4 Style LoRA — prose style evaluator},
  author = {Okita, Alex},
  year   = {2026},
  url    = {https://huggingface.co/alexokita/gemma4-style-lora-step3000}
}
```

## License

Use of the base Gemma 4 weights is subject to the [Gemma license](https://ai.google.dev/gemma/terms). This adapter inherits those terms for combined use.
