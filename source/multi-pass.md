# Multi-pass LLM classification (Phase 2)

Design note for speeding up bulk chunk classification on a 24 GB GPU (RTX 3090) using LM Studio.

## Problem

Today, Phase 2 runs **one LLM call per chunk** via `metrics_llm.assess()`, asking for all semantic fields in a single JSON response (~16 keys: 10 `llm_dimensions` + 6 `textual_principles`, plus computable metrics from spaCy/textstat).

With **Qwen3.6-32B-A3B** (~20 tok/s observed):

- Prompt + passage + full schema is heavy; generation is slow.
- **Parallelism is limited to 2 workers** (`--workers 2`). 3 workers becomes unstable after a few hours; 4 can freeze the machine.

A **2–4B model** (e.g. Mistral 3 3B) is faster (~3× tok/s in practice) and fits **4 parallel workers** safely.

## Recommended strategy: two-pass, stateless (not 6-turn chat)

CoPilot’s 6-turn *chat chain* per chunk is valid in LM Studio but **not what we should build first**:

| Chat-chain issue | Why it matters here |
|------------------|---------------------|
| Full history reprocessed every turn | 6× cost *per chunk*, on top of 6× calls |
| `llm_client.complete()` is single-turn only | Would need a new `complete_messages()` API |
| Small models barely use prior turns | Gains are mostly from smaller prompts, not memory |
| Parallelism is per-chunk | Turns within a chunk must stay sequential |

**Better fit for this repo:** run the corpus in **two sequential passes**, merging JSON into `metadata.style_profile`. Pass 2 receives Pass 1 labels as **structured context in one user message** — not as chat history.

```
Pass 1 (small model, --workers 4)
  → classify "easy" dimensions + textual principles
  → write partial style_profile to output JSONL

Pass 2 (32B MoE, --workers 2)
  → read partial profiles + passage
  → classify "hard" dimensions only
  → merge into final style_profile
```

Resume semantics stay the same as `run_pipeline.py`: skip records that already have the fields that pass cares about.

### Durability

Every pass **appends one line and flushes after each chunk** — safe to interrupt at any time. Resume reloads the output file using last-wins per record key (so duplicate lines from an interrupted deep pass are harmless).

When `--pass deep` **finishes successfully**, the pipeline rewrites the file once to remove duplicate keys and restore a single line per chunk. If deep is interrupted before that compact step, rerun `--pass deep`; resume still works, and the next successful completion compacts the file.

## Field split (aligned to `style_rubric.json`)

CoPilot’s example turns mixed in **computable** fields. Those never go to the LLM — they come from `metrics_computable.py`:

| Field | Source |
|-------|--------|
| `lexical_density`, `sentence_length_mean`, `subordination_ratio`, `dialogue_ratio` | spaCy / textstat (always) |

### Pass 1 — small model (Mistral 3 3B or similar)

~4 parallel workers. **Four groups → up to four short requests per chunk**, or **two requests** if 3B handles ~5 keys reliably in one shot (validate on a `--limit 50` sample first).

| Group | Keys | Notes |
|-------|------|-------|
| **1 — Lexical surface** | `lexical_complexity`, `register`, `figurative_density` | Most stable; good 3B task |
| **2 — Syntax (semantic)** | `sentence_complexity` | Do **not** ask LLM for subordination or sentence length — already computed |
| **3 — Discourse** | `pov`, `cohesion` | Do **not** ask LLM for `dialogue_ratio` — already computed |
| **4 — Textual dynamics** | `segmentation`, `prose_rhythm`, `end_focus`, `subordination_salience`, `textual_relations`, `climax` | `textual_principles` in rubric |

Optional consolidation: groups 1+2 in one call, groups 3+4 in one call → **2 requests/chunk** on the small model.

### Pass 2 — 32B MoE (Qwen3.6-32B-A3B)

~2 parallel workers. **One request per chunk** — prompt includes Pass 1 JSON + passage + rubric context for only these keys:

| Group | Keys | Notes |
|-------|------|-------|
| **5 — Tone** | `tone`, `climax` | `climax` may already be set in Pass 1; Pass 2 can refine or skip if present |
| **6 — Viewpoint / mind style** | `narrative_distance`, `mind_style`, `free_indirect_discourse` | Needs holistic reading; 32B MoE |

If Pass 1 already scored `climax`, Pass 2 can omit it and only fill `tone`, `narrative_distance`, `mind_style`, `free_indirect_discourse` (4 keys) — the main win for the large model.

## Hardware and LM Studio settings

| Model | Safe `--workers` | LM Studio concurrent slots |
|-------|------------------|----------------------------|
| Qwen3.6-32B-A3B (MoE) | **2** | 2 |
| Mistral 3 3B (or 2–4B) | **4** | 4 |

Between passes: unload the small model and load the 32B in LM Studio (or point `LLM_MODEL` / `--model` at the loaded id). Only one model needs to be in VRAM at a time.

### Same model for both passes (`--pass both`)

If a single model handles both field sets well (e.g. Mistral 3 3B on your box), use **`--pass both`** — one LM Studio load, **two smaller LLM requests per chunk** (fast fields, then deep fields with pass-1 context). No model swap between passes.

```bash
LLM_MODEL=mistralai/ministral-3-3b
python tools/style_classification/run_pipeline.py \
  --workers 4 \
  --pass both \
  --input source-data/processed/horror_novel_chunks/chunks.jsonl \
  --output train/romance_corpus/horror_styled.jsonl
```

Resume: skips chunks that already have **all** LLM fields. If you previously ran `--pass fast` only, rerun with `--pass both` — it fills in the missing deep fields only.

`--pass full` remains the single-call alternative (one large schema request per chunk).

Env pattern (two separate runs, different models):

```bash
# Pass 1
LLM_MODEL=mistralai/ministral-3-3b
python tools/style_classification/run_pipeline.py \
  --workers 4 \
  --pass fast \
  --input source-data/processed/horror_novel_chunks/chunks.jsonl \
  --output train/romance_corpus/horror_styled.jsonl

$env:LLM_MODEL="mistralai/ministral-3-3b"
python tools/style_classification/run_pipeline.py --pass fast --workers 4 --input source-data/processed/horror_novel_chunks/chunks.jsonl --output train/romance_corpus/horror_styled.jsonl

# Pass 2 (after swapping model in LM Studio)
LLM_MODEL=qwen3.6-35b-a3b-abliterated-heretic
python tools/style_classification/run_pipeline.py \
  --workers 2 \
  --pass deep \
  --input source-data/processed/horror_novel_chunks/chunks.jsonl \
  --output train/romance_corpus/horror_styled.jsonl
```

(`--pass fast|deep|both|full` is implemented in `run_pipeline.py` and `classify_passage.py`.)

## What exists today

| Component | Status |
|-----------|--------|
| `run_pipeline.py --workers N` | Parallel **chunks**, one model per run |
| `run_pipeline.py --pass fast\|deep\|both\|full` | Two-pass + same-model both + single-shot modes |
| `metrics_llm.assess()` | Field-restricted requests + prior context |
| `metrics_computable.compute()` | Always runs first |
| Resume | Per-pass field completion checks |
| Multi-turn chat in `llm_client` | **Not implemented** (not needed for two-pass) |

## Implementation checklist

Shipped:

1. **`metrics_llm.assess()`** — `pass_mode`, `fields`, and `prior` for restricted keys and Pass 1 context.
2. **`classify_passage.classify()`** — `pass_mode` and `prior_profile`.
3. **`run_pipeline.py`** — `--pass fast|deep|both|full`, per-pass resume, append+flush after each chunk, compact at end for deep/both.
4. **`pass_config.py`** — `PASS1_LLM_FIELDS`, `PASS2_LLM_FIELDS`, `pass_complete()`.

Still optional:

- `llm_client.complete_messages()` for true multi-turn chains.
- Confidence routing: send ambiguous Pass 1 rows to 32B for all fields.
- Single command that orchestrates both passes with a model swap prompt. **Shipped as `--pass both`** (same model, no swap).

## Throughput expectations (realistic)

Rough orders of magnitude — measure on your box:

| Mode | Workers | Calls / chunk | Notes |
|------|---------|---------------|-------|
| Current 32B single-shot | 2 | 1 large | ~20 tok/s, full schema |
| Pass 1 only (3B, 2 calls) | 4 | 2 small | Higher chunk throughput |
| Pass 2 only (32B) | 2 | 1 small schema | Most prompt tokens dropped |

Do **not** expect literal 10× end-to-end speedup: Pass 1 adds extra calls, and Pass 2 still touches every chunk. The win is **Pass 2 prompt size** (4 keys vs 16) plus **4-wide parallelism on Pass 1**. Validate with wall-clock on `--limit 200` before committing to a full corpus run.

## When *not* to use multi-pass

- **`--no-llm`** — computable metrics only; unchanged.
- **`--llm-sample-rate < 1`** — sample rate applies per pass if you run both; document that double sampling is usually wrong (run fast pass on 100%, deep pass on sample, or vice versa).
- **Quality spot-checks** — keep a `--pass full` code path identical to today’s single 32B call for regression comparison.

## CoPilot takeaways (kept, corrected)

- **Grouping 2–3 dimensions per request** helps small models; our Pass 1 groups follow that.
- **Chaining helps large models** when prior labels are supplied; Pass 2 should get Pass 1 JSON in the prompt, not rely on chat memory.
- **Multi-turn chat** trades accuracy for latency when every turn re-reads history — avoid as the default automation path.
- Several CoPilot turn labels (`subordination`, `sentence length`, `dialogue ratio`) are **computable in this pipeline** — asking the LLM for them would duplicate or contradict spaCy metrics.

## Related files

- `tools/style_classification/metrics_llm.py` — prompts and parsing
- `tools/style_classification/run_pipeline.py` — bulk run and `--workers`
- `tools/llm_client.py` — HTTP client (extend for multi-turn if needed)
- `source/extracted/style_analysis_system.json` — canonical dimension list
- `source/style_rubric.json` — rubric + RAG context for classification
- `docs/GPU_RUNBOOK.md` — Phase 2 runbook (update worker guidance after implementation)
