# Phases 5–6 — From style judge to long-form novel

**Status:** Draft plan (Jul 2026)  
**Depends on:** Phase 4 Gemma 4 style LoRA reaching **3000 steps on DGX Spark (`spark-4f07`)** + GGUF export for LM Studio on the 3090

---

## Progress status

**Last updated:** 2026-07-17 (Spark local)  
**Current focus:** Phase **5A** bake-off — Gemma GGUF as judge on the 3090.  
**Blocked on:** GGUF download/load in LM Studio on the PC (3090); then run agree/reject.

### What's next (ordered)

1. Finish Hugging Face GGUF download on the PC → load **Q4** (or Q5) in LM Studio (temp 1.0, top_p 0.95, top_k 64, thinking **off**).
2. Point `LLM_BASE_URL` / `LLM_MODEL` at that endpoint and run `python3 tools/style_evaluation/bakeoff_5a.py run`.
3. Read go/no-go from `*.summary.json`. If green → **5B** steering cards **and** kick off **6A** bulk Gemma classification on a pilot corpus.
4. Single-chapter **6C** pilot → aim at **6D** novel merge (north-star checkpoint).

### Checklist

| Phase | Item | State |
|---|---|---|
| — | North-star + Phases 5–6 plan written | **Done** |
| — | Docs: Spark = training, 3090 = inference only | **Done** |
| — | Active config `train_config.gemma4_spark.toml` + Docker Gemma overlay | **Done** |
| — | Style training JSONL present (`style_training/`, ~430k train pairs) | **Done** |
| 4 | Gemma 4 LoRA training started on Spark | **Done** |
| 4 | Checkpoints through **2750** (250…2750) | **Done** |
| 4 | Resume with less-frequent eval (milestones 2000 / 3000) | **Done** |
| 4 | Training → **3000** steps | **Done** |
| 4 | GGUF export (f16 / q5 / q4) for LM Studio | **Done** (on Spark; HF download to PC in progress) |
| 5A | Bake-off harness + fixed eval set (30 gold + 10 traps) | **Done** (`tools/style_evaluation/bakeoff_5a.py`, `eval/bakeoff_5a/`) |
| 5A | Evaluator bake-off on 3090 | **In progress** (waiting on GGUF load in LM Studio) |
| 5B | `steering_card` enrichment on styled corpus | **Not started** |
| 5C | Passage-level writer ↔ evaluator loop | **Not started** |
| 5D | Writer LoRA SFT on Spark | **Not started** |
| 5E | Preference / DPO layer | **Not started** (optional) |
| 6A | Bulk re-classify with Gemma step-3000 | **Not started** (gated on 5A) |
| 6B | Novel-level steering contract | **Not started** |
| 6C | Chapter draft → evaluate → revise | **Not started** |
| 6D | Full long-form novel merge + style report | **Not started** ← north-star checkpoint |
| 6E | Hard negatives → writer SFT | **Not started** (optional) |

### Accomplished (context)

- Phase 1–3 pipeline exists (rubric, knowledge, styled corpora, instruction pairs).
- Phase 4 target locked to **Gemma 4 26B-A4B QAT** on **spark-4f07** (~128 GB unified); 3090 reserved for quantized inference.
- Train loss ~1.21; eval loss **2.51 → 2.38** by checkpoint-2750.
- Post-training roadmap (judge → steer → bulk classify → novel merge) documented in this file.

### How to update this section

When a row changes state, bump **Last updated**, move the item in **What's next**, and adjust the checklist. Keep **Current focus** to a single line so the next action stays obvious.

---

## North-star goal (keep this in view)

**Produce long-form fiction that holds a chosen voice, style, and tone across a full novel** — not just clever single-passage rewrites.

That means the stack has to do three jobs well:

| Job | What it is | Where it lives |
|---|---|---|
| **Judge** | Classify / score prose against Leech & Short dimensions + a compact steering card | Gemma 4 style LoRA (Phase 4 → 5A) |
| **Steer** | Tell a writer model *how* to write (voice / style / tone / do-don’t) | Steering cards (5B) + rewrite/writer SFT (5D) |
| **Compose** | Plan and assemble chapter- and novel-scale text that stays on-card | Phase **6** (bulk re-label → novel merge / long-form pipeline) |

Phase 5 proves the **judge + steer** loop on passages.  
Phase 6 uses a **trusted** judge to re-label at scale and to **merge** passage-level control into a **full long-form novel**.

Longer-term productization of that loop as a **multi-grain MoE style editor** (sentence / span / act experts) — including how today’s training data feeds it, what is done, and what is still missing — is documented in [`MOE_STYLE_EDITOR.md`](MOE_STYLE_EDITOR.md).

If a task doesn’t serve “consistent style across a book,” it’s secondary.

---

## Can the step-3000 model be an evaluator?

**Yes — as a structured style evaluator, not as a general “is this good writing?” judge.**

| Suitable now | Not suitable yet |
|---|---|
| Score prose on the existing profile dimensions (register, POV, tone, FID, figurative density, …) | Overall literary quality / preference ranking |
| Check whether a writer model hit a **target style card** | Plot coherence, factuality, character consistency (those are Phase 6 / factory concerns) |
| Dimension judgments (“how much dialogue?”, “what tone?”) | Calibrated numeric rewards without a human agreement study |

**Why it’s plausible after 3000 steps**

- Training data is almost entirely **classification** (full JSON profiles) and **judgment** (one-dimension labels + short rationale).
- Held-out eval loss has been trending down (~2.51 → ~2.43 by step 1250) while train loss sits ~1.2 — enough signal to try, not enough to trust blindly.
- Chat template is non-thinking Gemma-4, aimed at JSON / short analytic answers — good fit for eval prompts.

**Gate before treating it as production evaluator (Phase 5A) — required before Phase 6 bulk replace**

1. Export Q4/Q5 GGUF and smoke-test in LM Studio on 20–50 held-out passages.
2. Compare model profiles vs existing `metadata.style_profile` labels (agreement on categorical fields).
3. Blind check: generate 10 passages with another model under an explicit target card; ask this model to classify; measure hit rate on voice / style / tone knobs.

If agreement is weak on tone/POV/FID, continue Phase 4-style SFT or add a small preference/contrast set — **do not** start Phase 6 bulk classification or rewriter training.

---

## Steering card (generation contract)

Keep the full Leech & Short profile for analysis. For generation and eval-against-target, promote a **compact card**:

### Voice
- `pov` — first / second / third_limited / third_omniscient / mixed  
- `narrative_distance` — intimate / moderate / distant  
- `free_indirect_discourse` — none / sparse / moderate / heavy  
- `mind_style` — standard / distinct / deviant  

### Style
- `register` — formal_literary / formal_technical / neutral_narrative / colloquial / dialect / archaic  
- `sentence_complexity` — simple_paratactic / moderate / complex_hypotactic  
- `figurative_density` — low / moderate / high  
- `dialogue_ratio` — bin continuous → low / mid / high  
- `cohesion`, `prose_rhythm` — optional secondary knobs  

### Tone
- `tone` — neutral / lyrical / sardonic / melancholic / comedic / tense / contemplative  

### Prompt glue (Phase 5B)
- `voice_summary` — one sentence  
- `style_summary` — one sentence  
- `do` / `dont` — 2–4 bullets each  

**De-emphasize as generation knobs:** `end_focus`, `subordination_salience`, `textual_relations`, fine graphology. Keep them in the full profile; omit from default writer prompts.

---

## Phase 5 — Style steering & generation loop

Prove the judge, attach steering cards, and (optionally) train a **passage-level** writer. Still chapter-scale at most.

### Phase 5A — Evaluator bake-off (immediately after GGUF)

**Owner:** Train/export on **DGX Spark (`spark-4f07`)** → load quantized GGUF in LM Studio on the **RTX 3090** (inference only)  

1. Finish Phase 4 at step 3000 on Spark; confirm GGUFs under `gemma4_style_q4/` and `gemma4_style_q5/`.
2. Copy Q4/Q5 to the 3090 and load in LM Studio: temp 1.0, top_p 0.95, top_k 64, thinking **off**.
3. Build a small eval set (`eval/style_benchmark/` or extend existing fixtures):
   - 30 passages with known profiles (from styled corpus)
   - 10 “trap” passages (mixed POV, heavy FID, archaic register)
4. Metrics:
   - Exact / relaxed match on categorical fields (tone, pov, register, FID, figurative_density)
   - JSON parse success rate for full-profile prompts
5. **Go / no-go:** if categorical agreement is usable on voice+style+tone, promote the GGUF to **default Phase 2 classifier** and unlock Phase **6A**. If not, more SFT / harder judgment pairs first.

### Phase 5B — Enrich styled corpus with steering cards

**Owner:** Enrichment via quantized models on the **3090** (LM Studio) and/or Spark-hosted APIs; merge + Phase 3 on **Spark**  

1. Add a post-pass that writes `metadata.steering_card` onto styled JSONL from existing `style_profile`.
2. Stabilize enums (no free-text drift on the core knobs).
3. Bin computables used in prompts (`dialogue_ratio`, sentence length → low/mid/high).
4. Optional card-aware Phase 3 tasks on Spark:
   - “Summarize this passage as a steering card”
   - “Does this passage match the following card? List mismatches.”

### Phase 5C — Closed-loop eval of another writer model

**Owner:** LM Studio on **3090** (writer + evaluator GGUFs)  

1. Pick a base writer (untuned instruct / existing romance model).
2. For each target card, prompt: *Write N words matching this steering card.*
3. Run Gemma evaluator → full profile + mismatch list vs target.
4. Score: fraction of knobs matched; qualitative spot-check.
5. Use failures to drive better writer prompts **or** Phase 5D training data.

### Phase 5D — Rewrite / style-conditioned generation SFT

**Owner:** pair generation on 3090 or frontier API → **all LoRA training on Spark (`spark-4f07`)**  

Only after 5A passes.

1. **Rewrite pairs:** same content, different target card → `Rewrite to match this steering card.`
2. **Optional scene→prose pairs:** outline + card → prose.
3. Train a **separate** writer LoRA on Spark (never overwrite the evaluator; never train on the 3090).
4. Iterate with 5C.

### Phase 5E — Optional reward / preference layer (later)

- Pairwise prefs: “which passage better matches card C?”
- Thin preference / DPO only after human spot checks
- Keep the classifier GGUF as the interpretable judge

---

## Phase 6 — Bulk Gemma classification & long-form novel merge

**Unlock:** Phase **5A go**.  
**Goal link:** Use the trusted Gemma judge as the **house style classifier**, then assemble **book-scale** output that stays on a novel-level steering contract.

### Phase 6A — Bulk re-classification with step-3000 Gemma 4

**Owner:** **3090 + LM Studio** (quantized Gemma evaluator GGUF) as the Phase 2 LLM backend; optional Spark vLLM later for throughput  

Replace (or dual-run against) the smaller instruct models used for historical Phase 2 labeling.

1. Point `run_pipeline.py` / `LLM_*` env at the Gemma style GGUF in LM Studio (thinking **off**, JSON-friendly settings).
2. Re-classify priority corpora (or new segments) with `--pass both` (or fast→deep) into `*_styled_seg_*.jsonl`.
3. Dual-run a sample against old labels: measure agreement drift; freeze a **label version** (`style_profile_version`, e.g. `gemma4-step3000-v1`).
4. Sync styled JSONL to Spark; rebuild Phase 3 mixes if training data should track the new judge.
5. Emit `steering_card` on every new styled row (reuse 5B logic).

**Success:** bulk pipeline runs stably on Gemma; categorical fields parse; agreement study documented; version tag on all new styled rows.

### Phase 6B — Novel-level style contract

Before generating a book, define **one primary card** (and optional per-arc variants):

- `novel_steering_card` — default voice / style / tone for the manuscript  
- `arc_overrides[]` — optional shifts (e.g. climax → `tense`, epilogue → `contemplative`) with allowed drift bounds  
- `consistency_rules` — POV lock, register lock, FID budget, dialogue density band  

The evaluator’s job at novel scale: **flag chapter/scene drift** from the contract, not invent plot.

### Phase 6C — Long-form generate → evaluate → revise loop

**Owner:** Writer model (base or 5D LoRA) on 3090; Gemma evaluator on 3090; planning/orchestration may use sibling **romance-factory** / Spark vLLM later  

1. **Plan** — outline / chapter beats (factory or human).  
2. **Draft** — scene or chapter under `novel_steering_card` (+ arc override).  
3. **Evaluate** — Gemma classifies draft; compare to contract; list mismatches.  
4. **Revise** — rewrite mismatched scenes (prompted or 5D rewriter) until within drift bounds.  
5. **Merge** — concatenate accepted scenes/chapters into a manuscript with stable metadata:
   - per-scene `style_profile` + `steering_card`  
   - rolling “manuscript style digest” (modal POV/register/tone so far)

### Phase 6D — Full long-form novel merge (deliverable)

**Done when** you can export a single long-form artifact, e.g.:

- `manuscripts/<title>/novel.md` (or `.jsonl` scene stream + rendered markdown)  
- `manuscripts/<title>/style_report.json` — chapter-level match rates vs `novel_steering_card`  
- human spot-check that voice/style/tone feel continuous across act boundaries  

This is the first milestone that proves the north-star: **a book that stays on style**, not only a passage that scores well.

### Phase 6E — Feedback into training (optional)

Hard negatives from 6C (scenes that failed the contract) → more 5D rewrite pairs / Spark SFT.  
Do **not** conflate the evaluator and writer adapters.

---

## Suggested prompt shapes

**Evaluator (Gemma style LoRA):**

```text
Provide a complete style profile for this passage as JSON.
```

```text
Does this passage match the following steering card?
List mismatches only (field: expected vs observed).

Card: { ... }
Passage: ...
```

**Writer (passage / scene):**

```text
Write ~400–800 words that match this steering card.
Voice: ...
Style: ...
Tone: ...
Do: ...
Don't: ...
```

**Novel merge (orchestrator):**

```text
You are assembling chapter N of a novel.
Novel steering card: { ... }
Arc override (if any): { ... }
Prior chapter style digest: { ... }
Draft the chapter, then self-check against the card before returning.
```

(Final authority remains the Gemma evaluator, not the writer’s self-check.)

---

## Success criteria

| Milestone | Done when |
|---|---|
| **5A** | Q4/Q5 GGUF loads; JSON profiles parse; categorical agreement acceptable → **unlocks 6A** |
| **5B** | Styled JSONL carries `steering_card`; enums stable |
| **5C** | Writer outputs can be auto-scored against cards with useful rankings |
| **5D** | Writer LoRA improves card-hit rate vs base on the passage eval suite |
| **6A** | Bulk Phase 2 runs on Gemma step-3000; label version tagged; sync to Spark |
| **6B** | Novel-level contract + optional arc overrides defined |
| **6C** | Chapter loop: draft → evaluate → revise converges within drift bounds |
| **6D** | **Full long-form novel merge** exported with style report — north-star checkpoint |

---

## Machine split

| Machine | Role |
|---|---|
| **DGX Spark (`spark-4f07`)** — ~128 GB unified Blackwell + CUDA | **All training** — Phase 4 Gemma, 5D writer LoRA, optional later SFT; optional high-throughput serving |
| **RTX 3090** (24 GB) + LM Studio | **Inference only** — Gemma evaluator GGUF, writer GGUF, Phase **6A** bulk classification, 6C draft/eval |

Do **not** run Phase 4/5/6 LoRA training on the 3090.

---

## Out of scope (for now)

- Merging evaluator and writer into one adapter  
- Treating Gemma as a plot/character bible (use romance-factory / separate memory for that)  
- Starting 6A bulk replace **before** 5A go  

---

## Pipeline at a glance

```text
Phase 4  Gemma style LoRA @ 3000 (Spark) → GGUF
    ↓
Phase 5A bake-off (3090) ──fail──▶ more SFT
    ↓ pass
Phase 5B–E  steering cards + passage writer loop
    ↓
Phase 6A  bulk classify corpora with Gemma (3090)
Phase 6B  novel steering contract
Phase 6C  draft → evaluate → revise (chapters)
Phase 6D  merge → long-form novel + style report   ← north-star checkpoint
Phase 6E  hard negatives → Spark writer SFT (optional)
```

---

## Immediate next steps

See **[Progress status](#progress-status)** at the top of this doc — that checklist is the live source of truth for what’s done and what’s next.
