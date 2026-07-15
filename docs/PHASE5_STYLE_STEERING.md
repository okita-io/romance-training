# Phase 5 — Style Steering & Generation Loop

**Status:** Draft plan (Jul 2026)  
**Depends on:** Phase 4 Gemma 4 style LoRA reaching **3000 steps on DGX Spark (`spark-4f07`)** + GGUF export for LM Studio on the 3090  
**Goal:** Turn classified “styled prose” into a **steering card** that can (1) evaluate generated text and (2) later train / prompt a writer model toward tone + style + voice.

---

## Can the step-3000 model be an evaluator?

**Yes — as a structured style evaluator, not as a general “is this good writing?” judge.**

| Suitable now | Not suitable yet |
|---|---|
| Score prose on the existing profile dimensions (register, POV, tone, FID, figurative density, …) | Overall literary quality / preference ranking |
| Check whether a writer model hit a **target style card** | Plot coherence, factuality, character consistency |
| Dimension judgments (“how much dialogue?”, “what tone?”) | Calibrated numeric rewards without a human agreement study |

**Why it’s plausible after 3000 steps**

- Training data is almost entirely **classification** (full JSON profiles) and **judgment** (one-dimension labels + short rationale).
- Held-out eval loss has been trending down (~2.51 → ~2.43 by step 1250) while train loss sits ~1.2 — enough signal to try, not enough to trust blindly.
- Chat template is non-thinking Gemma-4, aimed at JSON / short analytic answers — good fit for eval prompts.

**Gate before treating it as production evaluator**

1. Export Q4/Q5 GGUF and smoke-test in LM Studio on 20–50 held-out passages.
2. Compare model profiles vs existing `metadata.style_profile` labels (agreement on categorical fields).
3. Blind check: generate 10 passages with another model under an explicit target card; ask this model to classify; measure hit rate on voice / style / tone knobs.

If agreement is weak on tone/POV/FID, continue Phase 4-style SFT or add a small preference/contrast set — don’t jump straight to rewriter training.

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

### Prompt glue (new — Phase 5A)
- `voice_summary` — one sentence  
- `style_summary` — one sentence  
- `do` / `dont` — 2–4 bullets each  

**De-emphasize as generation knobs:** `end_focus`, `subordination_salience`, `textual_relations`, fine graphology. Keep them in the full profile; omit from default writer prompts.

---

## Plan of action

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
5. **Go / no-go:** if categorical agreement is usable on voice+style+tone, promote the GGUF to “evaluator” in the generation loop. If not, schedule more SFT / harder judgment pairs before 5C.

### Phase 5B — Enrich styled corpus with steering cards

**Owner:** Enrichment via quantized models on the **3090** (LM Studio) and/or Spark-hosted APIs; merge + Phase 3 on **Spark**  

1. Add a post-pass (or LLM enrichment job) that writes `metadata.steering_card` onto styled JSONL from existing `style_profile`.
2. Stabilize enums (no free-text drift on the core knobs).
3. Bin computables used in prompts (`dialogue_ratio`, sentence length → low/mid/high).
4. Re-run / extend Phase 3 on Spark only for card-aware tasks if needed:
   - “Summarize this passage as a steering card”
   - “Does this passage match the following card? List mismatches.”

### Phase 5C — Closed-loop eval of another writer model

**Owner:** LM Studio on **3090** (writer + evaluator GGUFs)  

1. Pick a base writer (untuned instruct / existing romance model).
2. For each target card, prompt: *Write N words matching this steering card.*
3. Run Gemma evaluator → full profile + mismatch list vs target.
4. Score: fraction of knobs matched; qualitative spot-check.
5. Use failures to drive either better writer prompts **or** Phase 5D training data.

### Phase 5D — Phase 3B rewrite / style-conditioned generation SFT

**Owner:** pair generation on 3090 or frontier API → **all LoRA training on Spark (`spark-4f07`)**  

Only after 5A says the evaluator is trustworthy enough to label mismatches.

1. **Rewrite pairs:** same content, different target card → `Rewrite to match this steering card.`
2. **Optional scene→prose pairs:** outline + card → prose (smaller, higher effort).
3. Train a **separate** writer LoRA on Spark (do not overwrite the evaluator adapter; do not train on the 3090).
4. Iterate with 5C: writer generates → evaluator scores → hard negatives into next mix.

### Phase 5E — Optional reward / preference layer (later)

If discrete card matching isn’t enough:

- Pairwise prefs: “which passage better matches card C?”
- Train a thin preference head or DPO set **using evaluator + human spot checks**
- Keep the classifier GGUF as the interpretable judge; don’t replace it with a black-box reward-only model until needed

---

## Suggested prompt shapes

**Evaluator (current Gemma LoRA):**

```text
Provide a complete style profile for this passage as JSON.
```

```text
Does this passage match the following steering card?
List mismatches only (field: expected vs observed).

Card: { ... }
Passage: ...
```

**Writer (future / base model):**

```text
Write ~400 words that match this steering card.
Voice: ...
Style: ...
Tone: ...
Do: ...
Don't: ...
```

---

## Success criteria

| Milestone | Done when |
|---|---|
| 5A | Q4/Q5 GGUF loads; JSON profiles parse; categorical agreement acceptable on held-out set |
| 5B | Styled JSONL carries `steering_card`; enums stable |
| 5C | Writer outputs can be auto-scored against cards with human-confirmed useful rankings |
| 5D | Writer LoRA improves card-hit rate vs base on the same eval suite |

---

## Machine split

| Machine | Role in Phase 5 |
|---|---|
| **DGX Spark (`spark-4f07`)** — ~128 GB unified Blackwell + CUDA | **All training** — finish Gemma Phase 4, GGUF export, later writer LoRA / Phase 5D SFT |
| **RTX 3090** (24 GB) + LM Studio | **Inference only** — quantized evaluator/writer GGUFs, smoke tests, optional enrichment / pair generation |

Do **not** run Phase 4/5 LoRA training on the 3090.

---

## Out of scope for Phase 5

- Replacing Phase 2 bulk classification with the new Gemma model until 5A passes  
- Merging evaluator and writer into one adapter  
- Full novel-length consistency (character bible, plot) — separate stack  

---

## Immediate next steps (when 3000 lands)

1. Confirm GGUF export (auto or `--export-only`).
2. Run Phase **5A** bake-off in LM Studio.
3. If green, start **5B** steering_card enrichment on new/ongoing styled segments.
4. Stand up a tiny **5C** loop with any available writer model — prove the evaluator loop before investing in 5D SFT.
