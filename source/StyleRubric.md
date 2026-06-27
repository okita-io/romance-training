# Style Rubric — Extraction from *Style in Fiction*

This document describes how **Leech & Short’s *Style in Fiction*** (2nd ed., 2007) was turned into the structured analysis framework in [`style_rubric.json`](style_rubric.json), and how that rubric is used downstream in the training pipeline.

The rubric is the canonical reference for Phase 2 classification and Phase 3 instruction-pair generation. It is **already committed** in the repo; you only need to regenerate it if you change the parsed manuscript or extend the taxonomy.

---

## Source material

| Artifact | Role |
|----------|------|
| `style-guide/Style-in-Fiction.pdf` | Bundled reference PDF (Pearson/Longman, ISBN 978-0-582-78409-3) |
| `source/Style-in-Fiction.parsed.md` | Full-book markdown transcription (~4,800 lines, `<PAGE>`-delimited) |
| `source/style_rubric.json` | Distilled taxonomy (version **2.0**) |
| `source/extracted/style_knowledge.jsonl` | RAG chunks for classification (231 records) |
| `source/extracted/style_analysis_system.json` | LLM system prompt + analysis protocol derived from the rubric |

The book is organised in two parts:

- **Part One — Approaches and Methods** (Chapters 1–4): stylistic theory, quantitative vs qualitative analysis, the multilevel linguistic model.
- **Part Two — Aspects of Style** (Chapters 5–12): applied analysis — lexis, grammar, figures, cohesion, viewpoint, speech presentation, etc.

The rubric’s `framework.parts` and `framework.levels` fields mirror this structure directly.

---

## End-to-end pipeline

```
Style-in-Fiction.pdf
    │
    ├─[Phase 1A] Vision transcription ──▶ source/extracted/pages/page_*.md
    │                                      source/extracted/Style-in-Fiction.md (merged)
    │                                      source/Style-in-Fiction.parsed.md (canonical input)
    │
    └─[Phase 1C] distill_style_system.py ──▶ source/style_rubric.json
                                              source/extracted/style_knowledge.jsonl
                                              source/extracted/style_analysis_system.json
                                                    │
                                                    └─[Phase 2] classify_passage / run_pipeline.py
```

Phase 1 prioritises **analytic fidelity** over speed: the book is transcribed once, parsed deterministically, and distilled into a fixed rubric rather than re-deriving dimensions from an LLM on every run.

### Phase 1A — PDF → parsed markdown

**Tool:** `tools/style_extraction/pdf_vision_harness.py`

Each PDF page is sent to a vision-capable LLM (Qwen-VL / Qwen3.6 in LM Studio, or OpenRouter). Flowcharts in the book are converted to fenced `mermaid` blocks. Output is written per page under `source/extracted/pages/`, then concatenated.

The committed canonical input is `source/Style-in-Fiction.parsed.md`, which uses `<PAGE>` markers to preserve page boundaries for provenance tagging.

Quality controls in the vision harness:

- Rejects reasoning dumps, repetition spam, and one-word outputs before saving a page.
- Resume support: finished pages are skipped; bad pages are re-transcribed on audit.

```bash
export LLM_VISION_MODEL=your-vision-model
python tools/style_extraction/pdf_vision_harness.py --disable-thinking --concat
```

An alternative OCR path (`marker-pdf` via legacy `extract_rubric.py`) exists but is not the source of the committed rubric.

### Phase 1C — Distillation (current method)

**Tool:** `tools/style_extraction/distill_style_system.py`  
**Parser:** `tools/style_extraction/manuscript_parser.py`

This is the **preferred** path when `Style-in-Fiction.parsed.md` exists. It does **not** call an LLM to invent dimensions; instead it:

1. Parses the manuscript into filtered, tagged sections.
2. Extracts Section **3.1** checklist questions verbatim (structured prompts).
3. Assembles a **curated** set of core dimensions and textual principles anchored in specific book sections.
4. Writes the rubric, knowledge base, and analysis-system JSON in one pass.

```bash
python tools/style_extraction/distill_style_system.py --force
```

Use `--force` to overwrite an existing `style_rubric.json`.

### Legacy path — LLM extraction

**Tool:** `tools/style_extraction/extract_rubric.py` (version 1.0 rubric shape)

The legacy script sends style-relevant markdown sections to an LLM with a dimension-extraction prompt, merges duplicates, and supplements gaps from hard-coded seed dimensions. It produces a flatter rubric without `framework`, `textual_principles`, or the Section 3.1 checklist.

The repo’s committed rubric is **v2.0** from `distill_style_system.py`. The legacy script remains for experimentation or when only vision-merged markdown (not the parsed manuscript) is available.

---

## Manuscript parsing

`manuscript_parser.parse_manuscript()` reads `Style-in-Fiction.parsed.md` and returns a list of `ManuscriptSection` objects.

### Noise filtering

Front matter is dropped until the parser sees **Introduction**, **Part One**, or **Aim**. Lines removed as noise include:

- Page images (`![](images/...)`)
- Bare page numbers and TOC entries (`"Foreword x"`, `"2.9 Features of style 52"`)
- Longman series boilerplate (“English Language Series”, sibling book titles)
- Publisher/copyright blocks

Sections shorter than 200 characters (configurable via `--min-section-chars`) are discarded.

### Structure detection

Headings are recognised by regex:

| Pattern | Example | Fields set |
|---------|---------|------------|
| `PART ONE` … | Part One | `part` |
| `CHAPTER N` | Chapter 7 | `chapter` |
| `N.N.N Title` | `7.3 The principle of end-focus` | `section_id` |
| Checklist heading | `3.1 A checklist of…` | `section_id = "3.1"`, `is_checklist = true` |

### Category tagging

Each section receives one or more category labels via keyword matching over title + first 1,200 characters of body:

`lexical`, `grammatical`, `figurative`, `cohesion`, `context`, `viewpoint`, `textual`

Special rules:

- Section IDs starting with `3.1` → all four checklist groups (lexical, grammatical, figurative, cohesion).
- Section IDs starting with `7.` → `textual`.
- Section IDs starting with `6.` or “mind style” in text → `viewpoint`.

**Current parse stats** (from `style_rubric.json` → `stats`):

| Metric | Count |
|--------|------:|
| Manuscript sections | 162 |
| RAG knowledge chunks | 231 |
| Checklist items | 21 |
| Scoring dimensions | 14 |
| Textual principles | 6 |

---

## Section 3.1 checklist extraction

Chapter 3, Section **3.1** (“A checklist of linguistic and stylistic categories”) is the book’s primary **heuristic analysis instrument**. Leech & Short organise it under four headings:

| Group | Book label | Rubric `checklist[].group` |
|-------|------------|----------------------------|
| A | Lexical categories | `a` |
| B | Grammatical categories | `b` |
| C | Figures of speech, etc. | `c` |
| D | Context and cohesion | `d` |

`extract_checklist_items()` walks the Section 3.1 body:

- Lines matching `A:` … `D:` start a new group/subgroup.
- Numbered sub-items (`1 GENERAL`, `2 NOUNS`, `1 SENTENCE TYPES`, …) start a new prompt within the group.
- Continuation lines are concatenated into a single `prompt` string.

Each checklist entry in the rubric has:

```json
{
  "id": "a_general_is_the_vocabulary_simple_or_complex",
  "group": "a",
  "subgroup": "General",
  "prompt": "Is the vocabulary simple or complex …",
  "source_section": "3.1"
}
```

These prompts are **not paraphrased** — they preserve the book’s wording (including footnote markers like `^{10}`). They are injected into LLM classification context and listed in `style_analysis_system.json` for the analysis protocol.

The book explicitly notes (and the first checklist item records) that categories overlap by design and that semantic categories are reached indirectly through lexical and grammatical analysis — consistent with Section 2.9’s discussion of semantic measurement difficulty.

---

## Linguistic framework (`framework`)

The rubric encodes Leech & Short’s multilevel model from **Chapter 4** (especially §4.2–4.3):

### Levels of organisation

| `framework.levels[].id` | Book term | Scope |
|------------------------|-----------|-------|
| `semantic` | Semantics | Meaning — sense vs significance |
| `syntax` | Syntax / lexigrammar | Grammar + lexis (“double articulation”) |
| `phonology` | Phonology | Sound pattern — stress, rhythm, intonation |
| `graphology` | Graphology | Writing system; implied phonology in silent reading |

Written prose is analysed at the graphological level, but phonological effects (rhythm, alliteration) remain analytically relevant — reflected in checklist group C (“Phonological schemes”) and the `prose_rhythm` textual principle.

### Analysis layers

The six `framework.analysis_layers` map operational layers used in classification to book sections:

| Layer ID | Label | Primary source |
|----------|-------|----------------|
| `lexical` | Lexis | §3.1.A |
| `grammatical` | Grammar and syntax | §3.1.B |
| `figurative` | Figures of speech | §3.1.C |
| `cohesion` | Cohesion and context | §3.1.D; Ch. 7 |
| `textual` | Textual dynamics | §7.1–7.5 |
| `viewpoint` | Viewpoint and mind style | §6.1–6.4; Ch. 10 |

These layers appear in the LLM system prompt in `style_analysis_system.json` and guide the ordered analysis protocol (lexis → grammar → figures → cohesion → textual dynamics → viewpoint).

---

## Scoring dimensions (`dimensions`)

The 14 `dimensions` are a **curated operational subset** of the full checklist — chosen for trainable, repeatable scoring in Phase 2. Each dimension links back to book sections where the underlying concepts are defined.

### Lexical (§3.1.A)

| ID | Name | Computation | Book anchor |
|----|------|-------------|-------------|
| `lexical_complexity` | Lexical Complexity | LLM | §3.1.A — vocabulary simplicity, register, collocation, semantic fields |
| `lexical_density` | Lexical Density | **Computable** (content-word ratio) | Quantitative tradition, Ch. 2; checklist asks about vocabulary complexity |
| `register` | Register | LLM | §3.1.A — formal/colloquial, dialect, specialised vocabulary |

### Grammatical (§3.1.B)

| ID | Name | Computation | Book anchor |
|----|------|-------------|-------------|
| `sentence_complexity` | Sentence Complexity | LLM | §3.1.B.2 — coordination vs subordination vs parataxis |
| `sentence_length_mean` | Mean Sentence Length | **Computable** | Checklist B.2 explicitly asks for average sentence length |
| `subordination_ratio` | Subordination Ratio | **Computable** | Checklist B.2 — ratio of dependent to independent clauses |
| `dialogue_ratio` | Dialogue Ratio | **Computable** | §3.1.D.2 — direct vs indirect speech presentation |

### Figurative (§3.1.C)

| ID | Name | Computation | Book anchor |
|----|------|-------------|-------------|
| `figurative_density` | Figurative Language | LLM | §3.1.C — schemes, tropes, foregrounding, deviance from the code |

### Cohesion (§3.1.D.1, Ch. 7)

| ID | Name | Computation | Book anchor |
|----|------|-------------|-------------|
| `cohesion` | Cohesion | LLM | §3.1.D.1 — conjunction, reference, substitution, ellipsis, elegant variation |

### Viewpoint (Ch. 6, §3.1.D.2)

| ID | Name | Computation | Book anchor |
|----|------|-------------|-------------|
| `mind_style` | Mind Style | LLM | §6.1 — worldview encoded in language |
| `pov` | Narrative Point of View | LLM | §3.1.D.2 — grammatical person, access to interiority |
| `narrative_distance` | Narrative Distance | LLM | Ch. 6 — psychological proximity of narrator |
| `free_indirect_discourse` | Free Indirect Discourse | LLM | §3.1.D.2 — blending narrator and character voice |

### Context (§3.1.D.2)

| ID | Name | Computation | Book anchor |
|----|------|-------------|-------------|
| `tone` | Tone | LLM | §3.1.D.2 — authorial attitude, affective register |

### Computation split

| Type | Count | Dimensions |
|------|------:|------------|
| **Computable** | 4 | `lexical_density`, `sentence_length_mean`, `subordination_ratio`, `dialogue_ratio` |
| **LLM** | 10 | All others |

Computable metrics are implemented in `tools/style_classification/metrics_computable.py` (spaCy + textstat). LLM metrics use rubric definitions plus RAG retrieval from `style_knowledge.jsonl` via `tools/style_classification/metrics_llm.py`.

Each dimension specifies:

- `metric_type`: `continuous`, `ordinal`, or `categorical`
- `values`: allowed labels or scale description
- `scoring`: optional `{low, mid, high}` human-readable bands for continuous/ordinal dims

---

## Textual principles (`textual_principles`)

These six principles capture **Part Two’s textual dynamics** — especially Chapter 7 (segmentation, end-focus, climax) and §6.4.4 (given/new information). They are scored ordinally by the LLM alongside the main dimensions.

| ID | Name | Source | What it measures |
|----|------|--------|------------------|
| `end_focus` | Principle of End-Focus | §7.3 | Given → new ordering; salience at clause/sentence end |
| `segmentation` | Segmentation and Graphic Units | §7.4 | How prose is chunked into tone/graphic units |
| `prose_rhythm` | Rhythm of Prose | §7.4.1 | Tempo from graphic-unit length and stress patterns |
| `climax` | Principle of Climax | §7.5.2 | “Last is most important” across interrelated units |
| `subordination_salience` | Subordination and Backgrounding | §7.5.1 | Foregrounding vs backgrounding via clause hierarchy |
| `textual_relations` | Textual Relations (Given/New) | §6.4.4 | Reference, ellipsis, coordination linking old and new information |

Each principle includes an `analysis_prompt` used in LLM scoring prompts.

---

## RAG knowledge base

During distillation, every parsed section is also converted to JSONL records via `sections_to_knowledge_records()`:

- Long sections are split at paragraph boundaries (max ~3,500 characters per chunk).
- Each record carries `title`, `section_id`, `chapter`, `part`, `categories`, `page`, and full `text` (with `## title` header).

Phase 2 retrieves the top-*k* relevant chunks for each passage (`style_knowledge.py`, default *k*=2) and concatenates them with rubric dimension definitions into the LLM user prompt. This grounds semantic judgments in the book’s own exposition rather than model priors alone.

---

## `style_rubric.json` schema (v2.0)

```json
{
  "version": "2.0",
  "source": "Style in Fiction — Leech & Short (2nd ed., parsed manuscript)",
  "source_file": "Style-in-Fiction.parsed.md",
  "framework": { "levels": [...], "parts": [...], "analysis_layers": [...] },
  "textual_principles": [ ... ],
  "checklist": [ ... ],
  "categories": { "lexical": "...", ... },
  "dimensions": [ ... ],
  "stats": { "sections": 162, "knowledge_chunks": 231, ... }
}
```

| Top-level key | Purpose |
|---------------|---------|
| `framework` | Book structure and multilevel linguistic model |
| `textual_principles` | Chapter 6–7 dynamics scored ordinally |
| `checklist` | Verbatim Section 3.1 prompts (21 items) |
| `categories` | Human-readable category blurbs with section refs |
| `dimensions` | Trainable scoring dimensions (14 items) |
| `stats` | Parse/distillation counts for reproducibility |

Companion file `source/extracted/style_analysis_system.json` exports:

- `system_prompt` — role + layer summary + checklist/principle/dimension IDs
- `analysis_protocol` — ordered steps mirroring the book’s analytic sequence
- `llm_dimensions` / `computable_dimensions` — splits for prompt construction

---

## Downstream usage

### Phase 2 — Corpus classification

```bash
python tools/style_classification/run_pipeline.py          # LLM + computable
python tools/style_classification/run_pipeline.py --no-llm # computable only
```

- `classify_passage.classify()` loads `style_rubric.json`, runs computable metrics, then optionally calls `metrics_llm.assess()`.
- LLM path builds a JSON schema from rubric dimensions + textual principles and retrieves knowledge chunks.
- Output: `style_profile` dict per passage (flat key/value metrics).

### Phase 3 — Instruction pairs

`tools/training_formats/generate_instruction_pairs.py` uses the rubric categories and dimension names to build multi-task training examples (classify, judge, explain).

### Phase 4 — Fine-tuning

Training data references the same dimension IDs, so the fine-tuned model learns the Leech & Short vocabulary and scale definitions from the rubric.

---

## Regenerating or extending the rubric

1. **Re-transcribe PDF** (only if OCR quality needs improvement):
   ```bash
   python tools/style_extraction/pdf_vision_harness.py --disable-thinking --concat
   ```

2. **Re-distil** from parsed markdown:
   ```bash
   python tools/style_extraction/distill_style_system.py --force
   ```

3. **Manual edits**: Review `source/style_rubric.json` — add dimensions, adjust `values`/`scoring`, or correct checklist prompts. Re-run Phase 2 with `--no-resume` after substantive changes.

4. **Tests**:
   ```bash
   pytest train/tests/test_distill_style_system.py -q
   ```

When adding a dimension, follow existing conventions:

- `id`: snake_case
- `category`: one of the seven category keys
- `computation`: `"computable"` or `"llm"`
- `source_section`: book section reference where possible

---

## Design rationale

| Decision | Reason |
|----------|--------|
| Deterministic distillation over LLM extraction | Reproducible rubric; avoids model drift and hallucinated dimensions |
| Verbatim Section 3.1 checklist | Preserves the book’s heuristic questions as analyst prompts |
| Curated 14 dimensions vs full checklist | Balances coverage with labelling cost and trainability at corpus scale |
| Separate textual principles | Chapter 7 dynamics are cross-cutting and don’t map cleanly to single lexical/grammatical bins |
| RAG alongside rubric | LLM scores cite book definitions; reduces generic “literary analysis” answers |
| Four-level framework metadata | Keeps the rubric aligned with Leech & Short’s explicit linguistic model for future extensions (e.g. phonological/graphological features) |

---

## Related files

| Path | Description |
|------|-------------|
| [`style_rubric.json`](style_rubric.json) | Canonical rubric (this document’s subject) |
| [`Style-in-Fiction.parsed.md`](Style-in-Fiction.parsed.md) | Parsed manuscript input |
| [`extracted/style_knowledge.jsonl`](extracted/style_knowledge.jsonl) | RAG chunks |
| [`extracted/style_analysis_system.json`](extracted/style_analysis_system.json) | LLM system prompt bundle |
| [`../tools/style_extraction/distill_style_system.py`](../tools/style_extraction/distill_style_system.py) | Distillation script |
| [`../tools/style_extraction/manuscript_parser.py`](../tools/style_extraction/manuscript_parser.py) | Parser and checklist extractor |
| [`../tools/style_classification/metrics_computable.py`](../tools/style_classification/metrics_computable.py) | spaCy/textstat metrics |
| [`../tools/style_classification/metrics_llm.py`](../tools/style_classification/metrics_llm.py) | Rubric-grounded LLM assessment |
