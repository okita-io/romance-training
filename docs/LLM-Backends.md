# LLM backends, chat templates, and model detection

Reference for **romance-factory** (sibling repo) and this repo’s **`tools/llm_client.py`** when swapping models (Mistral, Gemma, Qwen, Llama 3 fine-tunes, community merges).

**Related:** [GPU_RUNBOOK.md](GPU_RUNBOOK.md) (3090 / LM Studio setup) · [vLLM-multi-agent-plan.md](vLLM-multi-agent-plan.md) (DGX Spark serving)

---

## Three layers (don’t conflate them)

People say “backend” for three different things:

| Layer | What it is | Romance-factory / this repo |
|-------|------------|----------------------------|
| **Client API backend** | Which HTTP API shape the client sends | `LLM_BACKEND`: `openai_chat` (default), `openai_completions`, `ollama`, `openrouter` |
| **Serving engine** | vLLM, Ollama, LM Studio, llama.cpp, KoboldCpp | Whatever you point `--ollama-url` / `LLM_BASE_URL` at |
| **Chat template** | How `system` / `user` / `assistant` become tokens | Applied **on the server** when using the chat API — not in application prompt code |

Romance-factory builds `(system_prompt, user_prompt)` and sends OpenAI-style `messages`. It does **not** emit family-specific tokens (`<start_of_turn>`, `[INST]`, etc.) itself — the **server** applies the model’s Hugging Face Jinja chat template.

That is why romance-factory can run without Kobold skins or per-family client templates: **`openai_chat` + a server that knows the model is the normal modern path.**

---

## Romance-factory client backends

From integration tests in `train/tests/test_openrouter_integration*.py`:

| Backend | Payload | Who formats the prompt |
|---------|---------|------------------------|
| **`openai_chat`** (default) | `{ "messages": [{role, content}, ...] }` | Server applies chat template ✅ |
| **`openai_completions`** | `{ "prompt": "..." }` | Client or LM Studio’s selected template ⚠️ |
| **`ollama`** | Native Ollama API | Ollama applies its template |
| **`openrouter`** | Chat API to remote provider | Provider applies template |

**Recommendation:** stay on **`openai_chat`** unless you deliberately want raw completions.

This repo’s `tools/llm_client.py` uses the same chat-completions shape and optionally disables thinking for Qwen/Gemma-style models:

```python
# When LLM_DISABLE_THINKING=1 in .env:
extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
```

---

## Chat template variance by model family

Templates differ a lot at the **token** level, but look the same at the **client** level when you use the chat completions API.

| Family | Typical wire format | Gotchas for creative / JSON tasks |
|--------|---------------------|-----------------------------------|
| **Mistral / Llama 3** | `[INST]` or header tokens + roles | System-in-user vs dedicated system role (model-dependent) |
| **Gemma 2/3/4** | `<start_of_turn>user/model` | Thinking blocks; tool-call tokens; reasoning on by default on some servers |
| **Qwen 2.5/3** | ChatML `<\|im_start\|>user/assistant` | Qwen3 thinking in `` tags |
| **Generic ChatML** | Shared across many instruct models | Wrong template on wrong model → refusals, meta-commentary, broken JSON |

What changes between models on the **same** OpenAI-compatible server:

- Stop / special tokens
- Whether `system` is first-class or merged into `user`
- Thinking / reasoning defaults (Gemma 4, Qwen3)
- Tool-call syntax (only if you use tools)

You do **not** need application code to emit `<start_of_turn>` manually. You need the **server to load the correct template for the weights.**

---

## Kobold chat templates

**Kobold templates** are legacy **prompt-string** formats from the KoboldAI / KoboldCpp ecosystem, e.g.:

```text
{Instruction} Write the next scene...
{Input} Previous paragraph...
{Response}
```

They made sense when:

- Models were base + manual instruct wrapping
- You hit `/v1/completions` with a raw string
- The server did **not** have per-model HF chat templates

### On modern instruct models + chat API

Kobold templates are usually **wrong or redundant**:

- **Double wrapping** — server adds Gemma/Qwen/Llama tokens *and* Kobold `{Instruction}` noise
- **Wrong stop behavior** — model generates past where it thinks the “response” ended
- **System prompt drift** — romance-factory’s system block may not land where the model expects

Kobold still helps when:

- You deliberately use **`openai_completions`** or KoboldCpp’s native API
- You run an **old instruct** model without a proper chat template
- You manually A/B prompt skins in SillyTavern-style UIs

For romance-factory act generation, editorial JSON, and rewrite passes: **not an upgrade path** on the current architecture.

---

## Will “proper backends/templates” improve success rate?

### Unlikely to move the needle much

| Change | Why |
|--------|-----|
| Kobold template on chat API | Redundant or harmful double-formatting |
| Switching vLLM ↔ LM Studio ↔ Ollama | Throughput/latency; template should match if same model + chat API |
| Per-family client templates in romance-factory | Duplicates server work; high maintenance |

### Likely to help (especially when swapping models)

| Change | Why |
|--------|-----|
| Stay on **`openai_chat`**, not `openai_completions` | Server applies the correct HF template |
| **Match template to weights** in LM Studio (Auto is usually fine) | Wrong template → weird refusals, meta-commentary, broken JSON |
| **Disable thinking** for prose/JSON on Qwen3 / Gemma 4 | Thinking tokens eat budget and leak into output; set `LLM_DISABLE_THINKING=1` or vLLM `--default-chat-template-kwargs '{"enable_thinking": false}'` |
| **Model-specific sampling** | Families differ in `temperature` / `top_p` / repetition penalty for long fiction |
| **JSON editorial passes** | Lower temperature + explicit “JSON only” matters more than template skin |
| **Context truncation** | Long system prompts can silently truncate the user block — looks like “model ignored instructions” |

### Decision rule

- **Chat API + Auto template** → you’re already doing it right
- Quality drops after a model swap → check thinking, template auto-detection, temperature, truncation **before** Kobold
- **Kobold** → only if you move to raw completions / KoboldCpp

---

## Detecting chat templates for fine-tuned models

**Question:** Can you detect which template to use for a fine-tune (e.g. [Vortex5/Silver-Siren-12B](https://huggingface.co/Vortex5/Silver-Siren-12B), Llama 3 lineage)?

**Answer:** **Partially.** Reliable detection uses **metadata**, not weights. You can determine what to **try first**, then validate with a probe.

### What you can detect automatically

#### 1. Embedded template (best source when present)

The authoritative definition lives in the model repo:

- `tokenizer_config.json` → `chat_template` (Jinja string)
- sometimes `chat_template.jinja` as a separate file

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Vortex5/Silver-Siren-12B", trust_remote_code=True)
print(tok.chat_template is not None)
print(tok.apply_chat_template(
    [{"role": "user", "content": "Hi"}],
    tokenize=False,
    add_generation_prompt=True,
))
```

Without downloading the full weights:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Vortex5/Silver-Siren-12B tokenizer_config.json --local-dir ./probe
# inspect probe/tokenizer_config.json for "chat_template"
```

If the formatted string contains Llama-3-style headers (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`, etc.), that is what the repo author intended.

#### 2. Base model / architecture metadata (good fallback)

| Source | Field | Tells you |
|--------|-------|-----------|
| HF model card | **Base model** link | e.g. “fine-tuned from Meta-Llama-3.1-8B-Instruct” |
| `config.json` | `model_type`, `architectures` | `llama`, `LlamaForCausalLM` |
| Merge YAML (Silver-Siren) | `tokenizer.source` | Which tokenizer/template lineage was copied |
| GGUF metadata (LM Studio/Ollama) | `general.architecture`, chat template key | What the quant packer embedded |

For a **straight Llama 3 instruct fine-tune** (single base, tokenizer copied from instruct checkpoint): use the **base instruct template** — usually correct.

For **merges** like Silver-Siren (LunaMaid + Stellar-Witch + …): architecture is still Llama-class, but training format may have been mixed. Default assumption: **Llama 3 chat template from the tokenizer source model** (Silver-Siren uses `Vortex5/LunaMaid-12B`), not “guess from merge name.”

#### 3. Heuristic name / family mapping (weak alone)

Useful as a **first guess**, not proof:

| Name pattern | Likely template family |
|--------------|------------------------|
| `*llama*3*`, `*Llama-3*` | Llama 3 chat |
| `*mistral*`, `*nemo*` | Mistral / ChatML variant |
| `*gemma*` | Gemma turn tokens |
| `*qwen*` | ChatML / Qwen |

Silver-Siren-12B → treat as **Llama 3 family** (tokenizer from LunaMaid-12B). **Confirm** via `tokenizer_config.json`; do not stop at the name.

#### 4. Empirical probe (when metadata is missing or `auto`)

Send a tiny chat request:

```python
messages = [
    {"role": "system", "content": "Reply with exactly: TEMPLATE_OK"},
    {"role": "user", "content": "Go."},
]
```

| Symptom | Likely wrong template |
|---------|------------------------|
| Meta-commentary (“Sure, here’s…”) before answer | ChatML on Llama, or double-wrapped prompt |
| Ignores system instruction | System role not supported in template used |
| Immediate special-token garbage | Mismatch or completions API without template |
| Repeats `{Instruction}` / `{Input}` literally | Kobold/completions skin on chat model |
| Clean `TEMPLATE_OK` | Template is probably fine |

### What you cannot reliably auto-detect

- **Training format from weights alone** — a Llama 3 LoRA might have been trained on ShareGPT, Alpaca, or custom RP formats
- **“Base model = same template”** — fine-tunes sometimes omit or change `chat_template`
- **Merge models** — Silver-Siren’s merge config says `chat_template: auto` (“let the loader decide”), usually the **tokenizer source**, not a guarantee every parent was trained the same way
- **Server-side override** — LM Studio manual template can disagree with the HF repo

Hugging Face removed silent class-default templates because missing `chat_template` used to “work” with the **wrong** format and silently hurt quality.

### Detection ladder (recommended)

```text
1. Read tokenizer_config.json chat_template from the fine-tune repo
      ↓ present?
   YES → use it (server: Auto / model default)
      ↓ missing?
2. Read base_model / tokenizer.source → load THAT repo's chat_template
      ↓ still missing?
3. Fall back to family default (Llama-3-Instruct for Llama 3 fine-tunes)
      ↓ quality still off?
4. Run probe; A/B Llama 3 vs ChatML vs Alpaca on 2–3 romance-factory prompts
```

### Silver-Siren-12B specifically

From the [model card](https://huggingface.co/Vortex5/Silver-Siren-12B):

- Multi-stage merge (Stellar-Witch, LunaMaid, etc.)
- `tokenizer.source: Vortex5/LunaMaid-12B`
- Merge config: `chat_template: auto`

Steps:

1. Inspect `Vortex5/LunaMaid-12B` → `tokenizer_config.json`
2. Assume **Llama 3 chat** unless the card says otherwise
3. LM Studio: **Chat template = Auto**, not Kobold
4. Romance-factory: **`openai_chat`**

### Publishing your own fine-tunes

When merging LoRA or shipping a custom checkpoint:

- Copy `chat_template` from the **exact instruct base** used during SFT
- Save it in `tokenizer_config.json`
- Document base model + template in the model card

Future detection is then trivial: **read the repo, don’t guess.**

---

## Quick reference: romance-factory + style pipeline

```bash
# romance-factory (sibling repo)
LLM_BACKEND=openai_chat
# --ollama-url http://localhost:1234/v1   # LM Studio
# --ollama-url http://localhost:8000/v1   # vLLM on Spark

# This repo (tools/llm_client.py)
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=your-model-id-as-shown-in-server
LLM_DISABLE_THINKING=1    # Qwen3 / Gemma 4 prose + JSON tasks
```

| Setting | LM Studio | vLLM (Spark) |
|---------|-----------|--------------|
| Chat template | **Auto** / model default | Baked into image + model; force `--moe-backend marlin` for NVFP4 MoE |
| Client backend | `openai_chat` | `openai_chat` |
| Thinking | Off for production prose | `--default-chat-template-kwargs '{"enable_thinking": false}'` or per-request kwargs |

---

## Inspect script (optional utility)

To add to this repo later: `tools/inspect_chat_template.py` that, given a Hugging Face model id:

1. Downloads `tokenizer_config.json` (and optionally `config.json`) only
2. Reports whether `chat_template` is set
3. Prints `base_model` / `model_type` if present
4. Renders a sample one-turn prompt via `apply_chat_template`
5. Suggests LM Studio setting: **Auto** vs manual family fallback

Example one-liner until that script exists:

```bash
python -c "
from transformers import AutoTokenizer
m = 'Vortex5/Silver-Siren-12B'
t = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
print('has_template:', bool(t.chat_template))
print(t.apply_chat_template([{'role':'user','content':'Hi'}], tokenize=False, add_generation_prompt=True)[:500])
"
```

---

## Session notes

_Space for model-swap log:_

```
Date:
Model loaded:
HF repo / GGUF path:
chat_template source (embedded / base model / fallback):
LLM_BACKEND:
LLM_DISABLE_THINKING:
Probe result (TEMPLATE_OK):
Notes:
```
