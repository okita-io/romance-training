# vLLM setup — local 3090 (checkpoint)

Process guide and checklist for replacing **LM Studio** with **vLLM** on the **local RTX 3090** as the Phase 2 classification backend.

**Status:** checkpoint — local machine only  
**Last updated:** 2026-07-11

**This session:** Track A (3090) only — standard Docker image, no custom builds.  
**Deferred:** DGX Spark (MTP, DFlash, from-source vLLM) — separate agent; see [vLLM-multi-agent-plan.md](vLLM-multi-agent-plan.md).

**Related docs**

| Doc | Scope |
|-----|-------|
| [GPU_RUNBOOK.md](GPU_RUNBOOK.md) | Phase 2–4 on the 3090 (LM Studio path today) |
| [LLM-Backends.md](LLM-Backends.md) | Client API, chat templates, env vars |
| [vLLM-multi-agent-plan.md](vLLM-multi-agent-plan.md) | Spark only — **not in scope here** |

---

## Checkbox legend

| Marker | Meaning |
|--------|---------|
| `[ ]` | Not started |
| `[-]` | In progress (partially done or blocked) |
| `[x]` | Done |

Update markers in **Track A** as you work. Spark items stay `[ ]` until a later checkpoint.

---

## Current checkpoint — what we're doing now

**Goal:** Get vLLM serving your **current classifier model** on the 3090 so `run_pipeline.py --pass both --workers 4` runs faster and more stably than LM Studio.

**In scope**

- WSL2 + Docker on the 3090 PC
- Pull `vllm/vllm-openai:latest` (stock x86 image — no compile)
- Serve **Profile A** (Ultra Instruct 4X3B 10B Q4_K_M GGUF via `vllm-gguf-plugin`, 4 concurrent sequences)
- Point `.env` at `http://localhost:8000/v1`
- Smoke test → optional benchmark → resume literotica classification

**Out of scope (this session)**

- Spark vLLM, Gemma 4 NVFP4, Marlin MoE flags
- MTP / DFlash speculative decoding (custom builds on Spark)
- Remote classification via `10.0.1.4:8000`
- Phase 4 training (stop vLLM first when you get there)

**Exit criteria for this checkpoint**

- [ ] vLLM healthy at `localhost:8000`
- [ ] `classify_passage.py` returns real LLM fields
- [ ] Pipeline runs `--limit 50` on vLLM without errors
- [ ] Decision: promote vLLM to default over LM Studio (after optional A5 benchmark)

---

## Where this is headed (big picture)

```text
  TODAY (LM Studio)              THIS CHECKPOINT                 LATER (Spark — other agent)
  ─────────────────              ───────────────                 ───────────────────────────
  run_pipeline.py                run_pipeline.py                 Gemma 4 + MTP/DFlash builds
        │                              │                              │
        ▼                              ▼                              ▼
  :1234 llama.cpp                :8000 vLLM Docker               Agent serving / training
  unstable @ 4 workers           Profile A, 4 seqs               (deferred)
```

Phase 2 code is unchanged — only `LLM_BASE_URL` and `LLM_MODEL` switch. Resume semantics, incremental segments, and event logs stay the same.

After local classification finishes, **stop vLLM** before Phase 4 fine-tuning on the same GPU.

---

## Local architecture (3090 only)

```text
┌──────────────────────────────────────────────────────────────┐
│  3090 PC — Windows host + WSL2                               │
│                                                              │
│  PowerShell (repo)              WSL2 Docker                  │
│  ┌────────────────────┐        ┌──────────────────────────┐  │
│  │ run_pipeline.py    │─HTTP──▶│ vllm-classifier-3090     │  │
│  │ manage.py          │ :8000  │ Ultra Instruct 10B GGUF   │  │
│  │ .env → localhost   │        │ --max-num-seqs 4         │  │
│  └─────────┬──────────┘        └──────────────────────────┘  │
│            ▼                                                 │
│  train/romance_corpus/literotica_stories_deep_seg_*.jsonl    │
└──────────────────────────────────────────────────────────────┘

  Styled JSONL → scp to Spark when ready (training handoff — not this session)
```

---

## Model profile (3090 — Profile A)

Use a smaller DavidAU MoE first so the local vLLM checkpoint is about **proving GGUF + vLLM serving**, not fighting the 18.4B Dark Champion memory path.

| Setting | Value |
|---------|-------|
| Model (GGUF) | `DavidAU/Llama-3.2-4X3B-MOE-Ultra-Instruct-10B-GGUF:Q4_K_M` |
| Tokenizer / config | `DavidAU/Llama-3.2-4X3B-MOE-Ultra-Instruct-10B` if available; otherwise use the GGUF repo and add `--hf-config-path` only if vLLM cannot infer config |
| Architecture | mergekit MoE — 4× Llama 3.2 3B experts, likely 2 active per token |
| GGUF size | Q4_K_M, roughly 6–7 GB expected for a 10B MoE |
| vLLM support | `vllm-gguf-plugin` (out-of-tree GGUF quantization plugin) |
| Served name | `ultra-instruct-10b` (via `--served-model-name`; keeps `LLM_MODEL` short) |
| Pipeline | `--pass both --workers 4` |
| vLLM | `--max-num-seqs 4 --max-model-len 8192 --gpu-memory-utilization 0.90` |
| Image | custom local image based on `vllm/vllm-openai:latest` with `vllm-gguf-plugin` installed |
| Thinking | n/a for Llama 3 (leave `LLM_DISABLE_THINKING=1`; harmless) |

Two LLM calls per chunk (fast fields, then deep). One model load, no swap.

**Why this model first:** `DavidAU/Llama-3.2-4X3B-MOE-Ultra-Instruct-10B-GGUF` is closer to the model id already used in the corpus README, smaller than Dark Champion, and already available as the Q4_K_M GGUF style you trust in LM Studio. If GGUF works under vLLM, this is the fastest path to a useful local backend.

**GGUF plugin caveat:** vLLM's GGUF path is official but still experimental and under-optimized. Search results show prior MoE GGUF issues, so treat this as a trial. If Ultra Instruct 10B Q4_K_M fails to load, fall back to the bitsandbytes source-weight path or a dense 3B model.

**Profile B — later comparison:** `DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B-GGUF:Q4_K_M`. Try only after Profile A proves the plugin works.

---

## Progress — this checkpoint

| Step | Status |
|------|--------|
| A1 Prerequisites | `[ ]` |
| A2 vLLM container running | `[ ]` |
| A3 `.env` wired | `[ ]` |
| A4 Smoke tests | `[ ]` |
| A5 Benchmark vs LM Studio (optional) | `[ ]` |
| A6 Production segment | `[ ]` |

**Deferred (Spark — do not work here)**

| Step | Status |
|------|--------|
| Spark vLLM / Gemma 4 | `[ ]` deferred |
| MTP / DFlash custom builds | `[ ]` deferred |
| Remote classify from 3090 → Spark | `[ ]` deferred |

---

## Track A — RTX 3090 (Windows + WSL2)

**Why WSL2 + Docker:** vLLM is Linux-first. Stock Docker image — no source build on the 3090.

### A1. Prerequisites

- [ ] WSL2 Ubuntu installed and updated
- [ ] Docker Desktop installed with WSL integration enabled
- [ ] `nvidia-smi` works in WSL
- [ ] `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` succeeds
- [ ] Hugging Face cache dir: `mkdir -p ~/vllm/hf-cache`
- [ ] **~15 GB free disk** in WSL for the Q4_K_M GGUF download + cache (model is not gated; `HF_TOKEN` optional)
- [ ] `docker pull vllm/vllm-openai:latest`

### A2. GGUF plugin image + launch script

Build a small local image that layers the GGUF plugin onto the stock vLLM image. This avoids installing the plugin every time the container starts.

Save in WSL as `~/vllm/Dockerfile.gguf`:

```dockerfile
FROM vllm/vllm-openai:latest
RUN python3 -m pip install --no-cache-dir "vllm-gguf-plugin==0.0.3"
```

Build it:

```bash
docker build -t vllm-openai-gguf:latest -f ~/vllm/Dockerfile.gguf ~/vllm
```

- [ ] `Dockerfile.gguf` saved
- [ ] `docker build -t vllm-openai-gguf:latest ...` succeeds

Save in WSL as `~/vllm/serve-classifier-3090.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-DavidAU/Llama-3.2-4X3B-MOE-Ultra-Instruct-10B-GGUF:Q4_K_M}"
TOKENIZER="${VLLM_TOKENIZER:-DavidAU/Llama-3.2-4X3B-MOE-Ultra-Instruct-10B}"
SERVED_NAME="${VLLM_SERVED_NAME:-ultra-instruct-10b}"
CONTAINER="${VLLM_CONTAINER:-vllm-classifier-3090}"
PORT="${VLLM_PORT:-8000}"

docker rm -f "$CONTAINER" 2>/dev/null || true

docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  --restart unless-stopped \
  -p "${PORT}:8000" \
  -v ~/vllm/hf-cache:/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  vllm-openai-gguf:latest \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --disable-log-requests

echo "Waiting for health (first launch downloads the Q4_K_M GGUF)..."
until curl -sf "http://localhost:${PORT}/health" >/dev/null; do sleep 10; done
curl -s "http://localhost:${PORT}/v1/models" | python3 -m json.tool
```

- [ ] Script saved; `chmod +x ~/vllm/serve-classifier-3090.sh`
- [ ] First launch OK (Q4_K_M GGUF downloaded)
- [ ] `/v1/models` shows expected model id

If vLLM cannot find the tokenizer repo, temporarily set:

```bash
export VLLM_TOKENIZER="DavidAU/Llama-3.2-4X3B-MOE-Ultra-Instruct-10B-GGUF"
```

If vLLM cannot infer the model config from GGUF metadata, add this extra flag to the script after `--tokenizer "$TOKENIZER"`:

```bash
--hf-config-path "$TOKENIZER" \
```

```bash
~/vllm/serve-classifier-3090.sh          # start
docker stop vllm-classifier-3090         # stop (before Phase 4 training)
docker logs -f vllm-classifier-3090      # debug
```

### A3. Wire the pipeline (Windows `.env`)

```dotenv
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=not-needed
LLM_MODEL=ultra-instruct-10b
LLM_DISABLE_THINKING=1
```

`ultra-instruct-10b` is the `--served-model-name` alias — confirm it in `GET /v1/models`. LM Studio fallback: `http://localhost:1234/v1` with the old GGUF model id.

- [ ] `.env` updated
- [ ] `LLM_MODEL` matches server id

### A4. Smoke tests

PowerShell, repo root:

```powershell
python -c "from tools.llm_client import check_connection; print(check_connection())"

echo "She wakened early, the room still dark." | python tools/style_classification/classify_passage.py --pass fast
```

Quick pipeline slice:

```powershell
python tools/style_classification/run_pipeline.py `
  --pass both --workers 4 --limit 50 `
  --input .\train\incremental\segments\literotica_stories\input\seg_001.jsonl `
  --output .\train\romance_corpus\temp_vllm_smoke.jsonl `
  --no-resume
```

- [ ] `check_connection()` OK
- [ ] `classify_passage.py` returns LLM fields
- [ ] `--limit 50` pipeline completes

**4-way concurrent curl (WSL):**

```bash
MODEL="ultra-instruct-10b"
for i in 1 2 3 4; do
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Reply with JSON: {\"ok\": true}"}],"max_tokens":64}' &
done
wait
```

- [ ] Four parallel requests finish without OOM

### A5. Benchmark vs LM Studio (optional)

Same input, `--limit 200`:

| Run | `LLM_BASE_URL` | rec/s |
|-----|----------------|-------|
| LM Studio | `:1234/v1` | _____ |
| vLLM | `:8000/v1` | _____ |

- [ ] Compared in `train/incremental/logs/*.events.jsonl`
- [ ] vLLM stable at 4 workers ≥30 min OR faster than LM Studio
- [ ] 10-row quality spot check

Skip if A4 `--limit 50` already looks good and you're eager to resume production.

### A6. Production classification

```powershell
python tools/style_classification/run_pipeline.py `
  --pass both --workers 4 `
  --input .\train\incremental\segments\literotica_stories\input\seg_001.jsonl `
  --output .\train\romance_corpus\literotica_stories_deep_seg_001.jsonl
```

Or: `python tools/incremental/manage.py classify-next --corpus literotica_stories --pass both --workers 4`

- [ ] Segment runs on vLLM without intervention
- [ ] Ctrl+C resume works
- [ ] LM Studio demoted to fallback (keep installed)

### A7. Troubleshooting (3090)

| Symptom | Fix |
|---------|-----|
| Cannot reach LLM from PowerShell | `127.0.0.1:8000`; check Docker + WSL port forward |
| `vllm-gguf-plugin` import/load error | Rebuild `vllm-openai-gguf:latest`; confirm plugin version 0.0.3 installed inside the image |
| GGUF MoE load error | GGUF support is experimental; try `--hf-config-path "$TOKENIZER"`; if still broken, use BF16 source + BNB or dense 3B fallback |
| OOM at 4 seqs | `--gpu-memory-utilization 0.85`; `--max-model-len 4096` |
| Slow first start | Normal — Q4_K_M GGUF download + tokenizer/config load; watch `docker logs -f` |
| Throughput below LM Studio | GGUF path is under-optimized in vLLM; try source BF16 + BNB, or continue LM Studio for this model |
| Bad JSON / wrong template | Confirm chat API (not completions); Llama 3 template comes from repo tokenizer config |

---

## Pipeline reference

```text
run_pipeline.py (--workers N)
  └─ classify_passage → metrics_llm.assess → llm_client.complete
                                              POST /v1/chat/completions
```

| `--pass` | Workers | LLM calls / chunk |
|----------|---------|-------------------|
| `both` | 4 | 2 |
| `fast` | 4 | 1 |
| `deep` | 2 | 1 |
| `full` | 2 | 1 |

Per-run override: `--base-url http://localhost:8000/v1`

---

## Environment (local only)

| Variable | Value |
|----------|-------|
| `LLM_BASE_URL` | `http://localhost:8000/v1` |
| `LLM_MODEL` | `ultra-instruct-10b` (served alias; confirm via `/v1/models`) |
| `LLM_DISABLE_THINKING` | `1` |
| `LLM_API_KEY` | any string |

---

## Operational rules (3090)

1. Stop vLLM before Phase 4 training on the same GPU.
2. `--workers` ≤ vLLM `--max-num-seqs` (both 4 for Profile A).
3. Safe to switch LM Studio → vLLM mid-corpus; resume skips completed chunks.
4. LM Studio stays installed as rollback — flip `LLM_BASE_URL` back to `:1234`.

---

## Decision log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-07-11 | 3090: WSL2 Docker + stock `vllm/vllm-openai:latest` | No custom build on Windows |
| 2026-07-11 | Profile A first (`--pass both`, 4 workers) | Active literotica workload |
| 2026-07-11 | Try `DavidAU/Llama-3.2-4X3B-MOE-Ultra-Instruct-10B-GGUF:Q4_K_M` via `vllm-gguf-plugin` first | Smaller than Dark Champion, same DavidAU Llama 3.2 MoE family, matches trusted GGUF quant workflow |
| 2026-07-11 | GGUF Q4_k_m local path on Docker Desktop **failed** | Engine init: MoE GGUF loaded as dense `LlamaForCausalLM`; tensor shape mismatch. Resume classification on LM Studio; next vLLM path = BF16 source + bitsandbytes |
| 2026-07-11 | **Checkpoint scoped to local 3090 only** | Spark agent on MTP/DFlash custom builds — separate track |
| 2026-07-11 | Spark work deferred to [vLLM-multi-agent-plan.md](vLLM-multi-agent-plan.md) | Avoid duplicating effort this session |

---

## Next actions (this session only)

1. `[ ]` A1 — verify WSL2 + Docker GPU in WSL
2. `[ ]` A2 — pull image, run `serve-classifier-3090.sh`
3. `[ ]` A3 — update `.env`
4. `[ ]` A4 — smoke test + `--limit 50`
5. `[ ]` A5 — benchmark (optional)
6. `[ ]` A6 — resume literotica seg_001 on vLLM

---

## Deferred — Spark (other agent)

Do **not** start these on the 3090 session. Checkbox mirror for when Spark is ready.

- [ ] Base vLLM on Spark (Gemma 4, Marlin MoE) — [vLLM-multi-agent-plan.md](vLLM-multi-agent-plan.md)
- [ ] MTP speculative decoding (from-source)
- [ ] DFlash draft model integration
- [ ] Remote `LLM_BASE_URL=http://10.0.1.4:8000/v1` from Windows
- [ ] scp styled JSONL + Phase 4 on Spark
