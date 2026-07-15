# vLLM multi-agent plan — DGX Spark (GB10)

Implementation reference for running **vLLM** on **NVIDIA DGX Spark** with **Gemma 4 MoE** and **~16 concurrent agent sessions** (OpenClaw-style). Written for resumable setup when SSH from Windows is flaky.

**Status:** planning / not yet deployed  
**Last updated:** 2026-07-05

**Rollout checklist (both machines):** [vLLM-setup.md](vLLM-setup.md)

---

## Goal

Serve **Gemma 4 26B-A4B** (MoE, ~3.8B active params) via vLLM’s OpenAI-compatible API on a single DGX Spark, with enough concurrency for **16 parallel agents** without OOM on unified memory.

The “16 agents” pattern is **one vLLM server** with `--max-num-seqs 16`, not 16 separate model instances.

---

## Hardware facts (drive every config choice)

| Property | Value | Implication |
|----------|-------|-------------|
| SoC | GB10 Grace Blackwell | sm_121 compute capability |
| Host arch | **ARM64 (aarch64)** | Need ARM64 containers or from-source build |
| Memory | **128 GB unified** (CPU + GPU) | `--gpu-memory-utilization` is a fraction of shared pool; OOM can happen with “free” RAM still showing |
| Bandwidth | ~273 GB/s | Decode is bandwidth-bound; small-active MoE models win |
| Single-node TP | **TP=1 only** | No tensor parallelism across GPUs on one Spark |

**Model naming note:** Community “Gemma 4 30B A4B” setups usually mean **`google/gemma-4-26B-A4B-it`** (25.2B total / 3.8B active MoE). For a literal ~30B MoE alternative, see **`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`**.

---

## What does *not* work

- Generic `pip install vllm` on x86 wheels — wrong arch.
- Stock/stable upstream vLLM images without **sm_121** fixes — FP4/MoE kernels crash or emit garbage.
- **`vllm/vllm-openai:gemma4`** (non-`-cu130`) — crashes on GB10 with `FP4 gemm ... sm120/sm121`.
- Native FP4 MoE on GB10 — **must use Marlin W4A16 fallback** (`--moe-backend marlin`).
- High defaults from HF model cards (`gpu-memory-utilization 0.85`, `max-num-seqs 256`) — OOM or unusable KV cache at 16+ concurrent agents.

---

## Prerequisites (host)

From [NVIDIA DGX Spark vLLM playbook](https://build.nvidia.com/spark/vllm):

| Requirement | Verify |
|-------------|--------|
| CUDA 13.0+ | `nvcc --version` |
| NVIDIA driver | `nvidia-smi` (driver ~580.95+ typical on Spark) |
| Docker | `docker --version` |
| NVIDIA Container Toolkit | GPU visible inside containers |
| Python 3.12 (host) | `python3.12 --version` — for HF pre-download |
| Git | `git --version` — only if building from source |
| Network | NGC + Hugging Face access |
| Disk | ~50 GB+ free (image + one NVFP4 model ~15–20 GB) |
| Hugging Face token | Gated Gemma models — accept license on HF first |

### One-time host setup

```bash
# GPU + Docker sanity
nvidia-smi
docker ps || { sudo usermod -aG docker $USER && newgrp docker; }
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi

# Model cache directory (mount into every container)
mkdir -p ~/vllm

# Hugging Face CLI (one-time on host)
pip install -U "huggingface_hub[cli]"
export HF_TOKEN=hf_xxx   # store in ~/.bashrc or pass at runtime

# Optional: NGC login if pulling private NVIDIA images
# docker login nvcr.io   # user: $oauthtoken, password: NGC API key
```

### Unified-memory OOM valve

If a large model fails to load after heavy file I/O:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

---

## Container strategy

Use **Docker** — do not build from source unless you need bleeding-edge Gemma 4 MTP/speculative decoding.

### Two image families (do not mix launch syntax)

| Image | vLLM | Use for | Launch syntax |
|-------|------|---------|---------------|
| **`nvcr.io/nvidia/vllm:26.04-py3`** | 0.19.0 | Stable NGC Spark; many `nvidia/...` checkpoints | Image tag, **then** `vllm serve MODEL ...` |
| **`vllm/vllm-openai:gemma4-cu130`** | Gemma-4 build | **Gemma 4 on GB10 (required)** | Model + flags only — entrypoint is already `vllm serve` |

Other useful tags:

- `vllm/vllm-openai:cu130-nightly` / `:nightly` — newest archs, moves frequently; pin digest for production.
- `nvcr.io/nvidia/vllm:25.12.post1-py3` — Nemotron-3-Nano Spark-tested image.

### Pull commands

```bash
# Gemma 4 (primary plan)
docker pull vllm/vllm-openai:gemma4-cu130

# Stable NGC fallback / non-Gemma models
docker pull nvcr.io/nvidia/vllm:26.04-py3
```

---

## Model options

| Model | Type | Active params | Notes |
|-------|------|---------------|-------|
| **`google/gemma-4-26B-A4B-it`** | MoE BF16 | 3.8B | Gated; baseline, larger memory footprint |
| **`bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4`** | MoE NVFP4 | 3.8B | Community NVFP4 + ships `gemma4_patched.py` (fixes vLLM #38912 scale-key bug) |
| **`nvidia/Gemma-4-26B-A4B-NVFP4`** | MoE NVFP4 | 3.8B | Official NVIDIA quant; may need patched loader mount |
| **`google/gemma-4-31B-it`** | Dense | 31B | ~6 tok/s decode — too slow for interactive multi-agent |
| **`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`** | MoE omni | ~3B | Alternative ~30B-class; separate recipe (audio install, v0.20.0 pin) |

**Recommendation for 16 agents:** Gemma-4-26B-A4B **NVFP4** MoE — fast decode (~50 tok/s single-stream measured on Spark), native tool calling, 256K context window (tune `max-model-len` down for agent workloads).

---

## Pre-stage model weights (strongly recommended)

Avoid re-downloading multi-GB weights on every container recreate.

```bash
export HF_HOME=~/vllm
export HF_TOKEN=hf_xxx

# BF16 (gated — accept license on Hugging Face first)
hf download google/gemma-4-26B-A4B-it

# NVFP4 community build with patched loader (recommended for NVFP4)
hf download bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4 \
  --local-dir ~/vllm/gemma4-26b-a4b-nvfp4
```

Every `docker run` should include:

```bash
  -e HF_HOME=/models \
  -v ~/vllm:/models \
```

Optional compile cache (speeds cold starts after container recreate):

```bash
  -v ~/vllm-compile-cache:/root/.cache/vllm \
```

Air-gapped after staging:

```bash
  -e HF_HUB_OFFLINE=1
```

---

## Multi-agent tuning (16 concurrent sessions)

Key flags for OpenClaw / multi-agent workloads:

| Flag | Recommended | Why |
|------|-------------|-----|
| `--max-num-seqs` | **16** | Caps concurrent requests and KV cache size |
| `--gpu-memory-utilization` | **0.60** | Leaves headroom for 16 KV slots on 128 GB unified memory (community-tested; HF cards often suggest 0.85) |
| `--max-model-len` | **65536** start; up to **262144** if memory allows | Long context vs KV memory tradeoff |
| `--max-num-batched-tokens` | **131072** | Throughput tuning (optional) |
| `--moe-backend` | **marlin** | Mandatory on sm_121 for NVFP4 MoE |
| `--enable-prefix-caching` | on | Reuse system prompts across agents |
| `--enable-chunked-prefill` | on | Better prefill under load |
| `--ipc=host` | on | Shared memory for vLLM workers |

**Quantization flags:**

- NVIDIA ModelOpt NVFP4 (`nvidia/...` repos): `--quantization modelopt`
- Community compressed-tensors NVFP4: auto-detected — **do not** pass `--quantization`

**Tool / reasoning (Gemma 4 agents):**

```bash
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --default-chat-template-kwargs '{"enable_thinking": true}'
```

Per-request override to disable thinking: `chat_template_kwargs={"enable_thinking": false}`.

---

## Launch recipes

### Recipe A — Gemma 4 26B-A4B BF16 (simplest first test)

```bash
export HF_TOKEN=hf_xxx

docker run -d --name gemma4-agents --ipc=host --restart unless-stopped \
  --gpus all -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_HOME=/models \
  -v ~/vllm:/models \
  vllm/vllm-openai:gemma4-cu130 \
  google/gemma-4-26B-A4B-it \
    --moe-backend marlin \
    --gpu-memory-utilization 0.60 \
    --max-model-len 65536 \
    --max-num-seqs 16 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --host 0.0.0.0 --port 8000
```

> BF16 uses more memory than NVFP4 — if OOM, drop `max-model-len` or `max-num-seqs` before switching to NVFP4.

### Recipe B — Gemma 4 26B-A4B NVFP4 (production target)

Uses community checkpoint with patched loader overlay (fixes expert scale-key `KeyError` at load).

```bash
export HF_TOKEN=hf_xxx

# Pre-download (if not done)
hf download bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4 \
  --local-dir ~/vllm/gemma4-26b-a4b-nvfp4

docker run -d --name gemma4-agents --ipc=host --restart unless-stopped \
  --gpus all -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/vllm/gemma4-26b-a4b-nvfp4:/models/gemma4 \
  -v ~/vllm/gemma4-26b-a4b-nvfp4/gemma4_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py \
  vllm/vllm-openai:gemma4-cu130 \
  --model /models/gemma4 --served-model-name gemma-4-26b \
    --quantization modelopt \
    --moe-backend marlin \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.60 \
    --max-model-len 65536 \
    --max-num-seqs 16 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --host 0.0.0.0 --port 8000
```

**Healthy startup log markers:**

- MoE: `Using 'MARLIN' NvFp4 MoE backend`
- Dense: `FLASHINFER_CUTLASS`
- **Bad:** MoE shows `CUTLASS_FP4` or output is `!!!!!` — Marlin flag did not apply

### Recipe C — NGC stable image (non-Gemma or baseline)

```bash
export HF_TOKEN=hf_xxx
export LATEST_VLLM_VERSION=26.04-py3
export HF_MODEL_HANDLE=openai/gpt-oss-20b   # example

docker run -d --name vllm --ipc=host --restart unless-stopped \
  --gpus all -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_HOME=/models \
  -v ~/vllm:/models \
  nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} \
  vllm serve ${HF_MODEL_HANDLE} \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 4
```

Note: NGC images take the full `vllm serve ...` command after the image tag.

---

## Verification checklist

Run after container starts (model load can take several minutes).

```bash
# Wait for health (up to 15 min on first run — JIT warmup)
timeout 900 bash -c 'until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 10; done'

# List models
curl -sS http://localhost:8000/v1/models | jq -r '.data[0].id'

# Math smoke test (expect "204")
curl -sS http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-26b",
    "messages": [{"role": "user", "content": "12*17"}],
    "max_tokens": 50
  }' | jq -r '.choices[0].message.content'

# Tool-call smoke (optional)
curl -sS http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-26b",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
    "max_tokens": 200
  }'

# Container logs
docker logs -f gemma4-agents
```

---

## Wiring agents (OpenClaw / OpenAI-compatible clients)

Point any OpenAI-compatible client at:

| Setting | Value |
|---------|-------|
| Base URL | `http://<spark-host>:8000/v1` |
| API key | dummy (vLLM ignores by default) |
| Model | `gemma-4-26b` (or whatever `--served-model-name` is) |

For **16 parallel agents**, ensure the client-side concurrency matches `--max-num-seqs 16`. Additional requests queue or fail depending on vLLM config.

Example OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
resp = client.chat.completions.create(
    model="gemma-4-26b",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
print(resp.choices[0].message.content)
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `FP4 gemm ... sm120/sm121` | Wrong container tag | Use `gemma4-cu130`, not bare `:gemma4` |
| Output `!!!!!` or NaN scales | MoE not on Marlin | Add `--moe-backend marlin`; check logs |
| `KeyError: ...input_scale` at load | NVFP4 scale keys vs stock gemma4.py | Use patched loader mount (Recipe B) |
| OOM under 128 GB | Unified memory + page cache | Drop caches; lower `--gpu-memory-utilization` and/or `--max-num-seqs` |
| `vllm serve vllm serve` error | Wrong syntax for upstream image | Upstream: model + flags only; NGC: full `vllm serve` |
| Gated model 401/403 | Missing token or license | Accept HF license; set `HF_TOKEN` |
| Slow first request | JIT / CUDA graph warmup | Normal — wait 1–2 min after health OK |
| Permission denied on `~/vllm` | Container downloaded as root | Pre-stage on host, or `--user $(id -u):$(id -g)` |

---

## Optional upgrades (later session)

Not required for initial 16-agent deployment:

| Enhancement | Source | Benefit |
|-------------|--------|---------|
| MTP speculative decoding | [vllm-gb10-gemma4](https://github.com/atcuality2021/vllm-gb10-gemma4) from-source build | ~1.6× single-stream decode |
| DFlash draft model | [AEON-7/vllm-dflash](https://github.com/AEON-7/vllm-dflash) | Up to ~2.7× decode on some models |
| KV cache compression | ManthanQuant patch | ~5× KV shrink for longer context |
| Multi-Spark cluster | [NVIDIA dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) Ray bootstrap | Models that don’t fit one Spark |

---

## Implementation order (suggested)

1. [ ] Host sanity: `nvidia-smi`, Docker GPU test, `mkdir -p ~/vllm`
2. [ ] Set `HF_TOKEN`; accept Gemma license on Hugging Face
3. [ ] `docker pull vllm/vllm-openai:gemma4-cu130`
4. [ ] Pre-download model weights to `~/vllm`
5. [ ] Launch **Recipe A** (BF16) — verify smoke tests
6. [ ] If memory tight or want NVFP4 speed: switch to **Recipe B**
7. [ ] Tune `--max-num-seqs` / `--gpu-memory-utilization` under real agent load
8. [ ] Point agent framework (OpenClaw etc.) at `:8000/v1`
9. [ ] Pin container digest once stable; document final flags

---

## References

- [NVIDIA DGX Spark vLLM playbook](https://build.nvidia.com/spark/vllm)
- [NVIDIA dgx-spark-playbooks — vLLM README](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/vllm)
- [NGC vLLM container catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm)
- [vLLM DGX Spark blog (Jun 2026)](https://vllm.ai/blog/2026-06-01-vllm-dgx-spark)
- [Community Spark + vLLM playbook (Flaviu Vlaicu)](https://vlaicu.io/posts/dgx-vllm-playbook/)
- [vllm-gb10-gemma4 — from-source + benchmarks](https://github.com/atcuality2021/vllm-gb10-gemma4)
- [OpenClaw stack for GB10 (Gemma 4 NVFP4)](https://github.com/chestercs/dgx-openclaw-stack)

---

## Session notes

_Space for implementation log when SSH is stable again:_

```
Date:
Spark hostname:
Image digest used:
Model checkpoint:
Final flags:
Observed tok/s (single / 16 concurrent):
Issues hit:
```
