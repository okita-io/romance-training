#!/usr/bin/env python3
"""
Train Qwen (LoRA) with Unsloth for romance novel generation.
Exports directly to GGUF format for LM Studio.

Configuration (portable, git-friendly):
  - Defaults are merged from, in order:
      train_config.toml (repo root, optional if missing)
      train_config.local.toml (optional, gitignored — personal overrides)
  - CLI: --config /path/to/custom.toml
  - Env: TRAIN_CONFIG_PATH, or ROMANCE_BASE_MODEL (overrides model.base only)

Requirements:
  Python 3.10+ (Unsloth pulls peft>=0.18, which needs 3.10+).

  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install accelerate bitsandbytes "trl>=0.18.2,<=0.24.0,!=0.19.0"
  pip install xformers   # optional; or use install_training_deps.ps1
  pip install tomli      # only on Python <3.11 (see requirements-train.txt)
  pip install cli-charts # optional; terminal loss/LR charts when report_to includes "cli"

Restarts / resume:
  - By default each run retrains from step 0 (same max_steps). Hub model weights reuse the local HF cache.
  - --resume           continue from latest checkpoint in output_dir, or --resume path/to/checkpoint-N
  - --export-only      skip training; reload LoRA from output_dir and redo save + GGUF (if training finished
                       and adapter_config.json exists but GGUF downloads stalled)

Windows GGUF (llama.cpp): requires CMake + Visual Studio 2022 Build Tools (C++ workload).
  This script prepends standard Kitware CMake install dirs to PATH if cmake.exe is not found (conda shells
  often miss it). If Unsloth still says cmake is missing, run install_llama_cpp_windows.ps1 and reopen the terminal.
  Env UNSLOTH_LLAMA_CPP_PATH can point to a prebuilt llama.cpp repo; or place a working clone at ./llama.cpp
  and run Python from the repo root (Unsloth prefers that copy).

VRAM (4-bit QLoRA, bitsandbytes):
  - Qwen/Qwen3.5-35B-A3B is a large MoE + vision model; 4-bit weights must stay on GPU
    (no CPU/disk offload). Plan for ~40GB+ on a single GPU.
  - 24GB cards (e.g. RTX 3090/4090): set model.base in train_config.toml to e.g.
    "Qwen/Qwen2.5-14B-Instruct"
    If you see:
      ValueError: Some modules are dispatched on the CPU or the disk
    that means the full quantized model does not fit; use a smaller model or more VRAM.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path
from typing import Any

# Allow `python train_qwen_unsloth.py` without `pip install -e .`
_train_root = Path(__file__).resolve().parent
_train_src = _train_root / "src"
if _train_src.is_dir():
    _sp = str(_train_src)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _ensure_windows_build_tools_on_path() -> None:
    """Kitware CMake often installs without being on PATH in the same shell as Python/conda."""
    if sys.platform != "win32":
        return
    import shutil

    if shutil.which("cmake") is not None:
        return
    roots = [
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "CMake", "bin"),
        os.path.join(
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            "CMake",
            "bin",
        ),
    ]
    path = os.environ.get("PATH", "")
    for d in roots:
        exe = os.path.join(d, "cmake.exe")
        if os.path.isfile(exe):
            os.environ["PATH"] = d + os.pathsep + path
            path = os.environ["PATH"]
            break


_ensure_windows_build_tools_on_path()

import unsloth  # noqa: F401 — apply patches before other HF imports
from unsloth import FastLanguageModel
import torch
import transformers
from datasets import load_dataset
from packaging.version import Version
from trl import SFTConfig, SFTTrainer
import trl

from romance_factory.cli_training_charts import CliChartLoggerCallback, parse_report_to

_TRL_VER = Version(trl.__version__)
_TF_VER = Version(transformers.__version__)
if _TF_VER >= Version("5.0.0") and _TRL_VER < Version("0.18.0"):
    print(
        f"Incompatible stack: transformers {_TF_VER} requires trl>=0.18 "
        f"(you have trl {_TRL_VER}).\n"
        'Fix: pip install -U "trl>=0.18.2,<=0.24.0,!=0.19.0"\n'
        "Old TRL passes tokenizer= into Trainer; Transformers 5 uses processing_class=.",
        file=sys.stderr,
    )
    sys.exit(1)

_REPO_ROOT = Path(__file__).resolve().parent

_DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "base": "Qwen/Qwen3.5-35B-A3B",
        "max_seq_length": 4096,
        "load_in_4bit": True,
    },
    "lora": {
        "rank": 16,
        "alpha": 16,
    },
    "training": {
        "batch_size": 2,
        "grad_accumulation_steps": 4,
        "max_steps": 1000,
        "learning_rate": 2e-4,
        "sft_max_seq_length": 2048,
        "warmup_steps": 10,
        "logging_steps": 10,
        "save_steps": 250,
        "eval_steps": 250,
        "report_to": "none",
    },
    "paths": {
        "data_dir": "data/romance_corpus",
        "output_dir": "romance_qwen_lora",
    },
    "export": {
        "f16_dir": "romance_qwen_f16",
        "q5_dir": "romance_qwen_q5",
        "q4_dir": "romance_qwen_q4",
    },
}


def _toml_load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        return tomllib.load(f)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, dict)
        ):
            out[key] = _deep_merge(out[key], val)  # type: ignore[arg-type]
        else:
            out[key] = val
    return out


def _resolve_path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (_REPO_ROOT / q)


def load_train_config(cli_config: Path | None) -> dict[str, Any]:
    cfg = _deep_merge({}, _DEFAULT_CONFIG)

    primary: Path | None = cli_config
    if primary is None:
        env_p = os.environ.get("TRAIN_CONFIG_PATH")
        primary = Path(env_p) if env_p else None
    if primary is None:
        default_toml = _REPO_ROOT / "train_config.toml"
        primary = default_toml if default_toml.is_file() else None

    if primary is not None:
        loaded = _toml_load(primary)
        if not loaded and cli_config is not None:
            print(f"ERROR: --config file not found or empty: {primary}", file=sys.stderr)
            sys.exit(1)
        if not loaded and os.environ.get("TRAIN_CONFIG_PATH"):
            print(
                f"ERROR: TRAIN_CONFIG_PATH points to missing/empty file: {primary}",
                file=sys.stderr,
            )
            sys.exit(1)
        cfg = _deep_merge(cfg, loaded)

    local_path = _REPO_ROOT / "train_config.local.toml"
    cfg = _deep_merge(cfg, _toml_load(local_path))

    if os.environ.get("ROMANCE_BASE_MODEL"):
        cfg.setdefault("model", {})["base"] = os.environ["ROMANCE_BASE_MODEL"]

    return cfg


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Romance LoRA training with Unsloth")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML config path (default: train_config.toml or TRAIN_CONFIG_PATH)",
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        help=(
            "Resume training from a checkpoint: omit value for latest under output_dir, "
            "or pass a path (e.g. output_dir/checkpoint-500)"
        ),
    )
    p.add_argument(
        "--export-only",
        action="store_true",
        help=(
            "Skip training: load base model + LoRA from output_dir, then run LoRA save + GGUF only. "
            "Use after training finished and adapter was saved, if GGUF export stalled."
        ),
    )
    return p.parse_args()


args = _parse_args()
_cfg = load_train_config(args.config)

_model = _cfg["model"]
_lora = _cfg["lora"]
_train = _cfg["training"]
_paths = _cfg["paths"]
_export = _cfg["export"]

BASE_MODEL = str(_model["base"])
MAX_SEQ_LENGTH = int(_model["max_seq_length"])
LOAD_IN_4BIT = bool(_model["load_in_4bit"])

LORA_RANK = int(_lora["rank"])
LORA_ALPHA = int(_lora["alpha"])

BATCH_SIZE = int(_train["batch_size"])
GRAD_ACCUM = int(_train["grad_accumulation_steps"])
MAX_STEPS = int(_train["max_steps"])
LEARNING_RATE = float(_train["learning_rate"])
SFT_MAX_SEQ_LENGTH = int(_train["sft_max_seq_length"])
WARMUP_STEPS = int(_train["warmup_steps"])
LOGGING_STEPS = int(_train["logging_steps"])
SAVE_STEPS = int(_train["save_steps"])
EVAL_STEPS = int(_train["eval_steps"])
REPORT_TO_RAW = str(_train["report_to"])
REPORT_TO, USE_CLI_CHARTS = parse_report_to(REPORT_TO_RAW)

DATA_DIR = _resolve_path(str(_paths["data_dir"]))
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "validation.jsonl"
OUTPUT_DIR = str(_resolve_path(str(_paths["output_dir"])))

EXPORT_F16 = str(_resolve_path(str(_export["f16_dir"])))
EXPORT_Q5 = str(_resolve_path(str(_export["q5_dir"])))
EXPORT_Q4 = str(_resolve_path(str(_export["q4_dir"])))

print("=" * 60)
print("Romance fine-tuning with Unsloth")
print("=" * 60)

if torch.cuda.is_available():
    _vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if _vram_gb < 40 and "Qwen3.5-35B-A3B" in BASE_MODEL:
        print(
            f"\nWARNING: GPU reports ~{_vram_gb:.1f} GiB VRAM; Qwen3.5-35B-A3B (MoE+VLM) "
            "often needs ~40+ GiB for 4-bit QLoRA. If load fails, set model.base in "
            "train_config.toml to e.g. Qwen/Qwen2.5-14B-Instruct\n"
        )

# 1. Load model (and LoRA: new training vs. export-only resume)
print("\n[1/6] Loading base model...")
print(f"  Model: {BASE_MODEL}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect (bf16 if available)
    load_in_4bit=LOAD_IN_4BIT,
)

if args.export_only:
    _adapter_cfg = Path(OUTPUT_DIR) / "adapter_config.json"
    if not _adapter_cfg.is_file():
        print(
            f"ERROR: --export-only requires a saved LoRA at {OUTPUT_DIR} "
            f"(missing adapter_config.json). Train first, or use --resume if you only have checkpoint-*.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("[2/6] Loading saved LoRA adapter (skipping training)...")
    from peft import PeftModel
    from unsloth.save import unsloth_save_pretrained_gguf

    model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    # Unsloth attaches save_pretrained_gguf to the inner base; delegated calls use `self` = base,
    # so PEFT merge is skipped and 4-bit save hits Transformers NotImplementedError. Bind to PeftModel.
    model.save_pretrained_gguf = types.MethodType(unsloth_save_pretrained_gguf, model)
else:
    # 2. Add LoRA adapters
    print("[2/6] Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 3. Load training data
    print(f"[3/6] Loading data from {DATA_DIR}...")
    dataset = load_dataset("json", data_files={
        "train": str(TRAIN_FILE),
        "validation": str(VAL_FILE)
    })

    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Val samples: {len(dataset['validation'])}")

    # 4. Setup trainer
    print("[4/6] Setting up trainer...")
    if USE_CLI_CHARTS:
        print(
            f"  CLI charts enabled (report_to was {REPORT_TO_RAW!r}); "
            f"Hugging Face report_to={REPORT_TO!r}. Plots also saved under {OUTPUT_DIR}/cli_charts/"
        )
    sft_args = SFTConfig(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=LOGGING_STEPS,
        output_dir=OUTPUT_DIR,
        optim="adamw_8bit",
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        eval_strategy="steps",
        load_best_model_at_end=True,
        report_to=REPORT_TO,
        dataset_text_field="text",
        max_length=SFT_MAX_SEQ_LENGTH,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )
    if USE_CLI_CHARTS:
        trainer.add_callback(CliChartLoggerCallback(OUTPUT_DIR))

    # 5. Train!
    print(f"[5/6] Starting training for {MAX_STEPS} steps...")
    print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Estimated time: ~{MAX_STEPS * BATCH_SIZE * GRAD_ACCUM / 60:.1f} minutes on RTX 4090")
    if args.resume is not None:
        print(f"  resume_from_checkpoint={args.resume!r}")
    print()

    trainer.train(resume_from_checkpoint=args.resume)

# 6. Save and export
print("\n[6/6] Saving and exporting models...")

# Save LoRA adapter
print("  Saving LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Export to GGUF (multiple quantizations)
print("  Exporting to GGUF formats...")
print("    - F16 (full precision, ~65GB)")
model.save_pretrained_gguf(
    EXPORT_F16,
    tokenizer,
    quantization_method="f16"
)

print("    - Q5_K_M (high quality, ~24GB)")
model.save_pretrained_gguf(
    EXPORT_Q5,
    tokenizer,
    quantization_method="q5_k_m"
)

print("    - Q4_K_M (balanced, ~20GB)")
model.save_pretrained_gguf(
    EXPORT_Q4,
    tokenizer,
    quantization_method="q4_k_m"
)

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print("\nGGUF files ready for LM Studio:")
print(f"  - {EXPORT_F16}/ (full precision)")
print(f"  - {EXPORT_Q5}/ (recommended)")
print(f"  - {EXPORT_Q4}/ (smaller, faster)")
print("\nCopy any .gguf file to LM Studio and load it!")
print("\nRecommended LM Studio settings:")
print("  Temperature: 1.0")
print("  Top P: 0.95")
print("  Top K: 20")
print("  Presence Penalty: 1.5")
