#!/usr/bin/env python3
"""
Probe candidate Phase-2 teacher models via LM Studio CLI + bake-off harness.

Loads each candidate on the **Local** Spark host with an explicit context
window (`lms load -c`), runs bakeoff_5a, then unloads it.

Usage:
  python tools/style_evaluation/teacher_search.py \\
    --limit 10 --context 32768 --require-local \\
    --models gemma-4-31b-styletune-i1 'qwen/qwen3.6-27b'
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_EVAL_SET = ROOT / "eval" / "bakeoff_5a" / "eval_set.jsonl"
DEFAULT_OUT_DIR = ROOT / "eval" / "bakeoff_5a" / "results" / "teacher_search"

# Prefer models already present on Spark Local (see `lms ls` DEVICE column).
DEFAULT_MODELS = (
    "gemma-4-31b-styletune-i1",
    "qwen/qwen3.6-27b",
    "qwen/qwen3.6-35b-a3b",
    "qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive",
    "skyfall-31b-v4.2-heretic-i1",
)

# Known remote-only ids that need a Spark copy before fair testing.
COPY_HINTS: dict[str, str] = {
    "mistral-nemo-style-step2000": (
        "Mac-Mini only today (alexokita/mistral-nemo-style-step2000-GGUF). "
        "Copy the GGUF onto Spark and `lms import`, or publish/download via HF."
    ),
    "qwen3.6-35b-a3b-abliterated-heretic": (
        "DESKTOP only today. Download on Spark with:\n"
        "  lms get https://huggingface.co/Youssofal/Qwen3.6-35B-A3B-Abliterated-Heretic-GGUF --gguf -y"
    ),
}

# Style classify prompts are passage + rubric; LM Studio defaultContextLength is often 8k.
DEFAULT_CONTEXT = 32768


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=False)


def lms_ps_text() -> str:
    proc = subprocess.run(["lms", "ps"], check=False, text=True, capture_output=True)
    return (proc.stdout or "") + (proc.stderr or "")


def lms_ls_text() -> str:
    proc = subprocess.run(["lms", "ls"], check=False, text=True, capture_output=True)
    return (proc.stdout or "") + (proc.stderr or "")


def devices_for_model(model: str, ls_text: str | None = None) -> set[str]:
    """Return device names that have a copy of `model` according to `lms ls`."""
    text = ls_text if ls_text is not None else lms_ls_text()
    devices: set[str] = set()
    for line in text.splitlines():
        if not line.startswith(model):
            # Also allow exact first-token match when model has no slash quirks
            first = line.split(None, 1)[0] if line.strip() else ""
            if first != model:
                continue
        # DEVICE is the last whitespace field (sometimes preceded by ✓ LOADED)
        parts = line.split()
        if not parts:
            continue
        dev = parts[-1]
        if dev == "LOADED" and len(parts) >= 2:
            dev = parts[-2]
        if dev in {"Local", "DESKTOP-1SLNTMU"} or "Mac-Mini" in dev or "Kaido" in dev:
            devices.add(dev)
        # Some rows end with host.local
        elif "." in dev or "DESKTOP" in dev:
            devices.add(dev)
    return devices


def model_loaded(identifier: str) -> bool:
    for line in lms_ps_text().splitlines():
        parts = line.split()
        if parts and parts[0] == identifier:
            return True
    return False


def loaded_context(identifier: str) -> int | None:
    for line in lms_ps_text().splitlines():
        parts = line.split()
        if not parts or parts[0] != identifier:
            continue
        # IDENTIFIER MODEL STATUS SIZE CONTEXT PARALLEL DEVICE ...
        # CONTEXT is usually a bare integer column.
        for tok in parts:
            if tok.isdigit() and int(tok) >= 1024:
                # Prefer the context-looking value (after size like 17.48GB).
                # Heuristic: take the first integer >= 1024 after STATUS.
                return int(tok)
    # Fallback: regex on CONTEXT-ish number near end
    m = re.search(
        rf"^{re.escape(identifier)}\s+\S+\s+\S+\s+\S+\s+(\d+)\s+",
        lms_ps_text(),
        re.M,
    )
    return int(m.group(1)) if m else None


def _ps_row(identifier: str) -> dict[str, str] | None:
    for line in lms_ps_text().splitlines():
        parts = line.split()
        if not parts or parts[0] != identifier:
            continue
        # IDENTIFIER MODEL STATUS SIZE CONTEXT PARALLEL DEVICE ...
        status = parts[2] if len(parts) > 2 else ""
        device = ""
        for tok in reversed(parts):
            if tok in {"Local", "DESKTOP-1SLNTMU"} or "Mac-Mini" in tok or "Kaido" in tok:
                device = tok
                break
            if tok == "LOADED" and len(parts) >= 2:
                continue
        ctx = ""
        for tok in parts:
            if tok.isdigit() and int(tok) >= 1024:
                ctx = tok
                break
        return {"status": status, "device": device, "context": ctx, "raw": line}
    return None


def wait_until_ready(
    identifier: str,
    *,
    context: int,
    require_local: bool,
    timeout_s: float = 300.0,
) -> dict[str, str]:
    """Block until model is on Local (optional), context set, and not still loading."""
    deadline = time.time() + timeout_s
    last: dict[str, str] | None = None
    while time.time() < deadline:
        last = _ps_row(identifier)
        if last:
            status = last["status"].upper()
            ctx_ok = False
            if last["context"].isdigit():
                ctx_ok = int(last["context"]) >= context
            local_ok = (not require_local) or (last["device"] == "Local")
            # LOADING / etc. are not ready; IDLE / GENERATING mean weights are up.
            ready_status = status in {"IDLE", "GENERATING"}
            if ready_status and ctx_ok and local_ok:
                # Extra settle so the OpenAI server publishes the id.
                time.sleep(3)
                if _probe_chat(identifier):
                    return last
                print(f"waiting for API probe… status={status}", flush=True)
            else:
                print(
                    f"waiting for ready… status={status or '?'} "
                    f"device={last['device'] or '?'} context={last['context'] or '?'}",
                    flush=True,
                )
        else:
            print("waiting for model to appear in `lms ps`…", flush=True)
        time.sleep(2)
    raise RuntimeError(
        f"{identifier} not ready after {timeout_s:.0f}s "
        f"(last={last})"
    )


def _probe_chat(identifier: str, base_url: str | None = None) -> bool:
    """Tiny completion to confirm the server can serve this model id."""
    import json
    import urllib.error
    import urllib.request

    url = (base_url or os.environ.get("LLM_BASE_URL") or "http://10.0.1.4:1234/v1").rstrip(
        "/"
    ) + "/chat/completions"
    body = {
        "model": identifier,
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        choice = (payload.get("choices") or [{}])[0]
        msg = (choice.get("message") or {}).get("content") or ""
        print(f"API probe ok ({identifier}): {msg!r}"[:120], flush=True)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"API probe not ready ({identifier}): {exc}", flush=True)
        return False


def load_model(model: str, context: int, identifier: str, *, require_local: bool = True) -> None:
    _run(
        [
            "lms",
            "load",
            model,
            "-y",
            "-c",
            str(context),
            "--identifier",
            identifier,
        ]
    )
    row = wait_until_ready(
        identifier,
        context=context,
        require_local=require_local,
        timeout_s=600.0,
    )
    ctx = int(row["context"]) if row["context"].isdigit() else None
    on_local = row["device"] == "Local"
    print(
        f"Loaded {identifier}: context={ctx} local={on_local} status={row['status']}",
        flush=True,
    )
    if ctx is not None and ctx < context:
        raise RuntimeError(
            f"requested -c {context} but loaded context is {ctx}"
        )
    if require_local and not on_local:
        raise RuntimeError(
            f"{identifier} loaded on a non-Local device ({row['device']}). "
            "Copy the GGUF to Spark and reload (see --require-local)."
        )


def unload_model(identifier: str) -> None:
    if not model_loaded(identifier):
        print(f"(skip unload) {identifier} not loaded", flush=True)
        return
    _run(["lms", "unload", identifier], check=False)


def safe_name(model: str) -> str:
    return model.replace("/", "_").replace("@", "_").replace(":", "_")


def run_bakeoff(model: str, *, eval_set: Path, out: Path, limit: int) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(ROOT / "tools" / "style_evaluation" / "bakeoff_5a.py"),
            "run",
            "--model",
            model,
            "--limit",
            str(limit),
            "--eval-set",
            str(eval_set),
            "--output",
            str(out),
        ]
    )
    return out.with_suffix(".summary.json")


def print_table(summaries: list[dict]) -> None:
    hdr = (
        f"{'model':42} {'n':>3} {'parse':>7} {'gate':>7} "
        f"{'tone':>7} {'pov':>7} {'reg':>7} {'fid':>7} {'fig':>7}"
    )
    print("\n=== teacher search comparison ===")
    print(hdr)
    for r in summaries:
        f = r.get("field_hit_rate") or {}
        print(
            f"{str(r.get('model', ''))[:42]:42} "
            f"{r.get('n', 0):3} "
            f"{r.get('json_parse_rate', 0):7.1%} "
            f"{r.get('gate_exact_rate', 0):7.1%} "
            f"{f.get('tone', 0):7.1%} "
            f"{f.get('pov', 0):7.1%} "
            f"{f.get('register', 0):7.1%} "
            f"{f.get('free_indirect_discourse', 0):7.1%} "
            f"{f.get('figurative_density', 0):7.1%}"
        )


def filter_local(models: list[str]) -> list[str]:
    ls = lms_ls_text()
    kept: list[str] = []
    for model in models:
        devices = devices_for_model(model, ls)
        if "Local" in devices:
            kept.append(model)
            print(f"[local ok] {model} devices={sorted(devices)}")
        else:
            hint = COPY_HINTS.get(model, "Copy/download the GGUF onto Spark Local first.")
            print(f"[skip remote-only] {model} devices={sorted(devices) or ['?']}")
            print(f"  → {hint}")
    return kept


def main() -> None:
    p = argparse.ArgumentParser(description="LM Studio teacher-model bake-off search")
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--limit", type=int, default=10)
    p.add_argument(
        "--context",
        "-c",
        type=int,
        default=DEFAULT_CONTEXT,
        help=f"lms load context length (default {DEFAULT_CONTEXT}; app default is often 8k)",
    )
    p.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--require-local",
        action="store_true",
        default=True,
        help="Only test models present on Spark Local (default: on)",
    )
    p.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow loading on Mac Mini / DESKTOP linked hosts",
    )
    p.add_argument(
        "--keep-loaded",
        action="store_true",
        help="Do not unload the last model (still unloads between candidates)",
    )
    p.add_argument(
        "--skip-load",
        action="store_true",
        help="Assume models are already loaded",
    )
    args = p.parse_args()
    require_local = args.require_local and not args.allow_remote

    summaries: list[dict] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    models = list(args.models)
    if require_local:
        models = filter_local(models)
        if not models:
            raise SystemExit(
                "No candidates available on Local. Copy models to Spark first "
                "(see COPY_HINTS / lms get), then re-run."
            )

    print(f"Candidates: {models}")
    print(f"limit={args.limit}  context={args.context}  require_local={require_local}")
    print("Current lms ps:", flush=True)
    _run(["lms", "ps"], check=False)

    for i, model in enumerate(models):
        identifier = model
        print(f"\n===== [{i+1}/{len(models)}] {model} =====", flush=True)
        try:
            if not args.skip_load:
                unload_model(identifier)
                load_model(
                    model,
                    args.context,
                    identifier,
                    require_local=require_local,
                )

            out = args.out_dir / f"{safe_name(model)}_n{args.limit}_c{args.context}.jsonl"
            summary_path = run_bakeoff(
                identifier,
                eval_set=args.eval_set,
                out=out,
                limit=args.limit,
            )
            if summary_path.is_file():
                summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            print(f"FAILED {model}: {exc}", flush=True)
        finally:
            last = i == len(models) - 1
            if args.keep_loaded and last:
                print(f"(keep loaded) {identifier}", flush=True)
            else:
                unload_model(identifier)

    baseline_dir = ROOT / "eval" / "bakeoff_5a" / "results"
    for name in (
        "ultra_instruct_teacher_n30.summary.json",
        "gemma4_step3000_spark_n30.summary.json",
    ):
        bp = baseline_dir / name
        if bp.is_file():
            summaries.append(json.loads(bp.read_text(encoding="utf-8")))

    print_table(summaries)
    cmp_path = args.out_dir / f"comparison_n{args.limit}_c{args.context}.json"
    cmp_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWrote comparison → {cmp_path}")


if __name__ == "__main__":
    main()
