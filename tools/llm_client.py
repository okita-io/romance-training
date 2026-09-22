"""
Minimal OpenAI-compatible LLM client.

Works with:
  - LM Studio   http://localhost:1234/v1        (default)
  - Ollama      http://localhost:11434/v1
  - Any OpenAI-compatible endpoint

Configure via env vars or pass explicitly:
  LLM_BASE_URL   e.g. http://localhost:1234/v1
  LLM_API_KEY    e.g. lm-studio  (LM Studio ignores this; include anyway)
  LLM_MODEL      model name as shown in LM Studio / Ollama
  LLM_VISION_MODEL  optional override for vision calls (defaults to LLM_MODEL)

Usage:
    from tools.llm_client import complete, DEFAULT_BASE_URL

    text = complete("Summarise this passage.", system="You are a literary critic.")
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _dotenv_paths() -> list[Path]:
    """Repo, train/, and ~/train .env files. First wins; later files fill gaps."""
    return [
        _REPO_ROOT / ".env",
        _REPO_ROOT / "train" / ".env",
        Path.home() / "train" / ".env",
    ]


def _parse_dotenv(path: Path) -> None:
    """Load KEY=value lines without overriding existing environment variables."""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            # Unquoted inline comments: KEY=value # note
            if " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            elif "\t#" in value:
                value = value.split("\t#", 1)[0].rstrip()
        os.environ[key] = value


def _load_repo_dotenv() -> None:
    """Load project .env files if present (does not override existing env vars)."""
    paths = [p for p in _dotenv_paths() if p.is_file()]
    if not paths:
        return
    try:
        from dotenv import load_dotenv
        for env_path in paths:
            load_dotenv(env_path, override=False)
    except ImportError:
        for env_path in paths:
            _parse_dotenv(env_path)

    # train/.env on this Spark uses OPEN_ROUTER_KEY.
    if not (os.environ.get("OPENROUTER_API_KEY") or "").strip():
        alias = (os.environ.get("OPEN_ROUTER_KEY") or "").strip()
        if alias:
            os.environ["OPENROUTER_API_KEY"] = alias


_load_repo_dotenv()

DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "lm-studio")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "local-model")
DEFAULT_VISION_MODEL = os.environ.get("LLM_VISION_MODEL", DEFAULT_MODEL)
DISABLE_THINKING = os.environ.get("LLM_DISABLE_THINKING", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_NEMOTRON_VISION = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
LM_STUDIO_REMOTE_DEFAULT = os.environ.get("LM_STUDIO_BASE_URL", "http://10.0.1.8/:1234/v1")


def pick_vision_model(model_ids: list[str], explicit: str | None = None) -> str:
    """
    Choose a vision-capable model id from an OpenAI-compatible /models list.

    Prefers explicit arg, then LLM_VISION_MODEL env, then gemma/qwen-vl heuristics.
    """
    if explicit:
        return explicit
    env = (os.environ.get("LLM_VISION_MODEL") or "").strip()
    if env:
        return env
    if not model_ids:
        return DEFAULT_VISION_MODEL

    lowered = [(m, m.lower()) for m in model_ids]

    def _pick(predicate) -> str | None:
        for original, low in lowered:
            if predicate(low):
                return original
        return None

    # Vision-capable Gemma / Qwen (multimodal checkpoints)
    hit = _pick(lambda s: "gemma" in s and ("a4b" in s or "nvfp4" in s or "vl" in s or "vision" in s))
    if hit:
        return hit
    hit = _pick(lambda s: "gemma" in s and ("vl" in s or "vision" in s))
    hit = _pick(lambda s: "qwen" in s and ("vl" in s or "vision" in s))
    if hit:
        return hit
    hit = _pick(lambda s: "vision" in s or "vl" in s or "omni" in s)
    if hit:
        return hit
    # Last resort: first loaded model (may be text-only — caller should verify)
    return model_ids[0]


class LLMError(RuntimeError):
    pass


def is_openrouter(base_url: str) -> bool:
    return "openrouter.ai" in base_url


def openrouter_api_key() -> str:
    """Resolve the OpenRouter key. Do not fall back to LM Studio's LLM_API_KEY."""
    for name in ("OPENROUTER_API_KEY", "OPEN_ROUTER_KEY"):
        key = (os.environ.get(name) or "").strip()
        if key:
            return key
    lm = (os.environ.get("LLM_API_KEY") or "").strip()
    if lm.startswith("sk-or-"):
        return lm
    raise LLMError(
        "OPENROUTER_API_KEY (or OPEN_ROUTER_KEY) is required for OpenRouter.\n"
        "Get a key at https://openrouter.ai/keys and put it in train/.env"
    )


def openrouter_headers() -> dict[str, str]:
    """Optional attribution headers recommended by OpenRouter."""
    headers: dict[str, str] = {}
    referer = (os.environ.get("OPENROUTER_HTTP_REFERER") or os.environ.get("HTTP_REFERER") or "").strip()
    title = (os.environ.get("OPENROUTER_X_TITLE") or "romance-training-style-extraction").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def _extract_assistant_text(body: dict[str, Any]) -> str:
    """Pull assistant text from an OpenAI-compatible chat completion body."""
    try:
        choice = body["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected response shape: {body}") from exc

    for key in ("content", "text"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            return val

    # OpenRouter reasoning models may use `reasoning` when content is empty.
    reasoning = msg.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning

    finish = choice.get("finish_reason")
    raise LLMError(
        f"Empty model response (finish_reason={finish!r}). "
        "Try --no-reasoning or increase --max-tokens."
    )


def _chat_completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    stop: list[str] | None = None,
    extra_body: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    max_retries: int = 0,
    retry_backoff: float = 15.0,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if stop:
        payload["stop"] = stop
    if extra_body:
        payload.update(extra_body)

    data = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if extra_headers:
        headers.update(extra_headers)

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read())
            if isinstance(body, dict) and body.get("error"):
                err = body["error"]
                if isinstance(err, dict):
                    code = err.get("code") or 502
                    detail = str(err.get("message") or err)[:400]
                else:
                    code, detail = 502, str(err)[:400]
                raise LLMError(f"HTTP {code} from {url}: {detail}")
            return _extract_assistant_text(body)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:400]
            if exc.code == 429 and attempt < max_retries:
                wait = retry_backoff * (attempt + 1)
                last_exc = LLMError(f"HTTP 429 from {url}: {detail}")
                time.sleep(wait)
                continue
            raise LLMError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except OSError as exc:
            raise LLMError(
                f"Cannot reach LLM at {url}.\n"
                "Start LM Studio (enable local server) or Ollama, then retry.\n"
                f"Original error: {exc}"
            ) from exc

    raise last_exc or LLMError(f"Request failed after {max_retries + 1} attempts")


def image_to_data_url(path: Path) -> str:
    """Encode a local image file as a data: URL for vision APIs."""
    mime, _ = mimetypes.guess_type(path.name)
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def complete(
    user_prompt: str,
    *,
    system: str = "You are a helpful assistant.",
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    temperature: float = 0.05,
    max_tokens: int = 2048,
    timeout: int = 180,
    stop: list[str] | None = None,
    extra_body: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    max_retries: int = 0,
) -> str:
    """
    Send a chat completion request and return the assistant message as a string.
    Raises LLMError on connection failure or non-200 response.
    """
    body = extra_body
    if body is None and DISABLE_THINKING:
        body = {"chat_template_kwargs": {"enable_thinking": False}}

    return _chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        stop=stop,
        extra_body=body,
        extra_headers=extra_headers,
        max_retries=max_retries,
    )


def complete_with_images(
    user_prompt: str,
    image_paths: list[Path | str],
    *,
    system: str = "You are a helpful assistant.",
    model: str = DEFAULT_VISION_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    temperature: float = 0.05,
    max_tokens: int = 4096,
    timeout: int = 300,
    extra_body: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    max_retries: int = 0,
) -> str:
    """
    Send a vision chat completion with one or more local image files.
    Uses the OpenAI-compatible image_url content format.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for raw_path in image_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(path)},
            }
        )

    return _chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_body=extra_body,
        extra_headers=extra_headers,
        max_retries=max_retries,
    )


def check_connection(base_url: str = DEFAULT_BASE_URL, api_key: str = DEFAULT_API_KEY) -> list[str]:
    """
    Return a list of available model IDs, or raise LLMError if unreachable.
    """
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
            return [m["id"] for m in body.get("data", [])]
    except OSError as exc:
        raise LLMError(f"Cannot reach {url}: {exc}") from exc
