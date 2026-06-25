#!/usr/bin/env python3
"""
Convert PDF pages to markdown via a vision LLM (Qwen3.6 / Qwen-VL in LM Studio).

Renders each page as a PNG, sends it to an OpenAI-compatible vision endpoint,
and writes one markdown file per page. Optionally concatenates all pages into a
single markdown file for downstream rubric extraction.

Usage:
    # LM Studio with Qwen3.6 vision loaded (default http://localhost:1234/v1)
    python tools/style_extraction/pdf_vision_harness.py

    # Smoke test on first 3 pages
    python tools/style_extraction/pdf_vision_harness.py --limit 3

    # Explicit model name (as shown in LM Studio)
    export LLM_VISION_MODEL=qwen3.6-27b-mlx-4bit
    python tools/style_extraction/pdf_vision_harness.py

    # After all pages are done, rubric extraction can reuse the merged markdown:
    python tools/style_extraction/extract_rubric.py --skip-pdf

    # OpenRouter — Nemotron Omni vision (free tier, rate-limit friendly)
    export OPENROUTER_API_KEY=sk-or-...
    python tools/style_extraction/pdf_vision_harness.py --openrouter --limit 3
    python tools/style_extraction/pdf_vision_harness.py --openrouter --cooldown 10
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "style_extraction"))

PDF_PATH = ROOT / "source" / "Style-in-Fiction.pdf"
PAGES_DIR = ROOT / "source" / "extracted" / "pages"
IMAGES_DIR = ROOT / "source" / "extracted" / "page_images"
MERGED_PATH = ROOT / "source" / "extracted" / "Style-in-Fiction.md"

SYSTEM_PROMPT = (
    "You transcribe scanned book pages into clean markdown for a literary-stylistics "
    "knowledge base. Preserve headings, paragraph breaks, lists, tables, footnotes, "
    "and emphasis. Convert diagrams into mermaid flowcharts when they are simple "
    "decision trees or process charts. Do not summarize or paraphrase prose. "
    "Do not explain your process or include reasoning. Output only the transcribed markdown."
)

USER_PROMPT = """Transcribe this page from *Style in Fiction* (Leech & Short) into markdown.

Important: output ONLY the transcribed page content. No analysis, planning, or commentary.

Rules:
- Use `#`, `##`, `###` for headings visible on the page
- Preserve paragraph breaks and block quotes
- Render tables as markdown tables when present
- Include footnotes and page numbers if visible
- Use `*italic*` and `**bold**` where the source shows emphasis
- **Diagrams and flowcharts**: when the page shows a simple flowchart, decision tree,
  or boxed process diagram, convert it to a mermaid flowchart inside a fenced block:

  ```mermaid
  flowchart TD
      A[Box label] --> B[Next box]
  ```

  Use `flowchart TD` for top-down charts and `flowchart LR` for left-right ones.
  Preserve every box label and arrow direction. If a diagram is too complex or
  illegible, add a short `> [Figure: brief description]` blockquote instead.
- If the page is mostly blank or a figure plate, transcribe whatever text is visible
- Do not wrap the full page output in an outer code fence
- Return ONLY the markdown for this page"""


def _require_pymupdf():
    try:
        import fitz  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "PyMuPDF is required for PDF rendering.\n"
            "Install: pip install pymupdf"
        ) from exc


def page_count(pdf_path: Path) -> int:
    import fitz

    with fitz.open(pdf_path) as doc:
        return doc.page_count


def render_page(pdf_path: Path, page_index: int, image_path: Path, dpi: int) -> None:
    import fitz

    image_path.parent.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pix.save(image_path)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```(?:markdown|md)?\s*\n(.*)\n```\s*$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def page_markdown_path(output_dir: Path, page_number: int) -> Path:
    return output_dir / f"page_{page_number:04d}.md"


def page_image_path(images_dir: Path, page_number: int) -> Path:
    return images_dir / f"page_{page_number:04d}.png"


def transcribe_page(
    image_path: Path,
    *,
    model: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
    timeout: int,
    disable_thinking: bool,
    enable_reasoning: bool,
    extra_headers: dict[str, str] | None,
    max_retries: int,
) -> str:
    from llm_client import complete_with_images

    extra_body: dict | None = None
    if disable_thinking:
        # Qwen3.6 / vLLM-style servers; harmless on other backends.
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    elif enable_reasoning:
        # OpenRouter Nemotron reasoning models
        extra_body = {"reasoning": {"enabled": True}}

    raw = complete_with_images(
        USER_PROMPT,
        [image_path],
        system=SYSTEM_PROMPT,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.05,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_body=extra_body,
        extra_headers=extra_headers,
        max_retries=max_retries,
    )
    return strip_code_fences(raw)


def _page_is_usable(md_path: Path) -> bool:
    from page_quality import validate_page_markdown

    if not md_path.exists() or md_path.stat().st_size == 0:
        return False
    body = md_path.read_text(encoding="utf-8", errors="replace")
    return validate_page_markdown(body).ok


def audit_and_report(output_dir: Path) -> int:
    from page_quality import audit_pages_dir

    all_pages = sorted(output_dir.glob("page_*.md"))
    failures = audit_pages_dir(output_dir)
    ok_count = len(all_pages) - len(failures)
    print(f"Audited {len(all_pages)} pages — {ok_count} ok, {len(failures)} failed")
    for item in failures[:30]:
        issues = "; ".join(item["issues"])
        print(f"  page {item['page']:4d}  ({item['word_count']} words)  {issues}")
    if len(failures) > 30:
        print(f"  … and {len(failures) - 30} more")
    return len(failures)


def transcribe_with_validation(
    image_path: Path,
    *,
    model: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
    timeout: int,
    disable_thinking: bool,
    enable_reasoning: bool,
    extra_headers: dict[str, str] | None,
    max_retries: int,
    quality_retries: int,
) -> tuple[str, object]:
    from page_quality import clean_model_output, validate_page_markdown

    last_report = None
    for attempt in range(quality_retries + 1):
        raw = transcribe_page(
            image_path,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout=timeout,
            disable_thinking=disable_thinking,
            enable_reasoning=enable_reasoning,
            extra_headers=extra_headers,
            max_retries=max_retries,
        )
        markdown = clean_model_output(raw)
        report = validate_page_markdown(markdown)
        last_report = report
        if report.ok:
            return markdown, report
        if attempt < quality_retries:
            print(f" retry ({attempt + 1}/{quality_retries}: {report.issues[0]})", end="", flush=True)
    raise ValueError(
        f"quality check failed after {quality_retries + 1} attempts: "
        + "; ".join(last_report.issues if last_report else ())
    )


def concat_pages(output_dir: Path, merged_path: Path, total_pages: int) -> None:
    parts: list[str] = []
    for page_number in range(1, total_pages + 1):
        md_path = page_markdown_path(output_dir, page_number)
        if not md_path.exists():
            raise FileNotFoundError(f"Missing page markdown: {md_path}")
        body = md_path.read_text(encoding="utf-8").strip()
        parts.append(f"<!-- page {page_number} -->\n\n{body}")

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_path.write_text("\n\n---\n\n".join(parts) + "\n", encoding="utf-8")


def run(
    *,
    pdf_path: Path,
    output_dir: Path,
    images_dir: Path,
    merged_path: Path,
    model: str,
    base_url: str,
    api_key: str,
    dpi: int,
    start_page: int,
    end_page: int | None,
    limit: int | None,
    max_tokens: int,
    timeout: int,
    resume: bool,
    save_images: bool,
    concat: bool,
    disable_thinking: bool,
    enable_reasoning: bool,
    extra_headers: dict[str, str] | None,
    max_retries: int,
    cooldown: float,
    validate_existing: bool,
    quality_retries: int,
) -> None:
    from llm_client import LLMError

    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    _require_pymupdf()
    total = page_count(pdf_path)
    first = max(1, start_page)
    last = end_page if end_page is not None else total
    last = min(last, total)

    if first > last:
        raise SystemExit(f"Invalid page range: {first}..{last} (PDF has {total} pages)")

    page_numbers = list(range(first, last + 1))
    if limit is not None:
        page_numbers = page_numbers[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    print(f"PDF: {pdf_path.name} ({total} pages)")
    print(f"Processing pages {page_numbers[0]}..{page_numbers[-1]} ({len(page_numbers)} selected)")
    print(f"Vision model: {model}")
    print(f"Endpoint: {base_url}")
    if cooldown > 0:
        print(f"Cooldown: {cooldown:.0f}s between pages")
    print(f"Output: {output_dir}/page_NNNN.md")
    print()

    t0 = time.time()
    done = 0
    skipped = 0
    transcribed_this_run = 0

    failed = 0

    for page_number in page_numbers:
        md_path = page_markdown_path(output_dir, page_number)
        if resume and md_path.exists() and md_path.stat().st_size > 0:
            if validate_existing:
                if _page_is_usable(md_path):
                    skipped += 1
                    print(f"  page {page_number:4d}/{total}  skip (ok)")
                    continue
                print(f"  page {page_number:4d}/{total}  re-transcribe (quality failed)")
            else:
                skipped += 1
                print(f"  page {page_number:4d}/{total}  skip (exists)")
                continue

        if transcribed_this_run > 0 and cooldown > 0:
            print(f"  cooldown {cooldown:.0f}s …", flush=True)
            time.sleep(cooldown)

        image_path = page_image_path(images_dir, page_number)
        page_index = page_number - 1

        print(f"  page {page_number:4d}/{total}  render …", end="", flush=True)
        render_page(pdf_path, page_index, image_path, dpi)
        print(" transcribe …", end="", flush=True)

        try:
            markdown, report = transcribe_with_validation(
                image_path,
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_tokens=max_tokens,
                timeout=timeout,
                disable_thinking=disable_thinking,
                enable_reasoning=enable_reasoning,
                extra_headers=extra_headers,
                max_retries=max_retries,
                quality_retries=quality_retries,
            )
        except (ValueError, LLMError) as exc:
            failed += 1
            msg = str(exc)
            if "does not support image" in msg or "image inputs" in msg:
                print(
                    f" FAIL (model does not support vision — load a VL/vision model in LM Studio)"
                )
            else:
                print(f" FAIL ({exc})")
            if not save_images and image_path.exists():
                image_path.unlink()
            continue

        md_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
        done += 1
        transcribed_this_run += 1
        print(f" ok ({report.summary()})")

        if not save_images and image_path.exists():
            image_path.unlink()

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed / 60:.1f} min — transcribed {done}, skipped {skipped}, failed {failed}")
    if done or skipped:
        audit_and_report(output_dir)

    if concat:
        # Only merge contiguous pages that exist from 1..total when doing a full run.
        merge_end = last if limit is None else page_numbers[-1]
        print(f"Merging pages 1..{merge_end} → {merged_path}")
        concat_pages(output_dir, merged_path, merge_end)
        print("Merged markdown ready for extract_rubric.py --skip-pdf")


def main() -> None:
    from llm_client import (
        DEFAULT_API_KEY,
        DEFAULT_BASE_URL,
        DEFAULT_VISION_MODEL,
        LLMError,
        LM_STUDIO_REMOTE_DEFAULT,
        OPENROUTER_BASE_URL,
        OPENROUTER_NEMOTRON_VISION,
        check_connection,
        openrouter_api_key,
        openrouter_headers,
        pick_vision_model,
    )

    parser = argparse.ArgumentParser(
        description="Convert PDF pages to markdown via a vision LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "LM Studio remote (Gemma 4 vision):\n"
            "  python tools/style_extraction/pdf_vision_harness.py --lm-studio-remote --limit 3\n"
            "OpenRouter:\n"
            "  python tools/style_extraction/pdf_vision_harness.py --openrouter --limit 3\n"
            "Audit existing pages only:\n"
            "  python tools/style_extraction/pdf_vision_harness.py --audit-only\n"
        ),
    )
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--output-dir", type=Path, default=PAGES_DIR)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--merged-path", type=Path, default=MERGED_PATH)
    parser.add_argument("--model", default=None, help="Vision model id (default: env or --openrouter preset)")
    parser.add_argument("--base-url", default=None, help="API base URL (default: local or OpenRouter with --openrouter)")
    parser.add_argument("--api-key", default=None, help="API key (default: LLM_API_KEY or OPENROUTER_API_KEY)")
    parser.add_argument(
        "--openrouter",
        action="store_true",
        help=(
            "Use OpenRouter (https://openrouter.ai/api/v1) with Nemotron Omni vision. "
            "Sets 10s cooldown unless overridden."
        ),
    )
    parser.add_argument(
        "--lm-studio-remote",
        action="store_true",
        help=f"Use LM Studio at {LM_STUDIO_REMOTE_DEFAULT} (auto-pick Gemma vision model)",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit existing page markdown for quality; do not transcribe",
    )
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        default=True,
        help="On resume, re-transcribe pages that fail quality checks (default: on)",
    )
    parser.add_argument(
        "--no-validate-existing",
        action="store_false",
        dest="validate_existing",
        help="On resume, skip any page that already has a non-empty .md file",
    )
    parser.add_argument(
        "--quality-retries",
        type=int,
        default=2,
        help="Re-request transcription when output fails quality checks (default: 2)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=None,
        help="Seconds to wait between page requests (default: 10 with --openrouter, 0 locally)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Retry count on HTTP 429 (default: 3 with --openrouter, 0 locally)",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Render resolution (default: 180)")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, help="Last page (1-based, inclusive)")
    parser.add_argument("--limit", type=int, help="Process only first N pages in range")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=300, help="Per-page request timeout (seconds)")
    parser.add_argument("--no-resume", action="store_true", help="Re-transcribe even if .md exists")
    parser.add_argument("--save-images", action="store_true", help="Keep rendered PNGs")
    parser.add_argument(
        "--concat",
        action="store_true",
        help="Merge page markdown into source/extracted/Style-in-Fiction.md",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=False for Qwen3.6 servers",
    )
    parser.add_argument(
        "--enable-reasoning",
        action="store_true",
        help="Enable OpenRouter reasoning (auto with --openrouter)",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable reasoning even with --openrouter",
    )
    args = parser.parse_args()

    if args.audit_only:
        n_failed = audit_and_report(args.output_dir)
        sys.exit(1 if n_failed else 0)

    use_openrouter = args.openrouter or (
        args.base_url and "openrouter.ai" in args.base_url
    )
    use_lm_remote = args.lm_studio_remote or (
        args.base_url and "10.0.1.7" in args.base_url
    )

    if use_openrouter:
        base_url = args.base_url or OPENROUTER_BASE_URL
        api_key = args.api_key or openrouter_api_key()
        model = args.model or OPENROUTER_NEMOTRON_VISION
        cooldown = 10.0 if args.cooldown is None else args.cooldown
        max_retries = 3 if args.max_retries is None else args.max_retries
        enable_reasoning = not args.no_reasoning and args.enable_reasoning
        extra_headers = openrouter_headers()
        if args.max_tokens == 4096:
            args.max_tokens = 8192
    elif use_lm_remote:
        base_url = args.base_url or LM_STUDIO_REMOTE_DEFAULT
        api_key = args.api_key or DEFAULT_API_KEY
        model = args.model
        cooldown = 0.0 if args.cooldown is None else args.cooldown
        max_retries = 0 if args.max_retries is None else args.max_retries
        enable_reasoning = False
        extra_headers = None
        args.disable_thinking = True
    else:
        base_url = args.base_url or DEFAULT_BASE_URL
        api_key = args.api_key or DEFAULT_API_KEY
        model = args.model or DEFAULT_VISION_MODEL
        cooldown = 0.0 if args.cooldown is None else args.cooldown
        max_retries = 0 if args.max_retries is None else args.max_retries
        enable_reasoning = args.enable_reasoning and not args.no_reasoning
        extra_headers = None

    if args.disable_thinking and enable_reasoning:
        print("Note: --disable-thinking ignored when reasoning is enabled.", file=sys.stderr)

    print(f"Checking vision endpoint at {base_url} …")
    try:
        models = check_connection(base_url, api_key)
        if models:
            preview = ", ".join(models[:5])
            suffix = "…" if len(models) > 5 else ""
            print(f"  Available models: {preview}{suffix}")
        else:
            print("  (model list not returned — proceeding with --model)")
        if not model and models:
            model = pick_vision_model(models)
            print(f"  Selected model: {model}")
            if "vl" not in model.lower() and "vision" not in model.lower() and "omni" not in model.lower():
                print(
                    "  Warning: selected model may be text-only. "
                    "Load a vision/VL checkpoint in LM Studio if transcription fails.",
                    file=sys.stderr,
                )
    except LLMError as exc:
        print(f"\n  Vision server not reachable: {exc}", file=sys.stderr)
        if use_openrouter:
            print("  Check OPENROUTER_API_KEY and https://openrouter.ai/status", file=sys.stderr)
        elif use_lm_remote:
            print("  Check LM Studio is running on the remote host and the vision model is loaded.", file=sys.stderr)
        else:
            print("  Load a vision model in LM Studio and enable the local server.", file=sys.stderr)
        sys.exit(1)

    if not model:
        print("No vision model selected. Pass --model or load a model in LM Studio.", file=sys.stderr)
        sys.exit(1)

    run(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        merged_path=args.merged_path,
        model=model,
        base_url=base_url,
        api_key=api_key,
        dpi=args.dpi,
        start_page=args.start_page,
        end_page=args.end_page,
        limit=args.limit,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        resume=not args.no_resume,
        save_images=args.save_images,
        concat=args.concat,
        disable_thinking=args.disable_thinking and not enable_reasoning,
        enable_reasoning=enable_reasoning,
        extra_headers=extra_headers,
        max_retries=max_retries,
        cooldown=cooldown,
        validate_existing=args.validate_existing,
        quality_retries=args.quality_retries,
    )


if __name__ == "__main__":
    main()
