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
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

PDF_PATH = ROOT / "source" / "Style-in-Fiction.pdf"
PAGES_DIR = ROOT / "source" / "extracted" / "pages"
IMAGES_DIR = ROOT / "source" / "extracted" / "page_images"
MERGED_PATH = ROOT / "source" / "extracted" / "Style-in-Fiction.md"

SYSTEM_PROMPT = (
    "You transcribe scanned book pages into clean markdown. "
    "Preserve headings, paragraph breaks, lists, tables, footnotes, and emphasis. "
    "Do not summarize or paraphrase. Output only markdown."
)

USER_PROMPT = """Transcribe this page from *Style in Fiction* (Leech & Short) into markdown.

Rules:
- Use `#`, `##`, `###` for headings visible on the page
- Preserve paragraph breaks and block quotes
- Render tables as markdown tables when present
- Include footnotes and page numbers if visible
- Use `*italic*` and `**bold**` where the source shows emphasis
- If the page is mostly blank or a figure plate, transcribe whatever text is visible
- Do not wrap the output in code fences
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
) -> str:
    from llm_client import complete_with_images

    extra_body = None
    if disable_thinking:
        # Qwen3.6 / vLLM-style servers; harmless on other backends.
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

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
    )
    return strip_code_fences(raw)


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
) -> None:
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
    print(f"Output: {output_dir}/page_NNNN.md")
    print()

    t0 = time.time()
    done = 0
    skipped = 0

    for page_number in page_numbers:
        md_path = page_markdown_path(output_dir, page_number)
        if resume and md_path.exists() and md_path.stat().st_size > 0:
            skipped += 1
            print(f"  page {page_number:4d}/{total}  skip (exists)")
            continue

        image_path = page_image_path(images_dir, page_number)
        page_index = page_number - 1

        print(f"  page {page_number:4d}/{total}  render …", end="", flush=True)
        render_page(pdf_path, page_index, image_path, dpi)
        if not save_images:
            # Keep images only for the in-flight request unless user asked to retain them.
            pass
        print(" transcribe …", end="", flush=True)

        markdown = transcribe_page(
            image_path,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout=timeout,
            disable_thinking=disable_thinking,
        )
        md_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
        done += 1
        print(f" ok ({len(markdown.split())} words)")

        if not save_images and image_path.exists():
            image_path.unlink()

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed / 60:.1f} min — transcribed {done}, skipped {skipped}")

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
        check_connection,
    )

    parser = argparse.ArgumentParser(
        description="Convert PDF pages to markdown via a vision LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "LM Studio: load a Qwen3.6 / Qwen-VL vision model, enable the local server.\n"
            "Set LLM_VISION_MODEL to the exact model id shown in LM Studio.\n"
        ),
    )
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--output-dir", type=Path, default=PAGES_DIR)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--merged-path", type=Path, default=MERGED_PATH)
    parser.add_argument("--model", default=DEFAULT_VISION_MODEL, help="Vision model id")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
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
    args = parser.parse_args()

    print(f"Checking vision endpoint at {args.base_url} …")
    try:
        models = check_connection(args.base_url, args.api_key)
        print(f"  Available models: {models or ['(none listed — set --model explicitly)']}")
    except LLMError as exc:
        print(f"\n  Vision server not reachable: {exc}", file=sys.stderr)
        print("  Load a Qwen3.6 vision model in LM Studio and enable the local server.", file=sys.stderr)
        sys.exit(1)

    run(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        merged_path=args.merged_path,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        dpi=args.dpi,
        start_page=args.start_page,
        end_page=args.end_page,
        limit=args.limit,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        resume=not args.no_resume,
        save_images=args.save_images,
        concat=args.concat,
        disable_thinking=args.disable_thinking,
    )


if __name__ == "__main__":
    main()
