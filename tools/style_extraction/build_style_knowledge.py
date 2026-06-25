#!/usr/bin/env python3
"""
Build a RAG-ready knowledge base from vision-transcribed Style in Fiction markdown.

Reads per-page or merged markdown, splits into section-aware chunks with metadata,
and writes JSONL suitable for retrieval during rubric extraction and classification.

Usage:
    python tools/style_extraction/pdf_vision_harness.py --disable-thinking --concat
    python tools/style_extraction/build_style_knowledge.py
    python tools/style_extraction/build_style_knowledge.py --input source/extracted/Style-in-Fiction.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MERGED = ROOT / "source" / "extracted" / "Style-in-Fiction.md"
DEFAULT_PAGES_DIR = ROOT / "source" / "extracted" / "pages"
DEFAULT_OUTPUT = ROOT / "source" / "extracted" / "style_knowledge.jsonl"

_PAGE_MARKER_RE = re.compile(r"^<!--\s*page\s+(\d+)\s*-->\s*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_MERMAID_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Leech & Short chapter categories for retrieval routing.
_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "lexical": ("lexic", "vocabulary", "register", "diction", "word choice", "collocation"),
    "grammatical": ("grammar", "syntax", "sentence", "clause", "passive", "subordination", "coordination"),
    "figurative": ("figur", "metaphor", "simile", "irony", "trope", "scheme", "rhetoric"),
    "cohesion": ("cohes", "reference", "anaphora", "connect", "linking", "discourse"),
    "context": ("context", "setting", "time", "place", "tone", "atmosphere"),
    "viewpoint": ("viewpoint", "point of view", "narrator", "perspective", "voice", "fid"),
}

_MAX_CHUNK_CHARS = 3500
_MIN_CHUNK_CHARS = 120


def _infer_categories(title: str, body: str) -> list[str]:
    combined = (title + " " + body[:800]).lower()
    hits = [cat for cat, kws in _CATEGORY_KEYWORDS.items() if any(kw in combined for kw in kws)]
    return hits or ["general"]


def _page_from_marker(text: str) -> int | None:
    m = _PAGE_MARKER_RE.search(text)
    return int(m.group(1)) if m else None


def _split_sections(text: str) -> list[dict[str, Any]]:
    """Split markdown into sections on headings, preserving page markers."""
    sections: list[dict[str, Any]] = []
    current_title = "Introduction"
    current_lines: list[str] = []
    current_page: int | None = None

    for line in text.splitlines():
        page_m = _PAGE_MARKER_RE.match(line.strip())
        if page_m:
            current_page = int(page_m.group(1))
            continue

        heading_m = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading_m:
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append({
                        "title": current_title,
                        "content": body,
                        "page": current_page,
                        "level": sections[-1]["level"] if sections else 1,
                    })
            current_title = heading_m.group(2).strip()
            current_lines = []
            continue

        current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({
                "title": current_title,
                "content": body,
                "page": current_page,
                "level": 1,
            })

    return sections


def _split_long_section(section: dict[str, Any]) -> list[dict[str, Any]]:
    """Sub-split sections that exceed the retrieval chunk size."""
    body = section["content"]
    if len(body) <= _MAX_CHUNK_CHARS:
        return [section]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
    if len(paragraphs) <= 1:
        # Hard split on sentence boundaries as fallback
        parts = re.split(r"(?<=[.!?])\s+", body)
        paragraphs = []
        buf: list[str] = []
        for sent in parts:
            buf.append(sent)
            if sum(len(s) for s in buf) >= _MAX_CHUNK_CHARS:
                paragraphs.append(" ".join(buf))
                buf = []
        if buf:
            paragraphs.append(" ".join(buf))

    chunks: list[dict[str, Any]] = []
    buf_paras: list[str] = []
    buf_len = 0
    part_idx = 0

    for para in paragraphs:
        if buf_len + len(para) > _MAX_CHUNK_CHARS and buf_paras:
            chunks.append({
                **section,
                "title": f"{section['title']} (part {part_idx + 1})",
                "content": "\n\n".join(buf_paras),
                "part_index": part_idx,
            })
            part_idx += 1
            buf_paras = []
            buf_len = 0
        buf_paras.append(para)
        buf_len += len(para)

    if buf_paras:
        suffix = f" (part {part_idx + 1})" if part_idx else ""
        chunks.append({
            **section,
            "title": section["title"] + suffix,
            "content": "\n\n".join(buf_paras),
            "part_index": part_idx,
        })

    return chunks


def build_chunks(markdown_text: str, source_file: str) -> list[dict[str, Any]]:
    sections = _split_sections(markdown_text)
    records: list[dict[str, Any]] = []

    for i, section in enumerate(sections):
        for sub in _split_long_section(section):
            body = sub["content"]
            if len(body) < _MIN_CHUNK_CHARS:
                continue

            title = sub["title"]
            categories = _infer_categories(title, body)
            mermaid_blocks = [m.group(1).strip() for m in _MERMAID_RE.finditer(body)]

            records.append({
                "id": f"siF_{i:04d}_{sub.get('part_index', 0)}",
                "source": "style_in_fiction",
                "source_file": source_file,
                "title": title,
                "page": sub.get("page"),
                "categories": categories,
                "has_mermaid": bool(mermaid_blocks),
                "mermaid_count": len(mermaid_blocks),
                "text": f"## {title}\n\n{body}",
                "char_count": len(body),
            })

    return records


def load_markdown(input_path: Path, pages_dir: Path | None) -> tuple[str, str]:
    if input_path.is_file():
        return input_path.read_text(encoding="utf-8", errors="replace"), input_path.name

    if pages_dir and pages_dir.is_dir():
        parts: list[str] = []
        for md_path in sorted(pages_dir.glob("page_*.md")):
            page_num = int(md_path.stem.split("_")[1])
            body = md_path.read_text(encoding="utf-8", errors="replace").strip()
            parts.append(f"<!-- page {page_num} -->\n\n{body}")
        if parts:
            return "\n\n---\n\n".join(parts) + "\n", f"{pages_dir.name}/pages"

    raise FileNotFoundError(
        f"No markdown found at {input_path} or {pages_dir}.\n"
        "Run: python tools/style_extraction/pdf_vision_harness.py --concat"
    )


def write_knowledge_base(
    input_path: Path,
    output_path: Path,
    pages_dir: Path | None = DEFAULT_PAGES_DIR,
) -> int:
    text, source_label = load_markdown(input_path, pages_dir)
    records = build_chunks(text, source_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG knowledge base from Style in Fiction markdown")
    parser.add_argument("--input", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--pages-dir", type=Path, default=DEFAULT_PAGES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    n = write_knowledge_base(args.input, args.output, args.pages_dir)
    print(f"Wrote {n} knowledge chunks → {args.output}")
    print("Next: python tools/style_extraction/extract_rubric.py --skip-pdf --use-knowledge")


if __name__ == "__main__":
    main()
