"""Parse and filter Style in Fiction vision-transcribed markdown."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator

PAGE_MARKER = "<PAGE>"

_IMAGE_RE = re.compile(r"^!\[.*\]\(.*\)$")
_PAGE_NUM_ONLY_RE = re.compile(r"^\d{1,3}$")
_TOC_ENTRY_RE = re.compile(r"^[\w\s.'\-,&()]+ \d{1,3}$")
_PART_RE = re.compile(r"^PART (ONE|TWO|THREE|FOUR)\b", re.IGNORECASE)
_CHAPTER_RE = re.compile(r"^CHAPTER (\d+)\b", re.IGNORECASE)
_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+(.+)$")
_CHECKLIST_HEADING_RE = re.compile(r"^3\.1\b.*checklist", re.IGNORECASE)
_SUBSECTION_LETTER_RE = re.compile(r"^([A-D]):\s*(.+)$", re.IGNORECASE)
# Numbered checklist item: "5 NOUN PHRASES. Are they …" (uppercase heading,
# optional OCR footnote marker like \( ^{[43]} \) before the period).
_CHECKLIST_ITEM_RE = re.compile(r"^(\d+)\s+([A-Z][A-Z'\- ]*[A-Z])\s*(?:\\\(.*?\\\))?\s*\.\s*(.*)$")
_BRACKET_ONLY_RE = re.compile(r"^\[.*\]$")

_SERIES_NOISE = (
    "english language series",
    "general editor",
    "complex words in english",
    "investigating english style",
    "creating texts",
    "the language of humour",
    "cohesion in english",
    "linguistic guide to english poetry",
)

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "lexical": ("lexic", "vocabulary", "register", "diction", "word choice", "collocation", "nouns", "adjectives", "verbs", "adverbs"),
    "grammatical": ("grammar", "syntax", "sentence", "clause", "passive", "subordination", "coordination", "noun phrase", "verb phrase"),
    "figurative": ("figur", "metaphor", "simile", "irony", "trope", "scheme", "rhetoric", "foreground", "deviance", "phonological"),
    "cohesion": ("cohes", "reference", "anaphora", "ellipsis", "connect", "linking", "discourse", "textual relation", "given", "new information"),
    "context": ("context", "setting", "tone", "atmosphere", "speech presentation", "direct speech", "indirect speech"),
    "viewpoint": ("viewpoint", "point of view", "narrator", "perspective", "voice", "mind style", "fid", "free indirect"),
    "textual": ("segmentation", "salience", "end-focus", "end focus", "climax", "linearity", "graphic unit", "rhythm of prose", "rhetoric of text"),
}


@dataclass
class ManuscriptSection:
    title: str
    content: str
    page: int | None = None
    part: str | None = None
    chapter: int | None = None
    section_id: str | None = None
    categories: list[str] = field(default_factory=list)
    is_checklist: bool = False


def _normalize_line(line: str) -> str:
    return line.replace("\r\n", "\n").replace("\r", "\n").strip()


def _is_noise_line(line: str, *, in_front_matter: bool) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _IMAGE_RE.match(stripped):
        return True
    if _PAGE_NUM_ONLY_RE.match(stripped):
        return True
    if stripped in {"Contents", "Style in Fiction", "Notes", "Publisher's acknowledgements"}:
        return True
    lower = stripped.lower()
    if in_front_matter and any(p in lower for p in _SERIES_NOISE):
        return True
    if in_front_matter and _TOC_ENTRY_RE.match(stripped):
        return True
    if in_front_matter and re.match(r"^\d+\.\d+(\.\d+)?\s+", stripped):
        return True
    if len(stripped) < 4 and stripped.isupper():
        return True
    return False


def _infer_categories(title: str, body: str, *, section_id: str | None = None) -> list[str]:
    combined = f"{title} {body[:1200]}".lower()
    hits = [cat for cat, kws in _CATEGORY_KEYWORDS.items() if any(kw in combined for kw in kws)]
    if section_id and section_id.startswith("3.1"):
        hits.extend(["lexical", "grammatical", "figurative", "cohesion"])
    if section_id and section_id.startswith("7."):
        hits.append("textual")
    if "mind style" in combined or (section_id and section_id.startswith("6.")):
        hits.append("viewpoint")
    return sorted(set(hits)) or ["general"]


def _detect_heading(line: str) -> tuple[str, str | None, int | None, str | None] | None:
    """Return (title, part, chapter, section_id) when line opens a new section."""
    stripped = line.strip()
    if not stripped:
        return None

    part_m = _PART_RE.match(stripped)
    if part_m:
        return f"Part {part_m.group(1).title()}", part_m.group(1).upper(), None, None

    chapter_m = _CHAPTER_RE.match(stripped)
    if chapter_m:
        num = int(chapter_m.group(1))
        return f"Chapter {num}", None, num, None

    section_m = _SECTION_RE.match(stripped)
    if section_m:
        section_id = section_m.group(1)
        title = section_m.group(2).strip()
        return f"{section_id} {title}", None, None, section_id

    if _CHECKLIST_HEADING_RE.search(stripped):
        return stripped, None, None, "3.1"

    # Short standalone headings (Introduction, Aim, Foreword)
    if stripped in {"Introduction", "Aim", "Foreword", "Preface to the second edition"}:
        return stripped, None, None, None

    return None


def split_pages(text: str) -> list[tuple[int, str]]:
    """Split parsed manuscript on <PAGE> markers; page numbers are 1-based indices."""
    chunks = text.split(PAGE_MARKER)
    pages: list[tuple[int, str]] = []
    for idx, chunk in enumerate(chunks, start=1):
        body = chunk.strip()
        if body:
            pages.append((idx, body))
    return pages


def _detect_running_headers(pages: list[tuple[int, str]], *, min_repeats: int = 3) -> set[str]:
    """
    Identify running page headers (book/chapter titles repeated at the top of pages).

    The vision transcription keeps headers like 'Style in Fiction' or 'A method of
    analysis and some examples' as the first content line of each page; left in place
    they get spliced mid-sentence into body text that spans a page break.
    """
    counts: Counter[str] = Counter()
    for _, page_text in pages:
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line or _IMAGE_RE.match(line) or _PAGE_NUM_ONLY_RE.match(line):
                continue
            counts[line] += 1
            break
    return {
        line
        for line, n in counts.items()
        if n >= min_repeats and not _SECTION_RE.match(line) and not _CHAPTER_RE.match(line)
    }


def parse_manuscript(text: str, *, min_section_chars: int = 200) -> list[ManuscriptSection]:
    """
    Parse Style-in-Fiction.parsed.md into filtered, section-tagged blocks.

    Front matter (TOC, series list, copyright) is dropped once the body begins
    at Introduction / Part One.
    """
    in_front_matter = True
    current_part: str | None = None
    current_chapter: int | None = None
    current_section_id: str | None = None
    current_title = "Front matter"
    current_lines: list[str] = []
    current_page: int | None = None

    pages = split_pages(text)
    running_headers = _detect_running_headers(pages)

    sections: list[ManuscriptSection] = []

    def flush() -> None:
        nonlocal current_lines, current_title, current_page
        body = "\n".join(current_lines).strip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        if len(body) < min_section_chars:
            current_lines = []
            return
        categories = _infer_categories(current_title, body, section_id=current_section_id)
        sections.append(
            ManuscriptSection(
                title=current_title,
                content=body,
                page=current_page,
                part=current_part,
                chapter=current_chapter,
                section_id=current_section_id,
                categories=categories,
                is_checklist=current_section_id == "3.1" or "checklist" in current_title.lower(),
            )
        )
        current_lines = []

    for page_num, page_text in pages:
        at_page_top = True
        for raw_line in page_text.splitlines():
            line = _normalize_line(raw_line)
            if _is_noise_line(line, in_front_matter=in_front_matter):
                continue
            # Drop running headers, but only at the top of a page so genuine
            # in-text occurrences of the same phrase survive.
            if at_page_top and line in running_headers:
                continue
            if line:
                at_page_top = False

            if in_front_matter and (
                line == "Introduction"
                or _PART_RE.match(line)
                or line.startswith("Aim")
            ):
                in_front_matter = False

            if in_front_matter:
                continue

            heading = _detect_heading(line)
            if heading:
                flush()
                title, part, chapter, section_id = heading
                current_title = title
                current_page = page_num
                if part:
                    current_part = part
                if chapter is not None:
                    current_chapter = chapter
                if section_id is not None:
                    current_section_id = section_id
                continue

            if not current_lines:
                current_page = page_num
            current_lines.append(line)

    flush()
    return sections


def extract_checklist_items(section: ManuscriptSection) -> list[dict[str, Any]]:
    """Parse Section 3.1 checklist questions into structured prompts."""
    if not section.is_checklist and section.section_id != "3.1":
        return []

    items: list[dict[str, Any]] = []
    current_group = "general"
    current_sub = ""
    current_item: int | None = None
    buf: list[str] = []

    def flush_item() -> None:
        nonlocal buf
        text = " ".join(buf).strip()
        buf = []
        if len(text) < 20:
            return
        item_id = re.sub(r"[^a-z0-9]+", "_", f"{current_group}_{current_sub}_{text[:40]}".lower()).strip("_")
        items.append({
            "id": item_id[:80],
            "group": current_group,
            "subgroup": current_sub or None,
            "item": current_item,
            "prompt": text,
            "source_section": "3.1",
        })

    for line in section.content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Page-reference stubs like "[For notes (i–xiv) on the categories see pp. 66–7]"
        if _BRACKET_ONLY_RE.match(stripped):
            continue

        sub_m = _SUBSECTION_LETTER_RE.match(stripped)
        if sub_m:
            flush_item()
            current_group = sub_m.group(1).lower()
            current_sub = sub_m.group(2).strip()
            current_item = None
            continue

        num_m = _CHECKLIST_ITEM_RE.match(stripped)
        if num_m:
            flush_item()
            current_item = int(num_m.group(1))
            current_sub = num_m.group(2).capitalize()
            rest = num_m.group(3).strip()
            if rest:
                buf = [rest]
            continue

        buf.append(stripped)

    flush_item()
    return items


def sections_to_knowledge_records(
    sections: list[ManuscriptSection],
    *,
    source_file: str,
    max_chars: int = 3500,
) -> list[dict[str, Any]]:
    """Convert manuscript sections to RAG JSONL records."""
    records: list[dict[str, Any]] = []
    for idx, section in enumerate(sections):
        body = section.content
        if len(body) <= max_chars:
            chunks = [body]
        else:
            chunks = []
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
            buf: list[str] = []
            buf_len = 0
            for para in paragraphs:
                if buf_len + len(para) > max_chars and buf:
                    chunks.append("\n\n".join(buf))
                    buf = []
                    buf_len = 0
                buf.append(para)
                buf_len += len(para)
            if buf:
                chunks.append("\n\n".join(buf))

        for part_idx, chunk in enumerate(chunks):
            title = section.title if len(chunks) == 1 else f"{section.title} (part {part_idx + 1})"
            records.append({
                "id": f"sif_{idx:04d}_{part_idx}",
                "source": "style_in_fiction",
                "source_file": source_file,
                "title": title,
                "page": section.page,
                "part": section.part,
                "chapter": section.chapter,
                "section_id": section.section_id,
                "categories": section.categories,
                "is_checklist": section.is_checklist,
                "text": f"## {title}\n\n{chunk}",
                "char_count": len(chunk),
            })
    return records
