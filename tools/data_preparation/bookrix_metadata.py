"""Parse author/title from BookRix romance book records."""

from __future__ import annotations

import re
from urllib.parse import unquote

_SLUG_RE = re.compile(r"_ebook-(.+)$", re.IGNORECASE)
_CHAPTER_RE = re.compile(r"\b(chapter|prologue|contents|entry in)\b", re.IGNORECASE)


def _title_from_slug(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.split("-") if part)


def parse_bookrix_url(url: str) -> tuple[str | None, str | None]:
    """Extract author slug hint and title slug from bookrix URL."""
    if not url:
        return None, None
    part = url.rstrip("/").split("/")[-1]
    m = _SLUG_RE.search(part)
    if not m:
        return None, None
    slug = unquote(m.group(1))
    # Heuristic: first two hyphen segments are often first+last name when slug has 3+ parts
    parts = [p for p in slug.split("-") if p]
    if len(parts) >= 3:
        author = f"{parts[0].capitalize()} {parts[1].capitalize()}"
        title = _title_from_slug("-".join(parts[2:]))
        return author, title
    if len(parts) == 2:
        return parts[0].capitalize(), _title_from_slug(parts[1])
    return None, _title_from_slug(slug)


def parse_bookrix_header(text: str) -> tuple[str | None, str | None]:
    """Parse 'Author Name Title words...' from the start of book text."""
    head = text[:400].replace("\r\n", "\n")
    line = head.split("\n", 1)[0].strip()
    if not line:
        return None, None

    # BookRix exports often pad title with spaces before chapter heading
    line = re.split(r"\s{3,}", line)[0].strip()

    # Stop title at chapter/prologue marker
    m = _CHAPTER_RE.search(line)
    if m and m.start() > 10:
        line = line[: m.start()].strip()

    words = line.split()
    if len(words) < 3:
        return None, None

    # First two tokens as author when both look like name parts
    if words[0][0].isupper() and words[1][0].isupper():
        author = f"{words[0]} {words[1]}"
        title = " ".join(words[2:]).strip(" .")
        if title:
            return author, title
    return None, None


def parse_bookrix_metadata(url: str, text: str) -> tuple[str | None, str | None, str | None]:
    """Return (author, title, source_url). Prefer text header, fall back to URL slug."""
    author, title = parse_bookrix_header(text)
    if not author or not title:
        url_author, url_title = parse_bookrix_url(url)
        author = author or url_author
        title = title or url_title
    return author, title, url or None
