"""Unified corpus record schema for the style classifier pipeline."""

from __future__ import annotations

import re
from typing import Any

TEXT_FIELD_CANDIDATES = (
    "text",
    "content",
    "passage",
    "body",
    "story",
    "chapter",
    "paragraph",
    "document",
    "description",
    "blurb",
    "prompt",
    "completion",
)

METADATA_FIELD_CANDIDATES = ("metadata", "meta", "info", "attrs", "attributes")

TITLE_FIELD_CANDIDATES = ("title", "name", "heading", "story_title", "book_title")
AUTHOR_FIELD_CANDIDATES = ("author", "writer", "by")
GENRE_FIELD_CANDIDATES = ("genre", "genres", "category", "categories", "tags")


def slug_from_repo_id(repo_id: str) -> str:
    """TristanBehrens/lovecraftcorpus -> lovecraftcorpus."""
    return repo_id.split("/")[-1]


def repo_dir_name(repo_id: str) -> str:
    """TristanBehrens/lovecraftcorpus -> TristanBehrens__lovecraftcorpus."""
    return repo_id.replace("/", "__")


def word_count(text: str) -> int:
    return len(text.split())


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_prose_text(
    text: str,
    *,
    reflow_ocr: bool = True,
    strip_front_matter: bool = True,
    strip_victorian_section_titles: bool = True,
) -> str:
    """Normalize line endings, reflow OCR wraps, and strip leading front matter."""
    text = normalize_whitespace(text)
    if reflow_ocr:
        from tools.data_preparation.reflow_prose import reflow_ocr_prose

        text = reflow_ocr_prose(text)
    if strip_front_matter:
        from tools.data_preparation.strip_front_matter import strip_front_matter as strip_fm

        text = strip_fm(text).text
    if strip_victorian_section_titles:
        from tools.data_preparation.strip_victorian_section_title import (
            strip_victorian_section_title,
        )

        text = strip_victorian_section_title(text).text
    from tools.data_preparation.strip_dedication import strip_dedication_and_boilerplate

    text = strip_dedication_and_boilerplate(text).text
    from tools.data_preparation.strip_license_agreement import strip_license_agreement

    text = strip_license_agreement(text).text
    from tools.data_preparation.normalize_quotes import normalize_quotes

    text = normalize_quotes(text).text
    return text


def normalize_genres(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,;/|]", value)
        return [p.strip().lower().replace(" ", "_") for p in parts if p.strip()]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(normalize_genres(item))
        return out
    return [str(value).strip().lower().replace(" ", "_")]


def genres_from_tag_map(value: Any) -> list[str]:
    """Extract active tags from a {tag: 0|1} or {tag: bool} mapping."""
    if not isinstance(value, dict):
        return []
    out: list[str] = []
    for key, flag in value.items():
        if flag in (1, True, "1", "true", "True"):
            out.append(str(key).strip().lower().replace("-", "_").replace(" ", "_"))
    return out


def pick_text_field(row: dict[str, Any], preferred: str | None = None) -> str | None:
    if preferred and preferred in row and isinstance(row[preferred], str):
        return preferred
    for key in TEXT_FIELD_CANDIDATES:
        if key in row and isinstance(row[key], str) and row[key].strip():
            return key
    for key, value in row.items():
        if isinstance(value, str) and len(value.split()) >= 20:
            return key
    return None


def pick_metadata_blob(row: dict[str, Any], preferred: str | None = None) -> dict[str, Any]:
    if preferred and preferred in row and isinstance(row[preferred], dict):
        return dict(row[preferred])
    for key in METADATA_FIELD_CANDIDATES:
        if key in row and isinstance(row[key], dict):
            return dict(row[key])
    return {k: v for k, v in row.items() if k not in TEXT_FIELD_CANDIDATES}


def map_fields(blob: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    out = dict(blob)
    for target, source in mapping.items():
        if source in blob:
            out[target] = blob[source]
    return out


def first_str(blob: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = blob.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def normalize_record(
    text: str,
    *,
    source_dataset: str,
    source_slug: str,
    genres: list[str] | None = None,
    author: str | None = None,
    title: str | None = None,
    source_file: str | None = None,
    record_index: int | None = None,
    extra: dict[str, Any] | None = None,
    min_words: int = 30,
    reflow_ocr: bool = True,
) -> dict[str, Any] | None:
    """Return a unified corpus record, or None if the text is too short."""
    text = normalize_prose_text(text, reflow_ocr=reflow_ocr)
    if word_count(text) < min_words:
        return None

    metadata: dict[str, Any] = {
        "source": f"hf:{source_slug}",
        "source_dataset": source_dataset,
        "word_count": word_count(text),
    }
    if genres:
        metadata["genres"] = genres
    if author:
        metadata["author"] = author
    if title:
        metadata["title"] = title
    if source_file:
        metadata["source_file"] = source_file
    if record_index is not None:
        metadata["record_index"] = record_index
    if extra:
        reserved = set(metadata) | {"text"}
        metadata["extra"] = {k: v for k, v in extra.items() if k not in reserved}

    return {"text": text, "metadata": metadata}
