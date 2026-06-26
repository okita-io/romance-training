#!/usr/bin/env python3
"""
Scan source-data/hf/ datasets and write a unified JSONL corpus.

Each subdirectory under source-data/hf/ is treated as one HF dataset download.
Optional manifests in source-data/manifests/<slug>.json override field mapping.

Usage:
    python tools/data_preparation/convert_hf_sources.py
    python tools/data_preparation/convert_hf_sources.py --dataset lovecraftcorpus
    python tools/data_preparation/convert_hf_sources.py --output train/romance_corpus/hf_unified.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data_preparation.unified_corpus import (
    AUTHOR_FIELD_CANDIDATES,
    GENRE_FIELD_CANDIDATES,
    TITLE_FIELD_CANDIDATES,
    first_str,
    genres_from_tag_map,
    map_fields,
    normalize_genres,
    normalize_record,
    pick_metadata_blob,
    pick_text_field,
    repo_dir_name,
    slug_from_repo_id,
)

HF_ROOT = ROOT / "source-data" / "hf"
MANIFESTS = ROOT / "source-data" / "manifests"
DEFAULT_OUTPUT = ROOT / "source-data" / "unified" / "fiction_corpus.jsonl"

DATA_FILE_GLOBS = ("*.jsonl", "*.json", "*.csv", "*.tsv", "*.parquet", "*.txt")


def load_manifest(slug: str) -> dict[str, Any] | None:
    for suffix in (".json", ".yaml", ".yml"):
        path = MANIFESTS / f"{slug}{suffix}"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as fh:
            if suffix == ".json":
                data = json.load(fh)
            else:
                try:
                    import yaml
                except ImportError as exc:
                    raise ImportError(
                        f"Install PyYAML to read manifest {path.name}: pip install pyyaml"
                    ) from exc
                data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Manifest must be a mapping: {path}")
        return data
    return None


def infer_repo_id(dataset_dir: Path) -> str:
    name = dataset_dir.name
    if "__" in name:
        author, repo = name.split("__", 1)
        return f"{author}/{repo}"
    return name


def list_data_files(dataset_dir: Path, manifest: dict[str, Any] | None) -> list[Path]:
    if manifest and manifest.get("files"):
        return sorted(dataset_dir.glob(manifest["files"]))

    skip_names = {".gitattributes", "README.md", "dataset_infos.json"}
    files: list[Path] = []
    for pattern in DATA_FILE_GLOBS:
        for path in sorted(dataset_dir.rglob(pattern)):
            if path.name in skip_names:
                continue
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                continue
            files.append(path)
    return files


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def iter_json(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                yield row
    elif isinstance(data, dict):
        for key in ("data", "records", "items", "examples"):
            if isinstance(data.get(key), list):
                for row in data[key]:
                    if isinstance(row, dict):
                        yield row
                return
        yield data


def iter_csv(path: Path) -> Iterator[dict[str, Any]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        for row in reader:
            yield dict(row)


def iter_parquet(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        from datasets import load_dataset

        ds = load_dataset("parquet", data_files=str(path), split="train")
        for row in ds:
            yield dict(row)
        return

    table = pq.read_table(path)
    for row in table.to_pylist():
        if isinstance(row, dict):
            yield row


def iter_plain_text(path: Path) -> Iterator[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if text:
        yield {"text": text, "source_file": path.name}


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from iter_jsonl(path)
    elif suffix == ".json":
        yield from iter_json(path)
    elif suffix in {".csv", ".tsv"}:
        yield from iter_csv(path)
    elif suffix == ".parquet":
        yield from iter_parquet(path)
    elif suffix == ".txt":
        yield from iter_plain_text(path)


def manifest_for_dataset(dataset_dir: Path) -> dict[str, Any]:
    slug = slug_from_repo_id(infer_repo_id(dataset_dir))
    manifest = load_manifest(slug)
    if manifest:
        return manifest
    return {
        "repo_id": infer_repo_id(dataset_dir),
        "slug": slug,
        "genres": [],
    }


def convert_row(
    row: dict[str, Any],
    *,
    manifest: dict[str, Any],
    record_index: int,
) -> dict[str, Any] | None:
    text_field = manifest.get("text_field")
    metadata_field = manifest.get("metadata_field")

    text_key = pick_text_field(row, preferred=text_field)
    if not text_key:
        return None

    text = row[text_key]
    blob = pick_metadata_blob(row, preferred=metadata_field)
    if manifest.get("field_mapping"):
        blob = map_fields(blob, manifest["field_mapping"])

    title = manifest.get("title") or first_str(blob, TITLE_FIELD_CANDIDATES) or first_str(row, TITLE_FIELD_CANDIDATES)
    author = manifest.get("author") or first_str(blob, AUTHOR_FIELD_CANDIDATES) or first_str(row, AUTHOR_FIELD_CANDIDATES)
    genres = normalize_genres(manifest.get("genres"))
    tag_field = manifest.get("genre_tags_field")
    if tag_field and tag_field in row:
        genres.extend(genres_from_tag_map(row[tag_field]))
    genres.extend(normalize_genres(blob.get("genres") or blob.get("genre") or blob.get("category")))
    genres.extend(normalize_genres(row.get("genres") or row.get("genre") or row.get("category")))
    genres = sorted(set(genres))

    source_file = blob.get("source_file") or blob.get("source") or blob.get("file") or blob.get("filename")
    min_words = int(manifest.get("min_words", 30))

    if manifest.get("extra_fields"):
        extra = {k: row[k] for k in manifest["extra_fields"] if k in row}
    else:
        extra = {
            k: v
            for k, v in blob.items()
            if k
            not in {
                *TITLE_FIELD_CANDIDATES,
                *AUTHOR_FIELD_CANDIDATES,
                *GENRE_FIELD_CANDIDATES,
                "source",
                "source_file",
                "file",
                "filename",
            }
        }

    return normalize_record(
        text,
        source_dataset=manifest["repo_id"],
        source_slug=manifest["slug"],
        genres=genres,
        author=author,
        title=title,
        source_file=str(source_file) if source_file else None,
        record_index=record_index,
        extra=extra or None,
        min_words=min_words,
    )


def convert_dataset(dataset_dir: Path) -> list[dict[str, Any]]:
    manifest = manifest_for_dataset(dataset_dir)
    files = list_data_files(dataset_dir, manifest)
    if not files:
        print(f"  No data files found in {dataset_dir}", file=sys.stderr)
        return []

    records: list[dict[str, Any]] = []
    record_index = 0
    for path in files:
        print(f"  {path.relative_to(dataset_dir)}")
        for row in iter_rows(path):
            converted = convert_row(row, manifest=manifest, record_index=record_index)
            record_index += 1
            if converted:
                records.append(converted)
    return records


def discover_datasets(selected: set[str] | None = None) -> list[Path]:
    if not HF_ROOT.exists():
        return []
    dirs = sorted(p for p in HF_ROOT.iterdir() if p.is_dir())
    if not selected:
        return dirs
    out = []
    for path in dirs:
        slug = slug_from_repo_id(infer_repo_id(path))
        if slug in selected or path.name in selected:
            out.append(path)
    return out


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Convert only this slug or directory name (repeatable)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Unified JSONL output path (default: {DEFAULT_OUTPUT.relative_to(ROOT)})",
    )
    args = parser.parse_args()

    selected = set(args.datasets) if args.datasets else None
    dataset_dirs = discover_datasets(selected)
    if not dataset_dirs:
        raise SystemExit(
            f"No datasets found under {HF_ROOT.relative_to(ROOT)}. "
            "Download one first, e.g. python tools/data_preparation/download_hf_dataset.py TristanBehrens/lovecraftcorpus"
        )

    all_records: list[dict[str, Any]] = []
    for dataset_dir in dataset_dirs:
        repo_id = infer_repo_id(dataset_dir)
        print(f"Converting {repo_id} …")
        records = convert_dataset(dataset_dir)
        print(f"  → {len(records)} records")
        all_records.extend(records)

    write_jsonl(all_records, args.output)
    print(f"\nWrote {len(all_records)} records to {args.output}")


if __name__ == "__main__":
    main()
