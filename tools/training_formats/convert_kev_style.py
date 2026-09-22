#!/usr/bin/env python3
"""
Convert romance-training style labels into Kev System One JSONL.

Council rows (`metadata.style_council`, consensus=true) are the default gold
set. Optional `--include-profile` also converts `metadata.style_profile`
labels from bulk styled JSONL.

Usage:
    python tools/training_formats/convert_kev_style.py
    python tools/training_formats/convert_kev_style.py --include-profile --profile-limit 2000
    python tools/training_formats/convert_kev_style.py --report-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.style_classification.chunk_text import split_sentences  # noqa: E402
from tools.style_classification.pass_config import ALL_LLM_FIELDS  # noqa: E402

RUBRIC_PATH = ROOT / "source" / "style_rubric.json"
CORPUS_DIR = ROOT / "train" / "romance_corpus"
DEFAULT_OUT_DIR = ROOT / "train" / "style_training" / "kev"
STATE_CHARS_WARN = 1400  # ~384 English tokens; Kev training drops longer states

CHOICE_FIELDS = frozenset({"register", "pov", "tone"})


def load_rubric(path: Path = RUBRIC_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_specs(rubric: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Judgeable LLM fields: rubric dimensions + textual principles, keyed by id."""
    rubric = rubric or load_rubric()
    specs: dict[str, dict[str, Any]] = {}

    def _add(entry: dict[str, Any], *, metric_type: str | None = None) -> None:
        mid = entry.get("id")
        values = entry.get("values")
        if not mid or not isinstance(values, list) or not values:
            return
        if mid not in ALL_LLM_FIELDS:
            return
        if mid in specs:
            return
        specs[mid] = {
            "id": mid,
            "name": entry.get("name", mid),
            "definition": entry.get("definition", ""),
            "analysis_prompt": entry.get("analysis_prompt", ""),
            "metric_type": metric_type or entry.get("metric_type") or "ordinal",
            "values": list(values),
            "scoring": dict(entry.get("scoring") or {}),
        }

    for dim in rubric.get("dimensions") or []:
        if dim.get("computation") == "llm":
            _add(dim)
    for principle in rubric.get("textual_principles") or []:
        _add(principle, metric_type=principle.get("metric_type") or "ordinal")
    return specs


def _instructions(spec: dict[str, Any]) -> str:
    name = spec.get("name") or spec["id"]
    definition = (spec.get("definition") or "").strip()
    prompt = (spec.get("analysis_prompt") or "").strip()
    parts = [f"{name}."]
    if definition:
        parts.append(definition if definition.endswith(".") else definition + ".")
    if prompt:
        parts.append(prompt)
    return " ".join(parts)


def _is_prose_description(text: str, values: list[str]) -> bool:
    """True when rubric scoring text is a phrase, not another label name."""
    stripped = text.strip()
    if not stripped:
        return False
    allowed = {v.casefold() for v in values}
    if stripped.casefold() in allowed:
        return False
    if "_" in stripped:
        return False
    return (" " in stripped) or stripped[:1].isupper()


def _level_description(spec: dict[str, Any], index: int, value: str) -> str:
    scoring = spec.get("scoring") or {}
    values = spec["values"]
    if len(values) == 3:
        for key, idx in (("low", 0), ("mid", 1), ("high", 2)):
            text = scoring.get(key)
            if idx == index and isinstance(text, str) and _is_prose_description(text, values):
                return f"{value}: {text.strip()}"
    return value


def question_bank(specs: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """Stable Kev question templates (instructions + criteria, no labels)."""
    specs = specs or metric_specs()
    bank: dict[str, dict[str, Any]] = {}
    for field, spec in specs.items():
        qtype = "choice" if field in CHOICE_FIELDS or spec["metric_type"] == "categorical" else "score"
        question: dict[str, Any] = {
            "type": qtype,
            "instructions": _instructions(spec),
        }
        if qtype == "choice":
            question["criteria"] = {
                value: _level_description(spec, i, value)
                for i, value in enumerate(spec["values"])
            }
        else:
            question["criteria"] = [
                _level_description(spec, i, value) for i, value in enumerate(spec["values"])
            ]
        bank[field] = question
    return bank


def truncate_state(text: str, max_chars: int = STATE_CHARS_WARN) -> tuple[str, bool]:
    """Fit Kev's ~384-token training budget, preferring a sentence boundary."""
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    sentences = split_sentences(text)
    kept: list[str] = []
    size = 0
    for sent in sentences:
        extra = len(sent) if not kept else len(sent) + 1
        if kept and size + extra > max_chars:
            break
        if not kept and extra > max_chars:
            clipped = sent[: max_chars - 1].rstrip() + "…"
            return clipped, True
        kept.append(sent)
        size += extra
    if kept:
        return " ".join(kept), True
    return text[: max_chars - 1].rstrip() + "…", True


def _normalized_state(text: str) -> str:
    return " ".join(text.casefold().split())


def state_key(text: str) -> str:
    return hashlib.sha256(_normalized_state(text).encode()).hexdigest()


def _valid_label(spec: dict[str, Any], value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value in spec["values"] else None


def extract_consensus_labels(
    record: dict[str, Any],
    specs: dict[str, dict[str, Any]],
    *,
    min_agree: int = 2,
) -> dict[str, tuple[str, dict[str, Any]]]:
    """field -> (label, council_meta) for consensus council votes."""
    meta = record.get("metadata") or {}
    council = meta.get("style_council") or {}
    out: dict[str, tuple[str, dict[str, Any]]] = {}
    if not isinstance(council, dict):
        return out
    for field, spec in specs.items():
        entry = council.get(field)
        if not isinstance(entry, dict):
            continue
        if not entry.get("consensus"):
            continue
        agree_n = int(entry.get("agree_n") or 0)
        if agree_n < min_agree:
            continue
        label = _valid_label(spec, entry.get("value"))
        if label is None:
            continue
        out[field] = (label, entry)
    return out


def extract_profile_labels(
    record: dict[str, Any],
    specs: dict[str, dict[str, Any]],
) -> dict[str, tuple[str, dict[str, Any]]]:
    """field -> (label, stub meta) from style_profile."""
    meta = record.get("metadata") or {}
    profile = meta.get("style_profile") or {}
    out: dict[str, tuple[str, dict[str, Any]]] = {}
    if not isinstance(profile, dict):
        return out
    for field, spec in specs.items():
        label = _valid_label(spec, profile.get(field))
        if label is None:
            continue
        out[field] = (label, {"source": "style_profile"})
    return out


def labelled_question(
    field: str,
    label: str,
    spec: dict[str, Any],
    bank: dict[str, dict[str, Any]],
    *,
    src: str,
) -> dict[str, Any]:
    template = bank[field]
    question = {
        "type": template["type"],
        "instructions": template["instructions"],
        "criteria": template["criteria"],
        "src": src,
    }
    if template["type"] == "choice":
        question["label"] = label
    else:
        question["label"] = spec["values"].index(label)
    return question


def to_kev_record(
    record: dict[str, Any],
    labels: dict[str, tuple[str, dict[str, Any]]],
    specs: dict[str, dict[str, Any]],
    bank: dict[str, dict[str, Any]],
    *,
    src: str,
    max_state_chars: int = STATE_CHARS_WARN,
) -> dict[str, Any] | None:
    text = (record.get("text") or "").strip()
    if not text or not labels:
        return None
    questions: dict[str, dict[str, Any]] = {}
    agree_sum = 0
    for field, (label, entry) in labels.items():
        questions[field] = labelled_question(field, label, specs[field], bank, src=src)
        agree_sum += int(entry.get("agree_n") or 0)
    if not questions:
        return None
    state, truncated = truncate_state(text, max_state_chars)
    meta = record.get("metadata") or {}
    extra = meta.get("extra") or {}
    return {
        "state": state,
        "questions": questions,
        "_meta": {
            "source": src,
            "source_file": str(meta.get("source_file") or ""),
            "source_dataset": str(meta.get("source_dataset") or meta.get("source") or ""),
            "title": str(meta.get("title") or extra.get("title_slug") or ""),
            "author": str(meta.get("author") or ""),
            "story_key": str(meta.get("story_key") or extra.get("story_key") or ""),
            "chunk_index": meta.get("chunk_index", extra.get("chunk_index")),
            "original_chars": len(text),
            "truncated": truncated,
            "n_questions": len(questions),
            "agree_sum": agree_sum,
            "state_sha256": state_key(text),
        },
    }


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def discover_council_paths(corpus_dir: Path = CORPUS_DIR) -> list[Path]:
    return sorted(
        p
        for p in corpus_dir.glob("_council_*_out.jsonl")
        if p.is_file() and not p.name.endswith("_in.jsonl")
    )


def discover_profile_paths(corpus_dir: Path = CORPUS_DIR) -> list[Path]:
    return sorted(
        p
        for p in corpus_dir.glob("*_styled_seg_*.jsonl")
        if p.is_file() and not p.name.startswith("_")
    )


def convert_paths(
    paths: list[Path],
    *,
    specs: dict[str, dict[str, Any]],
    bank: dict[str, dict[str, Any]],
    mode: str,
    min_agree: int,
    max_state_chars: int,
    limit: int = 0,
    seed: int = 7,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {
        "files": [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in paths],
        "rows_read": 0,
        "rows_empty": 0,
        "rows_no_labels": 0,
        "kept": 0,
        "truncated": 0,
        "by_field": Counter(),
        "labels": {field: Counter() for field in specs},
        "questions_per_record": Counter(),
    }
    by_state: dict[str, dict[str, Any]] = {}
    rng = random.Random(seed)

    for path in paths:
        for record in iter_jsonl(path):
            stats["rows_read"] += 1
            text = (record.get("text") or "").strip()
            if not text:
                stats["rows_empty"] += 1
                continue
            if mode == "council":
                labels = extract_consensus_labels(record, specs, min_agree=min_agree)
            else:
                labels = extract_profile_labels(record, specs)
            if not labels:
                stats["rows_no_labels"] += 1
                continue
            converted = to_kev_record(
                record,
                labels,
                specs,
                bank,
                src=mode,
                max_state_chars=max_state_chars,
            )
            if converted is None:
                stats["rows_no_labels"] += 1
                continue
            key = converted["_meta"]["state_sha256"]
            previous = by_state.get(key)
            if previous is None:
                by_state[key] = converted
            else:
                prev_n = previous["_meta"]["n_questions"]
                new_n = converted["_meta"]["n_questions"]
                prev_agree = previous["_meta"]["agree_sum"]
                new_agree = converted["_meta"]["agree_sum"]
                if (new_n, new_agree) > (prev_n, prev_agree):
                    by_state[key] = converted

    records = list(by_state.values())
    if limit and len(records) > limit:
        records = rng.sample(records, limit)

    for converted in records:
        stats["kept"] += 1
        if converted["_meta"]["truncated"]:
            stats["truncated"] += 1
        nq = converted["_meta"]["n_questions"]
        stats["questions_per_record"][nq] += 1
        for field, question in converted["questions"].items():
            stats["by_field"][field] += 1
            if question["type"] == "choice":
                stats["labels"][field][str(question["label"])] += 1
            else:
                value = specs[field]["values"][question["label"]]
                stats["labels"][field][value] += 1

    stats["by_field"] = dict(stats["by_field"])
    stats["labels"] = {k: dict(v) for k, v in stats["labels"].items() if v}
    stats["questions_per_record"] = {
        str(k): v for k, v in sorted(stats["questions_per_record"].items())
    }
    stats["unique_states"] = len(by_state)
    stats["sampled"] = len(records)
    return records, stats


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_workload(path: Path, bank: dict[str, dict[str, Any]]) -> None:
    questions = {
        field: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
        for field, q in bank.items()
    }
    spec = {
        "name": "romance-style-council",
        "domain": (
            "Literary prose style classification using Leech & Short Style in Fiction. "
            "Each state is a fiction passage; questions are the Phase 2 LLM style fields."
        ),
        "state": (
            "A fiction passage of roughly 150-500 words. Training states are truncated "
            "to ~1400 characters (~384 tokens)."
        ),
        "questions": questions,
        "guidance": (
            "Use only observable linguistic evidence in the passage. "
            "choice fields (register, pov, tone) are unordered categories. "
            "score fields are ordered levels; label 0 is the first rubric value. "
            "Train on council consensus labels; style_profile labels are noisier."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _print_stats(title: str, stats: dict[str, Any], specs: dict[str, dict[str, Any]]) -> None:
    print(f"\n== {title} ==")
    print(
        f"read {stats['rows_read']} rows from {len(stats['files'])} files; "
        f"{stats['unique_states']} unique states; wrote {stats['sampled']}; "
        f"truncated {stats['truncated']}; no-label {stats['rows_no_labels']}"
    )
    print("questions per record:", stats["questions_per_record"] or "{}")
    for field in sorted(stats["by_field"], key=lambda f: (-stats["by_field"][f], f)):
        qtype = "choice" if field in CHOICE_FIELDS else "score"
        dist = stats["labels"].get(field) or {}
        parts = ", ".join(f"{k}={v}" for k, v in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])))
        print(f"  {field} ({qtype}, n={stats['by_field'][field]}): {parts}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--council", nargs="*", type=Path, default=None, help="Council *_out.jsonl (default: discover)")
    ap.add_argument("--include-profile", action="store_true", help="Also convert bulk style_profile JSONL")
    ap.add_argument("--profile", nargs="*", type=Path, default=None, help="Styled JSONL (default: *_styled_seg_*.jsonl)")
    ap.add_argument("--profile-limit", type=int, default=2000, help="Max unique profile records (0 = all)")
    ap.add_argument("--min-agree", type=int, default=2)
    ap.add_argument("--max-state-chars", type=int, default=STATE_CHARS_WARN)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--report-only", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specs = metric_specs()
    bank = question_bank(specs)
    out_dir: Path = args.out_dir

    council_paths = args.council if args.council is not None else discover_council_paths()
    if not council_paths:
        print("no council *_out.jsonl files found", file=sys.stderr)
        return 1

    council_records, council_stats = convert_paths(
        council_paths,
        specs=specs,
        bank=bank,
        mode="council",
        min_agree=args.min_agree,
        max_state_chars=args.max_state_chars,
    )
    _print_stats("council consensus", council_stats, specs)

    profile_records: list[dict[str, Any]] = []
    profile_stats: dict[str, Any] | None = None
    if args.include_profile:
        profile_paths = args.profile if args.profile is not None else discover_profile_paths()
        council_keys = {r["_meta"]["state_sha256"] for r in council_records}
        profile_records, profile_stats = convert_paths(
            profile_paths,
            specs=specs,
            bank=bank,
            mode="profile",
            min_agree=args.min_agree,
            max_state_chars=args.max_state_chars,
            limit=args.profile_limit,
            seed=args.seed,
        )
        profile_records = [r for r in profile_records if r["_meta"]["state_sha256"] not in council_keys]
        profile_stats["sampled"] = len(profile_records)
        _print_stats("style_profile (sampled, council-deduped)", profile_stats, specs)

    if args.report_only:
        return 0

    write_workload(out_dir / "workload.json", bank)
    write_jsonl(out_dir / "council.jsonl", council_records)
    (out_dir / "council.report.json").write_text(
        json.dumps(council_stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    preview = council_records[:2]
    (out_dir / "council.preview.json").write_text(
        json.dumps(preview, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {len(council_records)} council records to {out_dir / 'council.jsonl'}")
    print(f"wrote question spec to {out_dir / 'workload.json'}")
    print(f"wrote preview to {out_dir / 'council.preview.json'}")

    if profile_records:
        write_jsonl(out_dir / "profile.jsonl", profile_records)
        assert profile_stats is not None
        (out_dir / "profile.report.json").write_text(
            json.dumps(profile_stats, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {len(profile_records)} profile records to {out_dir / 'profile.jsonl'}")

    print(
        f"next: python /home/okita/git_repos/kev/skills/kev-finetune/scripts/split_data.py "
        f"{out_dir / 'council.jsonl'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
