#!/usr/bin/env python3
"""
Browse classified corpus JSONL files — text chunk plus metadata and style_profile.

Usage:
    python tools/corpus_preview/preview_corpus.py
    python tools/corpus_preview/preview_corpus.py --corpus train/romance_corpus/horror_styled.jsonl
    python tools/corpus_preview/preview_corpus.py --port 8765

Opens http://127.0.0.1:8765 in your browser (use --no-open to skip).
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data_preparation.prose_filter import classify_chunk_prose

DEFAULT_CORPUS_DIR = ROOT / "train" / "romance_corpus"

# style_profile keys that are computable (numeric / structural)
COMPUTABLE_KEYS = {
    "word_count",
    "sentence_count",
    "sentence_length_mean",
    "sentence_length_std",
    "sentence_length_min",
    "sentence_length_max",
    "type_token_ratio",
    "avg_word_length",
    "punctuation_density",
    "dialogue_ratio",
    "paragraph_count",
    "avg_paragraph_length",
}

# style_profile keys that are LLM-assigned labels
STYLE_LABEL_KEYS = {
    "register",
    "pov",
    "narrative_distance",
    "free_indirect_discourse",
    "figurative_density",
    "tone",
    "temporal_structure",
    "sentence_variety",
    "dialogue_style",
    "imagery_type",
    "lexical_complexity",
    "sentence_complexity",
    "cohesion",
    "mind_style",
    "end_focus",
    "segmentation",
    "prose_rhythm",
    "climax",
    "subordination_salience",
    "textual_relations",
}


class JsonlIndex:
    """Byte offsets for each line — random access without loading the whole file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offsets: list[int] = []
        self._build()

    def _build(self) -> None:
        offset = 0
        with self.path.open("rb") as fh:
            for line in fh:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line)

    def __len__(self) -> int:
        return len(self.offsets)

    def read(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.offsets):
            raise IndexError(index)
        with self.path.open("rb") as fh:
            fh.seek(self.offsets[index])
            return json.loads(fh.readline())


def discover_jsonl_files(corpus_dir: Path) -> list[Path]:
    if not corpus_dir.is_dir():
        return []
    return sorted(corpus_dir.glob("*.jsonl"), key=lambda p: p.name.lower())


def format_record(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    style = metadata.get("style_profile") or {}

    chunk_meta = {k: v for k, v in metadata.items() if k not in ("style_profile", "prose_quality")}
    prose_quality = metadata.get("prose_quality")
    if not isinstance(prose_quality, dict):
        quality = classify_chunk_prose(record.get("text") or "")
        prose_quality = {
            "keep": quality.verdict == "prose",
            "reason": quality.reason,
            "narrative_word_ratio": round(quality.narrative_word_ratio, 4),
        }
    computable = {k: style[k] for k in COMPUTABLE_KEYS if k in style}
    labels = {k: style[k] for k in STYLE_LABEL_KEYS if k in style}
    evidence = style.get("evidence") if isinstance(style.get("evidence"), dict) else {}

    # Any extra style_profile keys not in our known sets
    known = COMPUTABLE_KEYS | STYLE_LABEL_KEYS | {"evidence"}
    extra = {k: v for k, v in style.items() if k not in known}

    return {
        "text": record.get("text", ""),
        "chunk_metadata": chunk_meta,
        "prose_quality": prose_quality,
        "computable_metrics": computable,
        "style_labels": labels,
        "evidence": evidence,
        "extra_style_fields": extra,
    }


def render_fields_table(fields: dict[str, Any]) -> str:
    if not fields:
        return '<p class="empty">—</p>'
    rows = []
    for key in sorted(fields):
        value = fields[key]
        if isinstance(value, float):
            display = f"{value:.4g}" if abs(value) < 1000 else f"{value:.2f}"
        else:
            display = str(value)
        rows.append(
            f'<tr><th scope="row">{key}</th><td>{_escape(display)}</td></tr>'
        )
    return f"<table><tbody>{''.join(rows)}</tbody></table>"


def render_evidence(evidence: dict[str, Any]) -> str:
    if not evidence:
        return '<p class="empty">No LLM evidence quotes for this chunk.</p>'
    items = []
    for key in sorted(evidence):
        quote = evidence[key]
        items.append(
            f'<div class="evidence-item">'
            f'<div class="evidence-key">{_escape(key)}</div>'
            f'<blockquote>{_escape(str(quote))}</blockquote>'
            f"</div>"
        )
    return "".join(items)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Corpus preview</title>
  <style>
    :root {
      --bg: #0f1117;
      --surface: #1a1d27;
      --border: #2a2f3d;
      --text: #e8eaef;
      --muted: #9aa3b5;
      --accent: #7c9cff;
      --chunk-bg: #12151c;
      --quote-border: #4a5568;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0.75rem 1.25rem;
    }
    header h1 {
      margin: 0 0 0.5rem;
      font-size: 1.1rem;
      font-weight: 600;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem 1rem;
      align-items: center;
    }
    label { font-size: 0.85rem; color: var(--muted); }
    select, input[type="number"], button {
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.35rem 0.6rem;
      font: inherit;
    }
    button {
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); color: var(--accent); }
    .status { font-size: 0.85rem; color: var(--muted); margin-left: auto; }
    main { max-width: 960px; margin: 0 auto; padding: 1.25rem; }
    section {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      margin-bottom: 1rem;
      overflow: hidden;
    }
    section > h2 {
      margin: 0;
      padding: 0.65rem 1rem;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      background: rgba(255,255,255,0.03);
      border-bottom: 1px solid var(--border);
    }
    .chunk-text {
      margin: 0;
      padding: 1.25rem;
      background: var(--chunk-bg);
      white-space: pre-wrap;
      font-family: Georgia, "Times New Roman", serif;
      font-size: 1.05rem;
      line-height: 1.65;
      max-height: 420px;
      overflow-y: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }
    th, td {
      padding: 0.45rem 1rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }
    th {
      width: 38%;
      color: var(--muted);
      font-weight: 500;
    }
    tr:last-child th, tr:last-child td { border-bottom: none; }
    .empty { padding: 1rem; color: var(--muted); margin: 0; font-size: 0.9rem; }
    .evidence-item { padding: 0.75rem 1rem; border-bottom: 1px solid var(--border); }
    .evidence-item:last-child { border-bottom: none; }
    .evidence-key {
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--accent);
      margin-bottom: 0.35rem;
    }
    blockquote {
      margin: 0;
      padding-left: 0.75rem;
      border-left: 3px solid var(--quote-border);
      color: var(--text);
      font-style: italic;
      font-size: 0.95rem;
    }
    .error {
      background: #3d1f1f;
      border: 1px solid #7a3030;
      color: #ffb4b4;
      padding: 1rem;
      border-radius: 8px;
    }
    .prose-warn {
      background: #3d3218;
      border: 1px solid #8a7030;
      color: #ffe8a8;
      padding: 0.75rem 1rem;
      border-radius: 8px;
      margin-bottom: 1rem;
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <header>
    <h1>Corpus preview</h1>
    <div class="controls">
      <label>File
        <select id="file-select"></select>
      </label>
      <button type="button" id="prev-btn" title="Previous chunk">← Prev</button>
      <label>Chunk
        <input type="number" id="index-input" min="0" value="0" style="width:5rem">
      </label>
      <span id="total-label" style="color:var(--muted);font-size:0.85rem"></span>
      <button type="button" id="next-btn" title="Next chunk">Next →</button>
      <button type="button" id="random-btn">Random</button>
      <span class="status" id="status"></span>
    </div>
  </header>
  <main id="content">
    <p class="empty">Loading…</p>
  </main>
  <script>
    const fileSelect = document.getElementById("file-select");
    const indexInput = document.getElementById("index-input");
    const totalLabel = document.getElementById("total-label");
    const content = document.getElementById("content");
    const statusEl = document.getElementById("status");

    let totalRecords = 0;

    async function loadFiles() {
      const res = await fetch("/api/files");
      const data = await res.json();
      fileSelect.innerHTML = "";
      for (const f of data.files) {
        const opt = document.createElement("option");
        opt.value = f.name;
        opt.textContent = f.name + " (" + f.count.toLocaleString() + " chunks)";
        if (f.name === data.default) opt.selected = true;
        fileSelect.appendChild(opt);
      }
      await loadRecord(0);
    }

    async function loadRecord(index) {
      const name = fileSelect.value;
      statusEl.textContent = "Loading…";
      try {
        const res = await fetch("/api/record?" + new URLSearchParams({ file: name, index: String(index) }));
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.error || res.statusText);
        }
        const data = await res.json();
        totalRecords = data.total;
        indexInput.max = Math.max(0, totalRecords - 1);
        indexInput.value = data.index;
        totalLabel.textContent = "of " + totalRecords.toLocaleString();
        content.innerHTML = data.html;
        statusEl.textContent = data.index + 1 + " / " + totalRecords.toLocaleString();
      } catch (e) {
        content.innerHTML = '<div class="error">' + e.message + '</div>';
        statusEl.textContent = "";
      }
    }

    function currentIndex() {
      return parseInt(indexInput.value, 10) || 0;
    }

    fileSelect.addEventListener("change", () => loadRecord(0));
    document.getElementById("prev-btn").addEventListener("click", () => {
      loadRecord(Math.max(0, currentIndex() - 1));
    });
    document.getElementById("next-btn").addEventListener("click", () => {
      loadRecord(Math.min(totalRecords - 1, currentIndex() + 1));
    });
    document.getElementById("random-btn").addEventListener("click", () => {
      if (totalRecords > 0) loadRecord(Math.floor(Math.random() * totalRecords));
    });
    indexInput.addEventListener("change", () => {
      let idx = currentIndex();
      idx = Math.max(0, Math.min(totalRecords - 1, idx));
      loadRecord(idx);
    });
    indexInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") indexInput.dispatchEvent(new Event("change"));
    });

    document.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      if (e.key === "ArrowLeft") document.getElementById("prev-btn").click();
      if (e.key === "ArrowRight") document.getElementById("next-btn").click();
    });

    loadFiles();
  </script>
</body>
</html>
"""


class PreviewHandler(BaseHTTPRequestHandler):
    corpus_dir: Path
    default_file: str | None
    indices: dict[str, JsonlIndex] = {}

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _get_index(self, filename: str) -> JsonlIndex:
        path = self.corpus_dir / filename
        if not path.is_file() or path.suffix != ".jsonl":
            raise FileNotFoundError(filename)
        resolved = str(path.resolve())
        if resolved not in self.indices:
            self.indices[resolved] = JsonlIndex(path)
        return self.indices[resolved]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path

        if route in ("/", "/index.html"):
            self._send_html(INDEX_HTML)
            return

        if route == "/api/files":
            files = discover_jsonl_files(self.corpus_dir)
            default = self.default_file or (files[0].name if files else None)
            payload = {
                "default": default,
                "files": [
                    {"name": f.name, "count": len(self._get_index(f.name))}
                    for f in files
                ],
            }
            self._send_json(payload)
            return

        if route == "/api/record":
            params = parse_qs(parsed.query)
            filename = (params.get("file") or [""])[0]
            try:
                index_num = int((params.get("index") or ["0"])[0])
            except ValueError:
                self._send_json({"error": "index must be an integer"}, status=400)
                return

            try:
                idx = self._get_index(filename)
            except FileNotFoundError:
                self._send_json({"error": f"file not found: {filename}"}, status=404)
                return

            if len(idx) == 0:
                self._send_json({"error": "corpus file is empty"}, status=404)
                return

            index_num = max(0, min(len(idx) - 1, index_num))
            try:
                record = idx.read(index_num)
            except (IndexError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=500)
                return

            formatted = format_record(record)
            pq = formatted["prose_quality"]
            html = ""
            if not pq.get("keep", True):
                reason = pq.get("reason") or "non_prose"
                html += (
                    f'<div class="prose-warn">Non-prose chunk flagged: '
                    f'<strong>{_escape(str(reason))}</strong> '
                    f'(narrative word ratio: {pq.get("narrative_word_ratio", "?")})'
                    f"</div>"
                )
            html += (
                f'<section><h2>Text chunk</h2>'
                f'<pre class="chunk-text">{_escape(formatted["text"])}</pre></section>'
                f'<section><h2>Prose quality</h2>{render_fields_table(formatted["prose_quality"])}</section>'
                f'<section><h2>Chunk metadata</h2>{render_fields_table(formatted["chunk_metadata"])}</section>'
                f'<section><h2>Computable metrics</h2>{render_fields_table(formatted["computable_metrics"])}</section>'
                f'<section><h2>Style labels (LLM)</h2>{render_fields_table(formatted["style_labels"])}</section>'
                f'<section><h2>Evidence quotes</h2>{render_evidence(formatted["evidence"])}</section>'
            )
            if formatted["extra_style_fields"]:
                html += (
                    f'<section><h2>Other style fields</h2>'
                    f'{render_fields_table(formatted["extra_style_fields"])}</section>'
                )

            self._send_json(
                {
                    "index": index_num,
                    "total": len(idx),
                    "file": filename,
                    "html": html,
                }
            )
            return

        self._send_json({"error": "not found"}, status=404)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview classified corpus JSONL files.")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help=f"Directory containing *.jsonl files (default: {DEFAULT_CORPUS_DIR.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Default JSONL file to open (name or path under --corpus-dir)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser tab")
    args = parser.parse_args()

    corpus_dir = args.corpus_dir if args.corpus_dir.is_absolute() else ROOT / args.corpus_dir
    corpus_dir = corpus_dir.resolve()

    default_file: str | None = None
    if args.corpus:
        p = args.corpus if args.corpus.is_absolute() else ROOT / args.corpus
        if p.is_file():
            if p.parent.resolve() != corpus_dir:
                corpus_dir = p.parent.resolve()
            default_file = p.name
        else:
            sys.exit(f"Corpus file not found: {p}")

    files = discover_jsonl_files(corpus_dir)
    if not files:
        sys.exit(f"No *.jsonl files found in {corpus_dir}")

    PreviewHandler.corpus_dir = corpus_dir
    PreviewHandler.default_file = default_file or files[0].name
    PreviewHandler.indices = {}

    url = f"http://{args.host}:{args.port}/"
    server = ThreadingHTTPServer((args.host, args.port), PreviewHandler)
    print(f"Serving corpus preview at {url}")
    print(f"Corpus dir: {corpus_dir}")
    print(f"Files: {', '.join(f.name for f in files)}")
    print("Press Ctrl+C to stop.")

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
