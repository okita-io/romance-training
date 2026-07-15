"""Resume identity must survive prose normalization applied at write time."""

from __future__ import annotations

import json
from unittest.mock import patch

from tools.data_preparation.unified_corpus import normalize_prose_text
from tools.style_classification.pass_config import ALL_LLM_FIELDS
from tools.style_classification.run_pipeline import _record_key, run


def test_record_key_stable_across_normalize() -> None:
    raw = {
        "text": "'i wish,' he said, after a little hesitation, 'if only for \"a\"",
        "metadata": {"source": "hf:fiction_books", "chunk_index": 3},
    }
    saved = {
        "text": normalize_prose_text(raw["text"]),
        "metadata": dict(raw["metadata"]),
    }
    assert raw["text"][:60] != saved["text"][:60]
    assert _record_key(raw) == _record_key(saved)
    assert _record_key(raw) == _record_key(
        {"text": normalize_prose_text(saved["text"]), "metadata": raw["metadata"]}
    )


@patch("tools.style_classification.run_pipeline._enrich")
def test_resume_skips_when_saved_text_was_normalized(mock_enrich, tmp_path) -> None:
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    raw_text = "'Wait,' she said,  after a pause,  'are you sure?' " * 3
    record = {
        "text": raw_text,
        "metadata": {"source": "hf:test", "chunk_index": 0},
    }
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    saved = {
        "text": normalize_prose_text(raw_text),
        "metadata": {
            "source": "hf:test",
            "chunk_index": 0,
            "style_profile": {field: "x" for field in ALL_LLM_FIELDS},
        },
    }
    assert raw_text[:60] != saved["text"][:60]
    output_path.write_text(json.dumps(saved) + "\n", encoding="utf-8")

    run(
        input_path=input_path,
        output_path=output_path,
        use_llm=True,
        workers=1,
        resume=True,
        pass_mode="both",
        quiet=True,
    )

    mock_enrich.assert_not_called()
