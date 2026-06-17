"""Deterministic style metrics computed from text statistics.
No LLM required -- fast enough to run on multi-million chunk corpora.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

try:
    import spacy as _spacy_mod

    _nlp = None

    def _nlp_instance():
        global _nlp
        if _nlp is None:
            try:
                _nlp = _spacy_mod.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            except OSError:
                raise RuntimeError(
                    "spaCy model not found.\n"
                    "Fix: python -m spacy download en_core_web_sm"
                )
        return _nlp

    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

try:
    import textstat as _textstat
    _TEXTSTAT_AVAILABLE = True
except ImportError:
    _TEXTSTAT_AVAILABLE = False

_CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
_NOMINAL_SUFFIXES = ("tion", "sion", "ness", "ment", "ity", "ism", "ance", "ence", "hood", "ship")

# Build using chr() so no curly quote characters appear in the source file.
_LDQ = chr(0x201C)  # left double quotation mark
_RDQ = chr(0x201D)  # right double quotation mark
_LSQ = chr(0x2018)  # left single quotation mark
_RSQ = chr(0x2019)  # right single quotation mark

_QUOTED_RE = re.compile(
    '"[^"]{5,500}"'
    + "|" + _LDQ + "[^" + _RDQ + "]{5,500}" + _RDQ
    + "|" + _LSQ + "[^" + _RSQ + "]{5,500}" + _RSQ
)
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def compute(text: str) -> dict[str, Any]:
    """Compute all deterministic style metrics for a passage.
    Returns an empty dict for very short texts (< 20 words).
    """
    words = text.split()
    word_count = len(words)
    if word_count < 20:
        return {}

    metrics: dict[str, Any] = {"word_count": word_count}

    # Sentence-level
    sents = _split_sentences(text)
    n_sents = max(1, len(sents))
    sent_lengths = [len(s.split()) for s in sents if s.strip()]

    metrics["sentence_count"] = n_sents
    metrics["sentence_length_mean"] = round(word_count / n_sents, 2)
    if len(sent_lengths) > 1:
        metrics["sentence_length_std"] = round(statistics.stdev(sent_lengths), 2)
    else:
        metrics["sentence_length_std"] = 0.0
    metrics["sentence_length_min"] = min(sent_lengths) if sent_lengths else 0
    metrics["sentence_length_max"] = max(sent_lengths) if sent_lengths else 0

    # Vocabulary
    tokens = [w.lower().strip(".,!?;:\"'()[]{}--...") for w in words]
    tokens = [t for t in tokens if t]
    metrics["type_token_ratio"] = round(len(set(tokens)) / max(1, len(tokens)), 4)
    metrics["avg_word_length"] = round(
        sum(len(t) for t in tokens) / max(1, len(tokens)), 2
    )

    # Punctuation and dialogue
    punct_chars = sum(1 for c in text if c in ".,!?;:--...")
    metrics["punctuation_density"] = round(punct_chars / max(1, word_count), 4)

    dialogue_chars = sum(len(m.group()) for m in _QUOTED_RE.finditer(text))
    metrics["dialogue_ratio"] = round(dialogue_chars / max(1, len(text)), 4)

    # Paragraph structure
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    metrics["paragraph_count"] = len(paragraphs)
    metrics["avg_paragraph_length"] = round(word_count / max(1, len(paragraphs)), 2)

    # Readability
    if _TEXTSTAT_AVAILABLE:
        metrics["flesch_reading_ease"] = round(_textstat.flesch_reading_ease(text), 2)
        metrics["flesch_kincaid_grade"] = round(_textstat.flesch_kincaid_grade(text), 2)
        metrics["gunning_fog"] = round(_textstat.gunning_fog(text), 2)

    # spaCy POS / dependency metrics
    if _SPACY_AVAILABLE:
        nlp = _nlp_instance()
        doc = nlp(text[:12000])
        total = max(1, len(doc))

        pos_counts: dict[str, int] = {}
        for token in doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

        for pos in ("NOUN", "VERB", "ADJ", "ADV", "PROPN"):
            metrics["pos_ratio_" + pos.lower()] = round(pos_counts.get(pos, 0) / total, 4)

        content_n = sum(pos_counts.get(p, 0) for p in _CONTENT_POS)
        metrics["lexical_density"] = round(content_n / total, 4)

        passive_n = sum(1 for t in doc if t.dep_ in ("nsubjpass", "auxpass"))
        verb_n = max(1, pos_counts.get("VERB", 1))
        metrics["passive_rate"] = round(passive_n / verb_n, 4)

        metrics["coordination_ratio"] = round(pos_counts.get("CCONJ", 0) / total, 4)
        metrics["subordination_ratio"] = round(pos_counts.get("SCONJ", 0) / total, 4)

        nominal_n = sum(
            1 for t in doc
            if t.pos_ == "NOUN" and t.text.lower().endswith(_NOMINAL_SUFFIXES)
        )
        metrics["nominalization_ratio"] = round(nominal_n / total, 4)

        depths = []
        for token in doc:
            depth, t = 0, token
            while t.head != t and depth < 20:
                depth += 1
                t = t.head
            depths.append(depth)
        metrics["avg_dependency_depth"] = round(
            sum(depths) / max(1, len(depths)), 2
        ) if depths else 0.0

    return metrics


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]
