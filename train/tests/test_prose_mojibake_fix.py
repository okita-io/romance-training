"""Tests for deterministic prose mojibake repair."""

import romance_factory.story_core.prose_mojibake_fix as pmf


def test_lone_acirc_contraction():
    s, n_subs, n_chars = pmf.fix_prose_mojibake("Kaelen didnât answer.")
    assert n_subs >= 1
    assert n_chars >= 1
    assert "didn't" in s
    assert "â" not in s


def test_utf8_right_quote_as_latin1_triplet():
    triplet = "\u2019".encode("utf-8").decode("latin-1")
    s, n_subs, n_chars = pmf.fix_prose_mojibake(f"She said {triplet}no.{triplet}")
    assert n_subs >= 1
    assert n_chars >= 2 * len(triplet)
    assert "'" in s
    assert triplet not in s


def test_unicode_smart_quotes_normalized():
    s, n_subs, n_chars = pmf.fix_prose_mojibake("She didn\u2019t blink.")
    assert n_subs >= 1
    assert n_chars >= 1
    assert "didn't" in s


def test_idempotent_ascii():
    plain = "She didn't answer."
    s, n_subs, n_chars = pmf.fix_prose_mojibake(plain)
    assert s == plain
    assert n_subs == 0
    assert n_chars == 0


def test_compact_json_blob_gets_indented():
    raw = '{"scores":[{"rule_id":"CRAFT-01","score":8,"notes":"ok"}]}'
    s, n_subs, n_chars = pmf.fix_prose_mojibake(raw)
    assert "\n" in s
    assert '  "scores"' in s or '  "rule_id"' in s
    import json as _json

    assert _json.loads(s) == _json.loads(raw)


def test_prose_that_is_not_json_unchanged():
    plain = "She said {hello without closing the brace."
    s, _, _ = pmf.fix_prose_mojibake(plain)
    assert s == plain


def test_json_pretty_idempotent():
    import json as _json

    obj = {"a": 1, "b": [2, 3]}
    pretty = _json.dumps(obj, indent=2, ensure_ascii=False)
    s, _, _ = pmf.fix_prose_mojibake(pretty)
    assert s == pretty
