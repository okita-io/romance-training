"""Unit tests for TTS preprocessing and chapter/act segmentation (no Orpheus)."""

from __future__ import annotations

from romance_factory.text_to_speech.preprocess import (
    normalize_unicode,
    preprocess_act_tokens,
    split_paragraph_chunks,
    strip_markdown_for_tts,
)
from romance_factory.text_to_speech.manifest import resolve_voice_id
from romance_factory.text_to_speech.segments import (
    segments_for_chapter,
    iter_story_segments,
)


def test_strip_markdown_links_and_bold() -> None:
    raw = "## Chapter 1\n\n**Bold** and [link](http://x.com) here.\n\n---\n\nMore."
    out = strip_markdown_for_tts(raw, include_chapter_heading=False)
    assert "http" not in out
    assert "**" not in out
    assert "Bold" in out
    assert "link" in out


def test_normalize_unicode_quotes() -> None:
    t = normalize_unicode("\u201cHello\u201d")
    assert '"' in t


def test_split_paragraph_chunks_respects_max() -> None:
    paras = ["word " * 400, "end."]
    text = "\n\n".join(paras)
    chunks = split_paragraph_chunks(text, max_chars=500)
    assert len(chunks) >= 2


def test_preprocess_act_tokens_non_empty() -> None:
    chunks = preprocess_act_tokens("# Hi\n\nParagraph one.\n\nParagraph two.")
    assert len(chunks) >= 1
    assert "Paragraph" in chunks[0]


def test_segments_verified_acts(tmp_path) -> None:
    story = tmp_path / "story"
    (story / "chapters").mkdir(parents=True)
    acts = story / "chapters" / "chapter_01" / "acts"
    acts.mkdir(parents=True)
    a1 = "First act prose here.\n\nMore."
    a2 = "Second act."
    (acts / "Title_chapter1_act1.md").write_text(a1, encoding="utf-8")
    (acts / "Title_chapter1_act2.md").write_text(a2, encoding="utf-8")
    body = f"{a1}\n\n{a2}"
    (story / "chapters" / "chapter_01.md").write_text(
        f"# Chapter 1\n\n**Word Count:** 10\n\n---\n\n{body}",
        encoding="utf-8",
    )
    segs = segments_for_chapter(str(story), 1)
    assert len(segs) == 2
    assert segs[0].source == "acts_verified"
    assert segs[0].text.strip() == a1.strip()


def test_segments_proportional_when_acts_diverge(tmp_path) -> None:
    story = tmp_path / "story2"
    (story / "chapters").mkdir(parents=True)
    acts = story / "chapters" / "chapter_01" / "acts"
    acts.mkdir(parents=True)
    (acts / "Title_chapter1_act1.md").write_text("short", encoding="utf-8")
    (acts / "Title_chapter1_act2.md").write_text("also short", encoding="utf-8")
    final = "one two three four five six seven eight"
    (story / "chapters" / "chapter_01.md").write_text(
        f"# Chapter 1\n\n---\n\n{final}", encoding="utf-8"
    )
    segs = segments_for_chapter(str(story), 1)
    assert len(segs) == 2
    assert segs[0].source == "acts_proportional"
    joined = " ".join(s.text for s in segs)
    assert joined.split() == final.split()


def test_resolve_voice_override_and_profile(tmp_path) -> None:
    story = tmp_path / "st"
    story.mkdir()
    assert resolve_voice_id(str(story), override="  LEAH ") == "leah"
    (story / "author_profile.json").write_text('{"orpheus_voice": "zac"}', encoding="utf-8")
    assert resolve_voice_id(str(story)) == "zac"


def test_iter_story_segments_orders_chapters(tmp_path) -> None:
    story = tmp_path / "s"
    ch = story / "chapters"
    ch.mkdir(parents=True)
    for n, w in ((2, "aa bb"), (1, "cc dd")):
        (ch / f"chapter_{n:02d}.md").write_text(
            f"# Chapter {n}\n\n---\n\n{w}", encoding="utf-8"
        )
    segs = iter_story_segments(str(story))
    assert [s.chapter_num for s in segs] == [1, 2]
