"""Tests for non-prose chunk detection."""

from __future__ import annotations

from tools.data_preparation.prose_filter import classify_chunk_prose, is_narrative_prose

CATALOG_CHUNK = """\
MACKARNESS. Crown 8vo, cloth extra, 6s. *       *       *       *       *

=Plutarch's Lives of Illustrious Men.= Translated from the Greek, with
Notes Critical and Historical, and a Life of Plutarch, by JOHN and
WILLIAM LANGHORNE. Two Vols., 8vo, cloth extra, with Portraits, 10s. 6d.
"""

NARRATIVE_CHUNK = """\
The morning sun shone brightly over Raglan Castle. Dorothy walked slowly
through the courtyard, her thoughts far from the feast preparations.
Richard called her name from the gatehouse, but she did not turn at first.
When she did, his face was pale with worry. "They have sent word from London,"
he said quietly. She felt the old fear rise again, cold and familiar.
"""

MIXED_FRONT_MATTER = """\
Produced by Suzanne Shell and the Online Distributed Proofreading Team
at http://www.pgdp.net

The Sixteenth Century drew to a close. It was the last day of the last
year, and two hours only were wanting to the birth of another year.
The night was solemn and beautiful. Myriads of stars paved the deep
vault of heaven; the crescent moon hung like a silver lamp in the midst
of them. Crowds were collected in the open places, watching the wonders
in the heavens, and drawing auguries from them, chiefly sinister.
"""

TOC_CHUNK = """\
CONTENTS

CHAPTER I. THE RUINED HOUSE IN THE VAUXHALL ROAD                    1
CHAPTER II. THE DOG-FANCIER                                           12
CHAPTER III. THE HAND AND THE CLOAK                                   28
CHAPTER IV. THE IRON-MERCHANT'S DAUGHTER                              45
CHAPTER V. THE MEETING NEAR THE STATUE                                61
CHAPTER VI. THE CHARLES THE SECOND SPANIEL                            78
CHAPTER VII. THE HAND AGAIN!                                          94
CHAPTER VIII. THE BARBER OF LONDON                                   110
"""


def test_publisher_catalog_rejected() -> None:
    q = classify_chunk_prose(CATALOG_CHUNK)
    assert q.verdict == "non_prose"
    assert q.reason == "publisher_catalog"


def test_narrative_prose_kept() -> None:
    assert is_narrative_prose(NARRATIVE_CHUNK)
    q = classify_chunk_prose(NARRATIVE_CHUNK)
    assert q.verdict == "prose"
    assert q.reason is None


def test_mixed_front_matter_with_story_kept() -> None:
    q = classify_chunk_prose(MIXED_FRONT_MATTER)
    assert q.verdict == "prose"


def test_table_of_contents_rejected() -> None:
    q = classify_chunk_prose(TOC_CHUNK)
    assert q.verdict == "non_prose"
    assert q.reason == "table_of_contents"


def test_too_short_rejected() -> None:
    q = classify_chunk_prose("Crown 8vo, cloth extra, 6s.")
    assert q.verdict == "non_prose"
    assert q.reason == "too_short"


ERRATA_CHUNK = """\
THE END. CORRECTIONS

  page     original text                    correction
  ix       [missing from contents]          THE KEY TO GRIEF      185
  13       Yvette has gone to Bannelec.     Yvette has gone to Bannalec.
  23       It was crowded with Britons,     It was crowded with Bretons,
  29       doxens of similar red            dozens of similar red
  93       the great moth dated             the great moth darted
"""


def test_errata_table_rejected() -> None:
    q = classify_chunk_prose(ERRATA_CHUNK)
    assert q.verdict == "non_prose"
    assert q.reason == "errata_corrections"


VERSE_APPENDIX = """\
 _VI._

 _For at that word, the Sorcery
  Of Love shall change the earth and sky
  To Paradise, with cherubim
  Instead of birds on every limb._

 _VII._

 _Rivers shall sing our rhapsody;
  The vaulted forest, tree by tree,
  High hung with tapestry, shall glow
  With golden pillars all a-row._

 _VIII._

 _And down the gilded forest aisle
  Shy throngs of violets shall smile
  And kiss your feet from tree to tree
  While blue-bells droop in courtesy._
"""


def test_verse_stanza_appendix_rejected() -> None:
    q = classify_chunk_prose(VERSE_APPENDIX)
    assert q.verdict == "non_prose"
    assert q.reason == "verse_appendix"
