"""Tests for inline Gutenberg chapter index stripping."""

from __future__ import annotations

from tools.data_preparation.strip_chapter_index import strip_inline_chapter_index

MOBY_TOC = (
    "CHAPTER I.—Loomings CHAPTER II.—The Carpet Bag CHAPTER III.—The Spouter-Inn "
    "CHAPTER IV.—The Counterpane CHAPTER V.—Breakfast CHAPTER VI.—The Street "
    "CHAPTER VII.—The Chapel CHAPTER VIII.—The Pulpit CHAPTER IX.—The Sermon "
    "CHAPTER X.—A Bosom Friend CHAPTER XI.—Nightgown CHAPTER XII.—Biographical "
    "CHAPTER XIII.—Wheelbarrow CHAPTER XIV.—Nantucket CHAPTER XV.—Chowder "
    "CHAPTER XVI.—The Ship CHAPTER XVII.—The Ramadan CHAPTER XVIII.—His Mark "
    "CHAPTER XIX.—The Prophet CHAPTER XX.—All Astir CHAPTER XXI.—Going Aboard "
    "CHAPTER XXII.—Merry Christmas CHAPTER XXIII.—The Lee Shore "
    "CHAPTER XXIV.—The Advocate CHAPTER XXV.—Postscript "
    "CHAPTER XXVI.—Knights and Squires CHAPTER XXVII.—Knights and Squires "
    "CHAPTER XXVIII.—Ahab CHAPTER XXIX.—Enter Ahab; to him, Stubb "
    "CHAPTER XXX.—The Pipe CHAPTER XXXI.—Queen Mab CHAPTER XXXII.—Cetology "
    "CHAPTER XXXIII.—The Specksnyder CHAPTER XXXIV.—The Cabin Table "
    "CHAPTER XXXV.—The Mast-Head CHAPTER XXXVI.—The Quarter-Deck. Ahab and all "
    "CHAPTER XXXVII.—Sunset CHAPTER XXXVIII.—Dusk CHAPTER XXXIX.—First Night-Watch "
    "CHAPTER XL.—Forecastle—Midnight CHAPTER XLI.—Moby Dick "
    "CHAPTER XLII.—The Whiteness of the Whale CHAPTER XLIII.—Hark! "
    "CHAPTER XLIV.—The Chart CHAPTER XLV.—The Affidavit CHAPTER XLVI.—Surmises "
    "CHAPTER XLVII.—The Mat-Maker CHAPTER XLVIII.—The First Lowering "
    "CHAPTER XLIX.—The Hyena CHAPTER L.—Ahab’s Boat and Crew—Fedallah "
    "CHAPTER LI.—The Spirit-Spout CHAPTER LII.—The Pequod meets the Albatross "
    "CHAPTER LIII.—The Gam CHAPTER LIV.—The Town-Ho’s Story "
    "CHAPTER LV.—Monstrous Pictures of Whales CHAPTER LVI.—Less Erroneous Pictures of Whales "
    "CHAPTER LVII.—Of Whales in Paint, in Teeth, &c. CHAPTER LVIII.—Brit "
    "CHAPTER LIX.—Squid CHAPTER LX.—The Line CHAPTER LXI.—Stubb Kills a Whale "
    "CHAPTER LXII.—The Dart CHAPTER LXIII.—The Crotch CHAPTER LXIV.—Stubb’s Supper "
    "CHAPTER LXV.—The Whale as a Dish CHAPTER LXVI.—The Shark Massacre "
    "CHAPTER LXVII.—Cutting In CHAPTER LXVIII.—The Blanket CHAPTER LXIX.—The Funeral "
    "CHAPTER LXX.—The Sphynx CHAPTER LXXI.—The Pequod meets the Jeroboam. Her Story "
    "CHAPTER LXXII.—The Monkey-rope CHAPTER LXXIII.—Stubb & Flask kill a Right Whale "
    "CHAPTER LXXIV.—The Sperm Whale’s Head CHAPTER LXXV.—The Right Whale’s Head "
    "CHAPTER LXXVI.—The Battering Ram CHAPTER LXXVII.—The Great Heidelburgh Tun "
    "CHAPTER LXXVIII.—Cistern and Buckets CHAPTER LXXIX.—The Praire "
    "CHAPTER LXXX.—The Nut CHAPTER LXXXI.—The Pequod meets the Virgin "
    "CHAPTER LXXXII.—The Honor and Glory of Whaling "
    "CHAPTER LXXXIII.—Jonah Historically Regarded CHAPTER LXXXIV.—Pitchpoling "
    "CHAPTER LXXXV.—The Fountain CHAPTER LXXXVI.—The Tail "
    "CHAPTER LXXXVII.—The Grand Armada CHAPTER LXXXVIII.—Schools & Schoolmasters "
    "CHAPTER LXXXIX.—Fast Fish and Loose Fish CHAPTER XC.—Heads or Tails "
    "CHAPTER XCI.—The Pequod meets the Rose-Bud CHAPTER XCII.—Ambergris "
    "CHAPTER XCIII.—The Castaway CHAPTER XCIV.—A Squeeze of the Hand "
    "CHAPTER XCV.—The Cassock CHAPTER XCVI.—The Try-Works CHAPTER XCVII.—The Lamp "
    "CHAPTER XCVIII.—Stowing Down and Clearing Up CHAPTER XCIX.—The Doubloon "
    "CHAPTER C.—The Pequod meets the Samuel Enderby of London CHAPTER CI.—The Decanter "
    "CHAPTER CII.—A Bower in the Arsacides "
    "CHAPTER CIII.—Measurement of the Whale’s Skeleton CHAPTER CIV.—The Fossil Whale "
    "CHAPTER CV.—Does the Whale Diminish? CHAPTER CVI.—Ahab’s Leg "
    "CHAPTER CVII.—The Carpenter CHAPTER CVIII.—The Deck. Ahab and the Carpenter "
    "CHAPTER CIX.—The Cabin. Ahab and Starbuck CHAPTER CX.—Queequeg in his Coffin "
    "CHAPTER CXI.—The Pacific CHAPTER CXII.—The Blacksmith CHAPTER CXIII.—The Forge "
    "CHAPTER CXIV.—The Gilder CHAPTER CXV.—The Pequod meets the Bachelor "
    "CHAPTER CXVI.—The Dying Whale CHAPTER CXVII.—The Whale-Watch "
    "CHAPTER CXVIII.—The Quadrant CHAPTER CXIX.—The Candles CHAPTER CXX.—The Deck "
    "CHAPTER CXXI.—Midnight, on the Forecastle CHAPTER CXXII.—Midnight, Aloft "
    "CHAPTER CXXIII.—The Musket CHAPTER CXXIV.—The Needle CHAPTER CXXV.—The Log and Line "
    "CHAPTER CXXVI.—The Life-Buoy CHAPTER CXXVII.—Ahab and the Carpenter "
    "CHAPTER CXXVIII.—The Pequod meets the Rachel "
    "CHAPTER CXXIX.—The Cabin. Ahab and Pip CHAPTER CXXXI.—The Hat "
    "CHAPTER CXXXII.—The Pequod meets the Delight CHAPTER CXXXIII.—The Symphony "
    "CHAPTER CXXXIV.—The Chase."
)

ALICE_TOC = (
    "CHAPTER I. Down the Rabbit-Hole CHAPTER II. The Pool of Tears "
    "CHAPTER III. A Caucus-Race and a Long Tale CHAPTER IV. The Rabbit Sends in a Little Bill "
    "CHAPTER V. Advice from a Caterpillar CHAPTER VI. Pig and Pepper "
    "CHAPTER VII. A Mad Tea-Party CHAPTER VIII. The Queen’s Croquet-Ground "
    "CHAPTER IX. The Mock Turtle’s Story CHAPTER X. The Lobster Quadrille "
    "CHAPTER XI. Who Stole the Tarts? CHAPTER XII. Alice’s Evidence"
)


def test_strip_moby_dick_inline_toc() -> None:
    result = strip_inline_chapter_index(MOBY_TOC)
    assert result.stripped
    assert result.text == ""


def test_strip_alice_inline_toc_before_story() -> None:
    raw = (
        f"{ALICE_TOC}\n\n"
        "CHAPTER I. Down the Rabbit-Hole\n\n"
        "Alice was beginning to get very tired of sitting by her sister on the bank."
    )
    result = strip_inline_chapter_index(raw)
    assert result.stripped
    assert "CHAPTER II." not in result.text
    assert "CHAPTER I. Down the Rabbit-Hole" in result.text
    assert result.text.endswith("sitting by her sister on the bank.")


def test_strip_moby_dick_chunk_with_orphan_prefix() -> None:
    raw = (
        "Ahab and Starbuck CHAPTER CX.—Queequeg in his Coffin CHAPTER CXI.—The Pacific "
        "CHAPTER CXII.—The Blacksmith CHAPTER CXIII.—The Forge "
        "CHAPTER CXXXIV.—The Chase. First Day CHAPTER CXXXV.—The Chase. Second Day "
        "CHAPTER CXXXVI.—The Chase. Third Day EPILOGUE. ETYMOLOGY.\n\n"
        "The pale Usher—threadbare in coat, heart, body, and brain; I see him now."
    )
    result = strip_inline_chapter_index(raw)
    assert result.stripped
    assert "CHAPTER" not in result.text
    assert "Ahab and Starbuck" not in result.text
    assert result.text.startswith("The pale Usher")


def test_keep_standalone_chapter_heading_before_narrative() -> None:
    raw = (
        "CHAPTER X. Shaking\n\n"
        "She took her off the table as she spoke, and shook her backwards and forwards "
        "with all her might."
    )
    result = strip_inline_chapter_index(raw)
    assert not result.stripped
    assert result.text == raw


def test_keep_single_inline_chapter_before_narrative_paragraph() -> None:
    raw = (
        "—and it really _was_ a kitten, after all. CHAPTER XII. Which Dreamed it?\n\n"
        '"Your majesty shouldn\'t purr so loud," Alice said, rubbing her eyes.'
    )
    result = strip_inline_chapter_index(raw)
    assert not result.stripped
    assert "CHAPTER XII. Which Dreamed it?" in result.text
