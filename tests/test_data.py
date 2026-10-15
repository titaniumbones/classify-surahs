"""Sanity checks on the data. If one of these fails, every result is suspect."""

from surahs import load_surahs, load_verses
from surahs.data import BASMALA
from surahs.translit import translit


def test_counts():
    v = load_verses()
    s = load_surahs()
    assert len(s) == 114
    assert len(v) == 6236  # Kufan verse count
    assert (v.groupby("surah").size() == s.n_verses).all()


def test_noldeke_periods():
    counts = load_surahs().noldeke_period.value_counts()
    assert counts.to_dict() == {"Meccan I": 48, "Meccan II": 21, "Meccan III": 21, "Medinan": 24}


def test_label_disagreements():
    s = load_surahs()
    assert s.index[s.labels_disagree].tolist() == [13, 55, 76, 99]
    assert (s.cairo_place == "Medinan").sum() == 28


def test_basmala_removed_except_fatiha():
    v = load_verses().set_index("ref")
    assert v.loc["1:1", "text"] == BASMALA  # a numbered verse only in surah 1
    assert not v.loc["2:1", "text"].startswith("بسم")
    assert v.loc["2:1", "text"] == "الم"
    assert "بسم الله" in v.loc["27:30", "text"]  # the basmala inside Sulayman's letter stays


def test_every_verse_has_english():
    assert load_verses().english.notna().all()


def test_translit():
    assert translit("رَبّ") == "rabb"
    assert translit("اللَّه") == "allāh"


def test_verse_exceptions():
    v = load_verses().set_index("ref")
    assert v.is_exception.sum() == 157
    assert v.loc["73:20", "cairo_place_verse"] == "Medinan"
    assert v.loc["73:19", "cairo_place_verse"] == "Meccan"
    assert v.loc["8:30", "cairo_place_verse"] == "Meccan"
    assert v.loc["107:4", "cairo_place_verse"] == "Medinan"
