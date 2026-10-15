"""Load the Qur'an text, its morphology, and the competing chronologies.

Everything here reads from ``data/raw/`` (the verbatim downloads, see
``data/SOURCES.md``) and ``data/surahs.csv`` (the surah table built by
``scripts/build_surah_table.py``). The one exception is the Quranic Arabic
Corpus morphology, downloaded once to ``data/cache/`` (see
:func:`ensure_morphology`).

The two main entry points are:

- :func:`load_surahs` -- one row per surah, with the Cairo (1924) and
  Nöldeke-Schwally chronologies side by side.
- :func:`load_verses` -- one row per verse (ayah), with Arabic text,
  an English translation, lemmas, roots and both sets of labels.
"""

from __future__ import annotations

import gzip
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"

BASMALA = "بسم الله الرحمن الرحيم"

# Nöldeke-Schwally divide their chronological sequence (1-114) into four
# periods. The boundaries are fixed by the size of each period in
# Geschichte des Qorāns: 48 + 21 + 21 + 24 = 114.
NOLDEKE_PERIODS = [
    ("Meccan I", 1, 48),
    ("Meccan II", 49, 69),
    ("Meccan III", 70, 90),
    ("Medinan", 91, 114),
]
PERIOD_ORDER = [p for p, _, _ in NOLDEKE_PERIODS]

# Arabic letters only (no diacritics, no spaces); used to count verse length.
_ARABIC_LETTER = re.compile(r"[ء-غف-يٱ]")


def noldeke_period(order: int) -> str:
    for name, lo, hi in NOLDEKE_PERIODS:
        if lo <= order <= hi:
            return name
    raise ValueError(f"Nöldeke order out of range: {order}")


@lru_cache(maxsize=1)
def load_surahs() -> pd.DataFrame:
    """One row per surah (114 rows), indexed by surah number.

    Columns
    -------
    name_ar, name_translit, name_en : surah names (from Tanzil metadata)
    n_verses : number of verses in the Kufan count used by the Cairo edition
    cairo_order : chronological rank (1-114) in the 1924 Cairo edition
    cairo_place : "Meccan" or "Medinan" according to the Cairo edition
    noldeke_order : chronological rank (1-114) according to Nöldeke-Schwally
    noldeke_period : "Meccan I", "Meccan II", "Meccan III" or "Medinan"
    noldeke_place : "Meccan" or "Medinan", collapsing Nöldeke's three Meccan periods
    labels_disagree : True where Cairo and Nöldeke put the surah in different cities
    """
    df = pd.read_csv(DATA / "surahs.csv", index_col="surah")
    df["noldeke_period"] = pd.Categorical(df["noldeke_period"], categories=PERIOD_ORDER, ordered=True)
    return df


def _read_tanzil(path: Path) -> pd.DataFrame:
    """Read a Tanzil ``surah|ayah|text`` file, skipping the copyright footer."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            s, a, text = line.split("|", 2)
            rows.append((int(s), int(a), text.strip()))
    return pd.DataFrame(rows, columns=["surah", "ayah", "text"])


def strip_basmala(surah: int, ayah: int, text: str) -> str:
    """Remove the basmala that Tanzil prepends to verse 1 of surahs 2-114.

    In the standard Kufan count the basmala is a numbered verse only in
    surah 1. Everywhere else it is a heading, but Tanzil's plain-text export
    glues it onto the first verse. Left in place, it adds the same four words
    to 112 verses and distorts both word counts and word frequencies (the 2017
    version of this project had this problem).
    """
    if surah != 1 and ayah == 1 and text.startswith(BASMALA):
        return text[len(BASMALA):].strip()
    return text


MORPHOLOGY_URL = (
    "https://raw.githubusercontent.com/mustafa0x/quran-morphology/"
    "8f38b39016824284f9ed16ae15069ff9102c4acf/quran-morphology.txt"
)


def ensure_morphology() -> Path:
    """Return the path to the Quranic Arabic Corpus file, downloading it if needed.

    The corpus's terms allow only verbatim copies, so this repository does
    not ship it. The first call fetches the corrected Arabic-script version
    (about 6 MB, pinned to one commit so everyone gets identical data) and
    stores it compressed in ``data/cache/``.
    """
    path = DATA / "cache" / "qac-morphology.txt.gz"
    if not path.exists():
        import urllib.request

        print(f"Downloading the Quranic Arabic Corpus morphology (one time only) from\n  {MORPHOLOGY_URL}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MORPHOLOGY_URL) as r:
            raw = r.read()
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wb") as f:
            f.write(raw)
        tmp.rename(path)
    return path


@lru_cache(maxsize=1)
def load_verse_exceptions() -> pd.DataFrame:
    """Verses whose place of revelation differs from their surah's, per the Cairo tradition.

    Hand-coded from the "Note" column of Tanzil's revelation-order table
    (``raw/tanzil-revelation-order.csv``), which reproduces the exception
    lists in the 1924 Cairo edition's surah headings. One row per verse.
    Notes that name a place *between* Mecca and Medina (Juhfa, Mina, Arafat,
    the Hijra journey) are not coded as exceptions; see data/SOURCES.md.
    """
    ex = pd.read_csv(DATA / "verse_exceptions.csv")
    rows = []
    for r in ex.itertuples():
        for part in r.ayahs.split(";"):
            lo, _, hi = part.partition("-")
            for a in range(int(lo), int(hi or lo) + 1):
                rows.append((r.surah, a, r.place))
    return pd.DataFrame(rows, columns=["surah", "ayah", "exception_place"])


@lru_cache(maxsize=1)
def load_morphology() -> pd.DataFrame:
    """The Quranic Arabic Corpus morphology, one row per *segment*.

    A segment is a stem, prefix or suffix: the word وَقَالُوا ("and they
    said") is three segments: وَ (and) + قَالُ (said) + وا (they). Columns:
    surah, ayah, word, segment, form, tag, lemma, root, is_affix.
    """
    path = ensure_morphology()
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            loc, form, tag, feats = line.rstrip("\n").split("\t")
            s, a, w, g = (int(x) for x in loc.split(":"))
            lemma = root = None
            parts = feats.split("|")
            for p in parts:
                if p.startswith("LEM:"):
                    lemma = p[4:]
                elif p.startswith("ROOT:"):
                    root = p[5:]
            is_affix = "PREF" in parts or "SUFF" in parts
            rows.append((s, a, w, g, form, tag, lemma, root, is_affix))
    return pd.DataFrame(
        rows, columns=["surah", "ayah", "word", "segment", "form", "tag", "lemma", "root", "is_affix"]
    )


@lru_cache(maxsize=1)
def load_verses() -> pd.DataFrame:
    """One row per verse (6,236 rows, Kufan numbering).

    Columns
    -------
    surah, ayah : verse address, e.g. 2, 255 is "Q 2:255"
    text : Arabic, Tanzil "simple clean" (no vowel marks), basmala removed
    english : Pickthall's 1930 translation
    lemmas : space-separated dictionary forms of every segment, including
        prefixes such as "and" and "the" (useful for style)
    roots : space-separated three- or four-letter roots of content words
    n_words, n_letters : verse length
    plus every column from :func:`load_surahs`, copied down to each verse, and
    is_exception : True if the Cairo tradition assigns this verse to the other city
    cairo_place_verse : the Cairo label with those verse-level exceptions applied
    """
    ar = _read_tanzil(RAW / "tanzil-quran-simple-clean.txt")
    ar["text"] = [strip_basmala(s, a, t) for s, a, t in zip(ar.surah, ar.ayah, ar.text)]
    en = _read_tanzil(RAW / "tanzil-en-pickthall.txt").rename(columns={"text": "english"})

    morph = load_morphology()
    lem = morph.dropna(subset=["lemma"]).groupby(["surah", "ayah"])["lemma"].agg(" ".join)
    root = morph.dropna(subset=["root"]).groupby(["surah", "ayah"])["root"].agg(" ".join)

    v = ar.merge(en, on=["surah", "ayah"], how="left", validate="one_to_one")
    v = v.join(lem.rename("lemmas"), on=["surah", "ayah"]).join(root.rename("roots"), on=["surah", "ayah"])
    v[["lemmas", "roots"]] = v[["lemmas", "roots"]].fillna("")
    v["n_words"] = v["text"].str.split().str.len()
    v["n_letters"] = v["text"].map(lambda t: len(_ARABIC_LETTER.findall(t)))
    v = v.join(load_surahs(), on="surah")
    v = v.merge(load_verse_exceptions(), on=["surah", "ayah"], how="left", validate="one_to_one")
    v["is_exception"] = v["exception_place"].notna()
    v["cairo_place_verse"] = v["exception_place"].fillna(v["cairo_place"])
    v = v.drop(columns="exception_place")
    v.insert(0, "ref", v["surah"].astype(str) + ":" + v["ayah"].astype(str))
    return v


@lru_cache(maxsize=1)
def load_surah_texts() -> pd.DataFrame:
    """One row per surah with all its verses concatenated, plus length statistics.

    ``mvl`` is the *mean verse length* in letters, the variable that Bazargan,
    Sadeghi and Sinai all use as a rough chronological gauge.
    """
    v = load_verses()
    g = v.groupby("surah")
    out = pd.DataFrame(
        {
            "text": g["text"].agg(" ".join),
            "english": g["english"].agg(" ".join),
            "lemmas": g["lemmas"].agg(" ".join),
            "roots": g["roots"].agg(" ".join),
            "n_words": g["n_words"].sum(),
            "n_letters": g["n_letters"].sum(),
            "mvl": g["n_letters"].mean(),
            "mvl_words": g["n_words"].mean(),
        }
    )
    return out.join(load_surahs())
