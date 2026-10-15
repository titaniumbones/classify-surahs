"""Ways of turning a verse into numbers a model can use.

Each function returns a scikit-learn transformer that takes the verse table
from :func:`surahs.data.load_verses` (or the surah table from
:func:`surahs.data.load_surah_texts`) and produces a feature matrix. Keeping
them separate lets you ask *which kind of evidence* a classifier relies on:
length alone, a handful of hand-picked stylistic markers, vocabulary, or a
neural model's representation of the whole verse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from surahs.data import normalize

# --- hand-picked markers -----------------------------------------------------
# Each entry is (name, lemma). Lemmas are from the Quranic Arabic Corpus.
# The three divine names are among the variables in Dost's project and have a
# long history as chronological markers: rabb ("Lord", often "your Lord")
# dominates the earliest surahs, al-Rahman ("the Merciful") clusters in what
# Nöldeke called the second Meccan period, and Allah becomes overwhelming later.
DIVINE_NAMES = {
    "allah": "اللَّه",
    "rabb": "رَبّ",
    "rahman": "رَحْمٰن",
}

# Forms of address that Islamic tradition itself used as a rule of thumb:
# "O you who believe" is said to mark Medinan passages, "O people" Meccan ones.
ADDRESS_FORMULAE = {
    "ya_ayyuha_alladhina_amanu": "يا أيها الذين آمنوا",
    "ya_ayyuha_al_nas": "يا أيها الناس",
}

DIVINE_NAMES = {k: normalize(v) for k, v in DIVINE_NAMES.items()}

STYLE_NUMERIC = (
    ["log_letters", "log_words", "letters_per_word"]
    + [f"name_{k}" for k in DIVINE_NAMES]
    + list(ADDRESS_FORMULAE)
)


def add_style_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the hand-picked stylistic columns added.

    Works on verses or whole surahs. Divine-name columns are rates (per
    word), so that long verses and long surahs are comparable.
    """
    out = df.copy()
    words = out["n_words"].clip(lower=1)
    out["log_letters"] = np.log1p(out["n_letters"])
    out["log_words"] = np.log1p(out["n_words"])
    out["letters_per_word"] = out["n_letters"] / words
    lemma_lists = out["lemmas"].str.split()
    for key, lemma in DIVINE_NAMES.items():
        out[f"name_{key}"] = lemma_lists.map(lambda ls, l=lemma: ls.count(l)) / words
    for key, phrase in ADDRESS_FORMULAE.items():
        out[key] = out["text"].str.count(phrase)
    # Rhyme: the final letter of the verse. Early surahs often run on short
    # rhymes that change frequently; later ones settle into long runs of -ūn/-īn.
    out["rhyme"] = out["text"].str.strip().str[-1].fillna("")
    out["rhyme2"] = out["text"].str.strip().str[-2:].fillna("")
    return out


def _style_frame(df: pd.DataFrame) -> pd.DataFrame:
    return add_style_columns(df)


# --- feature sets ------------------------------------------------------------


def length_features():
    """Verse length only: letters, words and letters per word."""
    return make_pipeline(
        FunctionTransformer(_style_frame),
        ColumnTransformer([("num", StandardScaler(), ["log_letters", "log_words", "letters_per_word"])]),
    )


def style_features():
    """Length + rhyme + divine names + forms of address (about 60 columns)."""
    return make_pipeline(
        FunctionTransformer(_style_frame),
        ColumnTransformer(
            [
                ("num", StandardScaler(), STYLE_NUMERIC),
                ("rhyme", OneHotEncoder(handle_unknown="ignore", min_frequency=10), ["rhyme", "rhyme2"]),
            ]
        ),
    )


def _tfidf(column: str, **kw):
    return ColumnTransformer([(column, TfidfVectorizer(**kw), column)])


def word_features():
    """Bag of surface words, as in the 2017 version (but TF-IDF weighted).

    Arabic attaches "and", "the", "to", pronouns etc. to the word, so
    وقالوا ("and they said") and قالوا ("they said") count as different words.
    """
    return _tfidf("text", token_pattern=r"\S+", min_df=2, sublinear_tf=True)


def lemma_features():
    """Bag of lemmas (dictionary forms), including prefixes like "and"/"the".

    Function words are kept on purpose: in stylometry the frequencies of
    common little words are often more telling than the rare content words.
    """
    return _tfidf("lemmas", token_pattern=r"\S+", min_df=2, sublinear_tf=True)


def root_features():
    """Bag of consonantal roots, e.g. k-f-r covers kafara, kāfir, kufr..."""
    return _tfidf("roots", token_pattern=r"\S+", min_df=2, sublinear_tf=True)


def char_features():
    """Character n-grams (2-5 letters) of the unvowelled text.

    These catch prefixes, suffixes and rhyme endings without needing a
    morphological analysis.
    """
    return _tfidf("text", analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=60000)


def english_features():
    """Bag of words over Pickthall's English translation.

    Lets students who don't read Arabic run the same experiment, but note
    that this models Pickthall's English as much as the Qur'an's Arabic.
    """
    return _tfidf("english", lowercase=True, min_df=2, sublinear_tf=True, stop_words=None)


def combined_features():
    """Lemmas + character n-grams + stylistic markers, stacked side by side."""
    from sklearn.pipeline import FeatureUnion

    return FeatureUnion(
        [("lemmas", lemma_features()), ("chars", char_features()), ("style", style_features())]
    )


FEATURE_SETS = {
    "length": length_features,
    "style": style_features,
    "words": word_features,
    "lemmas": lemma_features,
    "roots": root_features,
    "chars": char_features,
    "english": english_features,
    "combined": combined_features,
}

FEATURE_DESCRIPTIONS = {
    "length": "Verse length only (letters, words)",
    "style": "Length + rhyme + divine names + forms of address",
    "words": "Bag of Arabic surface words (the 2017 approach)",
    "lemmas": "Bag of lemmas incl. function words",
    "roots": "Bag of consonantal roots",
    "chars": "Character 2-5-grams",
    "english": "Bag of English words (Pickthall translation)",
    "combined": "Lemmas + characters + style",
    "camelbert": "CAMeLBERT-CA sentence embeddings (Classical Arabic BERT)",
}
