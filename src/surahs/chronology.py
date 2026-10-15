"""Beyond Mecca vs Medina: relative chronology.

Binary classification inherits whatever its labels assume. The functions
here ask a different question: if we *ignore* the labels, does the text
itself arrange the surahs in an order, and how well does that order agree
with the chronologies scholars have proposed?

This is the logic of Behnam Sadeghi's stylometric work: if several
independent, low-level stylistic measures (verse length, the frequencies of
common morphemes) all sort the surahs the same way, and that sort agrees with
Nöldeke's, the agreement is evidence for a real chronological signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from surahs.data import load_morphology, load_surah_texts


def rank_correlations(surahs: pd.DataFrame | None = None, columns=("mvl",)) -> pd.DataFrame:
    """Spearman correlation of each column with the Nöldeke and Cairo orders.

    Spearman's rho compares *rankings*: +1 means the two orders agree
    perfectly, 0 means no relation, -1 means one is the reverse of the other.
    """
    s = load_surah_texts() if surahs is None else surahs
    rows = []
    for col in columns:
        for ref in ("noldeke_order", "cairo_order"):
            rho, p = spearmanr(s[col], s[ref])
            rows.append({"measure": col, "against": ref, "spearman_rho": rho, "p_value": p, "n_surahs": len(s)})
    return pd.DataFrame(rows)


def morpheme_profile(n_features: int = 100, min_words: int = 0) -> pd.DataFrame:
    """Relative frequency of the ``n_features`` commonest lemmas in each surah.

    One row per surah, one column per lemma; values are "occurrences per 1000
    segments". The commonest lemmas are mostly grammatical (and, the, in, that,
    he, not...), which is the point: authors and periods differ in how they use
    these without anyone deciding to.
    """
    m = load_morphology().dropna(subset=["lemma"])
    top = m["lemma"].value_counts().index[:n_features]
    counts = pd.crosstab(m["surah"], m["lemma"])[top]
    totals = m.groupby("surah").size()
    prof = counts.div(totals, axis=0) * 1000
    if min_words:
        keep = load_surah_texts()["n_words"] >= min_words
        prof = prof.loc[keep[keep].index]
    return prof


def pca_chronology(n_features: int = 100, min_words: int = 150, n_components: int = 3):
    """Principal component analysis of the morpheme profile.

    PCA finds the direction along which the surahs differ most, using no
    labels at all. Returns (scores DataFrame joined to the surah table,
    the fitted PCA, the loadings DataFrame).

    Short surahs are excluded by default (``min_words``): a surah of 20 words
    gives a very noisy estimate of how often it uses, say, "that".
    """
    prof = morpheme_profile(n_features, min_words)
    Z = StandardScaler().fit_transform(prof)
    pca = PCA(n_components=n_components, random_state=0).fit(Z)
    scores = pd.DataFrame(
        pca.transform(Z), index=prof.index, columns=[f"PC{i + 1}" for i in range(n_components)]
    )
    s = load_surah_texts()
    # PCA's sign is arbitrary; orient PC1 so that higher = later in Nöldeke.
    if spearmanr(scores["PC1"], s.loc[scores.index, "noldeke_order"])[0] < 0:
        scores["PC1"] *= -1
        pca.components_[0] *= -1
    loadings = pd.DataFrame(pca.components_.T, index=prof.columns, columns=scores.columns)
    return scores.join(s), pca, loadings


def within_period_correlations(scores: pd.DataFrame, column: str = "PC1") -> pd.DataFrame:
    """Does the measure still track Nöldeke's order *inside* each period?

    A measure could separate Meccan from Medinan perfectly and still say
    nothing about order within Mecca. This checks each period separately.
    """
    rows = []
    for period, g in scores.groupby("noldeke_period", observed=True):
        if len(g) >= 5:
            rho, p = spearmanr(g[column], g["noldeke_order"])
            rows.append({"period": period, "n_surahs": len(g), "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)


def chunk_verses(verses: pd.DataFrame, words_per_chunk: int = 300) -> pd.DataFrame:
    """Cut each surah into consecutive blocks of about ``words_per_chunk`` words.

    Long surahs like al-Baqara (2) may contain material from different
    moments. Splitting them into blocks, as Sadeghi does, lets a long surah
    show internal variation instead of being forced into one average.
    """
    rows = []
    for surah, g in verses.groupby("surah", sort=True):
        buf, start, n = [], None, 0
        for _, r in g.iterrows():
            if start is None:
                start = r["ayah"]
            buf.append(r)
            n += r["n_words"]
            if n >= words_per_chunk:
                rows.append(_block(surah, start, buf))
                buf, start, n = [], None, 0
        if buf:
            if rows and rows[-1]["surah"] == surah and n < words_per_chunk / 2:
                # fold a short tail into the previous block
                prev = rows.pop()
                buf = prev["_rows"] + buf
                start = prev["ayah_start"]
            rows.append(_block(surah, start, buf))
    out = pd.DataFrame(rows).drop(columns="_rows")
    return out


def _block(surah, start, buf):
    return {
        "surah": surah,
        "ayah_start": start,
        "ayah_end": buf[-1]["ayah"],
        "text": " ".join(r["text"] for r in buf),
        "lemmas": " ".join(r["lemmas"] for r in buf),
        "n_words": sum(r["n_words"] for r in buf),
        "n_letters": sum(r["n_letters"] for r in buf),
        "mvl": np.mean([r["n_letters"] for r in buf]),
        "_rows": buf,
    }
