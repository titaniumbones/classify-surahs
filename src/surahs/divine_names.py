"""Divine names as a chronological marker: testing Dost's three phases.

Dost (*Before the Qur'an*, 2026, ch. 3) takes Sinai's ordering of the surahs
by mean verse length and observes that the way the text names God changes
along it:

- **R-Qur'an** (about 40 early surahs): God is "Rabb-X", *your Lord*,
  *Lord of the worlds*. Where *Allāh* appears at all, it is mostly in
  verses that Sinai or Bell had already identified as later insertions.
- **AR-Qur'an**: *al-Raḥmān* is briefly used as God's proper name
  (the "Raḥmān surahs", e.g. 19, 20, 21, 25, 36, 43, 67), then *Allāh* and
  *Rabb-X* appear in roughly equal numbers.
- **A-Qur'an** (roughly the traditional Medinan surahs): *Allāh*
  dominates, with about one *Allāh* per verse or more.

The functions below let you check each of those claims against the counts,
and then ask the harder question: once you take the divine names *out*, do
the rest of the words still separate Dost's phases? If they do, the phases
are picking up something broader than naming habits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from surahs.data import load_morphology, load_surah_texts, load_verses
from surahs.features import DIVINE_NAMES


def divine_name_table(exclude_interpolations: bool = False) -> pd.DataFrame:
    """Per-surah counts of *Allāh*, *rabb* and *al-Raḥmān*, and rates per verse.

    *rabb* in the Qur'an almost always appears in a construct ("Lord of...",
    "your Lord", "our Lord"), so its count is a good stand-in for Dost's
    Rabb-X. Set ``exclude_interpolations`` to drop the verses Sinai treats
    as later insertions, as Dost does.

    The 1:1 basmala is counted as a verse of surah 1; the other surahs'
    basmalas are not in the text at all.
    """
    m = load_morphology()
    v = load_verses()
    if exclude_interpolations:
        keep = v.loc[~v.is_sinai_interpolation, ["surah", "ayah"]]
        m = m.merge(keep, on=["surah", "ayah"])
        v = v[~v.is_sinai_interpolation]
    names = {lemma: key for key, lemma in DIVINE_NAMES.items()}
    counts = (
        m[m.lemma.isin(names)]
        .groupby(["surah", "lemma"])
        .size()
        .unstack(fill_value=0)
        .rename(columns=names)
        .reindex(index=range(1, 115), columns=list(DIVINE_NAMES), fill_value=0)
    )
    counts["n_verses_used"] = v.groupby("surah").size()
    for k in DIVINE_NAMES:
        counts[f"{k}_per_verse"] = counts[k] / counts["n_verses_used"]
    return counts.join(load_surah_texts()[["name_translit", "mvl", "noldeke_period", "cairo_place",
                                           "sinai_mvl_rank", "dost_phase", "dost_note"]])


def check_r_qurans_lack_allah(table: pd.DataFrame) -> pd.DataFrame:
    """R-Qur'an surahs that still contain *Allāh* (after whatever exclusions ``table`` used)."""
    r = table[table.dost_phase == "R"]
    return r[r.allah > 0][["name_translit", "sinai_mvl_rank", "allah", "rabb", "n_verses_used"]]


def allah_threshold_check(table: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """Dost's rule of thumb: A-Qur'an surahs have about one *Allāh* per verse or more.

    Cross-tabulates his phase against "at or above the threshold" for the
    surahs he assigns to AR or A.
    """
    t = table[table.dost_phase.isin(["AR", "A"])]
    return pd.crosstab(t.dost_phase.astype(str), np.where(t.allah_per_verse >= threshold,
                                                          f">= {threshold}", f"< {threshold}"))


def mvl_agreement() -> dict:
    """How well our MVL (Arabic letters, Kufan count, nothing removed) matches Sinai's ranking.

    Sinai counts letters of a Latin transliteration and leaves out the
    verses he considers insertions, so some disagreement is expected.
    """
    s = load_surah_texts()
    rho, _ = spearmanr(s.mvl, s.sinai_mvl_rank)
    return {"spearman_rho": rho, "n_surahs": len(s)}


def lemmas_without_divine_names(verses: pd.DataFrame) -> pd.Series:
    """The verse lemma strings with *Allāh*, *rabb* and *al-Raḥmān* removed."""
    drop = set(DIVINE_NAMES.values())
    return verses["lemmas"].map(lambda s: " ".join(w for w in s.split() if w not in drop))


def phase_classifier(remove_divine_names: bool = True, n_splits: int = 5, seed: int = 0):
    """Predict Dost's phase (R / AR / A) from the vocabulary, one surah at a time held out.

    Returns (per-surah predictions, confusion matrix). With
    ``remove_divine_names=True`` the model never sees the names Dost used to
    define the phases, so success means the phases differ in other ways too.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline

    v = load_verses()
    v = v[v.dost_phase.isin(["R", "AR", "A"])].copy()
    text = lemmas_without_divine_names(v) if remove_divine_names else v["lemmas"]
    y = v.dost_phase.astype(str).to_numpy()
    model = make_pipeline(
        TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True),
        LogisticRegression(C=4.0, class_weight="balanced", max_iter=5000),
    )
    classes = ["R", "AR", "A"]
    proba = np.zeros((len(v), 3))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(text, y, v.surah):
        model.fit(text.iloc[tr], y[tr])
        order = [list(model.classes_).index(c) for c in classes]
        proba[te] = model.predict_proba(text.iloc[te])[:, order]
    P = pd.DataFrame(proba, columns=classes, index=v.index).groupby(v.surah).mean()
    P["predicted"] = P[classes].idxmax(axis=1)
    P["dost_phase"] = v.groupby("surah").dost_phase.first().astype(str)
    cm = pd.DataFrame(
        confusion_matrix(P.dost_phase, P.predicted, labels=classes),
        index=pd.Index(classes, name="Dost says"),
        columns=pd.Index(classes, name="model says"),
    )
    return P, cm
