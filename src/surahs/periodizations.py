"""Compare competing periodizations of the Qur'an: 2, 3 and 4 stages.

A *periodization* assigns every surah to one of k ordered stages. Scholars
have proposed several:

==========================  ===  ================================================
scheme                      k    source
==========================  ===  ================================================
cairo                       2    1924 Cairo edition (Islamic tradition)
noldeke2                    2    Nöldeke–Schwally, Meccan vs Medinan
dost                        3    Dost 2026 ch. 3: R-, AR-, A-Qur'an (divine names)
noldeke3                    3    Nöldeke with Meccan II + III merged; the same
                                 *shape* as Farrin's (2019) three periods, but
                                 not his surah list
noldeke4                    4    Nöldeke–Schwally's four periods
noldeke_early2, dost_early2 2    the earliest stage vs everything after it
mvl_k                       k    the best cut of the mean-verse-length ranking
                                 into k contiguous stages (a data-driven baseline)
==========================  ===  ================================================

How do you compare a 2-stage model with a 4-stage one? More stages always
fit the data at least as well, in the same way a more detailed map is
always "closer" to the territory. So every comparison here corrects for
the number of stages, in one of three ways:

1. **Penalised fit (BIC).** Each scheme is treated as a statistical model:
   each stage has its own average stylistic profile. BIC (Bayesian
   information criterion) rewards fit and charges a fixed price for every
   extra parameter. Lower is better; differences above about 10 are
   conventionally called "very strong" evidence.
2. **Against chance (permutation test).** Shuffle the surahs into random
   groups of the same sizes 1,000 times and see how often a random
   grouping fits as well as the scheme. That gives a z-score: how many
   standard deviations better than random the scheme is.
3. **Predictability (chance-corrected).** Train a classifier on all but a
   few surahs and predict the stage of the held-out ones. Cohen's kappa
   rescales accuracy so that 0 = chance and 1 = perfect, whatever k is.

**The circularity problem, again.** Nöldeke, Sinai and Dost all used style
(verse length, divine names) to build their schemes. If we then measure
fit with those same features, the scheme will look good by construction.
So the default feature space here is the frequency of common *function
words*, with divine names and verse length removed. None of the schemes
was built from these.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

from surahs.data import load_morphology, load_surah_texts
from surahs.features import DIVINE_NAMES

# --- the schemes ---------------------------------------------------------------


def periodizations(include_mvl_cuts: tuple[int, ...] = (2, 3, 4, 5)) -> pd.DataFrame:
    """One column per scheme, one row per surah; values are stage numbers 1..k.

    Surahs a scheme leaves unassigned (Dost leaves 1 and 112) are NaN.
    """
    s = load_surah_texts()
    out = pd.DataFrame(index=s.index)
    out["cairo"] = np.where(s.cairo_place == "Medinan", 2, 1)
    out["noldeke2"] = np.where(s.noldeke_period.astype(str) == "Medinan", 2, 1)
    out["noldeke3"] = s.noldeke_period.astype(str).map(
        {"Meccan I": 1, "Meccan II": 2, "Meccan III": 2, "Medinan": 3}
    )
    out["noldeke4"] = s.noldeke_period.astype(str).map(
        {"Meccan I": 1, "Meccan II": 2, "Meccan III": 3, "Medinan": 4}
    )
    out["dost"] = s.dost_phase.astype(str).map({"R": 1, "AR": 2, "A": 3})
    # Two-stage variants that put the break early instead of at the hijra
    out["noldeke_early2"] = np.where(s.noldeke_period.astype(str) == "Meccan I", 1, 2)
    out["dost_early2"] = s.dost_phase.astype(str).map({"R": 1, "AR": 2, "A": 2})
    for k in include_mvl_cuts:
        out[f"mvl_{k}"] = best_contiguous_cut(s.mvl, k)
    return out


SCHEME_INFO = {
    "cairo": "Cairo 1924: Meccan / Medinan",
    "noldeke2": "Nöldeke: Meccan / Medinan",
    "noldeke3": "Nöldeke, Meccan II+III merged",
    "noldeke4": "Nöldeke: Meccan I / II / III / Medinan",
    "dost": "Dost 2026: R / AR / A (divine names)",
    "noldeke_early2": "Nöldeke: Meccan I / everything later",
    "dost_early2": "Dost: R-Qur'an / everything later",
}


def best_contiguous_cut(values: pd.Series, k: int) -> pd.Series:
    """Split surahs, sorted by ``values``, into k contiguous stages minimising within-stage variance.

    This is the periodization that verse length *alone* would suggest. It is
    solved exactly by dynamic programming (Fisher's method for optimal
    one-dimensional grouping).
    """
    order = values.sort_values()
    x = np.log(order.to_numpy())  # MVL is right-skewed; cut on the log scale
    n = len(x)
    cs, cs2 = np.concatenate([[0], np.cumsum(x)]), np.concatenate([[0], np.cumsum(x**2)])

    def sse(i, j):  # cost of segment x[i:j]
        m = j - i
        s = cs[j] - cs[i]
        return cs2[j] - cs2[i] - s * s / m

    cost = np.full((k + 1, n + 1), np.inf)
    back = np.zeros((k + 1, n + 1), dtype=int)
    cost[0, 0] = 0
    for g in range(1, k + 1):
        for j in range(g, n + 1):
            best, arg = np.inf, 0
            for i in range(g - 1, j):
                c = cost[g - 1, i] + sse(i, j)
                if c < best:
                    best, arg = c, i
            cost[g, j], back[g, j] = best, arg
    labels = np.zeros(n, dtype=int)
    j = n
    for g in range(k, 0, -1):
        i = back[g, j]
        labels[i:j] = g
        j = i
    return pd.Series(labels, index=order.index).reindex(values.index)


# --- the feature space ------------------------------------------------------------


def function_word_profile(n_features: int = 60, drop_divine_names: bool = True) -> pd.DataFrame:
    """Per-surah frequencies (per 1000 segments) of the commonest *particles, pronouns and prepositions*.

    Restricted to segments the Quranic Arabic Corpus tags as particles ("P":
    prepositions, conjunctions, negations, the article, attached pronouns
    and the like), so no theme words like "believe" or "messenger" sneak in.
    """
    m = load_morphology().dropna(subset=["lemma"])
    closed = {"P"}  # the corpus tags every particle, preposition and conjunction "P"
    m = m[m.tag.isin(closed)]
    if drop_divine_names:
        m = m[~m.lemma.isin(set(DIVINE_NAMES.values()))]
    top = m.lemma.value_counts().index[:n_features]
    counts = pd.crosstab(m.surah, m.lemma)[top]
    totals = load_morphology().groupby("surah").size()
    return (counts.div(totals, axis=0) * 1000).reindex(range(1, 115), fill_value=0)


def feature_space(kind: str = "function_words", n_components: int = 8, min_words: int = 0) -> pd.DataFrame:
    """Standardised principal-component scores for each surah.

    ``kind="function_words"``: closed-class word frequencies (default; independent of how any
    scheme was built). ``kind="mvl"``: log mean verse length only (the variable most schemes
    were partly built on, so expect circular agreement).
    """
    s = load_surah_texts()
    if kind == "mvl":
        X = pd.DataFrame({"log_mvl": np.log(s.mvl)})
    elif kind == "function_words":
        prof = function_word_profile()
        Z = StandardScaler().fit_transform(prof)
        pcs = PCA(n_components=n_components, random_state=0).fit_transform(Z)
        X = pd.DataFrame(pcs, index=prof.index, columns=[f"PC{i + 1}" for i in range(n_components)])
    else:
        raise ValueError(kind)
    if min_words:
        X = X.loc[s.index[s.n_words >= min_words]]
    return pd.DataFrame(StandardScaler().fit_transform(X), index=X.index, columns=X.columns)


# --- the comparisons ----------------------------------------------------------------


def _gaussian_bic(X: np.ndarray, labels: np.ndarray) -> float:
    """BIC of a model where each stage has its own mean and all share one diagonal variance."""
    n, d = X.shape
    groups = np.unique(labels)
    resid = np.vstack([X[labels == g] - X[labels == g].mean(axis=0) for g in groups])
    var = resid.var(axis=0) + 1e-9
    loglik = -0.5 * n * np.sum(np.log(2 * np.pi * var) + 1)
    n_params = len(groups) * d + d
    return -2 * loglik + n_params * np.log(n)


def _between_ratio(X: np.ndarray, labels: np.ndarray) -> float:
    """Share of total variance explained by stage membership (R²)."""
    total = ((X - X.mean(axis=0)) ** 2).sum()
    within = sum(((X[labels == g] - X[labels == g].mean(axis=0)) ** 2).sum() for g in np.unique(labels))
    return 1 - within / total


def compare_schemes(
    X: pd.DataFrame | None = None,
    schemes: pd.DataFrame | None = None,
    n_permutations: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Score every scheme on the same surahs and the same features.

    Surahs that any scheme leaves unassigned are dropped for all schemes, so
    the comparison is like for like. Includes a 1-stage "no chronology" row
    as the floor.
    """
    X = feature_space() if X is None else X
    P = periodizations() if schemes is None else schemes
    common = X.index.intersection(P.dropna().index)
    X, P = X.loc[common].to_numpy(), P.loc[common]
    rng = np.random.default_rng(seed)
    rows = [{"scheme": "none (1 stage)", "k": 1, "bic": _gaussian_bic(X, np.zeros(len(X))), "r2": 0.0,
             "z_vs_random": np.nan, "p_vs_random": np.nan}]
    for name in P.columns:
        lab = P[name].to_numpy().astype(int)
        r2 = _between_ratio(X, lab)
        null = np.array([_between_ratio(X, rng.permutation(lab)) for _ in range(n_permutations)])
        rows.append(
            {
                "scheme": name,
                "k": len(np.unique(lab)),
                "bic": _gaussian_bic(X, lab),
                "r2": r2,
                "z_vs_random": (r2 - null.mean()) / null.std(),
                "p_vs_random": (1 + (null >= r2).sum()) / (1 + n_permutations),
            }
        )
    out = pd.DataFrame(rows)
    out["delta_bic"] = out.bic - out.bic.min()
    out["description"] = out.scheme.map(SCHEME_INFO).fillna(
        out.scheme.str.replace("mvl_", "best cut of MVL order into ", regex=False).str.cat([" stages"] * len(out))
    )
    out.loc[out.scheme == "none (1 stage)", "description"] = "no chronology at all"
    out["n_surahs"] = len(common)
    return out.sort_values("bic").reset_index(drop=True)


def predictability(schemes: pd.DataFrame | None = None, n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """Chance-corrected accuracy (Cohen's kappa) of predicting each scheme's stages from held-out surahs.

    Verse-level lemma model (divine names removed), predictions averaged per
    surah, surahs grouped in cross-validation.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline

    from surahs.data import load_verses
    from surahs.divine_names import lemmas_without_divine_names

    P = periodizations() if schemes is None else schemes
    P = P.dropna()
    v = load_verses()
    v = v[v.surah.isin(P.index)]
    text = lemmas_without_divine_names(v)
    rows = []
    for name in P.columns:
        y_surah = P[name].astype(int)
        y = v.surah.map(y_surah).to_numpy()
        classes = np.sort(y_surah.unique())
        proba = np.zeros((len(v), len(classes)))
        model = make_pipeline(TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True),
                              LogisticRegression(C=4.0, class_weight="balanced", max_iter=5000))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in cv.split(text, y, v.surah):
            model.fit(text.iloc[tr], y[tr])
            idx = [list(model.classes_).index(c) for c in classes]
            proba[te] = model.predict_proba(text.iloc[te])[:, idx]
        sp = pd.DataFrame(proba, index=v.index).groupby(v.surah).mean()
        pred = classes[sp.to_numpy().argmax(axis=1)]
        truth = y_surah.loc[sp.index].to_numpy()
        rows.append({"scheme": name, "k": len(classes), "accuracy": (pred == truth).mean(),
                     "kappa": cohen_kappa_score(truth, pred),
                     "kappa_weighted": cohen_kappa_score(truth, pred, weights="linear")})
    return pd.DataFrame(rows).sort_values("kappa_weighted", ascending=False).reset_index(drop=True)


def agreement_matrix(schemes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Adjusted Rand index between every pair of schemes: 1 = identical groupings, 0 = chance."""
    P = (periodizations() if schemes is None else schemes).dropna()
    cols = P.columns
    return pd.DataFrame([[adjusted_rand_score(P[a], P[b]) for b in cols] for a in cols], index=cols, columns=cols)
