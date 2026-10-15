"""Train and evaluate classifiers without fooling ourselves.

The central rule: **a surah's verses are never split between training and
testing.** Verses of the same surah share rhyme, vocabulary and topic, so a
model that has seen 2:1-2:200 will find 2:201-2:286 easy for reasons that
have nothing to do with Mecca or Medina. All cross-validation here groups by
surah (``StratifiedGroupKFold``), and verse-level predictions are averaged
into one prediction per surah.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from surahs.features import FEATURE_SETS

LABELS = {
    # column in load_verses() -> the class we call "positive" (=1)
    "cairo_place": "Medinan",
    "noldeke_place": "Medinan",
}


def make_model(features: str | object, C: float = 1.0):
    """A feature set followed by a class-balanced logistic regression.

    ``class_weight="balanced"`` matters: about 74% of verses are Meccan, so an
    unweighted model can score 74% by always answering "Meccan".
    """
    feats = FEATURE_SETS[features]() if isinstance(features, str) else features
    return make_pipeline(feats, LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def baseline_model():
    """Always predicts the majority class. Any real model has to beat this."""
    return DummyClassifier(strategy="most_frequent")


@dataclass
class CVResult:
    name: str
    label: str
    verse_proba: pd.Series  # out-of-fold P(Medinan) for every verse
    surah_proba: pd.Series  # mean of verse_proba within each surah
    verse_metrics: dict
    surah_metrics: dict


def _metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    out = {
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "accuracy": float((pred == y_true).mean()),
        "macro_f1": f1_score(y_true, pred, average="macro"),
    }
    out["roc_auc"] = roc_auc_score(y_true, proba) if len(np.unique(proba)) > 1 else 0.5
    return out


def cross_validate(
    model,
    verses: pd.DataFrame,
    label: str = "cairo_place",
    X: np.ndarray | pd.DataFrame | None = None,
    n_splits: int = 5,
    seeds: tuple[int, ...] = (0, 1, 2),
    name: str = "",
) -> CVResult:
    """Grouped cross-validation, repeated with several random splits.

    Every verse gets an out-of-fold probability from each repetition; we
    average them. ``X`` can be passed for precomputed features (e.g. neural
    embeddings); otherwise the model receives the verse DataFrame itself.
    """
    y = (verses[label] == LABELS[label]).astype(int).to_numpy()
    groups = verses["surah"].to_numpy()
    data = verses if X is None else X
    proba = np.zeros((len(seeds), len(verses)))
    for i, seed in enumerate(seeds):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train, test in cv.split(np.zeros(len(y)), y, groups):
            m = clone(model)
            Xtr = data.iloc[train] if hasattr(data, "iloc") else data[train]
            Xte = data.iloc[test] if hasattr(data, "iloc") else data[test]
            m.fit(Xtr, y[train])
            proba[i, test] = m.predict_proba(Xte)[:, 1]
    p = pd.Series(proba.mean(axis=0), index=verses.index, name="p_medinan")
    sp = p.groupby(verses["surah"]).mean()
    sy = (verses.groupby("surah")[label].first() == LABELS[label]).astype(int)
    return CVResult(
        name=name,
        label=label,
        verse_proba=p,
        surah_proba=sp,
        verse_metrics=_metrics(y, p.to_numpy()),
        surah_metrics=_metrics(sy.loc[sp.index].to_numpy(), sp.to_numpy()),
    )


def top_features(model, n: int = 20) -> pd.DataFrame:
    """The features with the largest positive (Medinan) and negative (Meccan) weights.

    Only meaningful for a fitted pipeline whose first step has
    ``get_feature_names_out`` and whose last step is a linear model.
    """
    names = model[:-1].get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]
    coef = model[-1].coef_.ravel()
    order = np.argsort(coef)
    rows = [("Meccan", names[i], coef[i]) for i in order[:n]] + [
        ("Medinan", names[i], coef[i]) for i in order[::-1][:n]
    ]
    return pd.DataFrame(rows, columns=["points_towards", "feature", "weight"])
