"""Run every experiment and write tables to results/ and figures to results/figures/.

    uv run surahs-experiments            # everything except the neural model
    uv run surahs-experiments --bert     # also CAMeLBERT-CA (needs the extra:
                                         #   uv sync --extra transformers)

Takes a few minutes on a laptop. The notebooks in notebooks/ walk through the
same analyses step by step; this script exists so the numbers quoted in the
documentation can be regenerated with one command.
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict, train_test_split

from surahs.chronology import pca_chronology, rank_correlations, within_period_correlations
from surahs.data import PERIOD_ORDER, ROOT, load_surah_texts, load_verses
from surahs.features import FEATURE_DESCRIPTIONS, FEATURE_SETS
from surahs.models import baseline_model, cross_validate, make_model, top_features
from surahs.plotting import PERIOD_STYLE, style_axes
from surahs.translit import translit

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"


def compare_feature_sets(v: pd.DataFrame, bert: np.ndarray | None) -> tuple[pd.DataFrame, dict]:
    rows, runs = [], {}
    names = ["baseline", *FEATURE_SETS] + (["camelbert"] if bert is not None else [])
    for label in ("cairo_place", "noldeke_place"):
        for name in names:
            if name == "baseline":
                r = cross_validate(baseline_model(), v, label, name=name)
            elif name == "camelbert":
                r = cross_validate(_embedding_model(), v, label, X=bert, name=name)
            else:
                r = cross_validate(make_model(name), v, label, name=name)
            runs[(label, name)] = r
            rows.append(
                {
                    "labels": label.split("_")[0],
                    "features": name,
                    "description": FEATURE_DESCRIPTIONS.get(name, "Always answers 'Meccan'"),
                    "verse_bal_acc": r.verse_metrics["balanced_accuracy"],
                    "verse_auc": r.verse_metrics["roc_auc"],
                    "surah_bal_acc": r.surah_metrics["balanced_accuracy"],
                    "surah_auc": r.surah_metrics["roc_auc"],
                }
            )
            print(f"  {label:14s} {name:10s} surah AUC {r.surah_metrics['roc_auc']:.3f}")
    return pd.DataFrame(rows), runs


def _embedding_model():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000))


def replicate_2017(v: pd.DataFrame) -> pd.DataFrame:
    """The 2017 set-up, and what changes when the evaluation is done properly.

    The 2017 notebook: raw word counts, unweighted logistic regression, one
    random 40/60 split of surahs, plain accuracy. We repeat that split 20
    times to show how much the single reported number could wobble, and
    contrast it with a *verse-level* random split (which leaks).
    """
    y = (v.cairo_place == "Medinan").astype(int).to_numpy()
    X = CountVectorizer(token_pattern=r"\S+").fit_transform(v.text)
    surahs = v.surah.to_numpy()
    rows = []
    for seed in range(20):
        train_s, _ = train_test_split(np.arange(1, 115), train_size=0.4, random_state=seed)
        tr = np.isin(surahs, train_s)
        m = LogisticRegression(max_iter=5000).fit(X[tr], y[tr])
        rows.append({"setup": "2017: one random 40/60 split by surah", "seed": seed, "accuracy": m.score(X[~tr], y[~tr])})
    for split, cv in [("verse-level 5-fold (leaky)", KFold(5, shuffle=True, random_state=0)), ("surah-grouped 5-fold", GroupKFold(5))]:
        pred = cross_val_predict(LogisticRegression(max_iter=5000), X, y, cv=cv, groups=surahs)
        rows.append({"setup": split, "seed": 0, "accuracy": (pred == y).mean()})
    rows.append({"setup": "always answer 'Meccan'", "seed": 0, "accuracy": 1 - y.mean()})
    return pd.DataFrame(rows)


def period_classifier(v: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Four-way classification into Nöldeke's periods, evaluated per surah."""
    from sklearn.model_selection import StratifiedGroupKFold

    y = v.noldeke_period.astype(str).to_numpy()
    groups = v.surah.to_numpy()
    model = make_model("lemmas")
    model.set_params(logisticregression__C=4.0)
    proba = np.zeros((len(v), 4))
    classes = None
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in cv.split(v, y, groups):
        model.fit(v.iloc[tr], y[tr])
        classes = list(model.classes_)
        proba[te] = model.predict_proba(v.iloc[te])
    P = pd.DataFrame(proba, columns=classes, index=v.index)[PERIOD_ORDER]
    sp = P.groupby(v.surah).mean()
    s = load_surah_texts()
    sp["predicted"] = sp[PERIOD_ORDER].idxmax(axis=1)
    sp["noldeke_period"] = s.noldeke_period.astype(str)
    cm = pd.DataFrame(
        confusion_matrix(sp.noldeke_period, sp.predicted, labels=PERIOD_ORDER),
        index=pd.Index(PERIOD_ORDER, name="Nöldeke says"),
        columns=pd.Index(PERIOD_ORDER, name="model says"),
    )
    return sp, cm


def exceptions_test(v: pd.DataFrame, runs: dict) -> pd.DataFrame:
    """Do models single out the verses that tradition calls exceptions?

    The models were trained on whole-surah labels and never saw the surah a
    test verse comes from, so they know nothing about the exception lists.
    Within the 33 Meccan surahs that the Cairo tradition says contain
    Medinan verses, we ask: are those verses scored as more Medinan than
    their neighbours? AUC = the chance that a random exception verse
    outscores a random ordinary verse from a surah with exceptions; 0.5 means
    the model can't tell them apart.
    """
    from sklearn.metrics import roc_auc_score

    has_ex = v[(v.cairo_place == "Meccan") & v.surah.isin(v.loc[v.is_exception, "surah"])]
    rows = []
    for (label, name), r in runs.items():
        if label != "cairo_place" or name == "baseline":
            continue
        p = r.verse_proba.loc[has_ex.index]
        # rank within surah, so a surah's overall Medinan-ness doesn't count
        within = p.groupby(has_ex.surah).rank(pct=True)
        rows.append(
            {
                "features": name,
                "auc_pooled": roc_auc_score(has_ex.is_exception, p),
                "auc_within_surah": roc_auc_score(has_ex.is_exception, within),
                "n_exception_verses": int(has_ex.is_exception.sum()),
                "n_surahs": has_ex.surah.nunique(),
            }
        )
    return pd.DataFrame(rows)


def interpret(v: pd.DataFrame) -> pd.DataFrame:
    m = make_model("lemmas").fit(v, (v.cairo_place == "Medinan").astype(int))
    tf = top_features(m, 20)
    tf["transliteration"] = tf.feature.map(translit)
    tf["look_up"] = "https://corpus.quran.com/search.jsp?q=lem:" + tf.feature
    return tf


# --- figures -----------------------------------------------------------------


def fig_mvl(s: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for period, st in PERIOD_STYLE.items():
        g = s[s.noldeke_period == period]
        ax.scatter(g.noldeke_order, g.mvl, label=period, **st)
    for n in s.index[s.labels_disagree]:
        ax.annotate(f"Q{n}", (s.noldeke_order[n], s.mvl[n]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("Nöldeke's chronological rank (1 = earliest)")
    ax.set_ylabel("Mean verse length (letters, log scale)")
    ax.set_title("Later surahs (by Nöldeke's order) have longer verses")
    style_axes(ax)
    ax.legend(frameon=False, title="Nöldeke period")
    fig.tight_layout()
    fig.savefig(FIGS / "mvl_vs_noldeke.png", dpi=150)
    plt.close(fig)


def fig_pca(scores: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for period, st in PERIOD_STYLE.items():
        g = scores[scores.noldeke_period == period]
        ax.scatter(g.noldeke_order, g.PC1, label=period, **st)
    ax.set_xlabel("Nöldeke's chronological rank (1 = earliest)")
    ax.set_ylabel("PC1 of the 100 commonest lemmas")
    ax.set_title("With no labels at all, word-frequency patterns line up with Nöldeke")
    style_axes(ax)
    ax.legend(frameon=False, title="Nöldeke period")
    fig.tight_layout()
    fig.savefig(FIGS / "pca_vs_noldeke.png", dpi=150)
    plt.close(fig)


def fig_surah_proba(s: pd.DataFrame, p: pd.Series, title: str, fname: str):
    s = s.assign(p=p)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for period, st in PERIOD_STYLE.items():
        g = s[s.noldeke_period == period]
        ax.scatter(g.noldeke_order, g.p, label=period, **st)
    for n in s.index[s.labels_disagree | ((s.p > 0.5) != (s.cairo_place == "Medinan"))]:
        ax.annotate(f"Q{n}", (s.noldeke_order[n], s.p[n]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.axhline(0.5, color="#8a8984", lw=1, ls="--")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Nöldeke's chronological rank (1 = earliest)")
    ax.set_ylabel("Model's P(Medinan), never trained on this surah")
    ax.set_title(title)
    style_axes(ax)
    ax.legend(frameon=False, title="Nöldeke period", loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bert", action="store_true", help="also evaluate CAMeLBERT-CA embeddings")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    FIGS.mkdir(parents=True, exist_ok=True)

    v = load_verses()
    s = load_surah_texts()

    print("1. Replicating the 2017 set-up")
    rep = replicate_2017(v)
    rep.to_csv(RESULTS / "replication_2017.csv", index=False)
    print(rep.groupby("setup").accuracy.describe()[["mean", "min", "max"]].round(3).to_string())

    bert = None
    if args.bert:
        from surahs.embeddings import verse_embeddings

        print("   computing / loading CAMeLBERT-CA embeddings")
        bert = verse_embeddings(v)

    print("2. Comparing feature sets (grouped 5-fold CV x 3 repeats)")
    table, runs = compare_feature_sets(v, bert)
    table.to_csv(RESULTS / "feature_comparison.csv", index=False)

    print("3. Per-surah predictions")
    per = s[["name_translit", "cairo_place", "noldeke_period", "noldeke_order", "cairo_order", "mvl", "labels_disagree"]].copy()
    for (label, name), r in runs.items():
        if name in ("length", "lemmas", "chars", "english", "camelbert"):
            per[f"p_medinan_{name}_{label.split('_')[0]}"] = r.surah_proba
    per.to_csv(RESULTS / "surah_predictions.csv")
    fig_surah_proba(s, runs[("cairo_place", "lemmas")].surah_proba,
                    "Trained on Cairo labels: where the model and the tradition part ways", "surah_proba_lemmas_cairo.png")
    fig_surah_proba(s, runs[("cairo_place", "length")].surah_proba,
                    "Verse length alone gets most of the way there", "surah_proba_length_cairo.png")

    print("3b. Verse-level exceptions")
    ex = exceptions_test(v, runs)
    ex.to_csv(RESULTS / "exceptions_test.csv", index=False)
    print(ex.round(3).to_string())

    print("4. Four-period classifier")
    sp, cm = period_classifier(v)
    sp.to_csv(RESULTS / "period_predictions.csv")
    cm.to_csv(RESULTS / "period_confusion.csv")
    print(cm.to_string())

    print("5. Unsupervised chronology")
    scores, pca, loadings = pca_chronology()
    rc = pd.concat([rank_correlations(columns=("mvl",)).assign(subset="all 114 surahs"),
                    rank_correlations(scores, ("PC1", "mvl")).assign(subset="79 surahs >= 150 words")])
    rc.to_csv(RESULTS / "rank_correlations.csv", index=False)
    print(rc.round(3).to_string())
    wp = pd.concat([within_period_correlations(scores, "PC1").assign(measure="PC1"),
                    within_period_correlations(s, "mvl").assign(measure="mvl")])
    wp.to_csv(RESULTS / "within_period_correlations.csv", index=False)
    lo = loadings.assign(transliteration=[translit(x) for x in loadings.index]).sort_values("PC1")
    lo.to_csv(RESULTS / "pca_loadings.csv")
    fig_mvl(s)
    fig_pca(scores)

    print("6. Most informative lemmas")
    interpret(v).to_csv(RESULTS / "top_lemmas.csv", index=False)
    print(f"done; see {RESULTS}")


if __name__ == "__main__":
    main()
