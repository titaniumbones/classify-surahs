# Dating the surahs

In what order were the 114 surahs (chapters) of the Qur'an first proclaimed? Islamic tradition and modern scholarship have both proposed answers, and they mostly agree but not entirely. This repository uses transparent machine learning and simple statistics to test those answers against the text itself, and to show where such tests can mislead.

It was built for **HCSB03H3 (Historical Research Methods), University of Toronto Scarborough**, as preparation for Prof. Suleyman Dost's research project on the Qur'an's inner chronology. It began as Abubakar Abid's 2017 Meccan/Medinan classifier, which is preserved in `legacy/`.

**Start here:** [`docs/guide-for-students.md`](docs/guide-for-students.md) explains the question, the scholarship and the methods in plain language. No Arabic, statistics or programming is assumed.

## What's in it

| Path | What it is |
|---|---|
| `docs/guide-for-students.md` | The plain-language introduction. Read first. |
| `docs/state-of-the-field.md` | A cited survey of scholarship on dating the surahs, from Nöldeke to Sinai, Sadeghi, the sceptics, Dost's project and the computational literature. |
| `docs/results.md` | What the experiments found, with all the numbers. |
| `docs/what-changed.md` | How and why this differs from the 2017 version. |
| `notebooks/` | Six teaching notebooks with questions, to be worked through in order. |
| `src/surahs/` | The Python package the notebooks use: data loading, features, models, chronology. |
| `data/` | The Qur'an text (Arabic and English), surah metadata, the Cairo, Nöldeke and Dost chronologies, Sinai's MVL ranking and interpolations, and the verse-level exceptions. Sources and licences are in `data/SOURCES.md`. |
| `results/` | Tables and figures produced by `uv run surahs-experiments`. |
| `legacy/` | The 2017 notebook and data. |

## Quick start

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs the right Python and every package for you.

```sh
uv sync                      # once: set up the environment
uv run jupyter lab           # open the notebooks
uv run surahs-experiments    # regenerate results/ (a few minutes)
uv run pytest                # check the data
```

The first time the data is loaded, the Quranic Arabic Corpus morphology (about 6 MB) is downloaded to `data/cache/`. Its terms allow only verbatim copies, so it is not shipped here.

The neural-network model in notebook 4 needs an extra 1.5 GB download:

```sh
uv sync --extra transformers
uv run surahs-experiments --bert
```

To rebuild `data/surahs.csv` from the raw sources: `uv run scripts/build_surah_table.py`.

## Headline findings

- A model that knows only how long each verse is separates Meccan from Medinan surahs with an AUC of 0.90. Vocabulary models add little on top of that at the surah level.
- Every model agrees with Nöldeke's dating better than with the Cairo edition's. The difference comes almost entirely from four surahs (13, 55, 76, 99). The 2017 version reported those as model errors. In fact the model was agreeing with Nöldeke.
- With no labels at all, the frequencies of the 100 commonest lemmas order the surahs much as Nöldeke did (Spearman's rho 0.91). Within his periods the agreement is weak, so these methods support only a coarse chronology.
- Prof. Dost's three divine-name phases (*Before the Qur'an*, ch. 3) hold up: with the divine names deleted, other vocabulary still puts about 90% of surahs in his phase.
- Comparing 2-, 3- and 4-stage schemes on particle use, the sharpest break falls between the earliest surahs and the rest, not at the hijra. Three-stage schemes beat the traditional Meccan/Medinan split.

See `docs/results.md` for details and caveats.

## Credits and licences

The original code is © 2017 Abubakar Abid (see `LICENSE`). The data comes from the Tanzil Project (text CC BY 3.0, verbatim; Pickthall translation, public domain in Canada), the Quranic Arabic Corpus (Kais Dukes; GNU GPL), and Wikipedia's tabulation of Nöldeke's order after Neal Robinson (CC BY-SA). Full details are in `data/SOURCES.md`.
