# What changed from the 2017 version

The original project, by Abubakar Abid (January 2017; files in `legacy/`), asked whether a computer could learn to tell Meccan from Medinan surahs. It trained a bag-of-words logistic regression on the Cairo edition's labels and reported 86% verse-level accuracy. It was a clear, honest notebook, and it split its data by surah, which many later studies did not. This rebuild keeps its question but changes how the question is framed, measured and interpreted.

## Framing

| 2017 | 2026 |
|---|---|
| The Meccan/Medinan labels are the right answers; disagreements are model errors. | The labels are scholarly claims. Two traditions are loaded side by side: the 1924 Cairo edition, which draws on early Islamic lists, and Nöldeke–Schwally's four periods. Where a model disagrees with a label, that is a question to investigate. |
| Binary only. | Binary; Dost's three divine-name phases; Nöldeke's four periods; continuous rank (Spearman correlation with chronological orders); and a head-to-head comparison of 2-, 3- and 4-stage schemes. |
| Supervised only. | Also unsupervised (PCA of common-lemma frequencies), following Sadeghi's argument that independent stylistic measures should agree with each other if a real chronological signal exists. |
| No reference to scholarship. | `docs/state-of-the-field.md` surveys Nöldeke, Bazargan, Neuwirth, Sadeghi, Sinai, Farrin, the sceptical critiques (Reynolds, Stefanidis, Shoemaker, Dye, Tesei), Dost's project and the computational literature, with verified citations. |

## Data

| 2017 | 2026 |
|---|---|
| Tanzil text with the basmala glued onto verse 1 of 112 surahs. | Basmala removed in memory (`strip_basmala`), raw file kept verbatim. |
| A second copy of the text (`verses.csv`) with the Arabic garbled to `????`. | Removed (kept in `legacy/` for the record). |
| Surface words only, so وقالوا and قالوا are unrelated tokens. | Lemmas and roots from the Quranic Arabic Corpus, character n-grams, verse length in letters and words, rhyme letters, divine-name rates and forms of address. |
| Arabic only. | Pickthall's English translation alongside, for students who don't read Arabic and as a test of how much signal survives translation. |
| Surah-level labels only. | Also the Cairo tradition's verse-level exceptions (157 verses), hand-coded from Tanzil's revelation-order table. |
| No provenance notes. | `data/SOURCES.md` gives source, licence and retrieval date for every file. |

## Evaluation

| 2017 | 2026 |
|---|---|
| One random 40/60 split; the reported 86% could have been anywhere from 77% to 89% depending on the split. | Grouped 5-fold cross-validation by surah, repeated three times. |
| Plain accuracy against a 74% majority baseline. | Balanced accuracy and ROC AUC, with the majority baseline shown explicitly. |
| Unweighted logistic regression on an imbalanced dataset. | `class_weight="balanced"`. |
| No baseline beyond "always Meccan". | A verse-length-only model reported next to every vocabulary model. It turns out to explain most of the performance. |
| Feature weights from a model fitted on all data, read as "Meccan words". | The same, but with the caveat that a model fitted on all data can memorise one surah's refrain (see *ālāʾ* in surah 55). Features shown with transliterations and links to the Quranic Arabic Corpus dictionary. |

## Tools

| 2017 | 2026 |
|---|---|
| A single notebook, Python 3 of the day, no dependency list; does not run today without changes. | A Python package (`src/surahs/`) managed with `uv`, with a lock file pinning every dependency. |
| | Four teaching notebooks with questions for students. |
| | One command (`uv run surahs-experiments`) that regenerates every number in `docs/results.md`. |
| | Tests (`uv run pytest`) that check the data: verse counts, Nöldeke's period sizes, basmala removal. |
| | An optional neural model: embeddings from CAMeLBERT-CA, a BERT model trained on Classical Arabic. |

## Things deliberately not done

- **Asking a chatbot.** Large language models have read the traditional labels many times over, so asking one to classify a surah measures its memory of the tradition, not the text.
- **Fine-tuning the neural model.** With 114 surahs, a 110-million-parameter network would overfit badly, and its gain over embeddings plus logistic regression would be hard to tell apart from noise.
- **Treating a high score as a discovery.** A classifier that agrees with the Cairo labels 95% of the time has learned to reproduce an early classification. It has not dated anything independently.
