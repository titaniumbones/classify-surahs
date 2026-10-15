# Results

These numbers come from `uv run surahs-experiments --bert`. The full tables are in `results/*.csv` and the figures in `results/figures/`. If you change the code or data, rerun the command and update this page.

All supervised results use **grouped cross-validation**: the surahs are divided into five groups, a model is trained on four and tested on the fifth, and this is repeated so every surah is tested once. The whole procedure is repeated with three different random groupings and the results averaged. No model ever sees any verse of the surah it is being tested on.

Two metrics appear throughout:

- **Balanced accuracy** is the average of the percentage of Meccan items called Meccan and the percentage of Medinan items called Medinan. Always guessing "Meccan" scores 0.50.
- **AUC** (area under the ROC curve) is the probability that the model gives a randomly chosen Medinan item a higher Medinan score than a randomly chosen Meccan one. 0.50 is chance and 1.00 is perfect. AUC doesn't depend on where you draw the line between "Meccan" and "Medinan", so it is the fairer way to compare models.

"Verse" columns score each of the 6,236 verses. "Surah" columns average the verse scores within each surah and score the 114 surahs.

## 1. The 2017 number in context

| Set-up | Accuracy |
|---|---|
| Always answer "Meccan" | 0.740 |
| 2017 method, one random 40/60 split by surah (20 different splits) | 0.766 – 0.892 (mean 0.828) |
| 2017 method, surah-grouped 5-fold cross-validation | 0.845 |
| 2017 method, *verse*-level random 5-fold (leaky) | 0.867 |

The 2017 notebook reported 86.3% validation accuracy from a single random split. Depending on which surahs land in the training set, the same method scores anywhere from 77% to 89%, against a do-nothing baseline of 74%. The 2017 code did avoid leakage, though: it split by surah. The last row shows the roughly two-point inflation you get if you split by verse instead.

## 2. Which evidence matters? (`feature_comparison.csv`)

Each feature set is fed to the same class-balanced logistic regression.

| Features | vs Cairo labels: verse AUC | surah AUC | vs Nöldeke labels: verse AUC | surah AUC |
|---|---|---|---|---|
| Always "Meccan" | 0.50 | 0.50 | 0.50 | 0.50 |
| Verse length only (3 numbers) | 0.77 | 0.90 | 0.81 | 0.94 |
| Style: length, rhyme, divine names, forms of address | 0.77 | 0.89 | 0.78 | 0.92 |
| Arabic words (the 2017 features) | 0.85 | 0.92 | 0.89 | 0.995 |
| Lemmas | 0.85 | 0.91 | 0.89 | 0.996 |
| Roots | 0.83 | 0.92 | 0.87 | 1.000 |
| Character 2–5-grams | 0.87 | 0.92 | 0.91 | 0.996 |
| English words (Pickthall) | 0.85 | 0.92 | 0.89 | 0.99 |
| Lemmas + characters + style | 0.86 | 0.93 | 0.90 | 0.98 |
| CAMeLBERT-CA embeddings | 0.89 | 0.96 | 0.93 | 0.99 |

What the table shows:

1. **Verse length does most of the work.** Three numbers per verse get a surah-level AUC of 0.90. Vocabulary helps more at the verse level, where a single verse's length is a noisy signal, than at the surah level.
2. **The text agrees with Nöldeke more than with Cairo.** Every feature set fits Nöldeke's division better. The vocabulary models separate his Meccan and Medinan surahs almost perfectly (AUC 0.99–1.00). Against the Cairo labels they top out at 0.92–0.96. Almost all of the gap comes from the four surahs the two systems disagree on (13, 55, 76, 99): leave those four out when scoring, and the Cairo-trained vocabulary models also reach surah AUC 0.995–1.00 (length alone: 0.95). There is a catch, though. Nöldeke drew his line partly on stylistic grounds, so a stylistic model agreeing with him is not independent confirmation of his line (see section 5).
3. **English works about as well as Arabic.** The regularities are coarse enough to survive translation, which tells you they are not subtle features of Qur'anic Arabic.
4. **The neural model is best, by a modest margin.** CAMeLBERT-CA adds about 0.04 AUC over the best counting model against Cairo. Caution: it was pretrained on classical Arabic texts that very probably include commentaries discussing which surahs are Meccan, and it is far harder to interpret.

## 3. Where models and the Cairo labels disagree (`surah_predictions.csv`, `figures/surah_proba_lemmas_cairo.png`)

The lemma model, trained on Cairo labels, never trained on the surah in question:

- **Predicts Meccan for Cairo-Medinan surahs 55, 99, 76** (P(Medinan) 0.17, 0.21, 0.24) and sits on the fence for **13** (0.50). These are exactly the four Nöldeke dates to Mecca.
- **Leans slightly Medinan for Cairo-Meccan surahs 31, 35 and 42** (0.51–0.55) and is on the fence for 14 (0.49). All four are in Nöldeke's late Meccan III period and have long verses.
- **Leans slightly Medinan for 112** (al-Ikhlāṣ, four very short verses, 0.54). Four verses give the model very little to go on; working out which words drive this is a good exercise.

The 2017 model's six "errors" (13, 22, 47, 55, 76, 99) are the same story: four are the Cairo/Nöldeke disagreements. Sinai argues that a fifth, 22, combines Meccan and Medinan material.

## 4. Verse-level exceptions (`exceptions_test.csv`)

The Cairo tradition lists 148 verses in 33 Meccan surahs as Medinan exceptions (plus 9 Meccan verses in two Medinan surahs). Models trained only on whole-surah labels were asked whether they score the exception verses as more Medinan than their surah-mates:

| Features | AUC (exception vs ordinary verses, same surahs) |
|---|---|
| Length only | 0.54 |
| Style | 0.55 |
| Lemmas | 0.61 |
| Character n-grams | 0.61 |
| CAMeLBERT-CA | 0.61 |

The signal is weak but not zero. The text gives only modest support to the tradition's verse-by-verse exceptions, and much of what support there is does not come from verse length.

## 5. Chronology without labels (`rank_correlations.csv`, `within_period_correlations.csv`, figures)

**Spearman's rho** compares two rankings: 1 means identical order, 0 means unrelated.

| Measure | vs Nöldeke order | vs Cairo order | Surahs |
|---|---|---|---|
| Mean verse length (MVL) | 0.91 | 0.69 | all 114 |
| MVL | 0.95 | 0.56 | 79 surahs ≥ 150 words |
| PC1 of the 100 commonest lemmas (no labels used) | 0.91 | 0.57 | 79 surahs ≥ 150 words |

PC1 is the main axis of variation in how often each surah uses the 100 commonest lemmas, found by principal component analysis with no information about dates. It puts the surahs in nearly the same order as Nöldeke. Its early end is marked by *yawm* (day), *rabb* (Lord), *fa-* (so/then), *khalaqa* (create), *kadhdhaba* (deny) and the question particle *a-*. Its late end is marked by *Allāh*, *āmana* (believe), *rasūl* (messenger), *in* (if), *kafara* (disbelieve) and *qalb* (heart).

**Within Nöldeke's periods** the picture is more mixed:

| Period | MVL rho | PC1 rho |
|---|---|---|
| Meccan I | 0.21 (n=48) | −0.49 (n=15) |
| Meccan II | 0.86 | 0.75 |
| Meccan III | 0.87 | 0.26 |
| Medinan | 0.40 | 0.08 |

The strong within-period MVL correlations in Meccan II and III are partly circular: Nöldeke ordered surahs inside his periods largely by stylistic impression, verse length included. The independent measure, PC1, tracks his within-period order only in Meccan II. This fits Sinai's and Sadeghi's own caution: these methods support a coarse relative chronology, not a precise ranking.

## 6. Four periods (`period_confusion.csv`)

A lemma model trained on Nöldeke's four periods, evaluated per surah:

| Nöldeke says ↓ / model says → | Meccan I | Meccan II | Meccan III | Medinan |
|---|---|---|---|---|
| Meccan I (48) | 40 | 7 | 0 | 1 |
| Meccan II (21) | 2 | 18 | 1 | 0 |
| Meccan III (21) | 0 | 1 | 20 | 0 |
| Medinan (24) | 1 | 0 | 0 | 23 |

Nearly all the errors fall in a neighbouring period, and most of them sit on the Meccan I/II boundary. That is consistent with the text changing gradually rather than in four sharp steps, which is how Sinai describes it.
