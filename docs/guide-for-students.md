# A guide for students: dating the surahs with a computer

This guide explains what this repository is for, the historical problem behind it, and how to read what the code produces. You don't need to know Arabic, statistics or programming to follow it. You will pick up a little of each along the way.

## 1. What is the question?

The Qur'an is divided into 114 chapters called **surahs**. Each surah is divided into verses called **āyāt** (singular **āya**); in the standard numbering there are 6,236. The surahs are not arranged in the order in which they were first proclaimed. Apart from the short opening surah, they run roughly from longest to shortest. Muslim tradition and modern scholars agree on this.

So in what order were they proclaimed? Muslim scholars asked this question very early. The science of distinguishing **Meccan** from **Medinan** revelation (*ʿilm al-makkī wa-l-madanī*) has been part of Qur'anic studies for over a thousand years. The labels refer to the two phases of the Prophet Muhammad's career:

- **Meccan**: proclaimed in Mecca, before the *hijra*, the emigration of Muhammad and his followers to Medina in 622 CE;
- **Medinan**: proclaimed after the hijra, when the believers formed a community in Medina with its own laws, alliances and wars.

Knowing the order matters to historians because it lets you read the text as a record of change. You can watch how its themes, its audience and its language develop over twenty-odd years. It matters to Muslim jurists for a related reason: when two verses seem to give different rulings, the later one may be taken to modify the earlier.

Prof. Suleyman Dost has already proposed one answer. In chapter 3 of his book *Before the Qur'an* (2026) he divides the Qur'an into three phases according to how it names God (section 3 below). His current project, "Bringing Order to Chaos: A Study of the Qur'an's Inner Chronology", tests whether measurable features of the text can sort out the order more fully. It looks at three features: how long the verses are, which names for God are used, and where words of possibly foreign origin occur. It uses correlation and regression, two basic statistical tools. This repository is a place to practise that kind of work and to see where it can go wrong.

A note on perspective. For many Muslims the Qur'an is the word of God, and its "date" is the moment of revelation. The methods here belong to *historical-critical* scholarship, which studies the text with the same tools historians use on any other document. That is one academic way of asking questions. It does not settle questions of faith, and you should be careful to say what your evidence shows rather than more.

## 2. Two answers already on the table

### The Cairo edition (1924)

Almost every printed Qur'an today descends from an edition published in Cairo in 1924. Its surah headings say whether each surah is Meccan or Medinan and give its place in an order of revelation. Many headings also list single verses said to come from the other city ("Meccan, except verses 10, 11 and 20, which are Medinan"). This information comes from early Islamic lists of the order of revelation, often attributed to the Prophet's cousin Ibn ʿAbbās, and from reports about the *asbāb al-nuzūl*, the "occasions of revelation".

### Nöldeke (1860, revised 1909–1938)

The German scholar Theodor Nöldeke, building on Gustav Weil, divided the surahs into four periods: **Meccan I** (48 surahs), **Meccan II** (21), **Meccan III** (21) and **Medinan** (24). He did not trust the traditional lists, which he thought disagreed with each other more than tradition admitted. He relied instead on the Qur'an's own language: verse length, style, vocabulary and references to events. His *History of the Qur'an* (*Geschichte des Qorāns*) is still the starting point for most Western scholarship, and it has been translated into English (Brill, 2013).

The two systems agree on 110 of the 114 surahs. They disagree on four, **13, 55, 76 and 99**, which Cairo calls Medinan and Nöldeke calls Meccan. Remember those four numbers. They come up again and again.

## 3. Why verse length?

If you read the Qur'an in order, you can't miss it: the surahs Islamic tradition calls early Meccan are short and urgent, with short rhyming lines. The ones it calls Medinan are long and discursive, with long verses that deal with laws and with the community's conflicts. Scholars have tried to turn that impression into a measurement.

- **Mehdi Bazargan**, an Iranian engineer and later prime minister, divided the Qur'an into 194 blocks and ordered them by verse length (1970s).
- **Behnam Sadeghi** (2011) took Bazargan's blocks and asked whether *other* features, chosen independently, change smoothly in the same order. The features were the frequencies of the commonest little words, of medium-frequency words and of rare words. They did. If one ordering lines up several unrelated measures at once, chance becomes a poor explanation and a real development over time a better one.
- **Nicolai Sinai** (2017) measured **mean verse length (MVL)** for each surah and showed that many other features line up with it. Oaths ("By the dawn!") appear only in short-verse surahs. References to the "hypocrites" appear only in long-verse ones. He is also careful about the limits: two surahs with similar MVL can't be put in order by MVL alone, and some short surahs, like prayers, may be short because of their genre rather than their date.

### Names of God: Dost's three phases

Prof. Dost (*Before the Qur'an*, 2026, chapter 3) takes Sinai's verse-length ordering and asks what the text calls God along it. He finds three phases:

- **R-Qur'an**: the earliest surahs call God *Rabb*, "your Lord" or "Lord of the worlds". The name *Allāh* is essentially absent. Where it does appear, it is usually in a verse that looks like a later addition, for example because it is much longer than its neighbours or breaks the rhyme.
- **AR-Qur'an**: for a short time *al-Raḥmān*, "the Merciful", is used as God's proper name (surah 19 uses it sixteen times). Then *Allāh* and *Rabb* appear about equally often.
- **A-Qur'an**: *Allāh* dominates, roughly once per verse or more. These are approximately the traditional Medinan surahs.

He then uses inscriptions to explain the shift: *Raḥmānān* was the name of God in Jewish and Christian inscriptions from South Arabia, while *Allāh* had a longer history in the north. Notebook 5 checks his counts and his rules, and asks whether other words change at the same boundaries.

### The sceptics

Not everyone accepts this picture. Gabriel Said Reynolds (2011) argues that the whole chronological approach takes its framework from the traditional biography of the Prophet, and that biography is itself partly an interpretation of the Qur'an, so the reasoning risks going in a circle. Stephen Shoemaker (2022) and Guillaume Dye argue that parts of the text may have taken shape after Muhammad's lifetime, which would loosen the link between "early/late in the text" and "early/late in Muhammad's career". You don't have to decide who is right. You do have to notice which assumptions your own analysis makes.

`state-of-the-field.md` covers all of this in much more detail, with full references.

## 4. What can a computer add?

### A classifier, in plain terms

A **classifier** is a program that learns to sort things into categories from examples. You give it thousands of verses, each labelled "Meccan" or "Medinan". It looks for features that tend to go with each label, such as particular words, lengths or endings, and gives each feature a **weight**. To classify a new verse, it adds up the weights of the features it finds there. The result is a probability, e.g. "72% Medinan".

The kind used most here is **logistic regression**. It is old, simple and transparent: you can list the weights and see exactly which features push a verse towards each label. The repository also tries a modern **neural language model**, CAMeLBERT-CA, trained on classical Arabic. It is a little more accurate and much harder to explain.

### Features: how a verse becomes numbers

The computer can't read. It needs each verse turned into numbers. The repository offers several ways, so you can ask which kind of evidence matters:

- **Length:** number of letters and words.
- **Style:** length plus the rhyme letter at the end of the verse, how often each of three names for God appears (*Allāh*, *rabb* "Lord", *al-Raḥmān* "the Merciful"), and two traditional forms of address ("O you who believe", "O people").
- **Words:** which words occur. Arabic attaches "and", "the" and pronouns to the word itself, so the repository also offers **lemmas** (dictionary forms, with those attachments split off) and **roots**. A root is the three-consonant core that carries a family of meanings: k-t-b gives *kataba* "he wrote", *kitāb* "book", *kātib* "scribe".
- **English:** the words of Pickthall's 1930 translation.

### Testing fairly

A model is only worth something if it works on material it hasn't seen. So we hide some surahs, train on the rest, and test on the hidden ones. Then we rotate until every surah has been hidden once. This is called **cross-validation**.

Two traps:

1. **Leakage.** If you hide individual *verses* rather than whole *surahs*, the model can recognise a surah from its neighbouring verses, its refrains and its rhyme, without learning anything about Mecca or Medina. Every test in this repository hides whole surahs.
2. **Imbalance.** About 74% of verses are labelled Meccan, so a "model" that always says Meccan is 74% accurate. We report **balanced accuracy** (where always-Meccan scores 50%) and **AUC** (where guessing scores 0.5 and perfection 1.0) instead.

### The biggest trap: circular labels

The classifier learns to reproduce its labels. If you train it on the Cairo labels and it agrees with them 95% of the time, it has *not* independently dated anything. It has learned to imitate an early classification. The interesting cases are the ones where it *disagrees*. Notice also that Nöldeke built his labels partly from style, so a style-based model agreeing with Nöldeke is not independent support for him either. One way out is to use methods that see no labels at all (notebook 3).

## 5. What the repository found

The details are in `results.md`. In brief:

1. **Verse length does most of the work.** A model that knows only each verse's length separates Meccan from Medinan surahs almost as well as models that know every word.
2. **The text sides with Nöldeke on the disputed surahs.** Every model fits Nöldeke's labels better than Cairo's, and the difference comes almost entirely from surahs 13, 55, 76 and 99. The 2017 version of this project reported those as model "errors". They were the model agreeing with Nöldeke.
3. **Without any labels**, the frequencies of the 100 commonest lemmas put the surahs in nearly the same order as Nöldeke (rank correlation 0.91).
4. **Fine-grained order is much shakier.** Inside Nöldeke's periods, the independent measures agree with his order only patchily. This matches the scholars' own caution that these methods give a coarse chronology, not a precise one.
5. **The tradition's single-verse exceptions get only weak support** from the models (AUC about 0.6).
6. **English translation works about as well as Arabic** for these coarse distinctions.
7. **Dost's three phases are more than naming habits.** With *Allāh*, *Rabb* and *al-Raḥmān* removed, a model still puts about 90% of surahs in Dost's phase from their other words. His "one *Allāh* per verse" rule cleanly separates his AR and A phases: no AR surah reaches it, and 19 of the 23 A surahs do.
8. **Two, three or four stages?** Measured by particle use (small grammatical words none of the schemes was built from), the biggest break falls *early*, between the earliest surahs (Nöldeke's Meccan I, Dost's R-Qur'an) and everything after, not at the move to Medina. Content vocabulary, on the other hand, marks the Meccan/Medinan divide very clearly. One way to put it: the hijra shows up in *what* the text talks about more than in *how* it talks. Three-stage schemes (Dost's, or Nöldeke's with Meccan II and III merged) beat the traditional two-stage one; whether Nöldeke's fourth stage is worth its extra complexity depends on which surahs you include.

## 6. Using the repository

### Setup (once)

1. Install **uv**, a tool that installs Python and all the packages this project needs. Follow the one-line instructions at https://docs.astral.sh/uv/getting-started/installation/.
2. Download this repository and open a terminal in its folder.
3. Run `uv sync`. This creates a private Python environment with exactly the right versions of everything. The first run takes a minute or two.
4. Run `uv run jupyter lab`. A browser tab opens.

The first time any code loads the data, it downloads the Quranic Arabic Corpus file (about 6 MB) into `data/cache/`. After that everything works offline.

### The notebooks, in order

Open the `notebooks/` folder in Jupyter and work through:

1. **`01_the_data.ipynb`**: the surah and verse tables, the two chronologies, and a first look at verse length.
2. **`02_classifying.ipynb`**: building and testing classifiers, leakage, which evidence matters, and the surahs where models and tradition disagree.
3. **`03_beyond_binary.ipynb`**: chronology without labels, the four periods, and long surahs cut into blocks.
4. **`04_neural_embeddings.ipynb`**: optional. Needs a 1.5 GB extra install: `uv sync --extra transformers`.
5. **`05_divine_names.ipynb`**: Prof. Dost's three phases. Checking his counts, his rules and his proposed insertions.
6. **`06_comparing_periodizations.ipynb`**: two, three or four stages? Comparing the competing schemes fairly.

Each notebook has questions in bold. Several ask you to read passages in English (https://quran.com is convenient) and come to your own judgement. The code points you to the interesting places; deciding what they mean is your job.

### If something breaks

Read the last line of the error first; it usually names the problem. "No module named surahs" means you started Jupyter without `uv run`. A network error on first run means the corpus download failed: check your connection and run the cell again.

## 7. Glossary

- **Āya / āyāt**: verse / verses of the Qur'an.
- **BIC (Bayesian information criterion)**: a score for comparing statistical models that rewards fit and penalises complexity; lower is better.
- **AUC**: a score from 0.5 (guessing) to 1.0 (perfect) for how well a model ranks one class above another.
- **Balanced accuracy**: the average of the percentage of each class correctly identified. Guessing the majority class scores 0.5.
- **Basmala**: the formula "In the name of God, the Merciful, the Compassionate", which opens every surah but one.
- **Classifier**: a program that learns to assign categories from labelled examples.
- **Cross-validation**: testing a model repeatedly on held-out parts of the data so that every item is tested once by a model that never saw it.
- **Embedding**: the list of numbers a neural network produces to represent a piece of text.
- **Feature**: one measurable property of a text that a model can use, e.g. the count of a word.
- **Hijra**: the emigration from Mecca to Medina, 622 CE.
- **Leakage**: when information about the test material slips into training, making results look better than they are.
- **Lemma**: the dictionary form of a word.
- **Logistic regression**: a simple, interpretable classifier that gives each feature a weight.
- **MVL (mean verse length)**: the average length of a surah's verses, in letters.
- **PCA (principal component analysis)**: a technique that finds the main directions along which items differ, without using any labels.
- **Root**: the (usually) three consonants that carry the core meaning of a family of Arabic words.
- **Spearman's rho**: a number from −1 to 1 measuring how similarly two lists rank the same items.
- **Surah**: a chapter of the Qur'an.

## 8. Where to read more

- Nicolai Sinai, *The Qur'an: A Historical-Critical Introduction* (Edinburgh University Press, 2017), chapter 5, "Inner-Qur'anic Chronology". The best single starting point.
- Behnam Sadeghi, "The Chronology of the Qurʾān: A Stylometric Research Program," *Arabica* 58 (2011): 210–299. Demanding but foundational.
- Gabriel Said Reynolds, "Le problème de la chronologie du Coran," *Arabica* 58 (2011): 477–502. The sceptical case (in French).
- Suleyman Dost, *Before the Qur'an: Material Sources at the Advent of Muslim Scripture* (Edinburgh University Press, 2026). Prof. Dost's book on the Qur'an's Arabian setting as seen through inscriptions. Chapter 3, "From Rabb to Allāh," is his chronology of the three phases; read it alongside notebook 5.
- Seyyed Hossein Nasr et al., eds., *The Study Quran* (HarperOne, 2015). An English translation whose introductions to each surah summarise what Islamic tradition says about where and when it was revealed.
