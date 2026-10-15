# Data sources

Everything the code reads is in this folder, except two things downloaded on first use into `data/cache/` (which git ignores): the Quranic Arabic Corpus morphology, and the optional neural model in notebook 4.

## `raw/` — verbatim downloads (do not edit)

| File | What it is | Source | Licence / terms | Retrieved |
|---|---|---|---|---|
| `tanzil-quran-simple-clean.txt` | Arabic text of the Qur'an, one verse per line (`surah\|ayah\|text`), no vowel marks, Kufan numbering (6,236 verses) | Tanzil Project, "Simple Clean" text v1.1, https://tanzil.net/download/ | Creative Commons Attribution 3.0; verbatim copies only, source must be credited with a link to tanzil.net. The copyright block at the end of the file must be kept. | 2026-10-14 |
| `tanzil-quran-data.xml` | Surah metadata: names, verse counts, the Cairo edition's chronological order and Meccan/Medinan label | Tanzil Project metadata v1.0, https://tanzil.net/res/text/metadata/quran-data.xml | CC BY | 2026-10-14 |
| `tanzil-en-pickthall.txt` | English translation by Marmaduke Pickthall (*The Meaning of the Glorious Koran*, 1930) | Tanzil Project translations, https://tanzil.net/trans/ | Pickthall died in 1936; the translation is in the public domain in Canada. Tanzil's digitisation is credited in the file footer. | 2026-10-14 |
| `qac-morphology-README.md` | Notes on the corrected Quranic Arabic Corpus file that the code downloads (see below) | https://github.com/mustafa0x/quran-morphology | — | 2026-10-14 |
| `tanzil-revelation-order.csv` | Tanzil's revelation-order table, including a "Note" column listing verse-level exceptions | https://tanzil.net/docs/revelation_order (HTML table, parsed with pandas, otherwise unaltered) | Tanzil terms as above | 2026-10-14 |
| `wikipedia-chapter-table.csv` | Cairo and Nöldeke chronological ranks for all 114 surahs | Table in Wikipedia, "List of chapters in the Quran", which cites Neal Robinson, *Discovering the Qur'an: A Contemporary Approach to a Veiled Text*, 2nd ed. (Georgetown University Press, 2003) | CC BY-SA 4.0 | 2026-10-14 |

## Downloaded on first use: Quranic Arabic Corpus morphology

`surahs.data.ensure_morphology()` downloads the word-by-word analysis (segments, part of speech, lemma, root) to `data/cache/qac-morphology.txt.gz` the first time it is needed. The source is the Quranic Arabic Corpus v0.4 by Kais Dukes (University of Leeds, https://corpus.quran.com), in the corrected Arabic-script version maintained at https://github.com/mustafa0x/quran-morphology, pinned to commit `8f38b39` (2018) so every user gets identical data. The corpus is released under the GNU GPL, but its header also allows only verbatim copies, and the corrected version is a modified copy with no licence of its own. So the repository does not redistribute it. The corpus's official download (https://corpus.quran.com/download/) requires an email address.

Sadeghi (2011) also used the Quranic Arabic Corpus segmentation for his morpheme counts.

## `verse_exceptions.csv` — hand-coded

The verse-level exceptions from the "Note" column of `raw/tanzil-revelation-order.csv`, one row per note, coded by hand: 35 surahs, 157 verses. Tanzil says these notes reproduce the traditional order of revelation (narrations going back to Ibn ʿAbbās, with details from al-Zanjānī's *History of the Qurʾān*). They match the kind of exception lists printed in the 1924 Cairo edition's surah headings; this has not been checked against a physical copy. Coding decisions:

- "Except X, from Medina" in a Meccan surah → verses X are Medinan. "Except X from Mecca" in a Medinan surah → Meccan.
- "Except the last three verses" (16) → 126–128; "Except last two verses" (9) → 128–129; "Only 1-3 from Mecca; the rest from Medina" (107) → 4–7 Medinan.
- Notes naming a place on the way or outside both cities (28:85 at Juhfa during the hijra; 47:13 during the hijra; 22:52–55 between Mecca and Medina; 2:281 at Mina; 5:3 at Arafat; 48 and 110 on journeys) are **not** coded as exceptions, because they don't move a verse from one period to the other.

## `surahs.csv` — built from the above

Built by `uv run scripts/build_surah_table.py`. The script checks that the Wikipedia table's Cairo order matches Tanzil's exactly (it does, for all 114 surahs) and that Nöldeke's ranks are a permutation of 1–114. Nöldeke's ranks are divided into his four periods: Meccan I = ranks 1–48, Meccan II = 49–69, Meccan III = 70–90, Medinan = 91–114.

The Cairo and Nöldeke labels disagree on four surahs: 13, 55, 76 and 99. The Cairo edition calls them Medinan; Nöldeke dates them to Mecca.

## Known limitations

- **Surah-level labels by default.** `cairo_place` gives every verse its surah's label. `cairo_place_verse` applies the hand-coded exceptions above. Nöldeke's labels are surah-level only, although he too discussed individual verses as later insertions.
- **One verse-numbering system.** Early Muslim centres counted verses differently (Kufan, Basran, Damascene, Medinan and Meccan counts). We use the Kufan count of the Cairo edition throughout. Since verse length is a key variable, a different count would shift some numbers slightly.
- **Basmala.** Tanzil prepends the basmala to verse 1 of surahs 2–114 (except 9). `surahs.data.strip_basmala` removes it in memory; the raw file is left untouched as Tanzil's terms require.
- **Nöldeke's order via a secondary source.** The ranks come from Robinson's tabulation as reproduced on Wikipedia, not directly from *Geschichte des Qorāns*. Spot-checks against the period lists in the Nöldeke–Schwally English translation (Brill, 2013) are a good exercise.
