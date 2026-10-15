# The 2017 version

These are the files from the original project by Abubakar Abid (January 2017), kept for comparison:

- `main.ipynb` and `README-2017.md`: the original notebook and its write-up.
- `verses.txt`: the Tanzil text the notebook used (basmala still attached to verse 1 of each surah).
- `verses.csv`: the same text with the Arabic garbled into `????` by an encoding error. It was never used.
- `surah-labels.csv`: the Cairo edition's Meccan (1) / Medinan (0) labels.

The notebook expects these files under `data/`, and it was written for Python 3 with the scikit-learn of 2017. It will not run unmodified today. `src/surahs/experiments.py` (`replicate_2017`) reproduces its method with current libraries.

What the rebuilt project changes, and why, is explained in `docs/what-changed.md`.
