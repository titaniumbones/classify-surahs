"""Rough Latin transliteration of Arabic lemmas and roots.

Good enough to let a reader who doesn't know the Arabic alphabet tell two
features apart and look them up; it is not a scholarly transliteration
(long vowels and case endings are not reconstructed from unvowelled text).
"""

_MAP = {
    "ا": "a", "أ": "ʾ", "إ": "ʾ", "آ": "ʾā", "ٱ": "a", "ء": "ʾ", "ؤ": "ʾ", "ئ": "ʾ",
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "ḥ", "خ": "kh", "د": "d", "ذ": "dh",
    "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "ṣ", "ض": "ḍ", "ط": "ṭ", "ظ": "ẓ",
    "ع": "ʿ", "غ": "gh", "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "ة": "", "و": "w", "ي": "y", "ى": "ā",
    # vowel marks in the Quranic Arabic Corpus lemmas
    "َ": "a", "ُ": "u", "ِ": "i", "ً": "an", "ٌ": "un", "ٍ": "in",
    "ٰ": "ā", "ْ": "", "ـ": "",
}


def translit(word: str) -> str:
    out = []
    last_consonant = None  # index in `out` of the most recent letter
    for ch in word:
        if ch == "ّ":  # shadda: double the most recent consonant
            if last_consonant is not None:
                out.insert(last_consonant + 1, out[last_consonant])
            continue
        if "ء" <= ch <= "ي" or ch == "ٱ":
            last_consonant = len(out)
        out.append(_MAP.get(ch, ch))
    s = "".join(out)
    if s in ("alllah", "allah"):
        return "allāh"
    # a short vowel followed by its matching letter is a long vowel
    s = s.replace("iy", "ī").replace("uw", "ū").replace("aa", "ā").replace("aā", "ā")
    if word.endswith("ة") and not s.endswith("a"):
        s += "a"
    return s[:-1] if s.endswith("ūa") else s  # silent alif after plural -ū
