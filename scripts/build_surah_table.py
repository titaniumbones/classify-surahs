#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas"]
# ///
"""Build data/surahs.csv from the raw sources in data/raw/.

- Names, verse counts, the Cairo chronological order and the Meccan/Medinan
  label come from Tanzil's metadata file (tanzil-quran-data.xml), which
  reproduces the 1924 Cairo (King Fuad) edition.
- Nöldeke-Schwally's chronological order comes from the table on
  Wikipedia's "List of chapters in the Quran" (retrieved 2026-10-14), which
  cites Neal Robinson, Discovering the Qur'an (2003). The script checks that
  it is a permutation of 1-114 and that the Cairo order in the same table
  matches Tanzil's exactly, as a guard against transcription errors.

Run:  uv run scripts/build_surah_table.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

PERIODS = [("Meccan I", 1, 48), ("Meccan II", 49, 69), ("Meccan III", 70, 90), ("Medinan", 91, 114)]


def period(order: int) -> str:
    return next(name for name, lo, hi in PERIODS if lo <= order <= hi)


def main() -> None:
    root = ET.parse(RAW / "tanzil-quran-data.xml").getroot()
    tanzil = pd.DataFrame(
        [
            {
                "surah": int(s.get("index")),
                "name_ar": s.get("name"),
                "name_translit": s.get("tname"),
                "name_en": s.get("ename"),
                "n_verses": int(s.get("ayas")),
                "cairo_order": int(s.get("order")),
                "cairo_place": s.get("type"),
            }
            for s in root.iter("sura")
        ]
    )

    wp = pd.read_csv(RAW / "wikipedia-chapter-table.csv").iloc[:114]
    wp.columns = ["surah", "verses", "place", "egypt_order", "noldeke_order"]
    wp = wp.astype({"surah": int, "egypt_order": int, "noldeke_order": int})

    assert sorted(wp.noldeke_order) == list(range(1, 115)), "Nöldeke order is not a permutation"
    df = tanzil.merge(wp[["surah", "egypt_order", "noldeke_order"]], on="surah", validate="one_to_one")
    assert (df.egypt_order == df.cairo_order).all(), "Wikipedia's Cairo order disagrees with Tanzil"
    df = df.drop(columns="egypt_order")

    df["noldeke_period"] = df.noldeke_order.map(period)
    df["noldeke_place"] = df.noldeke_period.where(df.noldeke_period == "Medinan", "Meccan")
    df["labels_disagree"] = df.cairo_place != df.noldeke_place

    out = ROOT / "data" / "surahs.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} surahs)")
    print("Cairo vs Nöldeke disagree on:", df.loc[df.labels_disagree, "surah"].tolist())
    print(df.noldeke_period.value_counts().reindex([p for p, _, _ in PERIODS]).to_string())


if __name__ == "__main__":
    main()
