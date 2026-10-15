#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Render the four docs/*.md files into one self-contained HTML page.

Each document becomes a section; links between the docs become in-page
anchors; result figures are embedded as data URIs. Needs pandoc on PATH.

Run:  uv run scripts/build_site.py OUTPUT.html
"""

import base64
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    ("guide", "guide-for-students.md", "Student guide"),
    ("field", "state-of-the-field.md", "State of the field"),
    ("results", "results.md", "Results"),
    ("changes", "what-changed.md", "What changed since 2017"),
]
# figure inserted after the heading whose text starts with the key
FIGURES = {
    "results": {
        "3. Where models": [("surah_proba_lemmas_cairo.png", "Each surah's probability of being Medinan, from a model trained on the Cairo labels that never saw that surah. Surahs are placed in Nöldeke's order.")],
        "5. Chronology without labels": [
            ("mvl_vs_noldeke.png", "Mean verse length against Nöldeke's rank. The four surahs Cairo and Nöldeke disagree on are labelled."),
            ("pca_vs_noldeke.png", "The first principal component of common-lemma frequencies, computed with no labels, against Nöldeke's rank."),
        ],
        "7. Dost": [("dost_allah_per_verse.png", "Allāh per verse along Sinai's verse-length ranking, coloured by Dost's phase. The dashed line marks one Allāh per verse.")],
    }
}
REPO = "https://github.com/titaniumbones/classify-surahs"


def md_to_html(path: Path) -> str:
    return subprocess.run(
        ["pandoc", "-f", "gfm", "-t", "html5", "--wrap=none", "--no-highlight", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout


def img_uri(name: str) -> str:
    data = (ROOT / "results" / "figures" / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def process(key: str, html: str) -> str:
    # demote headings one level so each document's h1 becomes the section title (h2)
    for n in range(5, 0, -1):
        html = re.sub(rf"<(/?)h{n}([ >])", rf"<\1h{n + 1}\2", html)
    # prefix heading ids so they are unique across documents
    html = re.sub(r'id="([^"]+)"', rf'id="{key}-\1"', html)
    # links to sibling docs -> section anchors; repo-relative links -> GitHub
    for k, fname, _ in DOCS:
        html = html.replace(f'href="{fname}"', f'href="#{k}"').replace(f'href="docs/{fname}"', f'href="#{k}"')
    html = re.sub(r'href="(?!https?:|#|mailto:)([^"]+)"', rf'href="{REPO}/blob/master/\1"', html)
    # wrap tables so they scroll on phones
    html = html.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    for start, figs in FIGURES.get(key, {}).items():
        m = re.search(rf"(<h3[^>]*>{re.escape(start)}.*?</h3>)", html)
        if not m:
            raise SystemExit(f"figure anchor not found: {start}")
        block = "".join(
            f'<figure class="plate"><img src="{img_uri(f)}" alt="{c}" loading="lazy"><figcaption>{c}</figcaption></figure>'
            for f, c in figs
        )
        html = html[: m.end()] + block + html[m.end():]
    return html


def main(out: str) -> None:
    template = (ROOT / "scripts" / "site_template.html").read_text()
    nav = "".join(f'<a href="#{k}">{label}</a>' for k, _, label in DOCS)
    sections = "".join(
        f'<section class="doc" id="{k}" aria-label="{label}"><p class="eyebrow">{label}</p>{process(k, md_to_html(ROOT / "docs" / f))}</section>'
        for k, f, label in DOCS
    )
    Path(out).write_text(template.replace("{{NAV}}", nav).replace("{{SECTIONS}}", sections).replace("{{REPO}}", REPO))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "site.html")
