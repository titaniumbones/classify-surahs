"""Shared chart styling.

Nöldeke's three Meccan periods are ordered, so they get three steps of one
blue (light = early, dark = late); Medinan gets a contrasting orange. Each
period also has its own marker shape, so the charts still read in greyscale
or for colour-blind readers.
"""

PERIOD_STYLE = {
    "Meccan I": dict(color="#86b6ef", marker="o", s=28, edgecolor="white", linewidth=0.6),
    "Meccan II": dict(color="#2a78d6", marker="s", s=26, edgecolor="white", linewidth=0.6),
    "Meccan III": dict(color="#104281", marker="D", s=24, edgecolor="white", linewidth=0.6),
    "Medinan": dict(color="#eb6834", marker="^", s=34, edgecolor="white", linewidth=0.6),
}


def style_axes(ax):
    """Recessive grid and spines so the data marks carry the chart."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8a8984")
    ax.tick_params(colors="#52514e")
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    return ax
