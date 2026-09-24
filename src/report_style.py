"""Shared chart style for the report, so every figure reads as one system.

Colors follow a fixed assignment: an entity (e.g. China) always has the same color in
every figure, and anything outside the named series is neutral gray. The categorical
hues come from a colorblind-validated palette.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
GRAY = "#b5b4ae"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

BLOC_COLORS = {
    "United States": BLUE,
    "China": ORANGE,
    "EU27": AQUA,
    "United Kingdom": YELLOW,
}
SEQUENTIAL_BLUE = ["#dbe8f8", "#a9c8ee", "#6fa2e2", "#2a78d6", "#1d56a0", "#123a6e"]


def apply() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "text.color": INK,
    })


def direct_label(ax, x, y, text, color=INK_2, dx=0.3, **kw):
    """Label a line at its end, in text ink (never the series color), next to the mark."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset fontsize",
                va="center", fontsize=8, color=color, **kw)


def pct_axis(ax, axis="y", decimals=0):
    fmt = mpl.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def source_note(fig, text="Source: OpenAlex; analysis by the author."):
    fig.text(0.0, -0.02, text, fontsize=7, color=MUTED, ha="left", va="top")


def new(figsize=(6.5, 3.2), **kw):
    apply()
    return plt.subplots(figsize=figsize, **kw)


def year_axis(ax, step=None):
    """Whole-number year ticks (matplotlib otherwise shows 2012.5)."""
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, nbins=6 if step is None else "auto"))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v)}"))


def spread(values, min_gap):
    """Nudge label positions apart so no two are closer than min_gap, keeping their order."""
    import numpy as np
    vals = np.asarray(values, dtype=float)
    order = np.argsort(vals)
    out = vals[order].copy()
    for _ in range(50):
        moved = False
        for i in range(1, len(out)):
            if out[i] - out[i - 1] < min_gap:
                mid = (out[i] + out[i - 1]) / 2
                out[i - 1], out[i] = mid - min_gap / 2, mid + min_gap / 2
                moved = True
        if not moved:
            break
    result = np.empty_like(out)
    result[order] = out
    return result


def label_line_ends(ax, items, min_gap_frac=0.06):
    """items: list of (x_end, y_end, text). Labels are placed at line ends without overlapping."""
    lo, hi = ax.get_ylim()
    ys = spread([y for _, y, _ in items], (hi - lo) * min_gap_frac)
    for (x, _, text), y in zip(items, ys):
        direct_label(ax, x, y, text)
