"""One look for ALF's figures (progress plots and run reports), sized to read on a screen at a glance.

A figure has a title and a short line on how to read it (or, for one panel, the panel's own title and a caption
underneath); each panel names what it shows (bold, left) and states its key numbers (muted, right). Lines are
labeled at their ends rather than in legend boxes. Margins are fixed in inches, so figures line up whatever their
number of panels. Colors come from the reference categorical palette, each in its fixed order.
"""
import textwrap

import numpy as np
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SHADE = "#ecebe6"
# The reference categorical palette, in its fixed order: series take colors in this order, never cycled.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100", "#e87ba4", "#24916b", "#898781")
WIDTH = 10.0
OVERFLOW = "#898781"


def series_colors(items):
    """``(color, item)`` for each item: the palette in its fixed order for the first eight, then a neutral gray for
    the rest, which keep their end labels, so no series is dropped and no color is made up."""
    return [(CATEGORICAL[index] if index < len(CATEGORICAL) else OVERFLOW, item) for index, item in enumerate(items)]

# Text sizes, in points, for every figure.
TITLE, SUBTITLE, PANEL, NUMBERS, LABEL, TICK, NOTE = 15, 10.5, 12.5, 10.5, 11, 10, 9.5

RC = {"font.size": LABEL, "axes.titlesize": PANEL, "axes.labelsize": LABEL, "xtick.labelsize": TICK,
      "ytick.labelsize": TICK, "legend.fontsize": NOTE, "axes.titleweight": "bold", "axes.titlecolor": TEXT,
      "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
      "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
      "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
      "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
      "lines.solid_capstyle": "round"}

SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of", "on", "or", "over", "per",
               "the", "to", "vs", "with"}


def title_case(text):
    """``text`` in Title Case: each word capitalized except short joining words after the first; words that
    already hold capitals or digits (QM, GPU, E0_RMSE, ps) are left as they are."""
    if not text:
        return text
    words = text.split(" ")
    out = []
    for index, word in enumerate(words):
        bare = word.strip("(),:;")
        if not bare or any(character.isupper() or character.isdigit() for character in bare) or \
                bare in ("ps", "eV", "eV/Å", "s", "min", "h", "GiB", "GB"):
            out.append(word)
        elif index and bare.lower() in SMALL_WORDS:
            out.append(word.lower())
        else:
            out.append(word.replace(bare, bare[:1].upper() + bare[1:], 1))
    return " ".join(out)


def figure(rows, title=None, subtitle=None, caption=None, height=2.0, width=WIDTH, right=1.7):
    """``rows`` stacked panels, with a title and a how-to-read line above them, or a caption below.

    Returns the figure and its axes as a list. ``right`` is the margin, in inches, for labels at line ends.
    """
    import matplotlib.pyplot as plt
    characters = int(11.5 * width)  # A line of the how-to-read text or caption, at its size.
    lines = textwrap.wrap(subtitle, characters) if subtitle else []
    notes = textwrap.wrap(caption, characters) if caption else []
    line = 0.21  # inches per line of that text
    top = (0.55 if title else 0.1) + line * len(lines) + 0.62  # the first panel's title and numbers
    gap, bottom, left = 1.05, 0.7 + line * len(notes), 1.3
    total = top + rows * height + (rows - 1) * gap + bottom
    fig, axes = plt.subplots(rows, 1, figsize=(width, total), squeeze=False)
    fig.subplots_adjust(left=left / width, right=1 - right / width, top=1 - top / total, bottom=bottom / total,
                        hspace=gap / height)
    if title:
        fig.text(0.12 / width, 1 - 0.18 / total, title_case(title), ha="left", va="top", fontsize=TITLE,
                 fontweight="bold", color=TEXT)
    if lines:
        fig.text(0.12 / width, 1 - (0.58 if title else 0.12) / total, "\n".join(lines), ha="left", va="top",
                 fontsize=SUBTITLE, color=MUTED, linespacing=1.35)
    if notes:
        fig.text(0.12 / width, 0.12 / total, "\n".join(notes), ha="left", va="bottom", fontsize=SUBTITLE,
                 color=MUTED, linespacing=1.35)
    return fig, list(axes[:, 0])


def panel(axis, title, numbers=None, ylabel=None, xlabel=None):
    """Name a panel (bold, in Title Case) with its key numbers on their own line beneath (muted), so the two never
    collide, and title its axes in Title Case."""
    axis.set_title(title_case(title), loc="left", pad=NUMBERS * 1.9 if numbers else 6)
    if numbers:
        axis.annotate(numbers, (0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                      ha="left", va="bottom", fontsize=NUMBERS, color=MUTED, annotation_clip=False)
    if ylabel:
        axis.set_ylabel(title_case(ylabel))
    if xlabel:
        axis.set_xlabel(title_case(xlabel))


def label_end(axis, x, y, text, color, dy=0):
    """Label a line at its last finite point, instead of a legend."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    finite = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if finite.size:
        axis.annotate(text, (x[finite[-1]], y[finite[-1]]), xytext=(5, dy), textcoords="offset points",
                      va="center", fontsize=NOTE, color=color, annotation_clip=False)


def label_ends(axis, series, gap=12.0):
    """Label several lines at their last finite points, pushed apart vertically so no two labels overlap.

    ``series``: ``(x, y, text, color)`` for each line. ``gap``: the least distance between labels, in points."""
    ends = []
    for x, y, text, color in series:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        finite = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
        if finite.size:
            ends.append((x[finite[-1]], y[finite[-1]], text, color))
    if not ends:
        return
    axis.figure.canvas.draw_idle()
    to_points = 72.0 / axis.figure.dpi
    heights = [axis.transData.transform((x, y))[1] * to_points for x, y, _, _ in ends]
    order = np.argsort(heights)
    placed = np.array(heights, dtype=float)[order]
    for index in range(1, len(placed)):  # Push each label up past the one below it.
        placed[index] = max(placed[index], placed[index - 1] + gap)
    placed -= max(0.0, (placed[-1] - np.array(heights)[order][-1]) / 2)  # Then center the stack on the lines.
    for rank, index in enumerate(order):
        x, y, text, color = ends[index]
        axis.annotate(text, (x, y), xytext=(5, placed[rank] - heights[index]), textcoords="offset points",
                      va="center", fontsize=NOTE, color=color, annotation_clip=False)


def thousands(axis, which="y"):
    """Write an axis's tick values with thousands separators."""
    formatter = FuncFormatter(lambda value, _: f"{value:,.0f}" if abs(value) >= 1000 else f"{value:g}")
    (axis.yaxis if which == "y" else axis.xaxis).set_major_formatter(formatter)


def save(fig, path, dpi=130):
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
