"""Shared typography, colors, and export settings for the study figures."""
from pathlib import Path

import matplotlib.pyplot as plt


DPI = 300
NAMES = ("SantoriniE", "NLL")
LABELS = {"SantoriniE": "SantoriniE", "NLL": "NLL-SSST"}
COLORS = {"SantoriniE": "#D47A35", "NLL": "#2C6E9F"}
MARKERS = {"SantoriniE": "o", "NLL": "s"}
INK = "#263640"
MUTED = "#60717D"
GRID = "#E4E9ED"
MODEL_COLORS = {"line": "#45535D", "quadratic": "#267F78", "piecewise": "#7A5598"}

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#92A0AA",
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.linewidth": 0.7,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "legend.handlelength": 2,
    "legend.handletextpad": 0.7,
    "legend.columnspacing": 1.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 120,
    "savefig.dpi": DPI,
    "lines.linewidth": 1.5,
    "mathtext.default": "regular",
}


def style_axes(ax, grid="both", boxed=False):
    ax.grid(visible=True, axis=grid, which="major", color=GRID, linewidth=0.6)
    ax.tick_params(pad=4)
    for name, spine in ax.spines.items():
        spine.set_visible(boxed or name in ("left", "bottom"))
        spine.set_linewidth(0.7)
        spine.set_color("#92A0AA")


def figure_heading(fig, title, subtitle, left=0.105):
    fig.text(left, 0.965, title, ha="left", va="top",
             fontsize=14, fontweight="semibold", color=INK)
    fig.text(left, 0.907, subtitle, ha="left", va="top",
             fontsize=9.2, color=MUTED)


def save_figure(fig, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=DPI, facecolor="white")
    plt.close(fig)
