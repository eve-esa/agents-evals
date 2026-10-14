"""Regenerate the figures the workshop notebook embeds: ``python figures/make_figures.py`` from this folder.

Pass@k curves use 1 - (1 - p)^k, the Pass@k of an agent that succeeds with
probability p on every independent attempt (Chen et al., 2021, with n -> infinity).
The run anatomy is the real trace of Earth-Bench question 11 from an EVE run.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
GREEN, RED, AMBER, BLUE, GREY = "#2ea043", "#d1242f", "#bf8700", "#0969da", "#8c959f"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": "white", "axes.facecolor": "white"})


def box(ax, x, y, w, h, text, face, edge, size=9.5, color=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=face, ec=edge, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, fontweight=weight)


def pass_at_k():
    fig = plt.figure(figsize=(11, 4.1))
    ax = fig.add_axes([0.02, 0.08, 0.40, 0.80])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("One task, three independent attempts", loc="left", fontsize=11, fontweight="bold")
    attempts = [("attempt 1", "wrong", RED), ("attempt 2", "correct", GREEN), ("attempt 3", "correct", GREEN)]
    for i, (name, verdict, colour) in enumerate(attempts):
        y = 4.5 - i * 1.55
        ax.text(0.1, y + 0.4, name, fontsize=10, color=MUTED, va="center")
        box(ax, 2.0, y, 2.6, 0.85, verdict, "#fff", colour, color=colour, weight="bold")
    rows = [("Pass@1", "0", "the first attempt is wrong", RED), ("Pass@2", "1", "one of the first two is right", GREEN),
            ("Pass@3", "1", "one of the three is right", GREEN)]
    for i, (name, value, why, colour) in enumerate(rows):
        y = 4.5 - i * 1.55
        ax.text(5.1, y + 0.62, f"{name} = {value}", fontsize=11, fontweight="bold", color=colour, va="center")
        ax.text(5.1, y + 0.18, why, fontsize=8.6, color=MUTED, va="center")
    ax.text(0.1, 0.05, "EO-Gym's ordered early stop: stop at the first correct attempt.", fontsize=8.6, color=MUTED)

    ax = fig.add_axes([0.52, 0.16, 0.46, 0.70])
    ks = list(range(1, 9))
    for p, colour in ((0.3, RED), (0.5, AMBER), (0.7, GREEN)):
        ax.plot(ks, [1 - (1 - p) ** k for k in ks], marker="o", ms=4.5, lw=2, color=colour, label=f"succeeds {int(p * 100)}% of the time per attempt")
    ax.text(1.15, 0.06, "k = 1 is Pass@1: how reliable one try is", fontsize=8.6, color=MUTED)
    ax.set_title("Pass@k grows with k, and the gap shows inconsistency", loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("k, attempts allowed")
    ax.set_ylabel("Pass@k")
    ax.set_ylim(0, 1.04)
    ax.set_xticks(ks)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.6, loc="lower right")
    fig.savefig(OUT / "pass_at_k.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_anatomy():
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 13.9)
    ax.set_ylim(0, 5.0)
    ax.axis("off")
    ax.set_title("Cost and reliability of one run: Earth-Bench question 11", loc="left", fontsize=11, fontweight="bold")
    w, h, step = 2.6, 0.85, 2.9
    ax.text(0.0, 4.0, "Expert", fontsize=10, color=MUTED, va="center")
    for i, name in enumerate(["get_filelist", "lst_multi_channel", "calculate_threshold_ratio"]):
        box(ax, 1.4 + i * step, 3.6, w, h, name, "#f6f8fa", GRID, size=8.6)
    ax.text(0.0, 2.4, "Agent", fontsize=10, color=MUTED, va="center")
    agent = [("lst_multi_channel", RED, "error: file not found"), ("get_filelist", GREEN, ""), ("lst_multi_channel", GREEN, ""),
             ("calculate_threshold_ratio", GREEN, "")]
    for i, (name, colour, note) in enumerate(agent):
        x = 1.4 + i * step
        box(ax, x, 2.0, w, h, name, "#fff", colour, size=8.6)
        if note:
            ax.text(x + w / 2, 1.75, note, ha="center", fontsize=8.4, color=RED)
    box(ax, 13.15, 2.0, 0.55, h, "C", "#dafbe1", GREEN, size=12, color=GREEN, weight="bold")
    ax.annotate("", xy=(13.1, 2.42), xytext=(12.8, 2.42), arrowprops=dict(arrowstyle="->", color=MUTED))
    cards = [
        ("Efficiency", "4 / 3 = 1.33", "calls made vs expert calls;\nabove 1 means detours", BLUE),
        ("Tool errors", "1", "calls that raised an error;\nthe agent recovered", RED),
        ("Termination", "completed", "not recursion_limit,\nnot max_model_calls", GREEN),
        ("Answer", "C, correct", "but TEM = 0: the path\nwas not the expert's", GREY),
    ]
    for i, (title, value, note, colour) in enumerate(cards):
        x = 0.1 + i * 3.2
        box(ax, x, 0.05, 3.0, 1.2, "", "#fff", GRID)
        ax.text(x + 0.15, 1.02, title, fontsize=8.8, color=MUTED, va="center")
        ax.text(x + 0.15, 0.72, value, fontsize=12, fontweight="bold", color=colour, va="center")
        ax.text(x + 0.15, 0.33, note, fontsize=7.6, color=MUTED, va="center", linespacing=1.3)
    fig.savefig(OUT / "run_anatomy.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pass_at_k()
    run_anatomy()
    print("wrote", OUT / "pass_at_k.png", OUT / "run_anatomy.png")
