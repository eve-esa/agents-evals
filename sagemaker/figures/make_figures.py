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
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ks = list(range(1, 9))
    for p, colour in ((0.3, RED), (0.5, AMBER), (0.7, GREEN)):
        ax.plot(ks, [1 - (1 - p) ** k for k in ks], marker="o", ms=4.5, lw=2, color=colour,
                label=f"succeeds {int(p * 100)}% of the time per attempt")
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


def pass_at_k_tree():
    """Tree of possible traces for one task. Edge probabilities are illustrative; the three drawn attempts are one sample."""
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(-0.3, 12.6)
    ax.set_ylim(-1.5, 5.9)
    ax.axis("off")
    ax.set_title("One task, many possible traces: Pass@k asks whether k sampled traces include a correct one",
                 loc="left", fontsize=11, fontweight="bold")
    X = [0.0, 3.3, 6.9, 10.6]
    W, H = [2.0, 2.6, 3.0, 1.9], 0.78
    nodes = {  # name: (depth, y, text, probability from parent)
        "root": (0, 2.25, "Question 11", None),
        "A": (1, 3.75, "calls get_filelist", 0.65),
        "B": (1, 0.75, "calls lst_multi_channel\nwith a guessed path", 0.35),
        "A1": (2, 4.5, "lst_multi_channel with\nthe right paths", 0.85),
        "A2": (2, 3.0, "answers without\ncomputing", 0.15),
        "B1": (2, 1.5, "reads the error, calls\nget_filelist, retries", 0.70),
        "B2": (2, 0.0, "gives up", 0.30),
        "L1": (3, 5.0, "answer C", 0.90),
        "L2": (3, 4.0, "answer wrong", 0.10),
        "L3": (3, 3.0, "answer wrong", 1.0),
        "L4": (3, 2.0, "answer C", 0.75),
        "L5": (3, 1.0, "answer wrong", 0.25),
        "L6": (3, 0.0, "answer wrong", 1.0),
    }
    parent = {"A": "root", "B": "root", "A1": "A", "A2": "A", "B1": "B", "B2": "B",
              "L1": "A1", "L2": "A1", "L3": "A2", "L4": "B1", "L5": "B1", "L6": "B2"}
    correct = {"L1", "L4"}
    prob = {}

    def reach(name):
        if name not in prob:
            prob[name] = 1.0 if name == "root" else reach(parent[name]) * nodes[name][3]
        return prob[name]

    for n in nodes:
        reach(n)
    # edges
    for child, par in parent.items():
        d, y, _, pr = nodes[child]
        pd_, py, _, _ = nodes[par]
        x0, x1 = X[pd_] + W[pd_], X[d]
        ax.plot([x0, x1], [py + H / 2, y + H / 2], color=GRID, lw=1.6, zorder=1)
        if pr < 1.0:
            t = 0.62
            ax.text(x0 + t * (x1 - x0), py + H / 2 + t * (y - py), f"{pr:.2f}", fontsize=8, color=INK, ha="center", va="center",
                    bbox=dict(fc="white", ec=GRID, boxstyle="round,pad=0.18"), zorder=6)
    # sampled attempts, drawn over the tree
    attempts = [("1", ["root", "B", "B2", "L6"], BLUE), ("2", ["root", "B", "B1", "L4"], "#8250df"),
                ("3", ["root", "A", "A1", "L1"], AMBER)]
    for label, path, colour in attempts:
        off = {"1": -0.09, "2": 0.0, "3": 0.09}[label]
        xs, ys = [], []
        for n in path:
            d, y, *_ = nodes[n]
            xs += [X[d] + W[d] / 2 + off * 4 if False else X[d] + W[d] * 0.5, ]
            ys += [y + H / 2 + off]
        for a_, b_ in zip(path, path[1:]):
            da, ya, *_ = nodes[a_]
            db, yb, *_ = nodes[b_]
            ax.plot([X[da] + W[da], X[db]], [ya + H / 2 + off, yb + H / 2 + off], color=colour, lw=3.2, alpha=0.9, zorder=3)
        d, y, *_ = nodes[path[-1]]
        ax.text(X[d] + W[d] + 0.25, y + H / 2, f"attempt {label}", fontsize=9, fontweight="bold", color=colour, va="center")
    # nodes
    for name, (d, y, text, pr) in nodes.items():
        leaf = name.startswith("L")
        if leaf:
            ok = name in correct
            face, edge, ink = ("#dafbe1", GREEN, GREEN) if ok else ("#ffebe9", RED, RED)
            box(ax, X[d], y, W[d], H, ("\u2713 " if ok else "\u2717 ") + text, face, edge, size=8.8, color=ink, weight="bold")
        else:
            box(ax, X[d], y, W[d], H, text, "#f6f8fa" if name != "root" else "#fff", GRID if name != "root" else INK,
                size=8.4, weight="bold" if name == "root" else "normal")
    p_ok = sum(prob[n] for n in correct)
    ps = [1 - (1 - p_ok) ** k for k in (1, 2, 3)]
    ax.text(0.0, -0.55, f"Chance that one attempt ends correct: {p_ok:.2f}  (sum over the green leaves; edge numbers are illustrative).",
            fontsize=9.5, color=INK)
    ax.text(0.0, -1.0, f"Expected Pass@1 = {ps[0]:.2f}     Pass@2 = {ps[1]:.2f}     Pass@3 = {ps[2]:.2f}",
            fontsize=10.5, fontweight="bold", color=GREEN)
    ax.text(0.0, -1.4, "This draw: attempt 1 gives up (wrong), attempts 2 and 3 are correct, so this task scores Pass@1 = 0, Pass@2 = 1, Pass@3 = 1.",
            fontsize=9.2, color=MUTED)
    fig.savefig(OUT / "pass_at_k_tree.png", dpi=170, bbox_inches="tight")
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
    pass_at_k_tree()
    run_anatomy()
    print("wrote", OUT / "pass_at_k.png", OUT / "run_anatomy.png")
