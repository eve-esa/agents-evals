"""Regenerate the figures the workshop notebook embeds: ``python figures/make_figures.py`` from this folder.

The styling matches the trace viewer in ``workshop.py`` (``compare_html``): the same status
colours, rounded cards with a coloured left bar, monospace tool names, and muted labels.

The Pass@k tree uses illustrative edge probabilities; Pass@k = 1 - (1 - p)^k.
The run anatomy is the real trace of Earth-Bench question 11 from an EVE run.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent

# Same palette as workshop._STATUS and the viewer's card styles.
INK, MUTED, RULE = "#1f2328", "#59636e", "#d0d7de"
CARD, CARD_EDGE = "#f6f8fa", "#d0d7de"
GREEN, AMBER, RED, GREY, PURPLE, BLUE = "#2ea043", "#d29922", "#f85149", "#8b949e", "#8250df", "#0969da"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
MONO = {"family": "DejaVu Sans Mono"}


def card(ax, x, y, w, h, bar=None, fill=CARD, edge=CARD_EDGE, lw=1.0):
    """A rounded card with an optional coloured left bar, the viewer's ``border-left``."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fill, ec=edge, lw=lw, zorder=2))
    if bar:
        ax.add_patch(Rectangle((x, y + 0.14), 0.08, h - 0.28, color=bar, lw=0, zorder=3))


def dot(ax, x, y, colour, glyph, r=0.24):
    """The round status marker the viewer draws between expert and agent."""
    ax.add_patch(Circle((x, y), r, fc=colour, ec="none", zorder=4))
    ax.text(x, y - 0.01, glyph, ha="center", va="center", color="white", fontsize=10, fontweight="bold", zorder=5)


def pill(ax, x, y, text, colour=INK):
    """A small bordered label, like the viewer's score chips."""
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color=colour, zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=RULE, lw=0.9))


def pass_at_k_tree():
    """Tree of possible traces for one task. Edge probabilities are illustrative; the three drawn attempts are one sample."""
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.set_xlim(-0.3, 14.0)
    ax.set_ylim(-2.1, 5.9)
    ax.axis("off")
    ax.text(0.0, 5.55, "One task, many possible traces", fontsize=13, fontweight="bold", color=INK, va="center")
    ax.text(0.0, 5.12, "Pass@k asks whether k sampled traces include a correct one.", fontsize=9.5, color=MUTED, va="center")

    X = [0.0, 3.3, 6.9, 10.6]
    W, H = [2.0, 2.6, 3.0, 2.1], 0.8
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

    # Edges, with their probability as a bordered label.
    for child, par in parent.items():
        d, y, _, pr = nodes[child]
        pd_, py, _, _ = nodes[par]
        x0, x1 = X[pd_] + W[pd_], X[d]
        ax.plot([x0, x1], [py + H / 2, y + H / 2], color=RULE, lw=1.6, zorder=1)
        t = 0.62
        pill(ax, x0 + t * (x1 - x0), py + H / 2 + t * (y - py), f"{pr:.2f}")

    # Sampled attempts, drawn over the tree.
    attempts = [("1", ["root", "B", "B2", "L6"], BLUE), ("2", ["root", "B", "B1", "L4"], PURPLE),
                ("3", ["root", "A", "A1", "L1"], AMBER)]
    for label, path, colour in attempts:
        off = {"1": -0.09, "2": 0.0, "3": 0.09}[label]
        for a_, b_ in zip(path, path[1:]):
            da, ya, *_ = nodes[a_]
            db, yb, *_ = nodes[b_]
            ax.plot([X[da] + W[da], X[db]], [ya + H / 2 + off, yb + H / 2 + off], color=colour, lw=3.0, alpha=0.9, zorder=3)
        d, y, *_ = nodes[path[-1]]
        ax.text(X[d] + W[d] + 0.25, y + H / 2, f"attempt {label}", fontsize=9, fontweight="bold", color=colour, va="center")

    # Nodes as cards: leaves coloured by outcome, internal nodes grey, the root outlined.
    for name, (d, y, text, pr) in nodes.items():
        if name.startswith("L"):
            ok = name in correct
            colour = GREEN if ok else RED
            card(ax, X[d], y, W[d], H, bar=colour)
            ax.text(X[d] + 0.22, y + H / 2, ("✓ " if ok else "✗ ") + text, fontsize=9.2,
                    fontweight="bold", color=colour, va="center", zorder=5)
        elif name == "root":
            card(ax, X[d], y, W[d], H, bar=None, fill="white", edge=INK, lw=1.4)
            ax.text(X[d] + W[d] / 2, y + H / 2, text, fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=5)
        else:
            card(ax, X[d], y, W[d], H, bar=GREY)
            ax.text(X[d] + 0.22, y + H / 2, text, fontsize=8.8, va="center", color=INK, zorder=5, linespacing=1.25)

    p_ok = sum(prob[n] for n in correct)
    ps = [1 - (1 - p_ok) ** k for k in (1, 2, 3)]
    ax.text(0.0, -0.55, "Numbers on the edges: chance the agent takes that branch from the node on its left "
            "(illustrative; the edges leaving a node add up to 1).", fontsize=9.2, color=INK)
    ax.text(0.0, -0.95, f"Multiply along a path for the chance of that trace; add the green leaves for the chance "
            f"p that one attempt is correct: p = {p_ok:.2f}.", fontsize=9.2, color=INK)
    ax.text(0.0, -1.45, f"Expected Pass@1 = {ps[0]:.2f}     Pass@2 = {ps[1]:.2f}     Pass@3 = {ps[2]:.2f}",
            fontsize=10.5, fontweight="bold", color=GREEN)
    ax.text(0.0, -1.85, "This draw: attempt 1 gives up (wrong), attempts 2 and 3 are correct, so this task scores "
            "Pass@1 = 0, Pass@2 = 1, Pass@3 = 1.", fontsize=9.2, color=MUTED)
    fig.savefig(OUT / "pass_at_k_tree.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_anatomy():
    """One real run as cards, like the trace viewer's rows, plus stat tiles."""
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.text(0.0, 5.2, "One run, Earth-Bench question 11", fontsize=13, fontweight="bold", color=INK, va="center")
    ax.text(0.0, 4.8, "Expert trajectory above, agent trajectory below. Each card is one tool call.",
            fontsize=9.5, color=MUTED, va="center")

    xs = [1.4, 4.6, 7.8, 11.0]
    w, h = 3.0, 0.9
    ax.text(0.0, 3.95, "EXPERT", fontsize=8.5, color=MUTED, va="center", fontweight="bold")
    for x, name in zip(xs, ["get_filelist", "lst_multi_channel", "calculate_threshold_ratio"]):
        card(ax, x, 3.5, w, h, bar=GREY)
        ax.text(x + 0.22, 3.5 + h / 2, name, fontsize=8.2, va="center", zorder=5, **MONO, fontweight="bold")

    ax.text(0.0, 2.45, "AGENT", fontsize=8.5, color=MUTED, va="center", fontweight="bold")
    agent = [("lst_multi_channel", RED), ("get_filelist", GREEN), ("lst_multi_channel", GREEN),
             ("calculate_threshold_ratio", GREEN)]
    for x, (name, colour) in zip(xs, agent):
        card(ax, x, 2.0, w, h, bar=colour)
        ax.text(x + 0.22, 2.0 + h / 2, name, fontsize=8.2, va="center", zorder=5, **MONO, fontweight="bold")
    ax.text(xs[0] + 0.22, 1.72, "error: file not found, call failed", fontsize=8.4, color=RED, va="center")

    tiles = [
        ("Efficiency", "1.33", "4 calls vs 3 expert.\nAbove 1 means detours.", BLUE),
        ("Tool errors", "1", "Calls that raised an error.\nThe agent recovered.", RED),
        ("Termination", "completed", "Not recursion_limit,\nnot max_model_calls.", GREEN),
        ("Answer", "C, correct", "But TEM = 0: the path\nwas not the expert's.", GREY),
    ]
    for i, (title, value, note, colour) in enumerate(tiles):
        x = 0.1 + i * 3.55
        card(ax, x, 0.05, 3.35, 1.45)
        ax.text(x + 0.2, 1.2, title, fontsize=9, color=MUTED, va="center")
        ax.text(x + 0.2, 0.8, value, fontsize=13, fontweight="bold", color=colour, va="center")
        ax.text(x + 0.2, 0.33, note, fontsize=7.8, color=MUTED, va="center", linespacing=1.3)
    fig.savefig(OUT / "run_anatomy.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pass_at_k_tree()
    run_anatomy()
    print("wrote", OUT / "pass_at_k_tree.png", OUT / "run_anatomy.png")
