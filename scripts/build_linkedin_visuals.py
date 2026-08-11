#!/usr/bin/env python3
"""Render the suggested visual for each LinkedIn draft as a standalone PNG.

Why: the post text itself must stay plain (LinkedIn kills formatting and math),
so every formula/diagram ships as a separate square image attached to the post.
Sanitized by construction: no real tickers, weights, dates, or dollar values.

Usage:  python3 scripts/build_linkedin_visuals.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "linkedin_drafts" / "visuals"
INK, MUTED, BLUE, GREEN, BG = "#1A1A1A", "#7A7A7A", "#0A66C2", "#2E8B6F", "#FBFAF8"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150})


def canvas(title: str, sub: str = ""):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(8, 92, title, fontsize=21, fontweight="bold", color=INK, va="top")
    if sub:
        ax.text(8, 85.5, sub, fontsize=12.5, color=MUTED, va="top")
    ax.plot([8, 92], [81, 81], color="#E2E0DC", lw=1.2)
    return fig, ax


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.35, facecolor=BG)
    plt.close(fig)
    print(OUT / name)


def box(ax, x, y, w, h, label, fc, tc="white", fs=11.5):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5", fc=fc, ec="none")
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fs,
        color=tc,
        fontweight="bold",
    )


def arrow(ax, p1, p2):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=13, color=MUTED, lw=1.4, shrinkA=2, shrinkB=2
        )
    )


# 1 — walk-forward testing
def walk_forward():
    fig, ax = canvas(
        "Walk-forward testing", "Fit on the past, judge on data the model has never seen."
    )
    y = 66
    for i in range(4):
        x0 = 8 + i * 6
        box(ax, x0, y, 34, 7, "TRAIN", BLUE)
        box(ax, x0 + 35, y, 14, 7, "TEST", GREEN)
        ax.text(93, y + 3.5, f"fold {i + 1}", fontsize=10, color=MUTED, ha="right", va="center")
        y -= 12
    ax.annotate(
        "", xy=(92, 14), xytext=(8, 14), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4)
    )
    ax.text(50, 9.5, "time", fontsize=11, color=MUTED, ha="center")
    ax.text(
        8,
        24,
        "Only the green windows count. Everything tuned on blue\n"
        "is a hypothesis until it survives green.",
        fontsize=11.5,
        color=INK,
        va="top",
        linespacing=1.5,
    )
    save(fig, "2026-07-12_walk-forward.png")


# 2 — committee architecture
def committee():
    fig, ax = canvas(
        "Design for conflict, not consensus",
        "A decision only earns its way through after being argued against.",
    )
    views = ["Trend view", "Valuation view", "Risk view", "Devil's advocate"]
    y = 62
    for i, v in enumerate(views):
        fc = "#B04A3E" if i == len(views) - 1 else BLUE
        box(ax, 6, y, 30, 8, v, fc, fs=11)
        arrow(ax, (37, y + 4), (48, 41))
        y -= 11
    box(ax, 48, 34, 22, 14, "Debate", "#33383D")
    arrow(ax, (70.5, 41), (78, 41))
    box(ax, 78, 34, 17, 14, "Decision", GREEN)
    ax.text(
        6,
        20,
        "Unanimous, instant agreement was the warning sign.\n"
        "It usually meant every voice read the same headline.",
        fontsize=11.5,
        color=INK,
        va="top",
        linespacing=1.5,
    )
    save(fig, "2026-07-20_committee.png")


# 3 — fast exits vs patient holds
def holding_winners():
    import numpy as np

    fig, ax = canvas(
        "Why trend-following holds winners",
        "Same signal, different patience. The gap is compounding minus friction.",
    )
    a = fig.add_axes([0.09, 0.16, 0.84, 0.52])
    a.set_facecolor(BG)
    t = np.linspace(0, 10, 400)
    patient = 100 * 1.11**t
    fast = 100 * 1.11**t
    for k in range(1, 10):  # each exit clips the compounding leg
        fast[t >= k] *= 0.965
    a.plot(t, patient, color=GREEN, lw=2.6, label="Patient holds")
    a.plot(t, fast, color="#C2703A", lw=2.6, ls="--", label="Fast exits")
    for k in range(1, 10):
        a.plot(k, 100 * 1.11**k * 0.965**k, "o", ms=4.5, color="#C2703A")
    for s in a.spines.values():
        s.set_color("#DDDAD5")
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlabel("time", color=MUTED, fontsize=11)
    a.set_ylabel("growth of the same idea", color=MUTED, fontsize=11)
    a.legend(frameon=False, fontsize=11.5, loc="upper left")
    ax.text(
        8,
        5,
        "Illustrative shape, not a backtest. Dots are exits.",
        fontsize=10.5,
        color=MUTED,
        va="top",
    )
    save(fig, "2026-07-26_holding-winners.png")


# 4 — effective number of bets
def effective_bets():
    fig, ax = canvas(
        "Effective number of bets", "Diversification is about shared drivers, not position count."
    )
    ax.text(
        50,
        60,
        r"$N_{\mathrm{eff}} \;=\; \left( \sum_i w_i^{2} \right)^{-1}$",
        fontsize=40,
        color=INK,
        ha="center",
        va="center",
    )
    ax.text(
        50,
        40,
        "10 positions, one driver   →   $N_{eff} \\approx 2$",
        fontsize=15,
        color=BLUE,
        ha="center",
        fontweight="bold",
    )
    ax.text(
        8,
        28,
        "Count what actually moves independently, not what sits in the list.\n"
        "Ten holdings that respond to the same rate story are one bet\n"
        "wearing ten costumes.",
        fontsize=12.5,
        color=INK,
        va="top",
        linespacing=1.6,
    )
    save(fig, "2026-08-02_effective-bets.png")


# 5 — attempted vs achieved
def attempted_vs_achieved():
    fig, ax = canvas(
        "Attempted is not achieved",
        "A job that reports success and produces nothing is a silent failure.",
    )
    ax.text(30, 74, "What the log said", fontsize=13, color=MUTED, ha="center", fontweight="bold")
    ax.text(74, 74, "What it produced", fontsize=13, color=MUTED, ha="center", fontweight="bold")
    rows = ["scheduled job A", "scheduled job B", "scheduled job C"]
    y = 58
    for r in rows:
        box(ax, 8, y, 44, 9, f"{r}   ·   ran, ok", GREEN, fs=11)
        ax.add_patch(
            FancyBboxPatch(
                (58, y),
                32,
                9,
                boxstyle="round,pad=0.6,rounding_size=1.5",
                fc="none",
                ec="#C9C5BF",
                lw=1.4,
                ls=(0, (4, 3)),
            )
        )
        ax.text(74, y + 4.5, "no artifact", fontsize=11, color=MUTED, ha="center", va="center")
        y -= 13
    ax.text(
        8,
        22,
        "The tooling made me much faster at building — and just as fast\n"
        "at building things I never verified. Now every job has to leave\n"
        "a timestamped artifact behind. No artifact, it didn't happen.",
        fontsize=12,
        color=INK,
        va="top",
        linespacing=1.6,
    )
    save(fig, "2026-08-09_attempted-vs-achieved.png")


if __name__ == "__main__":
    walk_forward()
    committee()
    holding_winners()
    effective_bets()
    attempted_vs_achieved()
