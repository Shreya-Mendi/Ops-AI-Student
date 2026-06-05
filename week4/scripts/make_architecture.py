"""
make_architecture.py — Render the monitoring/retraining architecture diagram as a PNG.

Produces: week4/figures/fig7_architecture.png

Run from week4/:
    python3 scripts/make_architecture.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_HERE = os.path.dirname(os.path.abspath(__file__))
_WEEK4 = os.path.dirname(_HERE)
_FIG = os.path.join(_WEEK4, "figures")
os.makedirs(_FIG, exist_ok=True)

# Colors by stage
C_DATA = "#4C72B0"      # data sources
C_MONITOR = "#55A868"   # monitoring jobs
C_DECISION = "#DD8452"  # decision/threshold
C_ALERT = "#C44E52"     # alert/retrain
C_DEPLOY = "#8172B3"    # deploy/rollback
C_TEXT = "white"


def box(ax, x, y, w, h, text, color, fontsize=10, text_color="white"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor="#333333", facecolor=color, alpha=0.95,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold", wrap=True)
    return (x + w / 2, y, x + w / 2, y + h, x, y + h / 2, x + w, y + h / 2)  # cx,bottom,cx,top,left_x,left_y,right_x,right_y


def arrow(ax, x1, y1, x2, y2, color="#333333", style="-|>", label=None, label_offset=(0, 0), ls="-"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=16,
        linewidth=1.6, color=color, linestyle=ls,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1],
                label, ha="center", va="center", fontsize=8.5,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


def main():
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis("off")

    ax.text(6, 13.5, "NYC Taxi Demand — Monitoring & Retraining Architecture",
            ha="center", fontsize=15, fontweight="bold")

    # ── Data sources ───────────────────────────────────────────────────────
    b_base = box(ax, 0.5, 11.7, 4.2, 1.0,
                 "Baseline Data\n(Jan 1-15, 2026)", C_DATA, 10)
    b_live = box(ax, 7.3, 11.7, 4.2, 1.0,
                 "Live Data Pipeline\n(daily ingestion)", C_DATA, 10)

    # ── compute_metrics ────────────────────────────────────────────────────
    b_metrics = box(ax, 3.4, 10.0, 5.2, 1.0,
                    "compute_metrics.py  (6 metrics)\nGHA cron: daily @ midnight UTC", C_MONITOR, 9.5)

    # ── detect_drift ───────────────────────────────────────────────────────
    b_drift = box(ax, 3.4, 8.4, 5.2, 1.0,
                  "detect_drift.py\nKS test · PSI · segment analysis", C_MONITOR, 9.5)

    # ── threshold check ────────────────────────────────────────────────────
    b_check = box(ax, 4.4, 6.9, 3.2, 0.9, "Threshold Check", C_DECISION, 10)

    # ── OK branch (left) ───────────────────────────────────────────────────
    b_ok = box(ax, 0.5, 5.3, 3.0, 0.9, "No action\n(log only)", C_MONITOR, 9.5)

    # ── Alert branch (right) ───────────────────────────────────────────────
    b_alert = box(ax, 8.0, 5.3, 3.5, 0.9,
                  "GitHub Issue + Artifact\nmetrics-*.json / drift-*.json", C_ALERT, 9)

    b_assess = box(ax, 7.6, 3.8, 4.3, 1.0,
                   "Ops Assessment\nPSI>0.25 or acc<70% → retrain\n(dual-condition trigger)", C_ALERT, 9)

    b_retrain = box(ax, 7.9, 2.3, 3.7, 0.9,
                    "Retraining Pipeline\n(30-day rolling window)", C_ALERT, 9.5)

    b_validate = box(ax, 4.1, 2.3, 3.4, 0.9,
                     "Validation\nOffline → Shadow → Canary", C_DEPLOY, 9.5)

    b_deploy = box(ax, 4.4, 0.7, 3.0, 0.9, "Deploy or Rollback", C_DEPLOY, 10)

    # ── Arrows ─────────────────────────────────────────────────────────────
    # data -> metrics
    arrow(ax, 2.6, 11.7, 5.0, 11.0)
    arrow(ax, 9.4, 11.7, 7.0, 11.0)
    # metrics -> drift
    arrow(ax, 6.0, 10.0, 6.0, 9.4)
    # drift -> check
    arrow(ax, 6.0, 8.4, 6.0, 7.8)
    # check -> OK (left)
    arrow(ax, 4.4, 7.2, 3.5, 6.0, color=C_MONITOR, label="OK", label_offset=(-0.1, 0.2))
    # check -> alert (right)
    arrow(ax, 7.6, 7.2, 9.0, 6.2, color=C_ALERT, label="Alert", label_offset=(0.3, 0.2))
    # alert -> assess
    arrow(ax, 9.7, 5.3, 9.7, 4.8, color=C_ALERT)
    # assess -> retrain
    arrow(ax, 9.7, 3.8, 9.7, 3.2, color=C_ALERT)
    # retrain -> validate
    arrow(ax, 7.9, 2.75, 7.5, 2.75, color=C_DEPLOY)
    # validate -> deploy
    arrow(ax, 5.8, 2.3, 5.9, 1.6, color=C_DEPLOY)
    # rollback loop back up (deploy -> retrain dashed)
    arrow(ax, 7.4, 1.15, 9.0, 2.3, color=C_DEPLOY, ls="--", style="-|>",
          label="rollback", label_offset=(0.5, -0.1))

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_items = [
        (C_DATA, "Data sources"),
        (C_MONITOR, "Monitoring (daily CI)"),
        (C_DECISION, "Decision point"),
        (C_ALERT, "Alert / retrain"),
        (C_DEPLOY, "Validate / deploy"),
    ]
    for i, (c, lbl) in enumerate(legend_items):
        ly = 0.9 - i * 0.0  # placed horizontally
    lx = 0.5
    for i, (c, lbl) in enumerate(legend_items):
        ax.add_patch(FancyBboxPatch((lx + i * 2.3, 0.05), 0.3, 0.25,
                                    boxstyle="round,pad=0.01", facecolor=c, edgecolor="none"))
        ax.text(lx + i * 2.3 + 0.4, 0.17, lbl, fontsize=8, va="center")

    fig.tight_layout()
    out = os.path.join(_FIG, "fig7_architecture.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
