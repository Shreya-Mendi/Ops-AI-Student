"""
make_visuals.py — Generate report figures from the drift analysis.

Produces PNG charts in week4/figures/ for inclusion in the docx report:
  1. fig1_temporal_peak_shift.png   — hourly demand baseline vs Feb
  2. fig2_manhattan_lag_deflation.png — lag feature means + PSI
  3. fig3_outer_borough_scramble.png  — feature-target correlation drop
  4. fig4_manhattan_weekend.png       — weekend vs weekday demand change
  5. fig5_psi_summary.png             — PSI across features vs thresholds
  6. fig6_drift_heatmap.png           — per-borough MAE degradation heatmap

Run from week4/:
    python3 scripts/make_visuals.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_WEEK4 = os.path.dirname(_HERE)
_FIG = os.path.join(_WEEK4, "figures")
os.makedirs(_FIG, exist_ok=True)

# Consistent style
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
BASELINE_C = "#4C72B0"
NEW_C = "#C44E52"


def load():
    df = pd.read_parquet(os.path.join(_WEEK4, "data", "demand_enriched_week4.parquet"))
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])
    base = df[(df["time_bucket"] >= "2026-01-01") & (df["time_bucket"] < "2026-01-16")].copy()
    new = df[(df["time_bucket"] >= "2026-02-02") & (df["time_bucket"] < "2026-03-01")].copy()
    return base, new


def fig1_temporal_peak_shift(base, new):
    """Hourly mean trip_count: baseline vs Feb, highlighting shifted windows."""
    b_hr = base.groupby("hour")["trip_count"].mean()
    n_hr = new.groupby("hour")["trip_count"].mean()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(b_hr.index, b_hr.values, "-o", color=BASELINE_C, label="Baseline (Jan 1-15)", lw=2)
    ax.plot(n_hr.index, n_hr.values, "-o", color=NEW_C, label="Feb 2-28", lw=2)

    # Highlight the two affected windows
    ax.axvspan(5, 7, color="green", alpha=0.10, label="Early window (5-7am): +28%")
    ax.axvspan(9, 11, color="red", alpha=0.10, label="Late window (9-11am): -42%")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Trip Count per Zone / 15min")
    ax.set_title("Pattern 1: Temporal Peak Shift\nEarly-morning demand up, late-morning demand down")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(_FIG, "fig1_temporal_peak_shift.png"))
    plt.close(fig)
    print("  fig1_temporal_peak_shift.png")


def fig2_manhattan_lag_deflation(base, new):
    """Manhattan lag feature means baseline vs Feb, with PSI annotations."""
    b_man = base[base["borough_id"] == 0]
    n_man = new[new["borough_id"] == 0]
    feats = ["lag_1day", "lag_1week", "roll_mean_1day"]
    psi_vals = {"lag_1day": 0.222, "lag_1week": 0.120, "roll_mean_1day": 0.484}

    b_means = [b_man[f].mean() for f in feats]
    n_means = [n_man[f].mean() for f in feats]

    x = np.arange(len(feats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.bar(x - w/2, b_means, w, label="Baseline", color=BASELINE_C)
    ax.bar(x + w/2, n_means, w, label="Feb 2-28", color=NEW_C)

    ax.set_ylim(0, max(b_means) * 1.30)  # headroom for annotations
    for i, f in enumerate(feats):
        pct = (n_means[i] - b_means[i]) / b_means[i] * 100
        # Annotate above the shorter (red) bar so it never collides with title/legend
        ax.text(i + w/2, n_means[i] + 0.4,
                f"{pct:+.0f}%\nPSI={psi_vals[f]:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="red" if psi_vals[f] > 0.25 else "black")

    ax.set_xticks(x)
    ax.set_xticklabels(feats)
    ax.set_ylabel("Mean Feature Value")
    ax.set_title("Pattern 2: Manhattan Lag Feature Deflation\nLag inputs dropped to ~55% (PSI > 0.25 = critical)")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(_FIG, "fig2_manhattan_lag_deflation.png"))
    plt.close(fig)
    print("  fig2_manhattan_lag_deflation.png")


def fig3_outer_borough_scramble(base, new):
    """Feature-target correlation (zone_slot_baseline vs trip_count) by borough."""
    boroughs = {0: "Manhattan", 1: "Queens", 2: "Brooklyn"}
    b_corrs, n_corrs, labels = [], [], []
    for bid, name in boroughs.items():
        b = base[base["borough_id"] == bid][["zone_slot_baseline", "trip_count"]].dropna()
        n = new[new["borough_id"] == bid][["zone_slot_baseline", "trip_count"]].dropna()
        b_corrs.append(b.corr().iloc[0, 1])
        n_corrs.append(n.corr().iloc[0, 1])
        labels.append(name)

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w/2, b_corrs, w, label="Baseline", color=BASELINE_C)
    ax.bar(x + w/2, n_corrs, w, label="Feb 2-28", color=NEW_C)

    for i in range(len(labels)):
        delta = n_corrs[i] - b_corrs[i]
        ax.text(i, max(b_corrs[i], n_corrs[i]) + 0.02,
                f"{delta:+.2f}", ha="center", fontsize=10,
                fontweight="bold", color="red" if abs(delta) > 0.05 else "black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pearson Corr (zone_slot_baseline vs trip_count)")
    ax.set_title("Pattern 3: Outer Borough Baseline Scramble\nBrooklyn feature-target correlation broke (-0.16)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(_FIG, "fig3_outer_borough_scramble.png"))
    plt.close(fig)
    print("  fig3_outer_borough_scramble.png")


def fig4_manhattan_weekend(base, new):
    """Manhattan weekend vs weekday demand: baseline vs Feb."""
    def seg_mean(df, weekend):
        return df[(df["borough_id"] == 0) & (df["is_weekend"] == weekend)]["trip_count"].mean()

    cats = ["Weekday", "Weekend"]
    b_vals = [seg_mean(base, 0), seg_mean(base, 1)]
    n_vals = [seg_mean(new, 0), seg_mean(new, 1)]

    x = np.arange(len(cats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(x - w/2, b_vals, w, label="Baseline", color=BASELINE_C)
    ax.bar(x + w/2, n_vals, w, label="Feb 2-28", color=NEW_C)

    for i in range(len(cats)):
        pct = (n_vals[i] - b_vals[i]) / b_vals[i] * 100
        ax.text(i, max(b_vals[i], n_vals[i]) + 0.2, f"{pct:+.0f}%",
                ha="center", fontsize=11, fontweight="bold",
                color="red" if pct < -10 else "black")

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, max(b_vals) * 1.18)
    ax.set_ylabel("Mean Trip Count (Manhattan)")
    ax.set_title("Pattern 4: Manhattan Weekend Concept Drift\nWeekend demand fell 31% while weekday stayed stable")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(_FIG, "fig4_manhattan_weekend.png"))
    plt.close(fig)
    print("  fig4_manhattan_weekend.png")


def fig5_psi_summary(base, new):
    """PSI across key features vs warning/critical thresholds."""
    from scipy.stats import ks_2samp  # noqa

    def psi(b, n, bins=10):
        edges = np.unique(np.percentile(b, np.linspace(0, 100, bins + 1)))
        if len(edges) < 2:
            return 0.0
        edges[0] = min(edges[0], n.min()) - 1e-6
        edges[-1] = max(edges[-1], n.max()) + 1e-6
        bc, _ = np.histogram(b, bins=edges)
        nc, _ = np.histogram(n, bins=edges)
        bp = np.clip(bc / bc.sum(), 1e-6, None)
        np_ = np.clip(nc / nc.sum(), 1e-6, None)
        return float(np.sum((np_ - bp) * np.log(np_ / bp)))

    feats = ["trip_count", "lag_1day", "lag_1week", "roll_mean_1day"]
    man_b = base[base["borough_id"] == 0]
    man_n = new[new["borough_id"] == 0]
    vals = []
    for f in feats:
        if f.startswith("roll") or f.startswith("lag"):
            vals.append(psi(man_b[f].dropna().values, man_n[f].dropna().values))
        else:
            vals.append(psi(base[f].dropna().values, new[f].dropna().values))

    colors = ["#55A868" if v < 0.10 else "#DD8452" if v < 0.25 else "#C44E52" for v in vals]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(feats, vals, color=colors)
    ax.axhline(0.10, color="orange", ls="--", lw=1.5, label="Warning (0.10)")
    ax.axhline(0.25, color="red", ls="--", lw=1.5, label="Critical (0.25)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("PSI")
    ax.set_title("PSI Summary Across Features\n(lag/roll features are Manhattan-segmented)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(_FIG, "fig5_psi_summary.png"))
    plt.close(fig)
    print("  fig5_psi_summary.png")


def fig6_drift_heatmap(base, new):
    """Heatmap of MAE-rate degradation by borough x weekend/weekday."""
    boroughs = {0: "Manhattan", 1: "Queens", 2: "Brooklyn"}
    segs = [("Weekday", 0), ("Weekend", 1)]
    grid = np.zeros((len(boroughs), len(segs)))

    for i, bid in enumerate(boroughs):
        for j, (_, wk) in enumerate(segs):
            b = base[(base["borough_id"] == bid) & (base["is_weekend"] == wk)][["lag_1day", "trip_count"]].dropna()
            n = new[(new["borough_id"] == bid) & (new["is_weekend"] == wk)][["lag_1day", "trip_count"]].dropna()
            if len(b) == 0 or len(n) == 0:
                grid[i, j] = 0
                continue
            b_err = (np.abs(b["lag_1day"] - b["trip_count"]) / np.maximum(b["trip_count"], 1)).mean()
            n_err = (np.abs(n["lag_1day"] - n["trip_count"]) / np.maximum(n["trip_count"], 1)).mean()
            grid[i, j] = n_err - b_err

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto", vmin=-0.2, vmax=0.5)
    ax.set_xticks(range(len(segs)))
    ax.set_xticklabels([s[0] for s in segs])
    ax.set_yticks(range(len(boroughs)))
    ax.set_yticklabels(list(boroughs.values()))
    for i in range(len(boroughs)):
        for j in range(len(segs)):
            ax.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold")
    ax.set_title("Accuracy Degradation Heatmap\n(MAE-rate increase: red = worse)")
    fig.colorbar(im, ax=ax, label="MAE-rate change (Feb - Baseline)")
    fig.tight_layout()
    fig.savefig(os.path.join(_FIG, "fig6_drift_heatmap.png"))
    plt.close(fig)
    print("  fig6_drift_heatmap.png")


def main():
    print("Loading data...")
    base, new = load()
    print(f"Baseline: {len(base):,} rows | Feb: {len(new):,} rows")
    print("Generating figures in week4/figures/:")
    fig1_temporal_peak_shift(base, new)
    fig2_manhattan_lag_deflation(base, new)
    fig3_outer_borough_scramble(base, new)
    fig4_manhattan_weekend(base, new)
    fig5_psi_summary(base, new)
    fig6_drift_heatmap(base, new)
    print("Done.")


if __name__ == "__main__":
    main()
