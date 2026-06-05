"""
detect_drift.py — Detect 4 distinct drift patterns in Feb 2-28 vs Jan 1-15 data.

Run from week4/ directory:
    python3 scripts/detect_drift.py

Each pattern is detected using statistical tests (KS, PSI, segment comparison)
and documented with quantitative evidence.
"""

import os
import sys
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

_HERE = os.path.dirname(os.path.abspath(__file__))
_WEEK4 = os.path.dirname(_HERE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_psi(base: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
    bin_edges = np.percentile(base, np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0
    # Extend edges to capture all new data (avoids NaN when distributions don't overlap)
    bin_edges[0] = min(bin_edges[0], new.min()) - 1e-6
    bin_edges[-1] = max(bin_edges[-1], new.max()) + 1e-6
    base_counts, _ = np.histogram(base, bins=bin_edges)
    new_counts, _ = np.histogram(new, bins=bin_edges)
    if new_counts.sum() == 0:
        return float("inf")
    base_pct = np.clip(base_counts / base_counts.sum(), 1e-6, None)
    new_pct = np.clip(new_counts / new_counts.sum(), 1e-6, None)
    return float(np.sum((new_pct - base_pct) * np.log(new_pct / base_pct)))


def ks_result(base_vals, new_vals, label: str) -> dict:
    stat, pval = ks_2samp(base_vals, new_vals)
    return {
        "feature": label,
        "ks_statistic": round(float(stat), 4),
        "p_value": float(pval),
        "drifted": bool(pval < 0.05),
        "interpretation": "significant drift" if pval < 0.05 else "no significant drift",
    }


def detect_feature_drift(baseline_df: pd.DataFrame, new_df: pd.DataFrame, feature: str) -> dict:
    """
    Detect drift in a single feature using KS test and PSI.
    Returns dict with test results and interpretation.
    """
    base_vals = baseline_df[feature].dropna().values
    new_vals = new_df[feature].dropna().values
    result = ks_result(base_vals, new_vals, feature)
    result["baseline_mean"] = round(float(base_vals.mean()), 4)
    result["new_mean"] = round(float(new_vals.mean()), 4)
    result["mean_change_pct"] = round(
        (float(new_vals.mean()) - float(base_vals.mean())) / max(abs(float(base_vals.mean())), 1e-6) * 100, 2
    )
    result["psi"] = round(compute_psi(base_vals, new_vals), 4)
    return result


def detect_concept_drift_by_segment(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """
    Detect concept drift (accuracy degradation) by borough and weekend flag.
    Uses lag_1day as proxy predictions; trip_count as actuals.
    """
    results = {}
    for borough in sorted(baseline_df["borough_id"].unique()):
        b = baseline_df[baseline_df["borough_id"] == borough].copy()
        n = new_df[new_df["borough_id"] == borough].copy()
        b_valid = b[["lag_1day", "trip_count"]].dropna()
        n_valid = n[["lag_1day", "trip_count"]].dropna()
        if len(b_valid) == 0 or len(n_valid) == 0:
            continue
        b_err = (np.abs(b_valid["lag_1day"] - b_valid["trip_count"]) /
                 np.maximum(b_valid["trip_count"], 1)).mean()
        n_err = (np.abs(n_valid["lag_1day"] - n_valid["trip_count"]) /
                 np.maximum(n_valid["trip_count"], 1)).mean()
        results[int(borough)] = {
            "baseline_mae_rate": round(float(b_err), 4),
            "new_mae_rate": round(float(n_err), 4),
            "degradation": round(float(n_err - b_err), 4),
        }
    return results


# ── Pattern 1: Temporal Peak Shift ───────────────────────────────────────────

def detect_temporal_peak_shift(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """
    Early-morning slots (5-7am, slot_of_day 20-27) boosted 45%.
    Late-morning slots (9-11am, slot_of_day 36-43) reduced 35%.
    """
    early_slots = list(range(20, 28))
    late_slots = list(range(36, 44))

    b_early = baseline_df[baseline_df["slot_of_day"].isin(early_slots)]["trip_count"]
    n_early = new_df[new_df["slot_of_day"].isin(early_slots)]["trip_count"]
    b_late = baseline_df[baseline_df["slot_of_day"].isin(late_slots)]["trip_count"]
    n_late = new_df[new_df["slot_of_day"].isin(late_slots)]["trip_count"]

    ks_early = ks_result(b_early.values, n_early.values, "trip_count_early_slots")
    ks_late = ks_result(b_late.values, n_late.values, "trip_count_late_slots")

    early_change = (n_early.mean() - b_early.mean()) / b_early.mean() * 100
    late_change = (n_late.mean() - b_late.mean()) / b_late.mean() * 100

    return {
        "pattern": "temporal_peak_shift",
        "description": "Early morning (5-7am) demand boosted; late morning (9-11am) reduced",
        "type": "data_drift",
        "early_slot_mean_baseline": round(float(b_early.mean()), 2),
        "early_slot_mean_new": round(float(n_early.mean()), 2),
        "early_slot_change_pct": round(float(early_change), 1),
        "late_slot_mean_baseline": round(float(b_late.mean()), 2),
        "late_slot_mean_new": round(float(n_late.mean()), 2),
        "late_slot_change_pct": round(float(late_change), 1),
        "ks_early": ks_early,
        "ks_late": ks_late,
        "drift_confirmed": ks_early["drifted"] or ks_late["drifted"],
        "impact": "Temporal demand pattern shifted; models trained on normal peaks will mispredict commute hours",
    }


# ── Pattern 2: Manhattan Lag Deflation ───────────────────────────────────────

def detect_manhattan_lag_deflation(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """
    Manhattan (borough_id=0) lag_1day, lag_1week, roll_mean_1day deflated to 55%.
    """
    b_man = baseline_df[baseline_df["borough_id"] == 0]
    n_man = new_df[new_df["borough_id"] == 0]

    features = ["lag_1day", "lag_1week", "roll_mean_1day"]
    feature_results = {}
    for feat in features:
        b_vals = b_man[feat].dropna().values
        n_vals = n_man[feat].dropna().values
        feature_results[feat] = {
            "baseline_mean": round(float(b_vals.mean()), 3),
            "new_mean": round(float(n_vals.mean()), 3),
            "change_pct": round(
                (n_vals.mean() - b_vals.mean()) / max(abs(b_vals.mean()), 1e-6) * 100, 1
            ),
            **ks_result(b_vals, n_vals, feat),
            "psi": round(compute_psi(b_vals, n_vals), 4),
        }

    all_drifted = all(r["drifted"] for r in feature_results.values())
    return {
        "pattern": "manhattan_lag_deflation",
        "description": "Manhattan lag features deflated to ~55% of baseline",
        "type": "data_drift",
        "borough": "Manhattan (borough_id=0)",
        "features": feature_results,
        "drift_confirmed": all_drifted,
        "impact": "Model underestimates Manhattan demand; lag inputs are systematically low",
    }


# ── Pattern 3: Outer Borough Baseline Scramble ───────────────────────────────

def detect_outer_borough_scramble(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """
    Queens (borough_id=1) and Brooklyn (borough_id=2): zone_slot_baseline alternates
    between 0.22x and 3.8x multipliers on 5 zones, breaking feature-target correlation.

    The scramble affects individual zones, so aggregate KS tests are insufficient.
    Detection strategy:
    1. Per-zone mean change in zone_slot_baseline (affected zones show >50% change)
    2. Feature-target Pearson correlation degradation (scramble breaks it)
    """
    ZONE_CHANGE_THRESHOLD = 0.50   # >50% mean change flags a scrambled zone
    results = {}
    for borough, name in [(1, "Queens"), (2, "Brooklyn")]:
        b = baseline_df[baseline_df["borough_id"] == borough]
        n = new_df[new_df["borough_id"] == borough]

        # Per-zone mean of zone_slot_baseline
        b_zone_baseline = b.groupby("PULocationID")["zone_slot_baseline"].mean()
        n_zone_baseline = n.groupby("PULocationID")["zone_slot_baseline"].mean()
        common_zones = b_zone_baseline.index.intersection(n_zone_baseline.index)

        if len(common_zones) == 0:
            results[name] = {"scrambled_zones": 0, "drifted": False}
            continue

        b_vals = b_zone_baseline[common_zones]
        n_vals = n_zone_baseline[common_zones]
        rel_change = np.abs((n_vals - b_vals) / (b_vals.clip(lower=1e-6)))
        scrambled_zones = int((rel_change > ZONE_CHANGE_THRESHOLD).sum())

        # Pearson correlation between zone_slot_baseline and trip_count
        b_corr = b[["zone_slot_baseline", "trip_count"]].dropna().corr().iloc[0, 1]
        n_corr = n[["zone_slot_baseline", "trip_count"]].dropna().corr().iloc[0, 1]

        # KS on per-zone relative changes (large changes indicate scramble)
        ks = ks_result(b_vals.values, n_vals.values, f"zone_slot_baseline_{name.lower()}")

        results[name] = {
            **ks,
            "scrambled_zones": scrambled_zones,
            "total_zones": len(common_zones),
            "baseline_feature_target_corr": round(float(b_corr), 4),
            "new_feature_target_corr": round(float(n_corr), 4),
            "corr_degradation": round(float(n_corr - b_corr), 4),
            "drifted": scrambled_zones > 0 or abs(float(n_corr - b_corr)) > 0.05,
        }

    both_drifted = any(r.get("drifted", False) for r in results.values())
    return {
        "pattern": "outer_borough_baseline_scramble",
        "description": "Queens/Brooklyn zone_slot_baseline scrambled (0.22x and 3.8x per zone)",
        "type": "data_drift",
        "boroughs": results,
        "drift_confirmed": both_drifted,
        "impact": "Feature-target correlation broken in outer boroughs; predictions unreliable",
    }


# ── Pattern 4: Manhattan Weekend Concept Drift ───────────────────────────────

def detect_manhattan_weekend_concept_drift(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """
    Manhattan (borough_id=0) weekend (is_weekend=1) trip_count reduced to 72%.
    Concept drift: demand behavior changed, not just feature distributions.
    """
    b_wknd = baseline_df[(baseline_df["borough_id"] == 0) & (baseline_df["is_weekend"] == 1)]
    n_wknd = new_df[(new_df["borough_id"] == 0) & (new_df["is_weekend"] == 1)]
    b_wkdy = baseline_df[(baseline_df["borough_id"] == 0) & (baseline_df["is_weekend"] == 0)]
    n_wkdy = new_df[(new_df["borough_id"] == 0) & (new_df["is_weekend"] == 0)]

    b_wknd_mean = float(b_wknd["trip_count"].mean())
    n_wknd_mean = float(n_wknd["trip_count"].mean())
    b_wkdy_mean = float(b_wkdy["trip_count"].mean())
    n_wkdy_mean = float(n_wkdy["trip_count"].mean())

    ks_wknd = ks_result(b_wknd["trip_count"].values, n_wknd["trip_count"].values, "trip_count_manhattan_weekend")
    ks_wkdy = ks_result(b_wkdy["trip_count"].values, n_wkdy["trip_count"].values, "trip_count_manhattan_weekday")

    weekend_change = (n_wknd_mean - b_wknd_mean) / b_wknd_mean * 100
    weekday_change = (n_wkdy_mean - b_wkdy_mean) / b_wkdy_mean * 100

    b_valid = b_wknd[["lag_1day", "trip_count"]].dropna()
    n_valid = n_wknd[["lag_1day", "trip_count"]].dropna()
    b_err = (np.abs(b_valid["lag_1day"] - b_valid["trip_count"]) / np.maximum(b_valid["trip_count"], 1)).mean()
    n_err = (np.abs(n_valid["lag_1day"] - n_valid["trip_count"]) / np.maximum(n_valid["trip_count"], 1)).mean()

    return {
        "pattern": "manhattan_weekend_concept_drift",
        "description": "Manhattan weekend demand fell to ~72% of baseline (tourism + hybrid work shift)",
        "type": "concept_drift",
        "borough": "Manhattan (borough_id=0)",
        "weekend_mean_baseline": round(b_wknd_mean, 2),
        "weekend_mean_new": round(n_wknd_mean, 2),
        "weekend_change_pct": round(float(weekend_change), 1),
        "weekday_change_pct": round(float(weekday_change), 1),
        "ks_weekend": ks_wknd,
        "ks_weekday": ks_wkdy,
        "baseline_mae_rate_weekend": round(float(b_err), 4),
        "new_mae_rate_weekend": round(float(n_err), 4),
        "accuracy_degradation": round(float(n_err - b_err), 4),
        "drift_confirmed": ks_wknd["drifted"],
        "impact": "Model over-predicts Manhattan weekends by ~28%; trained on pre-Feb patterns",
    }


def main():
    """Main drift detection analysis."""
    print("=" * 70)
    print("DRIFT DETECTION -- NYC Taxi Demand Forecast")
    print(f"Run at: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    week4_path = os.path.join(_WEEK4, "data", "demand_enriched_week4.parquet")
    df = pd.read_parquet(week4_path)
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])

    baseline_df = df[(df["time_bucket"] >= "2026-01-01") & (df["time_bucket"] < "2026-01-16")].copy()
    new_df = df[(df["time_bucket"] >= "2026-02-02") & (df["time_bucket"] < "2026-03-01")].copy()

    print(f"\nBaseline (Jan 1-15): {len(baseline_df):,} rows")
    print(f"New data (Feb 2-28): {len(new_df):,} rows")

    # Global feature drift
    print("\n-- Global Feature Drift (KS + PSI) --")
    for feat in ["trip_count", "lag_1day", "lag_1week", "zone_slot_baseline"]:
        r = detect_feature_drift(baseline_df, new_df, feat)
        print(f"  {feat}: mean {r['baseline_mean']} -> {r['new_mean']} "
              f"({r['mean_change_pct']:+.1f}%), KS p={r['p_value']:.2e}, PSI={r['psi']:.3f}  "
              f"[{'DRIFT' if r['drifted'] else 'OK'}]")

    # Concept drift by borough
    print("\n-- Concept Drift by Borough (MAE rate) --")
    concept = detect_concept_drift_by_segment(baseline_df, new_df)
    for borough, r in concept.items():
        names = {0: "Manhattan", 1: "Queens", 2: "Brooklyn"}
        print(f"  {names.get(borough, borough)}: MAE rate {r['baseline_mae_rate']:.4f} -> "
              f"{r['new_mae_rate']:.4f} (delta {r['degradation']:+.4f})")

    # Pattern 1
    print("\n-- Pattern 1: Temporal Peak Shift --")
    p1 = detect_temporal_peak_shift(baseline_df, new_df)
    print(f"  Early (5-7am):  {p1['early_slot_mean_baseline']:.1f} -> {p1['early_slot_mean_new']:.1f} "
          f"({p1['early_slot_change_pct']:+.1f}%)  KS p={p1['ks_early']['p_value']:.2e}")
    print(f"  Late  (9-11am): {p1['late_slot_mean_baseline']:.1f} -> {p1['late_slot_mean_new']:.1f} "
          f"({p1['late_slot_change_pct']:+.1f}%)  KS p={p1['ks_late']['p_value']:.2e}")
    print(f"  Confirmed: {p1['drift_confirmed']}")

    # Pattern 2
    print("\n-- Pattern 2: Manhattan Lag Deflation --")
    p2 = detect_manhattan_lag_deflation(baseline_df, new_df)
    for feat, r in p2["features"].items():
        print(f"  {feat}: {r['baseline_mean']:.3f} -> {r['new_mean']:.3f} "
              f"({r['change_pct']:+.1f}%), KS p={r['p_value']:.2e}, PSI={r['psi']:.3f}")
    print(f"  Confirmed: {p2['drift_confirmed']}")

    # Pattern 3
    print("\n-- Pattern 3: Outer Borough Baseline Scramble --")
    p3 = detect_outer_borough_scramble(baseline_df, new_df)
    for boro, r in p3["boroughs"].items():
        print(f"  {boro}: scrambled_zones={r.get('scrambled_zones',0)}/{r.get('total_zones','?')}, "
              f"corr {r.get('baseline_feature_target_corr','?')} -> {r.get('new_feature_target_corr','?')} "
              f"(delta {r.get('corr_degradation','?')}), KS p={r['p_value']:.2e}")
    print(f"  Confirmed: {p3['drift_confirmed']}")

    # Pattern 4
    print("\n-- Pattern 4: Manhattan Weekend Concept Drift --")
    p4 = detect_manhattan_weekend_concept_drift(baseline_df, new_df)
    print(f"  Weekend: {p4['weekend_mean_baseline']:.2f} -> {p4['weekend_mean_new']:.2f} "
          f"({p4['weekend_change_pct']:+.1f}%)  KS p={p4['ks_weekend']['p_value']:.2e}")
    print(f"  Weekday change: {p4['weekday_change_pct']:+.1f}%")
    print(f"  Weekend MAE degradation: {p4['accuracy_degradation']:+.4f}")
    print(f"  Confirmed: {p4['drift_confirmed']}")

    # Summary
    patterns = [p1, p2, p3, p4]
    n_confirmed = sum(1 for p in patterns if p["drift_confirmed"])
    print(f"\n{'='*70}")
    print(f"SUMMARY: {n_confirmed}/4 drift patterns confirmed")
    for p in patterns:
        status = "CONFIRMED" if p["drift_confirmed"] else "NOT DETECTED"
        print(f"  [{status}] {p['pattern']} ({p['type']})")

    # Write JSON
    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "baseline_period": "2026-01-01 to 2026-01-15",
        "new_data_period": "2026-02-02 to 2026-02-28",
        "patterns_confirmed": n_confirmed,
        "patterns": {
            "temporal_peak_shift": p1,
            "manhattan_lag_deflation": p2,
            "outer_borough_scramble": p3,
            "manhattan_weekend_concept_drift": p4,
        },
        "concept_drift_by_borough": concept,
    }

    out_file = os.path.join(_WEEK4, f"drift-{datetime.now().strftime('%Y%m%d')}.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote drift report to {os.path.basename(out_file)}")

    if n_confirmed < 4:
        print(f"WARNING: Only {n_confirmed}/4 patterns detected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
