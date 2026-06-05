"""
compute_metrics.py — Load baseline and new data, run all metrics, write results.

Run from week4/ directory:
    python3 scripts/compute_metrics.py

Writes: metrics-YYYYMMDD.json
Exits non-zero if any critical alert fires (lets CI detect drift via failure).
"""

import json
import sys
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Allow running from repo root (python3 week4/scripts/compute_metrics.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
_WEEK4 = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from metric_template import MetricComputer

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "null_rate_critical": 0.01,
    "psi_warning": 0.10,
    "psi_critical": 0.25,
    "ks_p_value": 0.05,
    "accuracy_warning": 0.80,
    "accuracy_critical": 0.70,
    "duplicate_rate_critical": 0.005,
}


def load_data(week4_dir: str):
    baseline_path = os.path.join(week4_dir, "data", "demand_enriched_baseline.parquet")
    week4_path = os.path.join(week4_dir, "data", "demand_enriched_week4.parquet")

    baseline = pd.read_parquet(baseline_path)

    week4_full = pd.read_parquet(week4_path)
    week4_full["time_bucket"] = pd.to_datetime(week4_full["time_bucket"])

    # New data = Feb 2-28, 2026 (the injected drift window)
    new_data = week4_full[
        (week4_full["time_bucket"] >= "2026-02-02")
        & (week4_full["time_bucket"] < "2026-03-01")
    ].copy()

    print(f"Baseline rows: {len(baseline):,}")
    print(f"New data rows (Feb 2-28): {len(new_data):,}")
    return baseline, new_data


def build_predictions(new_df: pd.DataFrame):
    """
    Use lag_1day as a proxy for model predictions.
    Actuals = trip_count.
    """
    valid = new_df[["lag_1day", "trip_count"]].dropna()
    predictions = valid["lag_1day"].values.astype(float)
    actuals = valid["trip_count"].values.astype(float)
    return predictions, actuals, valid.index


def check_alerts(results: dict) -> list:
    alerts = []

    # Null rates
    for field, rate in results.get("null_rates", {}).items():
        if rate is not None and rate > THRESHOLDS["null_rate_critical"]:
            alerts.append(f"CRITICAL null rate {field}: {rate:.3%}")

    # PSI
    psi = results.get("psi_trip_count")
    if psi is not None:
        if psi > THRESHOLDS["psi_critical"]:
            alerts.append(f"CRITICAL PSI trip_count={psi:.3f} (threshold {THRESHOLDS['psi_critical']})")
        elif psi > THRESHOLDS["psi_warning"]:
            alerts.append(f"WARNING PSI trip_count={psi:.3f} (threshold {THRESHOLDS['psi_warning']})")

    # KS test
    for feat, ks in results.get("ks_test", {}).items():
        if ks.get("drifted"):
            alerts.append(
                f"KS drift {feat}: stat={ks['statistic']:.3f}, p={ks['p_value']:.2e}"
            )

    # Accuracy
    acc = results.get("accuracy_overall")
    if acc is not None:
        if acc < THRESHOLDS["accuracy_critical"]:
            alerts.append(f"CRITICAL accuracy={acc:.1%} (threshold {THRESHOLDS['accuracy_critical']:.0%})")
        elif acc < THRESHOLDS["accuracy_warning"]:
            alerts.append(f"WARNING accuracy={acc:.1%} (threshold {THRESHOLDS['accuracy_warning']:.0%})")

    # Duplicates
    dup = results.get("duplicate_rate", {})
    if dup.get("alert"):
        alerts.append(f"CRITICAL duplicate rate={dup['rate']:.3%}")

    return alerts


def main():
    week4_dir = _WEEK4

    print("=" * 70)
    print("MONITORING METRICS -- NYC Taxi Demand Forecast")
    print(f"Run at: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    baseline, new_data = load_data(week4_dir)
    computer = MetricComputer(baseline)
    predictions, actuals, valid_idx = build_predictions(new_data)
    valid_df = new_data.loc[valid_idx]

    results = computer.compute_all_metrics(valid_df, predictions, actuals)

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n-- Null Rates --")
    for field, rate in results["null_rates"].items():
        flag = " WARNING" if rate and rate > THRESHOLDS["null_rate_critical"] else ""
        print(f"  {field}: {rate:.4%}{flag}")

    print("\n-- KS Tests --")
    for feat, ks in results["ks_test"].items():
        drift_str = "DRIFT DETECTED" if ks["drifted"] else "OK"
        print(f"  {feat}: stat={ks['statistic']:.4f}, p={ks['p_value']:.2e}  [{drift_str}]")

    psi = results["psi_trip_count"]
    psi_label = "CRITICAL" if psi > 0.25 else ("WARNING" if psi > 0.10 else "OK")
    print(f"\n-- PSI (trip_count) --\n  {psi:.4f}  [{psi_label}]")

    acc = results.get("accuracy_overall")
    if acc is not None:
        acc_label = "CRITICAL" if acc < 0.70 else ("WARNING" if acc < 0.80 else "OK")
        print(f"\n-- Overall Accuracy (lag_1day proxy) --\n  {acc:.1%}  [{acc_label}]")

    fresh = results["data_freshness"]
    stale_str = "STALE" if fresh.get("stale") else "OK"
    print(f"\n-- Data Freshness --\n  Latest: {fresh.get('latest_record')}  [{stale_str}]")

    dup = results["duplicate_rate"]
    dup_str = "ALERT" if dup.get("alert") else "OK"
    print(f"\n-- Duplicate Rate --\n  {dup.get('rate', 0):.4%} ({dup.get('count', 0)} rows)  [{dup_str}]")

    # ── Alerts ─────────────────────────────────────────────────────────────
    alerts = check_alerts(results)
    print("\n-- Alerts --")
    if alerts:
        for a in alerts:
            print(f"  ALERT: {a}")
    else:
        print("  No alerts fired")

    # ── Write JSON ─────────────────────────────────────────────────────────
    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "baseline_rows": len(baseline),
        "new_data_rows": len(new_data),
        "thresholds": THRESHOLDS,
        "metrics": results,
        "alerts": alerts,
        "drift_detected": len(alerts) > 0,
    }

    out_file = os.path.join(week4_dir, f"metrics-{datetime.now().strftime('%Y%m%d')}.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote results to {os.path.basename(out_file)}")

    if alerts:
        print("\nDrift detected -- exiting with code 1 to trigger CI alert.")
        sys.exit(1)
    else:
        print("\nNo critical drift -- monitoring complete.")


if __name__ == "__main__":
    main()
