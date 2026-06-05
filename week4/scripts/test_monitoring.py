"""
test_monitoring.py — Unit tests for MetricComputer and drift detection functions.

Run from week4/ directory:
    python3 -m pytest scripts/test_monitoring.py -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from metric_template import MetricComputer
from detect_drift import (
    compute_psi,
    detect_feature_drift,
    detect_temporal_peak_shift,
    detect_manhattan_lag_deflation,
    detect_outer_borough_scramble,
    detect_manhattan_weekend_concept_drift,
    detect_concept_drift_by_segment,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_baseline(n=500, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    zones = [4, 13, 24, 41, 79]
    n_zones = len(zones)
    rows_per_zone = n // n_zones

    dfs = []
    for i, zone in enumerate(zones):
        tb = pd.date_range("2026-01-01", periods=rows_per_zone, freq="15min")
        trip_count = rng.integers(2, 30, size=rows_per_zone)
        df = pd.DataFrame({
            "PULocationID": zone,
            "time_bucket": tb,
            "trip_count": trip_count,
            "hour": tb.hour,
            "slot_of_day": (tb.hour * 4 + tb.minute // 15),
            "is_weekend": (tb.dayofweek >= 5).astype(int),
            "borough_id": i % 3,
            "zone_slot_baseline": trip_count.astype(float) * 0.9 + rng.random(rows_per_zone),
            "lag_1day": trip_count.astype(float) + rng.normal(0, 1, rows_per_zone),
            "lag_1week": trip_count.astype(float) + rng.normal(0, 1, rows_per_zone),
            "roll_mean_1day": trip_count.astype(float) + rng.normal(0, 0.5, rows_per_zone),
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def make_drifted(baseline: pd.DataFrame, seed=99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = baseline.copy()
    # Inflate trip_count by ~50% to create clear drift
    df["trip_count"] = (df["trip_count"] * 1.5 + rng.integers(0, 5, len(df))).clip(0)
    df["trip_count"] = df["trip_count"].astype(int)
    return df


# ── MetricComputer tests ──────────────────────────────────────────────────────

class TestMetricComputer:

    def setup_method(self):
        self.baseline = make_baseline()
        self.computer = MetricComputer(self.baseline)

    def test_null_rates_clean_data(self):
        result = self.computer.metric_3_null_rates(self.baseline)
        assert isinstance(result, dict)
        for field, rate in result.items():
            if rate is not None:
                assert rate == 0.0, f"Expected 0% nulls, got {rate} for {field}"

    def test_null_rates_with_nulls(self):
        dirty = self.baseline.copy()
        dirty.loc[dirty.index[:50], "lag_1day"] = np.nan
        result = self.computer.metric_3_null_rates(dirty)
        assert result["lag_1day"] > 0.0

    def test_null_rates_above_threshold(self):
        dirty = self.baseline.copy()
        # Inject >1% nulls
        n_nulls = int(len(dirty) * 0.05)
        dirty.loc[dirty.index[:n_nulls], "trip_count"] = np.nan
        result = self.computer.metric_3_null_rates(dirty)
        assert result["trip_count"] > 0.01

    def test_ks_test_no_drift(self):
        # Same data should not show drift
        result = self.computer.metric_4_ks_test(self.baseline)
        for feat, r in result.items():
            assert not r["drifted"], f"Expected no drift on baseline vs itself for {feat}"

    def test_ks_test_detects_drift(self):
        drifted = make_drifted(self.baseline)
        result = self.computer.metric_4_ks_test(drifted)
        assert result["trip_count"]["drifted"], "Expected KS test to detect 50% trip_count inflation"

    def test_psi_stable_data(self):
        # Baseline vs itself should give near-zero PSI
        psi = self.computer.metric_5_psi(self.baseline)
        assert psi < 0.05, f"PSI for identical data should be near 0, got {psi}"

    def test_psi_drifted_data(self):
        drifted = make_drifted(self.baseline)
        psi = self.computer.metric_5_psi(drifted)
        assert psi > 0.10, f"PSI for heavily drifted data should exceed 0.10, got {psi}"

    def test_psi_returns_float(self):
        psi = self.computer.metric_5_psi(self.baseline)
        assert isinstance(psi, float)
        assert psi >= 0.0

    def test_data_freshness_recent(self):
        df = self.baseline.copy()
        # Use UTC-naive timestamp to match metric_7's comparison against utcnow()
        df["time_bucket"] = pd.Timestamp.utcnow().tz_localize(None)
        result = self.computer.metric_7_data_freshness(df)
        assert result["stale"] is False
        assert result["age_hours"] < 1

    def test_data_freshness_stale(self):
        df = self.baseline.copy()
        df["time_bucket"] = pd.Timestamp("2020-01-01")
        result = self.computer.metric_7_data_freshness(df)
        assert result["stale"] is True
        assert result["age_hours"] > 48

    def test_duplicate_rate_no_dupes(self):
        result = self.computer.metric_8_duplicate_rate(self.baseline)
        assert result["count"] == 0
        assert result["rate"] == 0.0
        assert result["alert"] is False

    def test_duplicate_rate_with_dupes(self):
        duped = pd.concat([self.baseline, self.baseline.head(50)], ignore_index=True)
        result = self.computer.metric_8_duplicate_rate(duped)
        assert result["count"] == 50
        assert result["rate"] > 0.0

    def test_accuracy_perfect(self):
        predictions = self.baseline["trip_count"].values.astype(float)
        actuals = self.baseline["trip_count"].values.astype(float)
        acc = self.computer.metric_1_accuracy(self.baseline, predictions, actuals)
        assert acc == 1.0

    def test_accuracy_terrible(self):
        predictions = np.zeros(len(self.baseline))
        actuals = np.full(len(self.baseline), 100.0)
        acc = self.computer.metric_1_accuracy(self.baseline, predictions, actuals)
        assert acc < 0.1, f"Expected near-zero accuracy for zero predictions vs 100 actuals, got {acc}"

    def test_accuracy_by_zone_returns_dict(self):
        predictions = self.baseline["lag_1day"].values.astype(float)
        actuals = self.baseline["trip_count"].values.astype(float)
        result = self.computer.metric_2_accuracy_by_zone(self.baseline, predictions, actuals)
        assert isinstance(result, dict)
        for zone in self.baseline["PULocationID"].unique():
            assert int(zone) in result

    def test_compute_all_metrics_returns_expected_keys(self):
        result = self.computer.compute_all_metrics(self.baseline)
        assert "null_rates" in result
        assert "ks_test" in result
        assert "psi_trip_count" in result
        assert "data_freshness" in result
        assert "duplicate_rate" in result

    def test_compute_all_metrics_with_predictions(self):
        preds = self.baseline["lag_1day"].values.astype(float)
        acts = self.baseline["trip_count"].values.astype(float)
        result = self.computer.compute_all_metrics(self.baseline, preds, acts)
        assert "accuracy_overall" in result
        assert "accuracy_by_zone" in result


# ── PSI helper tests ──────────────────────────────────────────────────────────

class TestPSI:

    def test_psi_identical_distributions(self):
        vals = np.random.default_rng(0).normal(10, 3, 1000)
        psi = compute_psi(vals, vals)
        assert psi < 0.01

    def test_psi_very_different_distributions(self):
        base = np.random.default_rng(0).normal(10, 2, 1000)
        new = np.random.default_rng(1).normal(30, 2, 1000)
        psi = compute_psi(base, new)
        assert psi > 0.25, f"Expected PSI > 0.25 for very different distributions, got {psi}"

    def test_psi_non_negative(self):
        base = np.random.default_rng(5).exponential(5, 500)
        new = np.random.default_rng(6).exponential(10, 500)
        psi = compute_psi(base, new)
        assert psi >= 0.0


# ── Drift detection function tests ───────────────────────────────────────────

class TestDriftDetection:

    def setup_method(self):
        self.baseline = make_baseline(n=1000)
        self.drifted = make_drifted(self.baseline)

    def test_detect_feature_drift_no_drift(self):
        result = detect_feature_drift(self.baseline, self.baseline, "trip_count")
        assert result["drifted"] is False
        assert result["ks_statistic"] < 0.05

    def test_detect_feature_drift_with_drift(self):
        result = detect_feature_drift(self.baseline, self.drifted, "trip_count")
        assert result["drifted"] is True
        assert result["p_value"] < 0.05

    def test_detect_feature_drift_returns_required_keys(self):
        result = detect_feature_drift(self.baseline, self.drifted, "trip_count")
        for key in ["ks_statistic", "p_value", "drifted", "baseline_mean", "new_mean", "psi"]:
            assert key in result, f"Missing key: {key}"

    def test_detect_feature_drift_psi_positive(self):
        result = detect_feature_drift(self.baseline, self.drifted, "trip_count")
        assert result["psi"] >= 0.0

    def test_concept_drift_by_segment_returns_boroughs(self):
        result = detect_concept_drift_by_segment(self.baseline, self.drifted)
        assert isinstance(result, dict)
        for borough_id in result:
            assert "baseline_mae_rate" in result[borough_id]
            assert "new_mae_rate" in result[borough_id]
            assert "degradation" in result[borough_id]


# ── Integration test ──────────────────────────────────────────────────────────

class TestIntegration:

    def test_temporal_peak_shift_structure(self):
        baseline = make_baseline(n=2000)
        new = make_drifted(baseline)
        result = detect_temporal_peak_shift(baseline, new)
        assert "early_slot_change_pct" in result
        assert "late_slot_change_pct" in result
        assert "drift_confirmed" in result
        assert isinstance(result["drift_confirmed"], bool)

    def test_full_pipeline_no_crash(self):
        """End-to-end: MetricComputer + drift detection on synthetic data."""
        baseline = make_baseline(n=1000)
        new_data = make_drifted(baseline)
        computer = MetricComputer(baseline)

        preds = new_data["lag_1day"].fillna(0).values.astype(float)
        acts = new_data["trip_count"].values.astype(float)

        results = computer.compute_all_metrics(new_data, preds, acts)
        assert results["psi_trip_count"] >= 0.0
        assert 0.0 <= results["accuracy_overall"] <= 1.0
        assert results["duplicate_rate"]["rate"] >= 0.0
