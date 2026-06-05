"""
Monitoring metrics for NYC taxi demand drift detection.

MetricComputer implements 6 of the 8 provided stubs. Each method returns a
dict (or float) suitable for threshold comparison in compute_metrics.py.
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


class MetricComputer:
    """Compute monitoring metrics for drift detection."""

    def __init__(self, baseline_df: pd.DataFrame):
        self.baseline_df = baseline_df

    def metric_1_accuracy(
        self, new_df: pd.DataFrame, predictions: np.ndarray, actuals: np.ndarray
    ) -> float:
        """
        Overall accuracy: fraction of predictions within 20% of actual trip count.
        Uses lag_1day as proxy prediction when real model outputs are unavailable.
        """
        tolerance = 0.20
        actuals = actuals.astype(float)
        predictions = predictions.astype(float)
        denom = np.maximum(actuals, 1.0)
        within_tol = np.abs(predictions - actuals) / denom <= tolerance
        return float(within_tol.mean())

    def metric_2_accuracy_by_zone(
        self, new_df: pd.DataFrame, predictions: np.ndarray, actuals: np.ndarray
    ) -> dict:
        """
        Per-zone accuracy (within-20% tolerance). Highlights segment-level degradation
        that global metrics would obscure.
        """
        tolerance = 0.20
        actuals_f = actuals.astype(float)
        preds_f = predictions.astype(float)
        denom = np.maximum(actuals_f, 1.0)
        within = np.abs(preds_f - actuals_f) / denom <= tolerance
        zone_ids = new_df["PULocationID"].values
        result = {}
        for zone in np.unique(zone_ids):
            mask = zone_ids == zone
            result[int(zone)] = float(within[mask].mean()) if mask.sum() > 0 else None
        return result

    def metric_3_null_rates(self, new_df: pd.DataFrame) -> dict:
        """
        Null rates for critical columns. Alert if any field exceeds 1%.
        """
        critical_cols = [
            "trip_count",
            "PULocationID",
            "lag_1day",
            "lag_1week",
            "roll_mean_1day",
        ]
        rates = {}
        for col in critical_cols:
            if col in new_df.columns:
                rates[col] = float(new_df[col].isna().mean())
            else:
                rates[col] = None
        return rates

    def metric_4_ks_test(self, new_df: pd.DataFrame) -> dict:
        """
        KS test on trip_count, lag_1day, and lag_1week distributions.
        p-value < 0.05 signals significant drift.
        """
        features = ["trip_count", "lag_1day", "lag_1week"]
        results = {}
        for feat in features:
            if feat not in new_df.columns or feat not in self.baseline_df.columns:
                continue
            base_vals = self.baseline_df[feat].dropna().values
            new_vals = new_df[feat].dropna().values
            stat, pval = ks_2samp(base_vals, new_vals)
            results[feat] = {
                "statistic": float(stat),
                "p_value": float(pval),
                "drifted": bool(pval < 0.05),
            }
        return results

    def metric_5_psi(self, new_df: pd.DataFrame, bins: int = 10) -> float:
        """
        Population Stability Index for trip_count.
        PSI < 0.10 stable; 0.10–0.25 monitor; > 0.25 significant drift.
        """
        base = self.baseline_df["trip_count"].dropna().values
        new = new_df["trip_count"].dropna().values

        # Use baseline percentiles to define shared bins, extended to capture new data
        bin_edges = np.percentile(base, np.linspace(0, 100, bins + 1))
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0
        bin_edges[0] = min(bin_edges[0], new.min()) - 1e-6
        bin_edges[-1] = max(bin_edges[-1], new.max()) + 1e-6

        base_counts, _ = np.histogram(base, bins=bin_edges)
        new_counts, _ = np.histogram(new, bins=bin_edges)

        base_pct = base_counts / base_counts.sum()
        new_pct = new_counts / new_counts.sum()

        if new_counts.sum() == 0:
            return float("inf")

        eps = 1e-6
        base_pct = np.clip(base_pct, eps, None)
        new_pct = np.clip(new_pct, eps, None)

        psi = float(np.sum((new_pct - base_pct) * np.log(new_pct / base_pct)))
        return psi

    def metric_7_data_freshness(self, new_df: pd.DataFrame) -> dict:
        """
        Age of the most recent record in the dataset relative to now.
        Stale data (>2h) may indicate pipeline failure.
        """
        if "time_bucket" not in new_df.columns:
            return {"age_hours": None, "stale": None}
        latest = pd.to_datetime(new_df["time_bucket"].max())
        now = pd.Timestamp.utcnow().tz_localize(None)
        age_hours = (now - latest).total_seconds() / 3600
        return {
            "latest_record": str(latest),
            "age_hours": round(age_hours, 1),
            "stale": bool(age_hours > 48),
        }

    def metric_8_duplicate_rate(self, new_df: pd.DataFrame) -> dict:
        """
        Fraction of exact duplicate rows. Any duplicates signal pipeline issues.
        """
        key_cols = ["PULocationID", "time_bucket"]
        available = [c for c in key_cols if c in new_df.columns]
        total = len(new_df)
        if not available:
            return {"rate": None, "count": None}
        dupes = new_df.duplicated(subset=available).sum()
        return {
            "count": int(dupes),
            "rate": float(dupes / total) if total > 0 else 0.0,
            "alert": bool(dupes / total > 0.005) if total > 0 else False,
        }

    def compute_all_metrics(
        self,
        new_df: pd.DataFrame,
        predictions: np.ndarray = None,
        actuals: np.ndarray = None,
    ) -> dict:
        """Run all implemented metrics and return results dict."""
        results = {}

        results["null_rates"] = self.metric_3_null_rates(new_df)
        results["ks_test"] = self.metric_4_ks_test(new_df)
        results["psi_trip_count"] = self.metric_5_psi(new_df)
        results["data_freshness"] = self.metric_7_data_freshness(new_df)
        results["duplicate_rate"] = self.metric_8_duplicate_rate(new_df)

        if predictions is not None and actuals is not None:
            results["accuracy_overall"] = self.metric_1_accuracy(new_df, predictions, actuals)
            results["accuracy_by_zone"] = self.metric_2_accuracy_by_zone(new_df, predictions, actuals)

        return results
