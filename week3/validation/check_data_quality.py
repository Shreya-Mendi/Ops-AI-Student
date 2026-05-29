"""
Data Quality Validation for NYC Taxi Demand Data

Detects 4 known corruption patterns:
  1. Duplicate (PULocationID, time_bucket) rows
  2. Out-of-range trip_count (negative or extreme outliers)
  3. Holiday mislabeling (is_holiday=1 on non-holiday dates)
  4. Lag feature contamination in lag_1week
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_KNOWN_HOLIDAYS = {
    (1, 1),   # New Year's Day
    (1, 15),  # MLK Day (2024 observed)
    (1, 16),  # MLK Day (2023 observed)
    (1, 20),  # MLK Day (2025/2026 observed)
    (2, 17),  # Presidents Day
    (3, 17),  # St. Patrick's Day
    (5, 26),  # Memorial Day
    (7, 4),   # Independence Day
    (9, 1),   # Labor Day
    (10, 13), # Columbus Day
    (10, 31), # Halloween
    (11, 11), # Veterans Day
    (11, 27), # Thanksgiving (approximate)
    (12, 24), # Christmas Eve
    (12, 25), # Christmas
    (12, 31), # New Year's Eve
}


class DataQualityValidator:
    """Validates NYC taxi demand data against quality expectations."""

    def __init__(self, baseline_df: Optional[pd.DataFrame] = None):
        self.baseline = baseline_df
        self.issues: List[Dict] = []

        if baseline_df is not None and not baseline_df.empty and 'trip_count' in baseline_df.columns:
            tc = baseline_df['trip_count']
            self._baseline_trip_mean = float(tc.mean())
            self._baseline_trip_std = float(tc.std())
            self._baseline_trip_max = float(tc.max())
        else:
            self._baseline_trip_mean = 5.0
            self._baseline_trip_std = 5.0
            self._baseline_trip_max = 18.0

    def validate(self, df: pd.DataFrame) -> Dict:
        self.issues = []
        self.check_schema(df)
        self.check_duplicates(df)
        self.check_value_ranges(df)
        self.check_holiday_mislabeling(df)
        self.check_lag_contamination(df)
        return {
            'is_valid': len(self.issues) == 0,
            'num_issues': len(self.issues),
            'issues': self.issues,
        }

    def check_schema(self, df: pd.DataFrame):
        required = ['PULocationID', 'time_bucket', 'trip_count', 'hour', 'dayofweek', 'is_holiday']
        missing = [c for c in required if c not in df.columns]
        if missing:
            self._add_issue('schema', 'critical',
                            f"Missing required columns: {missing}",
                            count=len(missing), missing_columns=missing)

    def check_duplicates(self, df: pd.DataFrame):
        key = ['PULocationID', 'time_bucket']
        if not all(c in df.columns for c in key):
            return
        n_dupes = len(df) - df.drop_duplicates(subset=key).shape[0]
        if n_dupes > 0:
            dupe_mask = df.duplicated(subset=key, keep=False)
            affected_zones = sorted(df.loc[dupe_mask, 'PULocationID'].unique().tolist())
            self._add_issue(
                'duplicates', 'high',
                f"{n_dupes:,} duplicate (PULocationID, time_bucket) rows across "
                f"{len(affected_zones)} zones — demand predictions will be inflated ~2x for those zones",
                count=n_dupes,
                affected_zones=affected_zones,
            )

    def check_value_ranges(self, df: pd.DataFrame):
        if 'trip_count' not in df.columns:
            return
        upper = max(self._baseline_trip_max * 50, 5000)

        bad_neg = int((df['trip_count'] < 0).sum())
        if bad_neg > 0:
            self._add_issue(
                'out_of_range_trip_count', 'critical',
                f"{bad_neg:,} rows with negative trip_count (min={int(df['trip_count'].min())}) — "
                "negative trip counts are physically impossible",
                count=bad_neg, direction='negative',
                min_value=int(df['trip_count'].min()),
            )

        bad_ext = int((df['trip_count'] > upper).sum())
        if bad_ext > 0:
            self._add_issue(
                'out_of_range_trip_count', 'critical',
                f"{bad_ext:,} rows with trip_count > {upper:.0f} "
                f"(max={int(df['trip_count'].max())}, baseline max={self._baseline_trip_max:.0f}) — "
                "corrupts LightGBM feature scaling and target distribution",
                count=bad_ext, direction='extreme_high',
                threshold=upper,
                max_value=int(df['trip_count'].max()),
            )

    def check_holiday_mislabeling(self, df: pd.DataFrame):
        if 'is_holiday' not in df.columns or 'time_bucket' not in df.columns:
            return

        holiday_rows = df[df['is_holiday'] == 1]
        if len(holiday_rows) == 0:
            return

        tb = pd.to_datetime(holiday_rows['time_bucket'])
        false_mask = ~tb.apply(lambda t: (t.month, t.day) in _KNOWN_HOLIDAYS)
        false_count = int(false_mask.sum())
        false_rate = false_count / len(df)

        if false_rate > 0.001:
            # Surface which dates are the main offenders
            false_dates = tb[false_mask].dt.date.value_counts().head(5).to_dict()
            self._add_issue(
                'holiday_mislabeling', 'medium',
                f"{false_count:,} rows ({false_rate:.2%} of dataset) have is_holiday=1 on non-holiday dates — "
                "holiday surge pricing logic fires on normal commuter demand",
                count=false_count,
                false_holiday_rate=round(false_rate, 5),
                example_false_dates={str(k): v for k, v in false_dates.items()},
            )

    def check_lag_contamination(self, df: pd.DataFrame):
        """
        Detect zones whose lag_1week values come from a different zone.
        Method: for each zone, compute the expected lag by shifting the zone's own
        trip_count 672 slots (7 days × 24 hours × 4 slots). Zones where the stored
        lag_1week disagrees substantially with the computed lag (MAE > 20, excluding
        duplicate-affected zones) are flagged as contaminated.
        """
        if 'lag_1week' not in df.columns or 'PULocationID' not in df.columns:
            return
        if 'time_bucket' not in df.columns:
            return

        # Zones already flagged for duplicates — their computed lag will be wrong
        # due to duplicated time indices, so exclude them from this check.
        dup_zones: set = set()
        for issue in self.issues:
            if issue['type'] == 'duplicates':
                dup_zones.update(issue.get('affected_zones', []))

        flagged_zones = []
        for zone_id, zone_df in df.groupby('PULocationID'):
            if zone_id in dup_zones:
                continue
            zd = zone_df.set_index('time_bucket').sort_index()
            expected = zd['trip_count'].shift(672)
            valid = zd['lag_1week'].notna() & expected.notna()
            if valid.sum() < 50:
                continue

            mae = float((zd.loc[valid, 'lag_1week'] - expected[valid]).abs().mean())
            mean_tc = float(zd['trip_count'].replace({v: np.nan for v in zd['trip_count'][zd['trip_count'] > 5000]}).mean())
            if mean_tc > 0 and mae / mean_tc > 0.5 and mae > 20:
                flagged_zones.append(int(zone_id))

        if flagged_zones:
            n_rows = int(df[df['PULocationID'].isin(flagged_zones)].shape[0])
            self._add_issue(
                'lag_contamination', 'high',
                f"lag_1week contamination detected in {len(flagged_zones)} zones: {flagged_zones} — "
                "temporal autoregressive signal uses another zone's history, breaking zone-specific weekly rhythms",
                count=n_rows,
                affected_zones=flagged_zones,
            )

    def _add_issue(self, issue_type: str, severity: str, description: str,
                   count: int = None, **details):
        self.issues.append({
            'type': issue_type,
            'severity': severity,
            'description': description,
            'count': count,
            **details,
        })


def apply_graceful_degradation(df: pd.DataFrame, issues: List[Dict]) -> pd.DataFrame:
    """
    Apply targeted fixes so the API keeps serving with degraded-but-usable data.
    Each fix is logged. Holiday mislabeling is flagged but not auto-corrected
    because fixing calendar labels requires external authority.
    """
    issue_types = {i['type'] for i in issues}
    df = df.copy()

    if 'duplicates' in issue_types:
        before = len(df)
        df = df.drop_duplicates(subset=['PULocationID', 'time_bucket'], keep='first')
        logger.warning(f"[degradation] Dropped {before - len(df):,} duplicate rows")

    if 'out_of_range_trip_count' in issue_types and 'trip_count' in df.columns:
        neg_mask = df['trip_count'] < 0
        ext_mask = df['trip_count'] > 5000
        if neg_mask.any():
            df.loc[neg_mask, 'trip_count'] = 0
            logger.warning(f"[degradation] Set {neg_mask.sum():,} negative trip_count values to 0")
        if ext_mask.any():
            zone_medians = df[~ext_mask].groupby('PULocationID')['trip_count'].median()
            global_median = int(df.loc[~ext_mask, 'trip_count'].median())
            for zone_id, idx in df[ext_mask].groupby('PULocationID').groups.items():
                replacement = int(zone_medians.get(zone_id, global_median))
                df.loc[idx, 'trip_count'] = replacement
            logger.warning(f"[degradation] Replaced {ext_mask.sum():,} extreme trip_count values with zone medians")

    if 'lag_contamination' in issue_types and 'lag_1week' in df.columns:
        for issue in issues:
            if issue['type'] == 'lag_contamination':
                zones = issue.get('affected_zones', [])
                for z in zones:
                    mask = df['PULocationID'] == z
                    zone_df = df[mask].set_index('time_bucket').sort_index()
                    recomputed = zone_df['trip_count'].shift(672)
                    df.loc[mask, 'lag_1week'] = recomputed.values
                logger.warning(
                    f"[degradation] Recomputed lag_1week from own trip_count for zones {zones}"
                )

    if 'holiday_mislabeling' in issue_types:
        logger.warning(
            "[degradation] Holiday mislabeling detected — not auto-corrected. "
            "Calendar authority needed. Operators: check is_holiday labels in the Jan 7–21 window."
        )

    return df


def load_and_validate_data(path: str, baseline_path: str = None) -> pd.DataFrame:
    """Load a parquet file, validate it, and apply graceful degradation if issues are found."""
    baseline_df = None
    if baseline_path:
        try:
            baseline_df = pd.read_parquet(baseline_path)
        except Exception as e:
            logger.warning(f"Could not load baseline for comparison: {e}")

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.error(f"Failed to load data from {path}: {e}")
        if baseline_df is not None:
            logger.warning("Falling back to baseline data")
            return baseline_df
        raise

    validator = DataQualityValidator(baseline_df)
    result = validator.validate(df)

    if not result['is_valid']:
        logger.warning(f"Data quality issues in {path}:")
        for issue in result['issues']:
            logger.warning(f"  [{issue['severity'].upper()}] {issue['type']}: {issue['description']}")
        df = apply_graceful_degradation(df, result['issues'])
        logger.info("Graceful degradation applied — API continuing with cleaned data")
    else:
        logger.info(f"Data validation passed for {path}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    baseline_path = data_dir / "demand_enriched_baseline.parquet"
    corrupted_path = data_dir / "demand_enriched_corrupted.parquet"

    print(f"Loading baseline from {baseline_path} ...")
    baseline = pd.read_parquet(baseline_path)
    validator = DataQualityValidator(baseline)

    print("\n=== Validating baseline ===")
    baseline_result = validator.validate(baseline)
    print(f"Valid: {baseline_result['is_valid']}  Issues: {baseline_result['num_issues']}")

    print(f"\nLoading corrupted data from {corrupted_path} ...")
    corrupted = pd.read_parquet(corrupted_path)

    print("\n=== Validating corrupted data ===")
    corrupted_result = validator.validate(corrupted)
    print(f"Valid: {corrupted_result['is_valid']}  Issues: {corrupted_result['num_issues']}")
    for issue in corrupted_result['issues']:
        sev = issue['severity'].upper()
        print(f"  [{sev}] {issue['type']} ({issue['count']} rows): {issue['description']}")

    output = {
        'status': 'pass' if corrupted_result['is_valid'] else 'fail',
        'checks_run': 4,
        'num_issues': corrupted_result['num_issues'],
        'issues': corrupted_result['issues'],
        'baseline_valid': baseline_result['is_valid'],
    }
    out_path = Path("validation-results.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    sys.exit(0 if corrupted_result['is_valid'] else 1)
