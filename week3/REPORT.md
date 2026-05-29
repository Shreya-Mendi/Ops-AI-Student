# Week 3 Data Quality Report

## Issues Found

| # | Issue | Rows Affected | Model / API Impact | Root Cause |
|---|-------|---------------|--------------------|------------|
| 1 | **Duplicate rows** — zones 4, 43, 87, 107, 229 appear twice in the Jan 16 – Feb 28 window | 10,085 extra rows removed | Inflates demand ~2x for those zones; LightGBM sees phantom spikes and overfits to them | Upstream ingestion pipeline likely processed the same batch twice — a retry-without-deduplication bug |
| 2 | **Out-of-range trip_count** — 353 negative values (−5, −1) and 311 extreme outliers (9,999; 99,999); baseline max is 18 | 664 rows | Corrupts feature scaling and the target distribution; the model's loss function anchors to impossible values | Sentinel error codes (−5, −1 for "no data"; 9,999/99,999 for "overflow") were not stripped before loading — a missing data-cleaning step in the ETL |
| 3 | **Holiday mislabeling** — `is_holiday=1` applied to the entire Jan 7–21 window, not just MLK Day (Jan 20); normal Tuesday and Wednesday commutes are flagged as holidays | 142,752 rows (2.26%) | Holiday surge-pricing logic fires on regular commuter demand, wrong behavioral profiles applied across two weeks | A date-range window was used instead of a point lookup — the MLK Day observance was flagged as a two-week block rather than a single date |
| 4 | **Lag contamination** — `lag_1week` for zones 161, 162, 186 replaced with values from zone 237 (Midtown South) | All rows for those 3 zones | Temporal autoregressive signal uses the wrong zone's history; model learns that Upper West Side demand follows Midtown patterns, breaking weekly rhythm predictions | Zone ID was used as a key in a join that had a misaligned index — zone 237's lag values were written to the wrong zone slots in the feature matrix |

I found the holiday mislabeling by looking at the `is_holiday` rate by date — the Jan 7–21 window had a near-100% holiday rate, which is implausible since MLK Day is one day. The lag contamination is subtler: zones 161/162/186 and source zone 237 all sit around 40–52 trips/slot, so simple magnitude checks don't catch it. The check works by computing each zone's expected `lag_1week` from its own trip count shifted 672 slots (7 days × 24 hours × 4 slots/hour) and flagging zones where the stored lag diverges substantially — a signal that values arrived from somewhere else.

## Validation Approach

Each issue maps to a dedicated check in `DataQualityValidator`:

- **check_duplicates**: `drop_duplicates(key=['PULocationID','time_bucket'])`, counts removed rows, reports affected zones
- **check_value_ranges**: flags `trip_count < 0` (critical) and `trip_count > 5000` (critical); thresholds derived from 50× baseline max
- **check_holiday_mislabeling**: cross-references `is_holiday=1` rows against a known-holidays set; flags if false-holiday rate > 0.1%
- **check_lag_contamination**: computes expected `lag_1week` via self-shift, flags zones where MAE > 20 and relative MAE > 0.5

The validator returns `{is_valid, num_issues, issues}`. Each issue carries `type`, `severity` (`critical` / `high` / `medium`), a plain-English description, and the count of affected rows. The CI workflow exits with code 1 if any issues are found, blocking deployment.

## Validation Schedule

The workflow runs **hourly** (`cron: '0 * * * *'`). NYC taxi data arrives 20–30 minutes after each 15-minute window closes. An hourly check catches a bad batch before it corrupts multiple hours of predictions. One bad hour of demand forecasting means mispriced surge pricing and misallocated fleet positioning across the city — operationally expensive but recoverable. Running every 15 minutes would catch failures faster, but burns GitHub Actions at 4× the rate for a 2–3 minute job, and the marginal detection gain is mostly eaten by the 30-minute data arrival lag. Daily validation misses the entire morning peak if a problem drops in at 8 AM.

## Graceful Degradation Strategy

When the validator catches bad data, it applies the smallest fix that makes data safe to use. The API never crashes — it degrades with full logging.

- **Duplicates**: `drop_duplicates(keep='first')` — duplicate rows are identical, so the first occurrence is authoritative
- **Negative trip_count**: set to 0 — negative trips are physically impossible; zero is the conservative floor
- **Extreme trip_count (> 5,000)**: replace with the zone's own median from clean rows — preserves zone-level demand character without propagating the injected spike
- **Lag contamination**: recompute `lag_1week` for affected zones by shifting their own `trip_count` 672 slots — deterministic and self-contained, no external data needed
- **Holiday mislabeling**: log warning and alert operators — do not auto-fix; the system cannot know which calendar authority is correct without human confirmation
- **Complete load failure**: fall back to the last known good dataset cached in memory from the previous successful startup

Each fix emits a `logger.warning` with exact row counts. The `/health/ready` endpoint (added per Kubernetes readiness probe pattern) returns the validation result so Kubernetes can pull a degraded pod from rotation when load fails entirely. The `validation-results.json` artifact uploaded on every CI run gives operators the full issue list without log-diving.

All three validation layers from the reading are implemented: CI/CD pipeline (the workflow), startup validation (in `_load()`), and readiness monitoring (the `/health/ready` endpoint).
