# Week 4 Report: Monitoring, Drift Detection & Retraining Strategy

**NYC Taxi Demand Forecasting System**
Shreya Mendi | Duke MEng AIPI

---

## 1. Drift Detection Report

### Overview

We compared **baseline data (Jan 1–15, 2026)** against **new data (Feb 2–28, 2026)** to identify performance degradation and distributional shifts. Four distinct drift patterns were confirmed using KS tests, PSI, and segment-level analysis.

---

### Pattern 1: Temporal Peak Shift (Data Drift)

**What drifted:** The intraday demand shape changed — early morning slots (5–7am, `slot_of_day` 20–27) increased while late morning slots (9–11am, `slot_of_day` 36–43) decreased.

| Segment | Baseline Mean | Feb Mean | Change | KS Statistic | p-value |
|---------|--------------|----------|--------|-------------|---------|
| Early slots (5–7am) | 2.8 trips | 3.6 trips | **+28.1%** | 0.051 | 1.7×10⁻¹⁰ |
| Late slots (9–11am) | 14.9 trips | 8.6 trips | **−42.4%** | 0.184 | 1.8×10⁻¹³⁰ |

**Type:** Data drift (input distribution shift)

**Root cause hypothesis:** Commuter behavior change — more pre-rush activity, fewer mid-morning trips. Could indicate hybrid work schedule shift (later office start times reducing the 9–11am peak).

**Impact:** Models trained on the historical peak pattern will over-predict 9–11am demand and under-predict 5–7am demand, leading to systematic misallocation of driver dispatch.

---

### Pattern 2: Manhattan Lag Feature Deflation (Data Drift)

**What drifted:** Lag features for Manhattan zones (borough_id=0) dropped to ~55% of baseline values. The actual trip counts are less affected, meaning the model's primary input features are biased downward.

| Feature | Baseline Mean | Feb Mean | Change | PSI |
|---------|--------------|----------|--------|-----|
| `lag_1day` | 13.491 | 7.573 | **−43.9%** | 0.222 |
| `lag_1week` | 11.554 | 7.798 | **−32.5%** | 0.120 |
| `roll_mean_1day` | 13.617 | 7.587 | **−44.3%** | **0.484** |

All three features show KS p-values of 0.00 (effectively 0). PSI for `roll_mean_1day` = 0.484 is far above the critical threshold of 0.25.

**Type:** Data drift (feature distribution shift)

**Root cause hypothesis:** Data pipeline issue in Manhattan — likely a lag computation bug or upstream data source that dropped some Manhattan records. The actual trip_count is less impacted than the lag features, suggesting the lag is computed from a different source.

**Impact:** The model consistently under-predicts Manhattan demand because its primary predictive features are artificially low.

---

### Pattern 3: Outer Borough Baseline Scramble (Data Drift)

**What drifted:** Feature-target correlation for `zone_slot_baseline` degraded in Brooklyn (0.787 → 0.630, delta −0.157). Affects Queens (4 zones) and Brooklyn (1 zone) where `zone_slot_baseline` was scrambled with alternating 0.22× and 3.8× zone-level multipliers.

| Borough | Baseline F-T Corr | New F-T Corr | Degradation |
|---------|--------------------|--------------|-------------|
| Queens | 0.411 | 0.422 | +0.011 |
| Brooklyn | 0.787 | 0.630 | **−0.157** |

Brooklyn's correlation drop was detected via threshold (>0.05 degradation). Queens correlation improved slightly due to the net effect of alternating multipliers. Queens MAE rate degraded significantly (+0.446 from 0.524 to 0.970).

**Type:** Data drift (feature-target relationship break)

**Root cause hypothesis:** Data quality issue in the zone baseline calculation — possibly a zone metadata table update that scrambled `zone_slot_baseline` values for select zones in outer boroughs.

**Impact:** `zone_slot_baseline` is no longer a reliable predictor for affected zones. The model's reliance on this feature in outer boroughs drives unpredictable errors.

---

### Pattern 4: Manhattan Weekend Concept Drift (Concept Drift)

**What drifted:** Manhattan weekend (`is_weekend=1`) trip counts fell to ~69% of baseline while weekday demand was nearly stable (−3.7%). This is concept drift — the demand behavior itself changed, not just a feature artifact.

| Segment | Baseline Mean | Feb Mean | Change | KS p-value |
|---------|--------------|----------|--------|-----------|
| Manhattan weekend | 13.02 | 8.95 | **−31.3%** | 2.5×10⁻¹²² |
| Manhattan weekday | (stable) | — | −3.7% | — |

**MAE rate on Manhattan weekends:** 0.852 → 0.979 (delta **+0.194**) — significant accuracy degradation.

**Type:** Concept drift (target-generating process changed)

**Root cause hypothesis:** Tourism decline and hybrid work adoption changed Manhattan weekend travel patterns. Fewer tourists visiting Manhattan → reduced taxi demand on weekends. The lag features still reflect past higher-demand weekends, so the model consistently over-predicts.

**Impact:** Systematic over-prediction of Manhattan weekend demand. Requires new training data that reflects the changed behavior — retraining on historical data alone will not fix this.

---

### Summary Table

| Pattern | Type | Severity | Confirmed | Key Metric |
|---------|------|----------|-----------|-----------|
| Temporal peak shift | Data drift | High | Yes | KS p=1.8×10⁻¹³⁰ |
| Manhattan lag deflation | Data drift | Critical | Yes | PSI=0.484 |
| Outer borough scramble | Data drift | Medium | Yes | Brooklyn corr −0.157 |
| Manhattan weekend concept drift | Concept drift | High | Yes | MAE delta +0.194 |

---

## 2. Monitoring Framework

### Metrics Defined

We implemented 6 of the 8 metrics in `metric_template.py`:

| # | Metric | Computation | Baseline | Alert Threshold | Frequency | Segmentation |
|---|--------|-------------|----------|----------------|-----------|-------------|
| 1 | Overall accuracy | % predictions within 20% of actual (`lag_1day` proxy) | ~91% | <80% warning, <70% critical | Daily | Global |
| 2 | Accuracy by zone | Per-zone within-20% accuracy | 85–95% per zone | <75% for any zone | Daily | Per zone (57) |
| 3 | Null rates | `isna().mean()` for critical fields | 0.0% | >1% critical | Daily | Per field |
| 4 | KS test | `scipy.stats.ks_2samp` on `trip_count`, `lag_1day`, `lag_1week` | p > 0.05 (no drift) | p < 0.05 | Daily | Global + by borough |
| 5 | PSI (trip_count) | Bin-based Population Stability Index | 0.0 | >0.10 warning, >0.25 critical | Daily | Global |
| 7 | Data freshness | Age of most recent `time_bucket` vs. UTC now | <2h | >48h stale | Every run | Global |
| 8 | Duplicate rate | Exact duplicate `(PULocationID, time_bucket)` pairs | 0.0% | >0.5% | Daily | Global |

### Alert Thresholds Rationale

- **Accuracy <80%:** Below this, driver dispatch efficiency degrades noticeably.
- **PSI >0.25:** Industry standard for "significant change — investigate and consider retraining."
- **KS p <0.05:** Standard statistical significance. With 147K rows, even small true shifts hit this — use alongside PSI to avoid over-alerting.
- **Null rate >1%:** Zero in healthy baseline; any null rate increase signals pipeline degradation.

### Monitoring Schedule: Daily at Midnight UTC

**Frequency chosen:** Daily (cron `0 0 * * *`)

**Justification:**

| Concern | Analysis |
|---------|----------|
| Ground truth lag | Actual trip counts finalized ~24h after collection. Daily monitoring aligns with data availability. |
| Drift speed | Demand patterns shift over days/weeks, not hours. Daily resolution sufficient. |
| CI cost | 365 runs/year vs. 2,920 for 4-hourly. 8× cost savings with no meaningful detection lag. |
| Ops action window | Daily alert at midnight gives ops a clear 9am briefing window to act before peak demand. |

For proactive detection (feature drift, before accuracy drops), the same daily job runs `detect_drift.py` separately — this catches distribution shifts 1–2 days before accuracy metrics degrade.

---

## 3. Retraining Strategy

### Trigger Conditions

Retraining is triggered when **any two** of the following conditions fire simultaneously (to avoid false-positive retraining):

| Condition | Threshold | Type |
|-----------|-----------|------|
| Overall accuracy | <80% for 3 consecutive days | Reactive |
| PSI (trip_count) | >0.25 | Proactive |
| KS p-value | <0.01 (very strong drift) on 2+ features | Proactive |
| Per-zone accuracy | <75% in 5+ zones for 2+ days | Reactive |
| Feature-target correlation | Drop >0.10 for any borough | Proactive |

**Why dual-condition:** Trip count shows natural weekly seasonality that can temporarily trigger single-metric thresholds. Requiring two signals reduces unnecessary retraining (cost: ~$50 compute + 2h engineer time per run).

### Retraining Pipeline

```
Drift detected → Assess severity (PSI, KS, accuracy) → If threshold exceeded:
  1. Collect last 30 days of labeled data
  2. Retrain model (same architecture) on rolling window
  3. Offline validation (holdout: most recent 7 days)
  4. If new_accuracy >= current_accuracy - 0.02:
       Deploy via shadow mode (1 week) → promote to primary
     Else:
       Keep current model, file investigation ticket
```

**Training data window:** Last 30 days of ground truth (not full history). Rationale: demand patterns change seasonally. Including 2023 data would train the model on patterns that no longer hold.

**Shadow mode validation:** New model runs in parallel with current model for 1 week. If shadow model accuracy matches or exceeds current model in production, promote. This catches regressions that offline validation misses.

### Validation Approach

| Stage | Method | Pass Criteria |
|-------|--------|--------------|
| Offline | Holdout on 7-day test set | Accuracy ≥ current − 2% |
| Shadow | Side-by-side with live model | Accuracy ≥ current − 1% over 7 days |
| Canary | 10% traffic, then 50%, then 100% | No p99 latency regression, accuracy holds |
| Rollback | Automatic if canary accuracy < 70% | Restore previous model version immediately |

### Model Versioning

| Concern | Approach |
|---------|----------|
| Storage | GCS bucket: `gs://<project>/models/demand/v{YYYYMMDD}/` |
| Metadata | `model_card.json`: training date, accuracy, data window, PSI at time of training, git SHA |
| Retention | Keep last 5 versions (rollback window ~5 retraining cycles, typically 5–10 weeks) |
| Rollback trigger | Automatic if canary accuracy drops below 70%; manual if stakeholder flags issues |

### Retraining Frequency

**Expected cadence:** Monthly (based on Feb data — drift built up over 4 weeks).

**On-demand:** Any time dual-condition threshold fires outside the scheduled window.

**Not weekly:** Model retraining without clear signal (PSI < 0.10, accuracy stable) deploys models that may perform worse. Always validate before deploying.

---

## 4. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────┘

  Baseline Data                   Live Data Pipeline
  (Jan 1-15, 2026)                (daily ingestion)
        │                               │
        └──────────┬────────────────────┘
                   │
          ┌────────▼────────┐
          │ compute_metrics  │  ← runs daily at midnight (GHA cron)
          │   (6 metrics)    │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │  detect_drift   │  ← KS test, PSI, segment analysis
          │  (4 patterns)   │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │  Threshold      │
          │  Check          │
          └────┬───────┬───┘
               │       │
          [OK] │       │ [Alert]
               │       │
               │  ┌────▼─────────┐
               │  │  GitHub Issue │  ← drift-alert label
               │  │  + Artifact   │    metrics-*.json
               │  └────┬─────────┘    drift-*.json
               │       │
               │  ┌────▼─────────────────────────────┐
               │  │  Ops Assessment                   │
               │  │  - Which patterns confirmed?      │
               │  │  - PSI > 0.25? → Must retrain     │
               │  │  - Accuracy < 80%? → Must retrain │
               │  └────┬─────────────────────────────┘
               │       │
               │  ┌────▼─────────┐
               │  │  Retraining  │
               │  │  Pipeline    │
               │  │  (30-day     │
               │  │   window)    │
               │  └────┬─────────┘
               │       │
               │  ┌────▼─────────────────────────────┐
               │  │  Validation                       │
               │  │  Offline → Shadow → Canary        │
               │  └────┬─────────────────────────────┘
               │       │
               │  ┌────▼─────────┐
               │  │  Deploy or   │
               │  │  Rollback    │
               │  └─────────────┘
               │
        ┌──────▼──────┐
        │  No action  │
        │  (log only) │
        └─────────────┘
```

---

## 5. Code Summary

| File | Description |
|------|-------------|
| `scripts/metric_template.py` | `MetricComputer` class: 6 metrics (accuracy overall, by zone, null rates, KS, PSI, freshness, duplicates) |
| `scripts/compute_metrics.py` | Loads baseline + Feb data, runs all metrics, writes `metrics-YYYYMMDD.json`, exits 1 on alerts |
| `scripts/detect_drift.py` | Detects 4 drift patterns with KS/PSI/correlation, writes `drift-YYYYMMDD.json` |
| `scripts/test_monitoring.py` | 27 pytest unit tests: all passing |
| `.github/workflows/monitor-drift.yml` | Daily cron at midnight UTC; runs tests → metrics → drift → alerts |

### Key Results from Feb 2-28 Analysis

- **Overall accuracy (lag proxy):** 25.2% — far below 70% threshold (lag deflation is primary driver)
- **KS drift detected:** `trip_count`, `lag_1day`, `lag_1week` (all p ≪ 0.05)
- **PSI for `roll_mean_1day` (Manhattan):** 0.484 — critical (>0.25)
- **All 4 drift patterns confirmed**
- **27/27 tests passing**

---

## 6. Key Decisions & Trade-offs

| Decision | Choice | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Monitoring frequency | Daily at midnight | Every 4 hours | Ground truth 24h lag makes sub-daily monitoring wasteful; 8× cost savings |
| Retraining trigger | Dual-condition (2 metrics) | Single KS threshold | Reduces false-positive retraining; taxi demand has natural seasonality |
| Training window | Last 30 days | Full history | Recent drift matters more; old data trains the model on defunct patterns |
| Validation | Shadow → canary | Offline only | Catches production regressions that holdout tests miss |
| Accuracy proxy | `lag_1day` vs `trip_count` | Real model outputs | No deployed model available; lag_1day is the strongest feature and makes drift visible |
