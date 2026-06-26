# Week 7 Report: Cost Optimization & Continuous Learning

**TechCorp Agent — Cost Analysis, Optimization & Feedback Loop**
Shreya Mendi | Duke MEng AIPI

---

## 1. Overview

This final week adds three systems on top of the Week 5–6 agent:

| System | Class | Purpose |
|--------|-------|---------|
| **Cost analysis** | `CostAnalyzer` | Track per-query cost by component; flag expensive outliers |
| **Optimization** | `OptimizationStrategy` | Cut cost via caching, retrieval reduction, model selection, compression |
| **Feedback loop** | `FeedbackLoop` | Collect user corrections, validate them, measure impact |

All three live in [`cost_optimization_starter.py`](cost_optimization_starter.py).
The Week 6 agent + guardrails (`app_starter.py`, `access_control_starter.py`) are
copied in as the foundation.

---

## 2. Design

### CostAnalyzer
Records each query's cost split into four components — **retrieval, LLM, tool,
error** — plus a total and timestamp. `get_cost_breakdown()` sums each component
across all queries (so you can see *which* component drives spend). 

**Spike detection** uses the standard statistical outlier rule: a query is a spike
if its total cost exceeds **mean + 2·stdev** of all query costs. A key practical
lesson surfaced here: with too few "normal" queries, a single huge outlier inflates
the standard deviation enough to hide *itself*. A realistic baseline of normal
traffic is required before the 2-sigma rule flags anything — so the test seeds 7
normal queries before the 1 spike.

### OptimizationStrategy
Five strategies, each independently testable:
- **`apply_caching`** — returns `(is_hit, response)`. Cache key is the **normalized**
  query (lowercased, whitespace-collapsed), so `"What is the   TRAVEL policy?"` hits
  the cache entry for `"what is the travel policy?"`.
- **`optimize_retrieval_count`** — caps retrieved docs at top-k (default 3), cutting
  token cost on the retrieval path.
- **`select_model_by_complexity`** — routes simple queries to the cheaper
  `gemini-2.5-flash`; complex ones (keywords like *analyze/compare/design*, or >20
  words) to `gemini-2.5-pro`.
- **`enable_response_compression`** — keeps the first N sentences of long answers.
- **`get_optimization_impact`** — estimates combined savings **multiplicatively**
  (each strategy acts on what remains), which is more honest than adding percentages.

### FeedbackLoop
Collects corrections to the agent's answers and **validates before storing** — the
reading's key warning is that unvalidated feedback can corrupt the system (the
"cascade" failure mode). Two checks:
1. **Authority** — the role must be manager-level or above (authority level ≥ 3).
   Engineers (level 1) cannot submit corrections.
2. **Detail** — the correction must be longer/more detailed than the original answer.

`get_feedback_metrics()` reports total corrections, validation rate, average
correction length, and the most-frequently-corrected queries (signals for retraining
or corpus updates).

---

## 3. Test Results — `python3 cost_optimization_starter.py`

```
Testing CostAnalyzer...
  get_cost_breakdown: PASSED  (total=$0.1816, llm=$0.0931, retrieval=$0.0542)
  identify_cost_spikes: PASSED  (1 spike, $0.1600 > threshold $0.1337)

Testing OptimizationStrategy...
  apply_caching: PASSED  (miss then normalized hit)
  optimize_retrieval_count: PASSED  (15 -> 3)
  select_model_by_complexity: PASSED  (simple->flash, complex->pro)
  enable_response_compression: PASSED  ('First point. Second point. Third point.')
  get_optimization_impact: PASSED  (67.6% savings, 4 strategies)

Testing FeedbackLoop...
  submit_correction (manager): PASSED  (Correction accepted)
  submit_correction (engineer): PASSED  (rejected: Insufficient authority: engineer (level 1); need level 3+)
  submit_correction (too short): PASSED  (rejected: Correction must be more detailed than the original)
  get_feedback_metrics: PASSED  (1 correction, 100.0% valid, avg_len=81.0)

All tests passed!
```

### What each result demonstrates

| Rubric criterion | Evidence |
|------------------|----------|
| Cost analysis (breakdown by component) | `get_cost_breakdown` — $0.18 total, split into llm/retrieval/tool/error |
| Spike detection (expensive queries) | `identify_cost_spikes` — flags the $0.16 analytical query above the $0.1337 threshold |
| Optimization strategies | caching (normalized hit), retrieval 15→3, flash/pro routing, compression, 67.6% combined savings |
| Feedback loop | manager accepted, engineer rejected (authority), too-short rejected (detail), 100% validation rate |

---

## 4. Key Takeaways

- **Breakdown beats a single number.** Knowing total cost tripled is useless; knowing
  *retrieval* tripled (5→10 docs) tells you what to fix. Component-level accounting is
  the whole point.
- **Outlier detection needs a baseline.** The 2-sigma rule only works once you have
  enough normal traffic — a single early outlier hides itself by inflating the stdev.
- **Optimizations compound, not add.** Stacking caching + model selection + retrieval
  reduction is modeled multiplicatively, avoiding the >100% "savings" that naive
  addition produces.
- **Validate feedback before trusting it.** Authority + detail checks stop low-quality
  or unauthorized corrections from polluting the system — the safeguard against the
  feedback-cascade failure mode from the reading.

---

## Course Complete

- **Week 5** — Agent with tools + LLM
- **Week 6** — Access control + guardrails
- **Week 7** — Cost optimization + feedback loops
