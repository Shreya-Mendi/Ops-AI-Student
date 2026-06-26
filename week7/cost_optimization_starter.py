"""
Week 7: Cost Optimization & Feedback Loop Starter Template

Implement three systems:
1. CostAnalyzer - analyze and track query costs
2. OptimizationStrategy - optimize costs through caching, model selection, etc.
3. FeedbackLoop - collect and validate user corrections
"""

import json
import logging
import statistics
from typing import Dict, List, Any
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# TASK 1: Implement CostAnalyzer
# ============================================================================


class CostAnalyzer:
    """Analyze and track query costs by component."""

    def __init__(self):
        """Initialize cost analyzer.

        TODO: Initialize empty query history list
        """
        self.query_history = []

    # Component cost fields tracked for every query.
    COMPONENTS = ["retrieval_cost", "llm_cost", "tool_cost", "error_cost"]

    def record_query(self, query: Dict[str, Any]):
        """Record a query and its per-component cost breakdown.

        Accepts a dict with any of: query_text, retrieval_cost, llm_cost,
        tool_cost, error_cost. Missing components default to 0.0. total_cost
        and timestamp are filled in automatically if absent.
        """
        entry = {
            "query_text": query.get("query_text", ""),
            "retrieval_cost": float(query.get("retrieval_cost", 0.0)),
            "llm_cost": float(query.get("llm_cost", 0.0)),
            "tool_cost": float(query.get("tool_cost", 0.0)),
            "error_cost": float(query.get("error_cost", 0.0)),
        }
        entry["total_cost"] = float(
            query.get("total_cost", sum(entry[c] for c in self.COMPONENTS))
        )
        entry["timestamp"] = query.get(
            "timestamp", datetime.now(timezone.utc).isoformat()
        )
        self.query_history.append(entry)

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Sum costs across all recorded queries, by component."""
        retrieval = sum(q["retrieval_cost"] for q in self.query_history)
        llm = sum(q["llm_cost"] for q in self.query_history)
        tool = sum(q["tool_cost"] for q in self.query_history)
        error = sum(q["error_cost"] for q in self.query_history)
        return {
            "retrieval_total": retrieval,
            "llm_total": llm,
            "tool_total": tool,
            "error_total": error,
            "total_daily": retrieval + llm + tool + error,
            "query_count": len(self.query_history),
        }

    def identify_cost_spikes(self) -> List[Dict]:
        """Return queries whose total cost exceeds mean + 2*stdev.

        Needs at least 2 queries to compute a standard deviation; returns
        an empty list otherwise.
        """
        if len(self.query_history) < 2:
            return []

        costs = [q["total_cost"] for q in self.query_history]
        mean = statistics.mean(costs)
        stdev = statistics.stdev(costs)
        threshold = mean + 2 * stdev

        spikes = []
        for q in self.query_history:
            if q["total_cost"] > threshold:
                spikes.append(
                    {
                        "query_text": q["query_text"],
                        "total_cost": q["total_cost"],
                        "threshold": round(threshold, 6),
                        "mean": round(mean, 6),
                        "excess_over_mean": round(q["total_cost"] - mean, 6),
                    }
                )
        return spikes


# ============================================================================
# TASK 2: Implement OptimizationStrategy
# ============================================================================


class OptimizationStrategy:
    """Optimize agent costs through multiple strategies."""

    def __init__(self):
        """Initialize optimization strategy.

        TODO: Initialize cache and strategy tracking
        """
        self.cache = {}  # {query: response}
        self.strategies_applied = []

    # Keywords that mark a query as "complex" (needs the stronger model).
    COMPLEX_KEYWORDS = [
        "analyze", "compare", "design", "explain why", "evaluate",
        "summarize", "recommend", "trade-off", "tradeoff", "implications",
        "pros and cons", "strategy",
    ]

    # Estimated per-strategy savings (used by get_optimization_impact).
    STRATEGY_SAVINGS_PCT = {
        "caching": 40.0,
        "retrieval_reduction": 20.0,
        "model_selection": 25.0,
        "response_compression": 10.0,
    }

    def _normalize(self, query: str) -> str:
        """Normalize a query for cache keys (case/whitespace-insensitive)."""
        return " ".join(query.lower().split())

    def apply_caching(self, query: str, response: str) -> tuple:
        """Return (is_cache_hit, response). Caches on first sight."""
        key = self._normalize(query)
        if key in self.cache:
            self._mark("caching")
            return (True, self.cache[key])
        self.cache[key] = response
        return (False, response)

    def optimize_retrieval_count(self, num_docs: int, top_k: int = 3) -> int:
        """Cap retrieved docs at top_k (default 3) to cut token cost."""
        if num_docs > top_k:
            self._mark("retrieval_reduction")
        return max(1, min(num_docs, top_k))

    def select_model_by_complexity(self, query: str) -> str:
        """Route simple queries to a cheaper model, complex ones to the strong one.

        Complex if the query contains a complex keyword OR is long (>20 words).
        """
        q = query.lower()
        is_complex = any(kw in q for kw in self.COMPLEX_KEYWORDS) or len(
            query.split()
        ) > 20
        self._mark("model_selection")
        if is_complex:
            return "gemini-2.5-pro"
        return "gemini-2.5-flash"  # cheaper/faster for simple queries

    def enable_response_compression(self, response: str, max_sentences: int = 3) -> str:
        """Keep only the first N sentences of a long response."""
        sentences = [s.strip() for s in response.split(".") if s.strip()]
        if len(sentences) <= max_sentences:
            return response
        self._mark("response_compression")
        compressed = ". ".join(sentences[:max_sentences])
        if not compressed.endswith("."):
            compressed += "."
        return compressed

    def _mark(self, strategy: str):
        """Record that a strategy was applied (once per kind)."""
        if strategy not in self.strategies_applied:
            self.strategies_applied.append(strategy)

    def get_optimization_impact(self) -> Dict[str, Any]:
        """Estimate combined cost savings from the strategies actually applied.

        Savings combine multiplicatively (each strategy acts on what remains),
        which is more realistic than naive addition.
        """
        breakdown = {
            s: self.STRATEGY_SAVINGS_PCT[s] for s in self.strategies_applied
        }
        remaining = 1.0
        for pct in breakdown.values():
            remaining *= 1 - pct / 100.0
        total_savings_pct = round((1 - remaining) * 100.0, 1)
        return {
            "total_savings_pct": total_savings_pct,
            "strategies_applied": self.strategies_applied,
            "breakdown": breakdown,
        }


# ============================================================================
# TASK 3: Implement FeedbackLoop
# ============================================================================


class FeedbackLoop:
    """Collect and validate user corrections for continuous improvement."""

    def __init__(self):
        """Initialize feedback loop.

        TODO: Initialize corrections list and validation rules
        """
        self.corrections = []
        # Authority hierarchy for role-based validation
        self.authority = {
            "engineer": 1,
            "hr": 2,
            "finance": 2,
            "manager": 3,
            "executive": 4,
        }

    # Minimum authority level allowed to submit corrections (manager+).
    MIN_AUTHORITY = 3

    def submit_correction(
        self,
        original_query: str,
        original_answer: str,
        corrected_answer: str,
        user_role: str,
    ) -> Dict[str, Any]:
        """Validate a correction and store it if it passes the checks.

        Checks (per the assignment + reading on validating feedback before
        integration):
        1. Role must be known and have manager-level authority (level 3+).
        2. The correction must be more detailed than the original answer.
        """
        level = self.authority.get(user_role)
        if level is None:
            return {"accepted": False, "reason": f"Unknown role: {user_role}"}
        if level < self.MIN_AUTHORITY:
            return {
                "accepted": False,
                "reason": (
                    f"Insufficient authority: {user_role} (level {level}); "
                    f"need level {self.MIN_AUTHORITY}+"
                ),
            }
        if len(corrected_answer.strip()) <= len(original_answer.strip()):
            return {
                "accepted": False,
                "reason": "Correction must be more detailed than the original",
            }

        self.corrections.append(
            {
                "original_query": original_query,
                "original_answer": original_answer,
                "corrected_answer": corrected_answer,
                "user_role": user_role,
                "authority_level": level,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "valid": True,
            }
        )
        return {"accepted": True, "reason": "Correction accepted"}

    def validate_correction(self, index: int) -> bool:
        """Re-check a stored correction's quality by index."""
        if index < 0 or index >= len(self.corrections):
            return False
        c = self.corrections[index]
        level = self.authority.get(c["user_role"], 0)
        detailed = len(c["corrected_answer"].strip()) > len(
            c["original_answer"].strip()
        )
        return level >= self.MIN_AUTHORITY and detailed

    def get_feedback_metrics(self) -> Dict[str, Any]:
        """Summary metrics over the stored (accepted) corrections."""
        n = len(self.corrections)
        if n == 0:
            return {
                "total_corrections": 0,
                "validation_rate": 0.0,
                "avg_correction_length": 0.0,
                "top_error_patterns": [],
            }

        valid_count = sum(1 for i in range(n) if self.validate_correction(i))
        avg_len = sum(len(c["corrected_answer"]) for c in self.corrections) / n

        # "Error patterns": most frequently corrected queries.
        counts: Dict[str, int] = {}
        for c in self.corrections:
            q = c["original_query"]
            counts[q] = counts.get(q, 0) + 1
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:3]

        return {
            "total_corrections": n,
            "validation_rate": round(valid_count / n * 100.0, 1),
            "avg_correction_length": round(avg_len, 1),
            "top_error_patterns": [{"query": q, "count": c} for q, c in top],
        }


if __name__ == "__main__":
    # Run with: python3 cost_optimization_starter.py
    logging.getLogger().setLevel(logging.WARNING)  # keep test output clean

    # ===================== CostAnalyzer =====================
    print("Testing CostAnalyzer...")
    analyzer = CostAnalyzer()

    # A baseline of normal queries plus one obvious spike. A real baseline is
    # needed for the 2-sigma rule: with too few normal points, a single huge
    # outlier inflates the stdev so much it hides itself.
    normal = [
        ("What is the PTO policy?", 0.001, 0.002, 0.0005),
        ("Who is employee 5?", 0.0, 0.0015, 0.001),
        ("What is the travel policy?", 0.0012, 0.0022, 0.0006),
        ("What is the expense limit for a manager?", 0.0, 0.0018, 0.0004),
        ("Look up Brian Yang", 0.0, 0.0016, 0.0009),
        ("What is the remote work policy?", 0.0011, 0.0021, 0.0005),
        ("How many PTO days do I get?", 0.0009, 0.0019, 0.0004),
    ]
    for text, r, l, t in normal:
        analyzer.record_query({"query_text": text, "retrieval_cost": r,
                               "llm_cost": l, "tool_cost": t})
    # The spike: an expensive analytical query with retries.
    analyzer.record_query({"query_text": "Analyze all expenses across every department in 2025",
                           "retrieval_cost": 0.05, "llm_cost": 0.08, "tool_cost": 0.02,
                           "error_cost": 0.01})  # spike

    breakdown = analyzer.get_cost_breakdown()
    assert breakdown["query_count"] == 8, "should have 8 queries"
    print(f"  get_cost_breakdown: PASSED  (total=${breakdown['total_daily']:.4f}, "
          f"llm=${breakdown['llm_total']:.4f}, retrieval=${breakdown['retrieval_total']:.4f})")

    spikes = analyzer.identify_cost_spikes()
    assert len(spikes) >= 1, "should detect the expensive query"
    assert "Analyze all expenses" in spikes[0]["query_text"], "wrong spike identified"
    print(f"  identify_cost_spikes: PASSED  ({len(spikes)} spike, "
          f"${spikes[0]['total_cost']:.4f} > threshold ${spikes[0]['threshold']:.4f})")

    # ===================== OptimizationStrategy =====================
    print("\nTesting OptimizationStrategy...")
    optimizer = OptimizationStrategy()

    hit1, _ = optimizer.apply_caching("What is the travel policy?", "Pre-approve travel.")
    hit2, resp2 = optimizer.apply_caching("what is the   TRAVEL policy?", "ignored")
    assert hit1 is False, "first call should miss"
    assert hit2 is True and resp2 == "Pre-approve travel.", "second (normalized) call should hit"
    print("  apply_caching: PASSED  (miss then normalized hit)")

    assert optimizer.optimize_retrieval_count(15) == 3, "15 docs -> top 3"
    assert optimizer.optimize_retrieval_count(2) == 2, "few docs unchanged"
    print("  optimize_retrieval_count: PASSED  (15 -> 3)")

    assert optimizer.select_model_by_complexity("What is the PTO policy?") == "gemini-2.5-flash"
    assert optimizer.select_model_by_complexity(
        "Analyze and compare Q3 vs Q4 expenses by department") == "gemini-2.5-pro"
    print("  select_model_by_complexity: PASSED  (simple->flash, complex->pro)")

    long_resp = "First point. Second point. Third point. Fourth point. Fifth point."
    compressed = optimizer.enable_response_compression(long_resp, max_sentences=3)
    assert compressed.count(".") == 3, "should keep 3 sentences"
    print(f"  enable_response_compression: PASSED  ('{compressed}')")

    impact = optimizer.get_optimization_impact()
    assert impact["total_savings_pct"] > 0, "should report savings"
    print(f"  get_optimization_impact: PASSED  ({impact['total_savings_pct']}% savings, "
          f"{len(impact['strategies_applied'])} strategies)")

    # ===================== FeedbackLoop =====================
    print("\nTesting FeedbackLoop...")
    feedback = FeedbackLoop()

    # Manager+ with a more detailed correction -> accepted.
    r1 = feedback.submit_correction(
        "What is the travel policy for flights over 8 hours?",
        "There is no specific policy for 8+ hour flights.",
        "Employees can book business class for flights over 8 hours with manager approval.",
        "manager",
    )
    assert r1["accepted"] is True, "manager correction should be accepted"
    print(f"  submit_correction (manager): PASSED  ({r1['reason']})")

    # Engineer lacks authority -> rejected.
    r2 = feedback.submit_correction(
        "What is the travel policy?", "Old answer.",
        "A much longer and more detailed corrected answer here.", "engineer",
    )
    assert r2["accepted"] is False, "engineer correction should be rejected"
    print(f"  submit_correction (engineer): PASSED  (rejected: {r2['reason']})")

    # Executive but correction not more detailed -> rejected.
    r3 = feedback.submit_correction(
        "What is X?", "A long original answer that is quite detailed.",
        "Short.", "executive",
    )
    assert r3["accepted"] is False, "too-short correction should be rejected"
    print(f"  submit_correction (too short): PASSED  (rejected: {r3['reason']})")

    metrics = feedback.get_feedback_metrics()
    assert metrics["total_corrections"] == 1, "only 1 accepted correction stored"
    assert metrics["validation_rate"] == 100.0, "the stored correction is valid"
    print(f"  get_feedback_metrics: PASSED  ({metrics['total_corrections']} correction, "
          f"{metrics['validation_rate']}% valid, "
          f"avg_len={metrics['avg_correction_length']})")

    print("\nAll tests passed!")
