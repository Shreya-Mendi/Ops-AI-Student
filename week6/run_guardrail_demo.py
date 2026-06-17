"""
run_guardrail_demo.py — Demonstrate the Week 6 guardrails end-to-end.

Three demos:
  A. Access control / redaction — same employee query under different roles
     (engineer sees salary [REDACTED]; hr/manager see full).  [uses LLM]
  B. Rate limiting — a low limit, queries blocked after the cap.  [no LLM]
  C. Cost enforcement — engineer budget exhausted, query blocked.  [no LLM]

Only Demo A calls the LLM (paced for the free tier). Demos B and C exercise
the guardrails directly, so they're instant and quota-free.

Run:
    python3 run_guardrail_demo.py
"""

import time
import logging

logging.getLogger().setLevel(logging.WARNING)

from app_starter import Agent
from access_control_starter import RateLimiter, CostEnforcer


def demo_access_control(agent):
    print("=" * 70)
    print("DEMO A — Access Control & Redaction (live agent)")
    print("=" * 70)
    q = "Look up the employee named Brian Yang and show their salary."
    for role in ["engineer", "hr"]:
        print(f"\n[{role}] Q: {q}")
        r = agent.query(q, user_id=f"{role}_user", user_role=role)
        if r.get("error"):
            print(f"     BLOCKED: {r['error']}")
        else:
            ans = r["answer"].replace("\n", " ").strip()
            print(f"     A: {ans[:260]}")
            print(
                f"     tools={r.get('tools_used')} cost=${r.get('cost', 0):.6f} "
                f"budget_left=${r.get('budget_remaining', 0):.2f}"
            )
        time.sleep(30)  # pace for free-tier rate limit


def demo_rate_limit():
    print("\n" + "=" * 70)
    print("DEMO B — Rate Limiting (max 3/min for this demo)")
    print("=" * 70)
    rl = RateLimiter(max_queries_per_minute=3)
    for i in range(1, 6):
        allowed = rl.is_allowed("bob")
        remaining = rl.get_remaining_queries("bob")
        status = "ALLOWED" if allowed else "BLOCKED (rate limit exceeded)"
        print(f"  Query {i}: {status}  (remaining this minute: {remaining})")


def demo_cost_enforcement():
    print("\n" + "=" * 70)
    print("DEMO C — Cost Enforcement (engineer budget = $100/month)")
    print("=" * 70)
    ce = CostEnforcer()
    print(f"  Engineer budget: ${ce.role_budgets['engineer']:.0f}")
    # Simulate spending toward the cap.
    ce.add_cost("eng1", "engineer", 95.0)
    print(f"  After spending $95 -> remaining: ${ce.get_budget_remaining('eng1'):.2f}")
    print(f"  Can afford a $4 query?  {ce.can_afford_query('eng1', 4.0)}  (yes)")
    print(f"  Can afford a $10 query? {ce.can_afford_query('eng1', 10.0)}  (no — over budget)")
    # Executive has a larger budget.
    ce.add_cost("exec1", "executive", 95.0)
    print(f"  Executive after $95 -> remaining: ${ce.get_budget_remaining('exec1'):.2f}")
    print(f"  Executive can afford a $10 query? {ce.can_afford_query('exec1', 10.0)}  (yes — $1000 budget)")


def main():
    # Demos B and C first — they're instant and need no API.
    demo_rate_limit()
    demo_cost_enforcement()

    # Demo A — live agent (uses LLM quota).
    print()
    agent = Agent("data/techcorp.db")
    demo_access_control(agent)

    # Show a sample of the audit log produced during Demo A.
    print("\n" + "=" * 70)
    print("AUDIT LOG SAMPLE (from redaction during Demo A)")
    print("=" * 70)
    log = agent.access_controller.get_audit_log()
    for entry in log[-6:]:
        print(f"  {entry}")


if __name__ == "__main__":
    main()
