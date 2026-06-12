"""
run_tests.py — Run 10 test queries against the agent and print a clean summary.

Covers all 3 tools (employee_lookup, policy_search, expense_query), multi-tool
queries, and a no-result case. Produces output suitable for report screenshots.

Run:
    python3 run_tests.py
"""

import time
import logging

# Quiet the INFO logs so the screenshot output is clean.
logging.getLogger().setLevel(logging.WARNING)

from app_starter import Agent, MODEL_NAME

TEST_QUERIES = [
    # (question, role)
    ("What is the expense approval limit for a manager?", "manager"),
    ("What is the travel policy?", "engineer"),
    ("Look up the employee named Brian Yang.", "engineer"),
    ("How much can a director approve in expenses?", "director"),
    ("What does the company policy say about remote work?", "engineer"),
    ("Find employee with ID 2.", "manager"),
    ("What is the PTO / vacation policy?", "engineer"),
    ("What's the spending limit for a VP versus an IC3?", "vp"),
    ("Tell me about the security or data protection policy.", "engineer"),
    ("What is the expense approval limit for an astronaut?", "engineer"),  # no-match
]


def main():
    print("=" * 70)
    print(f"WEEK 5 AGENT — 10 TEST QUERIES (model: {MODEL_NAME})")
    print("=" * 70)

    agent = Agent("data/techcorp.db")
    print("Agent initialized successfully\n")

    for i, (q, role) in enumerate(TEST_QUERIES, 1):
        print(f"[{i:02d}] Q ({role}): {q}")
        try:
            r = agent.query(q, user_role=role)
            answer = r["answer"].replace("\n", " ").strip()
            if len(answer) > 220:
                answer = answer[:220] + "..."
            print(f"     A: {answer}")
            print(
                f"     tools={r['tools_used']}  "
                f"tokens={r['tokens_used']}  cost=${r['cost']:.6f}"
            )
        except Exception as e:
            print(f"     ERROR: {e}")
        print()
        # Free tier allows 5 requests/min and each query uses ~2 LLM calls.
        # Pace ~30s between queries to stay under the limit. The agent also
        # auto-retries on 429, so this is belt-and-suspenders.
        if i < len(TEST_QUERIES):
            time.sleep(30)

    print("=" * 70)
    m = agent.get_metrics()
    print("FINAL METRICS")
    print(f"  Total queries:       {m['total_queries']}")
    print(f"  Total tokens:        {m['total_tokens']:,}")
    print(f"  Total cost:          ${m['total_cost']:.6f}")
    print(f"  Avg cost per query:  ${m['avg_cost_per_query']:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
