"""
Week 6: Access Control, Rate Limiting & Cost Enforcement Starter Template

Implement three guardrails:
1. AccessController - role-based document/field access control
2. RateLimiter - limit queries per minute per user
3. CostEnforcer - enforce budget limits per role
"""

import json
import re
import logging
from typing import Dict, Any, List
from datetime import datetime, timezone
from time import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# TASK 1: Implement AccessController
# ============================================================================


class AccessController:
    """Enforce role-based access control."""

    def __init__(self, access_policy_path: str):
        """Load access control policy and initialize the audit log."""
        with open(access_policy_path) as f:
            self.policy = json.load(f)
        self.audit_log: List[Dict[str, Any]] = []

    def can_view_document(self, role: str, document: Dict[str, Any]) -> bool:
        """Check if a role can view a document based on its sensitivity level."""
        sensitivity = document.get("sensitivity", "Public")
        allowed_roles = self.policy.get("document_access", {}).get(sensitivity, [])
        allowed = role in allowed_roles
        self.log_access(role, document.get("id", "document"), allowed,
                        field=f"sensitivity={sensitivity}")
        return allowed

    def can_view_field(self, role: str, field_name: str) -> bool:
        """Check if a role can view a sensitive field.

        Fields not listed in sensitive_fields are considered non-sensitive
        and viewable by any role.
        """
        sensitive = self.policy.get("sensitive_fields", {})
        if field_name not in sensitive:
            return True  # not a sensitive field → always viewable
        visibility = sensitive[field_name].get("visibility", [])
        return role in visibility

    def redact_response(self, role: str, response: str) -> str:
        """Redact values of sensitive fields the role cannot view.

        Looks for patterns like  "field": value ,  field: value , or
        field = value  in the response text and replaces the value with
        [REDACTED] when the role lacks visibility for that field.
        """
        redacted = response
        for field_name in self.policy.get("sensitive_fields", {}):
            if self.can_view_field(role, field_name):
                continue  # role allowed to see this field; leave it

            # Match: optional-quote field optional-quote, separator, then value.
            # Value forms: "quoted string", $1,234.56, 12345, or a bare token.
            # Capture group 1 = the "field:" / "field=" prefix (quotes optional).
            # Capture group 2 = the value, one of:
            #   "quoted string"  |  a run of non-delimiter chars (numbers with
            #   internal commas like 145,000 or SSNs like 111-22-3333 are kept
            #   whole; a trailing comma/brace/semicolon ends the value).
            pattern = re.compile(
                rf'(["\']?{re.escape(field_name)}["\']?\s*[:=]\s*)'
                r'("[^"]*"|(?:\$?[\w.\-]+(?:,\d{3})*))',
                re.IGNORECASE,
            )
            if pattern.search(redacted):
                redacted = pattern.sub(r"\1[REDACTED]", redacted)
                self.log_access(role, "response", allowed=False, field=field_name)
        return redacted

    def log_access(self, role: str, resource: str, allowed: bool, field: str = None):
        """Append an access attempt to the audit trail."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "resource": resource,
            "field": field,
            "allowed": allowed,
        }
        self.audit_log.append(entry)
        logger.info(
            f"AUDIT role={role} resource={resource} field={field} allowed={allowed}"
        )

    def filter_documents(
        self, role: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return only the documents the role is permitted to view."""
        visible = []
        for doc in documents:
            if self.can_view_document(role, doc):
                visible.append(doc)
        return visible

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return audit log entries."""
        return self.audit_log


# ============================================================================
# TASK 2: Implement RateLimiter
# ============================================================================


class RateLimiter:
    """Rate limit queries per user per minute."""

    def __init__(self, max_queries_per_minute: int = 30):
        """Initialize rate limiter.

        TODO: Store max limit and initialize per-user query tracking
        """
        self.max_queries_per_minute = max_queries_per_minute
        self.user_query_times = {}  # {user_id: [timestamps...]}

    def _recent_queries(self, user_id: str) -> List[float]:
        """Return this user's query timestamps within the last 60 seconds."""
        now = time()
        timestamps = self.user_query_times.get(user_id, [])
        recent = [t for t in timestamps if now - t < 60]
        self.user_query_times[user_id] = recent  # prune old entries
        return recent

    def is_allowed(self, user_id: str) -> bool:
        """Allow the query if under the per-minute limit; record it if so."""
        recent = self._recent_queries(user_id)
        if len(recent) < self.max_queries_per_minute:
            recent.append(time())
            self.user_query_times[user_id] = recent
            return True
        return False

    def get_remaining_queries(self, user_id: str) -> int:
        """Queries the user has left in the current 60-second window."""
        remaining = self.max_queries_per_minute - len(self._recent_queries(user_id))
        return max(0, remaining)


# ============================================================================
# TASK 3: Implement CostEnforcer
# ============================================================================


class CostEnforcer:
    """Enforce cost limits per user/role."""

    # Monthly budget per role (USD).
    DEFAULT_BUDGETS = {
        "engineer": 100.0,
        "manager": 500.0,
        "hr": 200.0,
        "finance": 500.0,
        "executive": 1000.0,
    }

    def __init__(self, policy_path: str = None):
        """Set up per-role budgets and per-user spending tracking."""
        self.role_budgets = dict(self.DEFAULT_BUDGETS)
        # Optionally override budgets from a policy file.
        if policy_path:
            try:
                with open(policy_path) as f:
                    data = json.load(f)
                self.role_budgets.update(data.get("role_budgets", {}))
            except Exception as e:
                logger.warning(f"Could not load cost policy: {e}")
        # {user_id: {"role": str, "total": float}}
        self.user_spending: Dict[str, Dict[str, Any]] = {}

    def add_cost(self, user_id: str, role: str, cost: float):
        """Record the cost of a query against the user's running total."""
        if user_id not in self.user_spending:
            self.user_spending[user_id] = {"role": role, "total": 0.0}
        # Keep role up to date in case it was inferred earlier.
        self.user_spending[user_id]["role"] = role
        self.user_spending[user_id]["total"] += cost

    def _budget_for(self, user_id: str, role: str = None) -> float:
        """Resolve the budget for a user from their known or supplied role."""
        known_role = self.user_spending.get(user_id, {}).get("role")
        effective_role = known_role or role or "engineer"
        return self.role_budgets.get(effective_role, self.DEFAULT_BUDGETS["engineer"])

    def can_afford_query(
        self, user_id: str, estimated_cost: float, role: str = None
    ) -> bool:
        """True if the estimated cost fits within the user's remaining budget.

        New users default to the engineer budget unless a role is supplied.
        """
        budget = self._budget_for(user_id, role)
        spent = self.user_spending.get(user_id, {}).get("total", 0.0)
        return estimated_cost <= (budget - spent)

    def get_budget_remaining(self, user_id: str, role: str = None) -> float:
        """Remaining budget for the user (never negative)."""
        budget = self._budget_for(user_id, role)
        spent = self.user_spending.get(user_id, {}).get("total", 0.0)
        return max(0.0, budget - spent)


# ============================================================================
# TASK 4: Integrate with Week 5 Agent
# ============================================================================

# Once you have implemented the three classes above, open your copied
# app_starter.py and update the Agent class to use them:
#
# 1. In Agent.__init__, add:
#       self.access_controller = AccessController("data/access_control.json")
#       self.rate_limiter = RateLimiter(max_queries_per_minute=30)
#       self.cost_enforcer = CostEnforcer()
#
# 2. Update Agent.query() to accept user_id and user_role parameters:
#       def query(self, user_query: str, user_id: str, user_role: str = "engineer")
#
# 3. At the start of query(), add guardrail checks:
#       if not self.rate_limiter.is_allowed(user_id):
#           return {"error": "Rate limit exceeded"}
#       if not self.cost_enforcer.can_afford_query(user_id, estimated_cost=0.01):
#           return {"error": "Budget exceeded"}
#
# 4. After getting the LLM answer, redact sensitive fields:
#       answer = self.access_controller.redact_response(user_role, answer)
#
# 5. After each query, track actual cost:
#       self.cost_enforcer.add_cost(user_id, user_role, actual_cost)


# ============================================================================
# TASK 5: Test Your Implementation
# ============================================================================

# A basic test suite is provided below to help you verify your implementation.
# Run it with: python3 access_control_starter.py
# You are free to modify or extend these tests as you see fit.

if __name__ == "__main__":
    """Quick test of access control functionality."""

    # Test AccessController
    print("Testing AccessController...")
    controller = AccessController("data/access_control.json")

    assert not controller.can_view_field(
        "engineer", "salary"
    ), "Engineer should not see salary"
    assert controller.can_view_field("hr", "salary"), "HR should see salary"
    assert controller.can_view_field("manager", "salary"), "Manager should see salary"
    assert not controller.can_view_field(
        "engineer", "ssn"
    ), "Engineer should not see SSN"
    print("  can_view_field: PASSED")

    docs = [
        {"id": "doc1", "sensitivity": "Public", "content": "Mission statement"},
        {"id": "doc2", "sensitivity": "Confidential", "content": "Salary ranges"},
    ]
    visible = controller.filter_documents("engineer", docs)
    assert (
        len(visible) == 1 and visible[0]["id"] == "doc1"
    ), "Engineer should only see Public doc"
    print("  filter_documents: PASSED")

    # Test RateLimiter
    print("\nTesting RateLimiter...")
    limiter = RateLimiter(max_queries_per_minute=3)
    assert limiter.is_allowed("user1"), "First query should be allowed"
    assert limiter.is_allowed("user1"), "Second query should be allowed"
    assert limiter.is_allowed("user1"), "Third query should be allowed"
    assert not limiter.is_allowed("user1"), "Fourth query should be blocked"
    print("  is_allowed: PASSED")

    # Test CostEnforcer
    print("\nTesting CostEnforcer...")
    enforcer = CostEnforcer()
    assert enforcer.can_afford_query(
        "user1", 50.0
    ), "Should afford $50 within $100 budget"
    enforcer.add_cost("user1", "engineer", 50.0)
    assert enforcer.can_afford_query(
        "user1", 49.0
    ), "Should afford $49 with $50 remaining"
    assert not enforcer.can_afford_query(
        "user1", 51.0
    ), "Should not afford $51 with $50 remaining"
    print("  can_afford_query: PASSED")

    assert enforcer.get_budget_remaining("user1") == 50.0, "Should have $50 left"
    # Per-role budgets resolve correctly.
    enforcer.add_cost("exec1", "executive", 100.0)
    assert enforcer.can_afford_query("exec1", 800.0), "Exec has $1000 budget"
    print("  budget_remaining + per-role budgets: PASSED")

    # --- Extra: redaction (graded but not in the default tests) ---
    print("\nTesting redaction...")
    text = 'Employee record: {"name": "Sarah", "salary": 145000, "ssn": "111-22-3333"}'
    eng_view = controller.redact_response("engineer", text)
    assert "[REDACTED]" in eng_view, "Engineer should have salary/ssn redacted"
    assert "145000" not in eng_view, "Salary value should be gone for engineer"
    hr_view = controller.redact_response("hr", text)
    assert "145000" in hr_view, "HR should see salary"
    assert "111-22-3333" in hr_view, "HR should see ssn"
    print("  redact_response (engineer redacted, hr full): PASSED")

    # --- Extra: audit log captures access attempts ---
    print("\nTesting audit log...")
    controller.can_view_document("engineer", {"id": "d1", "sensitivity": "Restricted"})
    log = controller.get_audit_log()
    assert len(log) > 0, "Audit log should record access attempts"
    assert all("timestamp" in e and "allowed" in e for e in log), "Log entries well-formed"
    print(f"  audit_log: PASSED ({len(log)} entries recorded)")

    print("\nAll tests passed!")
