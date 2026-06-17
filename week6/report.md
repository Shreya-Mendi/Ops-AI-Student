# Week 6 Report: Access Control & Guardrails

**TechCorp Agent — Access Control, Rate Limiting & Cost Enforcement**
Shreya Mendi | Duke MEng AIPI

---

## 1. Overview

This week adds three guardrails on top of the Week 5 agent:

| Guardrail | Class | Purpose |
|-----------|-------|---------|
| **Access control** | `AccessController` | Role-based document/field access + redaction of sensitive values + audit logging |
| **Rate limiting** | `RateLimiter` | Cap queries per user per minute (sliding 60s window) |
| **Cost enforcement** | `CostEnforcer` | Per-role monthly budgets; block queries that would exceed them |

All three live in [`access_control_starter.py`](access_control_starter.py) and are
integrated into the Week 5 agent in [`app_starter.py`](app_starter.py). The
guardrails run **before** the LLM where possible, so blocked requests cost nothing.

---

## 2. Design

### AccessController
Loads `data/access_control.json`, which defines:
- **`document_access`** — which roles can view each sensitivity tier
  (Public / Internal / Confidential / Restricted).
- **`sensitive_fields`** — per-field visibility lists (e.g. `salary` → manager, hr,
  finance, executive; `ssn` → hr, finance only).

Methods:
- `can_view_document(role, doc)` — checks sensitivity tier against role.
- `can_view_field(role, field)` — non-sensitive fields are always viewable; sensitive
  ones check the visibility list.
- `redact_response(role, text)` — regex-replaces values of fields the role can't see
  with `[REDACTED]` (handles `"field": value`, `field=value`, quoted strings, numbers
  with commas, SSNs).
- `filter_documents(role, docs)` — returns only documents the role may view.
- `log_access(...)` / `get_audit_log()` — timestamped audit trail of every check.

### RateLimiter
Sliding-window limiter: keeps each user's query timestamps, prunes entries older than
60 seconds, and allows a query only if the recent count is below the limit
(default 30/min). `get_remaining_queries(user_id)` reports headroom.

### CostEnforcer
Per-role monthly budgets — engineer $100, manager $500, hr $200, finance $500,
executive $1000. Tracks per-user spending; `can_afford_query` blocks a query whose
estimated cost would exceed the remaining budget. New users default to the engineer
budget unless a role is supplied.

---

## 3. Integration with the Agent

`Agent.query(user_query, user_id, user_role)` now enforces the guardrails in order:

```python
# 1. Rate limit — before the LLM
if not self.rate_limiter.is_allowed(user_id):
    return {"error": "Rate limit exceeded", ...}

# 2. Budget check — before the LLM
if not self.cost_enforcer.can_afford_query(user_id, 0.01, role=user_role):
    return {"error": "Budget exceeded", ...}

# ... run Week 5 reasoning loop ...

# 3. Redact sensitive fields the role cannot view
answer = self.access_controller.redact_response(user_role, answer)

# 4. Track actual cost against the user's budget
self.cost_enforcer.add_cost(user_id, user_role, actual_cost)
```

---

## 4. Test Results

### 4a. Guardrail unit tests — `python3 access_control_starter.py`

```
Testing AccessController...
  can_view_field: PASSED
  filter_documents: PASSED

Testing RateLimiter...
  is_allowed: PASSED

Testing CostEnforcer...
  can_afford_query: PASSED
  budget_remaining + per-role budgets: PASSED

Testing redaction...
  redact_response (engineer redacted, hr full): PASSED

Testing audit log...
  audit_log: PASSED (5 entries recorded)

All tests passed!
```

### 4b. Redaction by role (offline)

Same record, different roles:

```
RAW:  {"name": "Sarah Chen", "salary": 145000, "ssn": "111-22-3333"}
ENG:  {"name": "Sarah Chen", "salary": [REDACTED], "ssn": [REDACTED]}
HR:   {"name": "Sarah Chen", "salary": 145000, "ssn": "111-22-3333"}
```

`compensation` is visible only to executive/finance, so even HR has it redacted —
confirming per-field visibility, not just a blanket role tier.

### 4c. Rate limiting (Demo B)

```
DEMO B — Rate Limiting (max 3/min for this demo)
  Query 1: ALLOWED  (remaining this minute: 2)
  Query 2: ALLOWED  (remaining this minute: 1)
  Query 3: ALLOWED  (remaining this minute: 0)
  Query 4: BLOCKED (rate limit exceeded)  (remaining this minute: 0)
  Query 5: BLOCKED (rate limit exceeded)  (remaining this minute: 0)
```

### 4d. Cost enforcement (Demo C)

```
DEMO C — Cost Enforcement (engineer budget = $100/month)
  Engineer budget: $100
  After spending $95 -> remaining: $5.00
  Can afford a $4 query?  True  (yes)
  Can afford a $10 query? False  (no — over budget)
  Executive after $95 -> remaining: $905.00
  Executive can afford a $10 query? True  (yes — $1000 budget)
```

### 4e. Live agent — access control end-to-end (Demo A)

Same query, two roles. The salary is redacted from the tool result **before** the
LLM sees it, so an engineer's answer cannot contain the value:

```
[engineer] Q: Look up the employee named Brian Yang and show their salary.
    A: I found Brian Yang's information, but their salary is redacted and
       cannot be displayed.
    tools=['employee_lookup'] cost=$0.000059 budget_left=$100.00

[hr] Q: Look up the employee named Brian Yang and show their salary.
    A: Brian Yang's salary is 467621.
    tools=['employee_lookup'] cost=$0.000058 budget_left=$200.00
```

This is the key result: **the engineer literally cannot obtain the salary**, because
the `[REDACTED]` substitution happens on the tool output, not just on the final
answer. The LLM never receives the number, so it can't leak it in prose. HR, who is
in the `salary` visibility list, gets the full value.

### 4f. Audit log sample

Every denied access is logged with a timestamp:

```
{'timestamp': '2026-06-17T00:09:04+00:00', 'role': 'engineer',
 'resource': 'response', 'field': 'salary', 'allowed': False}
```

---

## 5. Key Takeaways

- **Guardrails belong before the LLM.** Rate-limit and budget checks block bad
  requests at zero token cost; only redaction has to run after the answer exists.
- **Field-level beats role-level.** `compensation` being hidden from HR (but salary
  visible) shows why per-field visibility lists matter — a single "HR sees all"
  rule would leak it.
- **Audit everything.** Every access decision is logged with a timestamp, so denied
  attempts are traceable — essential for a real access-control system.
