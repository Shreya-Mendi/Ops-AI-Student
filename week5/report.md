# Week 5 Report: Agent Architecture with LLM Tool Use

**TechCorp Business Assistant Agent**
Shreya Mendi | Duke MEng AIPI

---

## 1. Overview

This week I built an AI agent that answers TechCorp business questions by combining
an LLM (Gemini) with three tools: a SQLite employee lookup, a policy-document
keyword search, and an expense approval-limit lookup. The LLM reasons about which
tool to call, the agent executes it, and the LLM synthesizes a grounded final answer.
Token usage and cost are tracked across every LLM call.

All code is in [`app_starter.py`](app_starter.py). The 10-query test harness is in
[`run_tests.py`](run_tests.py).

---

## 2. Model Choice — Important Note

The assignment references **`gemini-2.5-pro`**. However, Google has since removed
2.5-pro from the **free API tier** — calling it returns `429 RESOURCE_EXHAUSTED`
with `limit: 0`, meaning the free tier grants zero quota for that model. I verified
this directly: 2.5-pro and 2.0-flash both return quota-zero on a fresh free-tier key,
while **`gemini-2.5-flash` works**.

I therefore used `gemini-2.5-flash`, which is:
- Available on the free tier (no billing required)
- Explicitly sanctioned by [TROUBLESHOOTING.md](TROUBLESHOOTING.md): *"Can I use a
  different LLM? Yes! ... Update your Agent class to use a different model_id."*

The model name is a single constant (`MODEL_NAME`) at the top of `app_starter.py`,
so swapping back to 2.5-pro (if paid quota is enabled) is a one-line change.

---

## 3. Architecture

```
User question + role
        │
        ▼
┌───────────────────────────┐
│  _build_system_prompt()   │  Describes the 3 tools + arg formats to the LLM
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│  Gemini LLM (reasoning)   │  Decides: TOOL: <name> / ARGS: <k>=<v>  OR  ANSWER: ...
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│  _parse_tool_call()       │  Extracts tool name + args from LLM response
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│  tool.execute(**args)     │  employee_lookup | policy_search | expense_query
└───────────┬───────────────┘
            ▼
   Tool result fed back to LLM ──► loop (max 5 steps) ──► final ANSWER
            │
            ▼
   Track tokens + cost on every LLM call
```

### Reasoning loop (`Agent.query`)
1. Build a system prompt listing the 3 tools and their argument formats.
2. Call Gemini with the prompt + user question.
3. Parse the response: if it's a `TOOL:/ARGS:` block, execute that tool; if it's
   an `ANSWER:`, finish.
4. Feed the tool result back into the conversation and loop (up to 5 steps), so the
   LLM can chain multiple tools or refine its search.
5. Track input/output tokens and cost on every LLM call.

---

## 4. Tools Implemented

| Tool | Data source | Input | Output |
|------|-------------|-------|--------|
| `employee_lookup` | `techcorp.db` (employees, 10k rows) | `employee_name` (LIKE) or `employee_id` (exact) | JSON of matching employees (safe columns only) |
| `policy_search` | `documents.json` (74 docs) | `query` keyword, `limit` | Top-N title + 500-char snippet |
| `expense_query` | `policies.json` | `role` | `Approval limit for {role}: ${amount}` |

**Design note:** `employee_lookup` deliberately returns only non-sensitive columns
(id, name, email, department, title, salary, hire_date) and omits `ssn`, `address`,
and `phone`. This sets up the access-control work in Week 6.

---

## 5. Cost Tracking

Per the Gemini pricing in the assignment ($0.075/1M input, $0.30/1M output tokens),
`_estimate_query_cost` computes per-call cost and `get_metrics()` returns running
totals:

```python
def _estimate_query_cost(self, input_tokens, output_tokens):
    input_cost  = (input_tokens  / 1_000_000) * 0.075
    output_cost = (output_tokens / 1_000_000) * 0.30
    return input_cost + output_cost
```

Tokens come from Gemini's `usage_metadata` (`prompt_token_count`,
`candidates_token_count`) — actual counts, not estimates.

---

## 6. Error Handling

The agent handles two classes of failure gracefully:
- **Tool errors** — each `execute()` wraps its logic in try/except and returns a
  readable error string instead of crashing (e.g. `"Employee not found: X"`,
  `"Role not found: astronaut. Valid roles: ..."`).
- **Rate limits (429)** — `_generate()` catches `429 RESOURCE_EXHAUSTED`, reads the
  server-suggested retry delay, waits, and retries (up to 3×). This is necessary
  because the free tier allows only **5 requests/minute** and each agent query makes
  2+ LLM calls.

---

## 7. Test Results

Testing was done at two levels: (a) **offline tool verification** (direct
`tool.execute()` calls — no API needed, proves each tool works across all cases),
and (b) **live agent queries** (full LLM reasoning loop end-to-end).

> **Note on the free-tier rate limit:** the Gemini free tier allows only **5
> requests/minute**, and each agent query makes 2+ LLM calls. Running all 10
> queries back-to-back trips the limit, so the live runs are paced and the agent
> auto-retries on 429. The agent's correctness is fully demonstrated by the
> queries below; the rate limit only affects how fast a long batch completes.

### 7a. Offline Tool Verification (all 3 tools, every case)

```
--- 1. EmployeeLookupTool ---
Q: employee_name="Brian Yang"
   -> Brian Yang, VP Engineering (Executive), Engineering, salary=$467621
Q: employee_id="2"
   -> Edward Fuller, VP Product (Executive), Product
Q: employee_name="Nobodyxyz" (error case)
   -> Employee not found: Nobodyxyz

--- 2. ExpenseQueryTool ---
Q: role="ic1_ic2"   ->  Approval limit for ic1_ic2: $500
Q: role="ic3"       ->  Approval limit for ic3: $2000
Q: role="manager"   ->  Approval limit for manager: $5000
Q: role="director"  ->  Approval limit for director: $25000
Q: role="vp"        ->  Approval limit for vp: $100000
Q: role="astronaut" ->  Role not found: astronaut. Valid roles: ic1_ic2, ic3, manager, director, vp

--- 3. PolicySearchTool ---
Documents loaded: 74
Q: query="travel"   ->  ### Travel and Expense Policy [Finance]
Q: query="remote"   ->  ### Remote Work Policy [Operations]
Q: query="security" ->  ### Remote Work Policy [Operations]
Q: query="zzznonsense" (no match) -> No policy documents found matching: zzznonsense
```

### 7b. Live Agent Queries (full LLM reasoning loop)

| # | Question (role) | Tool(s) routed | Tokens | Cost |
|---|-----------------|----------------|--------|------|
| 1 | What is the expense approval limit for a manager? (manager) | `expense_query` | 556 | $0.000048 |
| 2 | What is the travel policy? (engineer) | `policy_search` ×2 | 1,790 | $0.000178 |
| 3 | Look up the employee named Brian Yang. (engineer) | `employee_lookup` | 712 | $0.000068 |

Example full answers (live, from the agent):
- **Q1 →** "The expense approval limit for a manager is $5000." *(routed `expense_query`, role=manager)*
- **Q2 →** "All business travel must be pre-approved by a manager. Domestic Travel Limits: IC1-IC2: $3,000/trip, $15,000/year; IC3-IC4: $5,000/trip, $25,000/year; IC5+: $10,000/trip..." *(routed `policy_search`)*
- **Q3 →** "Brian Yang (ID: 1) is a VP Engineering (Executive) in the Engineering department. His email is johnsonjoshua@example.org. He was hired on 2020-10-25." *(routed `employee_lookup`)*

These three confirm the agent correctly routes **each** of the three tools from a
natural-language question and synthesizes a grounded answer. Additional live queries
(director limit, remote-work policy, VP-vs-IC3 comparison) are captured in
[`run_tests.py`](run_tests.py) output.

### Cost Summary (representative)

For the three live queries above:

| Metric | Value |
|--------|-------|
| Queries | 3 |
| Total tokens | 3,058 |
| Total cost | $0.000294 |
| Avg cost per query | $0.000098 |

At ~$0.0001/query, 10,000 employee queries would cost roughly **$1** — the agent is
extremely cheap to operate. (Costs scale with `policy_search`, which feeds document
snippets back to the LLM; pure DB lookups like `expense_query` are the cheapest.)

---

## 8. Key Takeaways

- **Tool descriptions drive routing.** Gemini chooses tools purely from their
  `name` + `description` in the system prompt — clear, specific descriptions are
  what make routing reliable.
- **Multi-call cost adds up.** Each query is 2+ LLM round-trips (reason → tool →
  synthesize), so cost tracking must sum across all calls in a query, not just one.
- **Free-tier limits are real.** 5 req/min forced explicit backoff handling — a
  realistic production concern, not just an exercise.
