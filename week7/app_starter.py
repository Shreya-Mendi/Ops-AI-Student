"""
Week 5: Agent Architecture with LLM Tool Use

An AI agent that answers TechCorp questions using:
- Gemini LLM (free tier via Google AI API)
- SQLite database queries (employees)
- Policy document retrieval (documents.json)
- Expense approval limits (policies.json)

The agent reasons about which tool to call, executes it, then synthesizes
a final answer. Token usage and cost are tracked across all LLM calls.

Note on model choice: the assignment references gemini-2.5-pro, but Google
removed 2.5-pro from the free API tier (limit: 0). gemini-2.5-flash is used
here because it is available on the free tier. The model name is a single
constant (MODEL_NAME) so it can be swapped if paid 2.5-pro quota is enabled.
"""

import json
import sqlite3
import time
import re
from typing import Dict, Any
from google import genai
from google.genai import errors as genai_errors
import logging
import os

# Week 6 guardrails
from access_control_starter import AccessController, RateLimiter, CostEnforcer

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv optional; env var may be set directly

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Model + pricing. gemini-2.5-flash free-tier pricing (per 1M tokens).
MODEL_NAME = "gemini-2.5-flash"
INPUT_COST_PER_1M = 0.075
OUTPUT_COST_PER_1M = 0.30


# ──────────────────────────────────────────────────────────────────────────
# TASK 1: Tool base class
# ──────────────────────────────────────────────────────────────────────────


class Tool:
    """Base class for tools the agent can call."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, **kwargs) -> str:
        """Execute the tool. Overridden by each subclass."""
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────
# TASK 2: EmployeeLookupTool
# ──────────────────────────────────────────────────────────────────────────


class EmployeeLookupTool(Tool):
    """Look up employee information from the SQLite database."""

    # Columns returned to the LLM. We omit highly sensitive fields (ssn,
    # address, phone) — those are handled by access control in Week 6.
    SAFE_COLUMNS = [
        "id",
        "name",
        "email",
        "department_name",
        "job_level",
        "title",
        "salary",
        "hire_date",
        "manager_id",
    ]

    def __init__(self, db_path: str):
        super().__init__("employee_lookup", "Find employee information by name or ID")
        self.db_path = db_path

    def execute(self, employee_name: str = None, employee_id: str = None) -> str:
        """Look up an employee by name (partial match) or ID (exact match).

        Returns a JSON string of matching employees, or a not-found message.
        """
        try:
            if not employee_name and not employee_id:
                return "Error: provide either employee_name or employee_id"

            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cols = ", ".join(self.SAFE_COLUMNS)
            if employee_id:
                cursor.execute(
                    f"SELECT {cols} FROM employees WHERE id = ?", (employee_id,)
                )
            else:
                cursor.execute(
                    f"SELECT {cols} FROM employees WHERE name LIKE ? LIMIT 10",
                    (f"%{employee_name}%",),
                )

            rows = [dict(r) for r in cursor.fetchall()]
            conn.close()

            if not rows:
                target = employee_id or employee_name
                return f"Employee not found: {target}"

            return json.dumps(rows, indent=2)
        except Exception as e:
            logger.error(f"Employee lookup error: {e}")
            return f"Error: {str(e)}"


# ──────────────────────────────────────────────────────────────────────────
# TASK 3: PolicySearchTool
# ──────────────────────────────────────────────────────────────────────────


class PolicySearchTool(Tool):
    """Search policy documents by keyword."""

    def __init__(self, documents_path: str = None):
        super().__init__("policy_search", "Search policy documents by keyword or topic")
        # Load documents once at init so disk is read a single time.
        if documents_path is None:
            documents_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data", "documents.json"
            )
        try:
            with open(documents_path) as f:
                self.documents = json.load(f)
        except Exception as e:
            logger.error(f"Could not load documents: {e}")
            self.documents = []

    def execute(self, query: str, limit: int = 5) -> str:
        """Search documents for a keyword and return top-N matches.

        Each match includes the title, category, and a 500-char snippet.
        """
        try:
            if not query:
                return "Error: query is required"

            q = query.lower()
            matches = []
            for doc in self.documents:
                content = doc.get("content", "")
                title = doc.get("title", "")
                if q in content.lower() or q in title.lower():
                    matches.append(doc)

            if not matches:
                return f"No policy documents found matching: {query}"

            matches = matches[:limit]
            out = [f"Found {len(matches)} document(s) matching '{query}':\n"]
            for doc in matches:
                snippet = doc.get("content", "").strip()[:500]
                out.append(
                    f"### {doc.get('title', 'Untitled')} "
                    f"[{doc.get('category', 'N/A')}]\n{snippet}...\n"
                )
            return "\n".join(out)
        except Exception as e:
            logger.error(f"Policy search error: {e}")
            return f"Error: {str(e)}"


# ──────────────────────────────────────────────────────────────────────────
# TASK 4: ExpenseQueryTool
# ──────────────────────────────────────────────────────────────────────────


class ExpenseQueryTool(Tool):
    """Query expense approval limits by role."""

    def __init__(self, policies_path: str = None):
        super().__init__("expense_query", "Query expense approval limits by role")
        if policies_path is None:
            policies_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data", "policies.json"
            )
        try:
            with open(policies_path) as f:
                self.policies = json.load(f)
        except Exception as e:
            logger.error(f"Could not load policies: {e}")
            self.policies = {}

    def execute(self, role: str) -> str:
        """Return the expense approval limit for a role.

        Valid roles: ic1_ic2, ic3, manager, director, vp.
        """
        try:
            if not role:
                return "Error: role is required"

            limits = self.policies.get("expense", {}).get("approval_limits", {})
            key = role.strip().lower()

            if key in limits:
                return f"Approval limit for {key}: ${limits[key]}"

            valid = ", ".join(limits.keys()) if limits else "none loaded"
            return f"Role not found: {role}. Valid roles: {valid}"
        except Exception as e:
            logger.error(f"Expense query error: {e}")
            return f"Error: {str(e)}"


# ──────────────────────────────────────────────────────────────────────────
# TASK 5: Agent
# ──────────────────────────────────────────────────────────────────────────


class Agent:
    """AI agent that answers questions using a Gemini LLM + tools."""

    def __init__(self, db_path: str, api_key: str = None):
        self.db_path = db_path
        self.api_key = api_key or GOOGLE_API_KEY

        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Get a free key at: "
                "https://aistudio.google.com/app/apikey"
            )

        self.client = genai.Client(api_key=self.api_key)

        self.tools = {
            "employee_lookup": EmployeeLookupTool(db_path),
            "policy_search": PolicySearchTool(),
            "expense_query": ExpenseQueryTool(),
        }

        # Week 6 guardrails
        ac_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "access_control.json"
        )
        self.access_controller = AccessController(ac_path)
        self.rate_limiter = RateLimiter(max_queries_per_minute=30)
        self.cost_enforcer = CostEnforcer()

        # Metrics
        self.token_count = 0
        self.total_cost = 0.0
        self.queries_run = 0

    def _build_system_prompt(self, user_role: str) -> str:
        """Describe the agent's purpose and available tools to the LLM."""
        tool_lines = "\n".join(
            f"- {name}: {tool.description}" for name, tool in self.tools.items()
        )
        return f"""You are a TechCorp business assistant. Answer employee questions \
accurately using the tools below. Only use the data the tools return — do not \
invent facts.

User role: {user_role}

Available tools:
{tool_lines}

Tool argument reference:
- employee_lookup: employee_name=<name> OR employee_id=<id>
- policy_search: query=<keyword or topic>
- expense_query: role=<one of: ic1_ic2, ic3, manager, director, vp>

To use a tool, respond with EXACTLY this format and nothing else:
TOOL: <tool_name>
ARGS: <arg_name>=<value>

If you can answer without a tool, or once you have what you need, respond with:
ANSWER: <your answer>

Use one tool at a time. After you see the tool result, either call another \
tool or give the final ANSWER."""

    def _parse_tool_call(self, text: str):
        """Parse a 'TOOL:/ARGS:' block. Returns (tool_name, args_dict) or None."""
        tool_name = None
        args = {}
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("TOOL:"):
                tool_name = line.split(":", 1)[1].strip()
            elif line.upper().startswith("ARGS:"):
                arg_str = line.split(":", 1)[1].strip()
                # Support comma-separated key=value pairs
                for pair in arg_str.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        args[k.strip()] = v.strip()
        if tool_name:
            return tool_name, args
        return None

    def _generate(self, contents: str, max_retries: int = 3):
        """Call the LLM once, track tokens/cost, return (text, in_tok, out_tok).

        Retries on 429 (rate limit) using the server-suggested retry delay,
        so the agent degrades gracefully under free-tier limits.
        """
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=MODEL_NAME, contents=contents
                )
                usage = resp.usage_metadata
                in_tok = usage.prompt_token_count or 0
                out_tok = usage.candidates_token_count or 0
                self.token_count += in_tok + out_tok
                self.total_cost += self._estimate_query_cost(in_tok, out_tok)
                return (resp.text or ""), in_tok, out_tok
            except genai_errors.ClientError as e:
                if getattr(e, "code", None) == 429 and attempt < max_retries:
                    delay = self._retry_delay_seconds(str(e), default=15)
                    logger.warning(
                        f"Rate limited (429). Retrying in {delay}s "
                        f"(attempt {attempt + 1}/{max_retries})."
                    )
                    time.sleep(delay)
                    continue
                raise

    @staticmethod
    def _retry_delay_seconds(err_msg: str, default: int = 15) -> int:
        """Extract the server-suggested retry delay from a 429 error message."""
        m = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg)
        if m:
            return int(float(m.group(1))) + 1
        return default

    def query(
        self,
        user_query: str,
        user_id: str = "anonymous",
        user_role: str = "engineer",
    ) -> Dict[str, Any]:
        """Answer a question using the LLM + tools, gated by guardrails.

        Guardrails run *before* the LLM so blocked requests cost nothing:
        1. Rate limit (per user, per minute)
        2. Budget check (per role monthly budget)
        Then after the answer is produced:
        3. Redact sensitive fields the role may not view
        4. Track actual cost against the user's budget
        """
        logger.info(f"Processing query: {user_query} (user={user_id}, role={user_role})")

        # --- Guardrail 1: rate limiting ---
        if not self.rate_limiter.is_allowed(user_id):
            return {
                "error": "Rate limit exceeded",
                "answer": None,
                "cost": 0.0,
                "role": user_role,
                "remaining_queries": self.rate_limiter.get_remaining_queries(user_id),
            }

        # --- Guardrail 2: budget check (estimate before spending) ---
        estimated_cost = 0.01
        if not self.cost_enforcer.can_afford_query(
            user_id, estimated_cost, role=user_role
        ):
            return {
                "error": "Budget exceeded",
                "answer": None,
                "cost": 0.0,
                "role": user_role,
                "budget_remaining": self.cost_enforcer.get_budget_remaining(
                    user_id, role=user_role
                ),
            }

        start_tokens = self.token_count
        start_cost = self.total_cost

        system_prompt = self._build_system_prompt(user_role)
        conversation = f"{system_prompt}\n\nUser question: {user_query}\n"

        max_steps = 5
        tools_used = []
        answer = None

        for _ in range(max_steps):
            text, _, _ = self._generate(conversation)

            # Final answer reached?
            if "ANSWER:" in text.upper():
                idx = text.upper().index("ANSWER:")
                answer = text[idx + len("ANSWER:"):].strip()
                break

            parsed = self._parse_tool_call(text)
            if not parsed:
                # No tool call and no ANSWER tag — treat response as the answer.
                answer = text.strip()
                break

            tool_name, args = parsed
            tool = self.tools.get(tool_name)
            if tool is None:
                tool_result = f"Error: unknown tool '{tool_name}'"
            else:
                tools_used.append(tool_name)
                tool_result = tool.execute(**args)
                # Guardrail (pre-LLM): redact sensitive fields from the tool
                # result BEFORE the model sees them, so the LLM can never echo
                # a value the role isn't allowed to view (even in prose).
                tool_result = self.access_controller.redact_response(
                    user_role, tool_result
                )

            # Feed the tool result back into the conversation.
            conversation += (
                f"\n{text.strip()}\n\nTOOL RESULT ({tool_name}):\n{tool_result}\n\n"
                "Based on this, call another tool or give the final ANSWER.\n"
            )

        if answer is None:
            answer = "Unable to produce an answer within the step limit."

        # --- Guardrail 3: redact sensitive fields the role cannot view ---
        answer = self.access_controller.redact_response(user_role, answer)

        actual_cost = self.total_cost - start_cost

        # --- Guardrail 4: track actual cost against the user's budget ---
        self.cost_enforcer.add_cost(user_id, user_role, actual_cost)

        self.queries_run += 1

        return {
            "answer": answer,
            "tokens_used": self.token_count - start_tokens,
            "cost": actual_cost,
            "role": user_role,
            "user_id": user_id,
            "tools_used": tools_used,
            "budget_remaining": self.cost_enforcer.get_budget_remaining(
                user_id, role=user_role
            ),
        }

    def _estimate_query_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on tokens."""
        input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_1M
        output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_1M
        return input_cost + output_cost

    def get_metrics(self) -> Dict[str, Any]:
        """Return running performance/cost metrics."""
        avg = self.total_cost / self.queries_run if self.queries_run else 0.0
        return {
            "total_queries": self.queries_run,
            "total_tokens": self.token_count,
            "total_cost": self.total_cost,
            "avg_cost_per_query": avg,
        }


# ──────────────────────────────────────────────────────────────────────────
# TASK 6: Quick test
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    try:
        agent = Agent("data/techcorp.db")
        print("Agent initialized successfully (with Week 6 guardrails)")

        print("\nTesting query: 'What is the travel policy?'")
        result = agent.query("What is the travel policy?", user_id="user1", user_role="engineer")
        print(f"Answer: {result['answer']}")
        print(f"Tools used: {result.get('tools_used')}")
        print(f"Tokens: {result.get('tokens_used')}")
        print(f"Cost: ${result.get('cost', 0):.6f}")
        print(f"Budget remaining: ${result.get('budget_remaining', 0):.2f}")

        metrics = agent.get_metrics()
        print(f"\nMetrics: {metrics}")

    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Error during test")
        sys.exit(1)
