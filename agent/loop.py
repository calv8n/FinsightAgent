"""
agent/loop.py — CodeAct agent loop.

Wires together:
  llm_request.py   — LLM calls (raw Groq HTTP)
  history.py       — rolling-window conversation history
  sandbox_client.py — Docker code execution

Flow per iteration:
  1. Build windowed message list from history
  2. Call Groq API → get <THOUGHT> + <CODE> (or <FINAL_ANSWER>)
  3. Append assistant response to history
  4. If <FINAL_ANSWER>: stop
  5. Extract <CODE>, run in Docker sandbox
  6. Build observation string from ExecutionResult
  7. Append observation as user message → go to 1
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from api.apis import llm_request
from .history import ConversationHistory
from .sandbox_client import ExecResult, SandboxClient

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are FinSight, an expert financial research agent.
You solve problems by writing and executing Python code.

## Response format — follow EXACTLY

Every response must contain exactly one of these two structures:

STRUCTURE A — when you need to compute something:
<THOUGHT>
One or two sentences: what you need to do and why.
</THOUGHT>
<CODE>
# Python code here. print() to show results.
</CODE>

STRUCTURE B — when you have the final answer:
<THOUGHT>
Brief summary of what the code showed.
</THOUGHT>
<FINAL_ANSWER>
Clear, complete answer to the user's question.
</FINAL_ANSWER>

## Execution environment

- Fresh Docker container each iteration — hardened, no network, read-only FS
- Available: numpy (np), pandas (pd), plus all standard Python builtins
- Variables you assign persist across iterations (passed via state dict)
- print() output is captured and shown to you in [Code Execution Result]

## Rules

- Write code first, reason from its output — do not guess
- Use print() liberally; it is the only way to see intermediate values
- If code raises an error, read the traceback and fix the approach
- Variables from previous iterations are available — reuse them
- Maximum 10 iterations per question; 60 s wall-clock budget
- Never fabricate numbers; always compute them

## Example (multi-step)

User: "What is the compound value of $1000 at 7% for 10 years?"

Iteration 1:
<THOUGHT>
I need to compute FV = PV * (1 + r)^n with PV=1000, r=0.07, n=10.
</THOUGHT>
<CODE>
pv, r, n = 1000, 0.07, 10
fv = pv * (1 + r) ** n
print(f"Future value: ${fv:,.2f}")
</CODE>

[Code Execution Result]
Stdout: Future value: $1,967.15

Iteration 2:
<THOUGHT>
The code returned $1,967.15 which is the correct compound value.
</THOUGHT>
<FINAL_ANSWER>
$1,000 invested at 7% annual interest compounded yearly grows to **$1,967.15** after 10 years.
</FINAL_ANSWER>
"""


# ============================================================================
# PARSER HELPERS
# ============================================================================


def _extract(tag: str, text: str) -> Optional[str]:
    """Return inner text of <tag>…</tag>, or None if not found."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


# ============================================================================
# RESULT DATACLASS
# ============================================================================


@dataclass
class AgentResult:
    query: str
    answer: Optional[str]
    success: bool
    iterations: int
    elapsed_s: float
    history: list[dict]  # full message log (for debugging)
    final_state: dict = field(default_factory=dict)

    def summary(self) -> str:
        status = "Pass" if self.success else "Fail"
        return (
            f"{status} {'Success' if self.success else 'Failed'} | "
            f"{self.iterations} iteration(s) | "
            f"{self.elapsed_s:.1f}s\n"
            f"Answer: {self.answer or '[none]'}"
        )


# ============================================================================
# AGENT LOOP
# ============================================================================

MAX_ITERATIONS = 10
WALL_CLOCK_TIMEOUT_S = 60


def run_agent(
    query: str,
    sandbox: SandboxClient,
    max_iterations: int = MAX_ITERATIONS,
    timeout_s: float = WALL_CLOCK_TIMEOUT_S,
    window_size: int = 20,
    verbose: bool = True,
) -> AgentResult:
    """
    Run the CodeAct loop for a single user query.

    Args:
        query:          The user's question.
        sandbox:        Initialised SandboxClient (Docker must be running).
        max_iterations: Hard cap on LLM + exec cycles.
        timeout_s:      Wall-clock budget in seconds.
        window_size:    Max messages sent to LLM per turn (rolling window).
        verbose:        Print iteration details to stdout.

    Returns:
        AgentResult with answer, metadata, and full message history.
    """
    history = ConversationHistory(max_messages=window_size)
    state: dict = {}
    answer: Optional[str] = None
    t0 = time.monotonic()

    history.add_user(query)

    if verbose:
        print(f"\n{'═'*72}")
        print(f"  FinSight Agent")
        print(f"  Query: {query}")
        print(f"{'═'*72}")

    for iteration in range(1, max_iterations + 1):

        # ── wall-clock guard ────────────────────────────────────────────
        elapsed = time.monotonic() - t0
        if elapsed >= timeout_s:
            if verbose:
                print(f"\n⏱  Timeout ({elapsed:.1f}s ≥ {timeout_s}s). Stopping.")
            break

        if verbose:
            print(
                f"\n── Iteration {iteration}/{max_iterations} "
                f"| {elapsed:.1f}s elapsed ─────────────────────"
            )

        # ── 1. Call LLM ─────────────────────────────────────────────────
        response = llm_request(
            system_prompt=SYSTEM_PROMPT,
            messages=history.as_dicts(),
        )

        if response is None:
            if verbose:
                print("  [loop] LLM call failed, stopping.")
            break

        history.add_assistant(response)

        thought = _extract("THOUGHT", response)
        code = _extract("CODE", response)
        final_answer = _extract("FINAL_ANSWER", response)

        if verbose and thought:
            print(f"\n🧠 THOUGHT: {thought}")

        # ── 2. Check for final answer ────────────────────────────────────
        if final_answer:
            answer = final_answer
            if verbose:
                print(f"\nFINAL ANSWER:\n{answer}")
            break

        # ── 3. Execute code ──────────────────────────────────────────────
        if not code:
            if verbose:
                print("  [loop] No <CODE> block found — stopping.")
            break

        if verbose:
            print(f"\nCODE:\n{code}")

        result: ExecResult = sandbox.run(code, state)
        state = result.state  # persist updated variables

        if verbose:
            if result.stdout.strip():
                print(f"\nSTDOUT:\n{result.stdout.rstrip()}")
            if result.stderr.strip():
                print(f"\nSTDERR:\n{result.stderr.rstrip()}")
            print(
                f"   ok={result.ok} | {result.elapsed*1000:.0f}ms | "
                f"vars={list(state.keys())}"
            )

        # ── 4. Build observation → feed back into history ─────────────────
        observation = _build_observation(result)
        history.add_user(observation)

    elapsed_total = time.monotonic() - t0

    return AgentResult(
        query=query,
        answer=answer,
        success=answer is not None,
        iterations=iteration,
        elapsed_s=elapsed_total,
        history=history.full_history(),
        final_state=state,
    )


def _build_observation(result: ExecResult) -> str:
    """Format ExecResult as a terse observation for the next LLM turn."""
    parts = [f"[Code Execution Result]\nSuccess: {result.ok}"]

    if result.stdout.strip():
        parts.append(f"Stdout:\n{result.stdout.rstrip()}")

    if result.stderr.strip():
        parts.append(f"Stderr:\n{result.stderr.rstrip()}")

    if result.state:
        keys = list(result.state.keys())
        parts.append(f"Available variables: {keys}")
        preview = {k: repr(v)[:60] for k, v in result.state.items()}
        lines = "\n".join(f"  {k} = {v}" for k, v in preview.items())
        parts.append(f"State:\n{lines}")

    return "\n\n".join(parts)
