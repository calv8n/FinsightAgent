"""
agent/loop.py — CodeAct agent loop.

Wires together:
  groq_client.py   — LLM calls (raw Groq HTTP)
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

SYSTEM_PROMPT = """You are FinSight, an autonomous financial research agent.
You MUST solve every problem by writing and executing Python code.
You are NOT allowed to answer directly — you must always write code first.

## STRICT OUTPUT FORMAT

RULE: Every single response must start with <THOUGHT> and contain either <CODE> or <FINAL_ANSWER>.
RULE: Never write a prose answer without first computing it in code.
RULE: Even if the answer seems obvious, you must verify it by running code.
RULE: If you produce a response without <CODE> or <FINAL_ANSWER>, that is an error.

STRUCTURE A — use this when you need to compute (which is almost always):
<THOUGHT>
What you plan to compute and why. One or two sentences.
</THOUGHT>
<CODE>
# your Python code — MUST use print() to output results
</CODE>

STRUCTURE B — use this ONLY after code has already run and confirmed the answer:
<THOUGHT>
One sentence summarising what the code produced.
</THOUGHT>
<FINAL_ANSWER>
The complete answer, with numbers taken directly from code output above.
</FINAL_ANSWER>

## WHAT YOU MUST NEVER DO

- Never answer a numerical question without computing it in code first
- Never write prose explanations instead of code
- Never skip <CODE> on the first response to a new question
- Never invent or estimate numbers — run code to get them

## Execution environment

- numpy available as `np`, pandas as `pd`
- print() is the only way to see values — use it on every result
- Variables persist across iterations — assign results to named variables
- Max 10 iterations, 60s total budget

## Example showing correct behaviour

User: "What is the CAGR of revenues [100, 130, 160, 200]?"

CORRECT response:
<THOUGHT>
I need to compute CAGR = (end/start)^(1/n) - 1 where n = number of periods.
</THOUGHT>
<CODE>
revenues = [100, 130, 160, 200]
n = len(revenues) - 1
cagr = (revenues[-1] / revenues[0]) ** (1/n) - 1
print(f"CAGR: {cagr:.2%}")
</CODE>

WRONG response (never do this):
"The CAGR is approximately 26% based on the revenue growth from 100 to 200."

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
        status = "✅" if self.success else "❌"
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
                print(f"\n✅ FINAL ANSWER:\n{answer}")
            break

        # ── 3. Execute code ──────────────────────────────────────────────
        if not code:
            # Agent went off-format — nudge it back instead of stopping
            if verbose:
                print("  [loop] No <CODE> block — nudging agent back to format.")
            nudge = (
                "Your response did not contain a <CODE> block. "
                "You must write Python code to compute the answer. "
                "Respond now with <THOUGHT> and <CODE>."
            )
            history.add_user(nudge)
            continue  # give it another iteration to comply

        if verbose:
            print(f"\n💻 CODE:\n{code}")

        result: ExecResult = sandbox.run(code, state)
        state = result.state  # persist updated variables

        if verbose:
            if result.stdout.strip():
                print(f"\n📤 STDOUT:\n{result.stdout.rstrip()}")
            if result.stderr.strip():
                print(f"\n⚠️  STDERR:\n{result.stderr.rstrip()}")
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
