#!/usr/bin/env python3
"""
cli.py — FinSight interactive CLI

Usage:
  python cli.py              # start interactive session
  python cli.py --build      # build Docker image, then start
  python cli.py --no-docker  # run without Docker (unsafe exec, dev only)

Session commands (type these at the prompt):
  /help      show available commands
  /clear     clear screen
  /reset     start a new conversation (wipe history + state)
  /history   show conversation so far
  /state     show variables currently in sandbox
  /exit      quit
"""

from __future__ import annotations

import os
import sys
import time
import textwrap

# ── project root on path ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.loop import (
    run_agent,
    AgentResult,
    SYSTEM_PROMPT,
    _extract,
    _build_observation,
)
from api.apis import llm_request
from agent.history import ConversationHistory
from agent.sandbox_client import SandboxClient, build_sandbox_image, ExecResult

# ============================================================================
# ANSI COLOURS  (graceful fallback if terminal doesn't support them)
# ============================================================================


def _supports_colour() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLOUR = _supports_colour()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOUR else text


def bold(t):
    return _c(t, "1")


def dim(t):
    return _c(t, "2")


def green(t):
    return _c(t, "32")


def yellow(t):
    return _c(t, "33")


def cyan(t):
    return _c(t, "36")


def red(t):
    return _c(t, "31")


def magenta(t):
    return _c(t, "35")


def blue(t):
    return _c(t, "34")


# ============================================================================
# PRETTY PRINTERS
# ============================================================================

WIDTH = 72


def _hr(char: str = "─") -> str:
    return dim(char * WIDTH)


def _banner():
    os.system("clear" if os.name != "nt" else "cls")
    print()
    print(bold(cyan("  ███████╗██╗███╗   ██╗███████╗██╗ ██████╗ ██╗  ██╗████████╗")))
    print(bold(cyan("  ██╔════╝██║████╗  ██║██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝")))
    print(bold(cyan("  █████╗  ██║██╔██╗ ██║███████╗██║██║  ███╗███████║   ██║   ")))
    print(bold(cyan("  ██╔══╝  ██║██║╚██╗██║╚════██║██║██║   ██║██╔══██║   ██║   ")))
    print(bold(cyan("  ██║     ██║██║ ╚████║███████║██║╚██████╔╝██║  ██║   ██║   ")))
    print(bold(cyan("  ╚═╝     ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ")))
    print()
    print(bold("  Financial Research Agent  ·  CodeAct + Docker Sandbox"))
    print(dim("  Type /help for commands  ·  Ctrl+C or /exit to quit"))
    print()
    print(_hr("═"))
    print()


def _print_thought(thought: str):
    print()
    print(f"  {bold(yellow('🧠 THOUGHT'))}")
    for line in textwrap.wrap(thought, WIDTH - 4):
        print(f"    {dim(line)}")


def _print_code(code: str):
    print()
    print(f"  {bold(blue('💻 CODE'))}")
    print(f"  {dim('┌' + '─' * (WIDTH - 2))}")
    for line in code.splitlines():
        print(f"  {dim('│')} {cyan(line)}")
    print(f"  {dim('└' + '─' * (WIDTH - 2))}")


def _print_exec(result: ExecResult, iteration: int, elapsed: float):
    ok_str = green("✓ ok") if result.ok else red("✗ error")
    print()
    print(
        f"  {bold('📤 EXECUTION')}  {dim(f'iter={iteration}  {elapsed*1000:.0f}ms')}  {ok_str}"
    )
    if result.stdout.strip():
        for line in result.stdout.rstrip().splitlines():
            print(f"    {line}")
    if result.stderr.strip():
        print(f"  {red('STDERR:')}")
        for line in result.stderr.rstrip().splitlines():
            print(f"    {red(line)}")
    if result.state:
        keys = list(result.state.keys())
        print(f"  {dim(f'vars: {keys}')}")


def _print_answer(answer: str, elapsed_total: float, iterations: int):
    print()
    print(_hr())
    print(
        f"  {bold(green('✅ ANSWER'))}  {dim(f'{iterations} iteration(s) · {elapsed_total:.1f}s')}"
    )
    print(_hr())
    print()
    # Word-wrap but preserve intentional newlines
    for para in answer.split("\n"):
        if para.strip():
            for line in textwrap.wrap(para, WIDTH - 2):
                print(f"  {line}")
        else:
            print()
    print()
    print(_hr())


def _print_error(msg: str):
    print(f"\n  {red('✗')} {msg}\n")


def _print_info(msg: str):
    print(f"\n  {dim(msg)}\n")


def _print_help():
    cmds = [
        ("/help", "Show this help"),
        ("/clear", "Clear the screen"),
        ("/reset", "New conversation — wipe history and sandbox state"),
        ("/history", "Print the full conversation so far"),
        ("/state", "Show variables currently held in the sandbox"),
        ("/exit", "Quit"),
    ]
    print()
    print(f"  {bold('Commands')}")
    for cmd, desc in cmds:
        print(f"    {cyan(cmd):<20} {dim(desc)}")
    print()
    print(f"  {bold('Tips')}")
    tips = [
        "Ask financial questions in plain English.",
        "The agent writes Python, runs it in Docker, and iterates.",
        "Variables persist within a session — use /reset to clear.",
        "Use /history to see the raw message log for debugging.",
    ]
    for tip in tips:
        print(f"    {dim('·')} {dim(tip)}")
    print()


# ============================================================================
# INLINE AGENT LOOP  (re-implemented from loop.py with live streaming output)
# ============================================================================


def run_interactive(
    query: str,
    history: ConversationHistory,
    state: dict,
    sandbox: SandboxClient,
    max_iterations: int = 10,
    timeout_s: float = 60.0,
) -> tuple[str | None, dict]:
    """
    Run one query through the CodeAct loop with live printed output.
    Mutates `history` and `state` in place so they persist across queries.

    Returns (answer_text_or_None, updated_state).
    """
    history.add_user(query)
    answer = None
    t0 = time.monotonic()

    for iteration in range(1, max_iterations + 1):
        elapsed = time.monotonic() - t0
        if elapsed >= timeout_s:
            _print_error(f"Timeout ({elapsed:.0f}s). Stopping.")
            break

        # ── spinner while waiting for LLM ───────────────────────────────
        print(
            f"\n  {dim(f'[ iter {iteration} · calling LLM... ]')}", end="", flush=True
        )

        response = llm_request(
            system_prompt=SYSTEM_PROMPT,
            messages=history.as_dicts(),
        )
        print(f"\r{' ' * 50}\r", end="", flush=True)  # clear spinner line

        if response is None:
            _print_error("LLM call failed.")
            break

        history.add_assistant(response)

        thought = _extract("THOUGHT", response)
        code = _extract("CODE", response)
        final_answer = _extract("FINAL_ANSWER", response)

        if thought:
            _print_thought(thought)

        # ── final answer ─────────────────────────────────────────────────
        if final_answer:
            answer = final_answer
            _print_answer(answer, time.monotonic() - t0, iteration)
            break

        # ── execute code ─────────────────────────────────────────────────
        if not code:
            _print_error("No <CODE> block in response — stopping.")
            break

        _print_code(code)

        print(f"  {dim('[ executing in Docker sandbox... ]')}", end="", flush=True)
        exec_t0 = time.monotonic()
        result: ExecResult = sandbox.run(code, state)
        exec_elapsed = time.monotonic() - exec_t0
        print(f"\r{' ' * 50}\r", end="", flush=True)

        state = result.state
        _print_exec(result, iteration, exec_elapsed)

        observation = _build_observation(result)
        history.add_user(observation)

    return answer, state


# ============================================================================
# COMMAND HANDLERS
# ============================================================================


def cmd_history(history: ConversationHistory):
    msgs = history.full_history()
    if not msgs:
        _print_info("No messages yet.")
        return
    print()
    print(f"  {bold('Conversation history')}  {dim(f'({len(msgs)} messages)')}")
    print(_hr())
    for i, m in enumerate(msgs):
        role_str = bold(cyan("YOU")) if m["role"] == "user" else bold(yellow("AGENT"))
        snippet = m["content"][:200].replace("\n", " ")
        ellipsis = "…" if len(m["content"]) > 200 else ""
        print(f"  [{i:02d}] {role_str}: {dim(snippet)}{ellipsis}")
    print(_hr())
    print()


def cmd_state(state: dict):
    if not state:
        _print_info("Sandbox state is empty (no variables yet).")
        return
    print()
    print(f"  {bold('Sandbox state')}  {dim(f'({len(state)} variable(s))')}")
    print(_hr())
    for k, v in state.items():
        v_repr = repr(v)
        if len(v_repr) > 80:
            v_repr = v_repr[:77] + "…"
        print(f"  {cyan(k)} = {v_repr}")
    print(_hr())
    print()


# ============================================================================
# MAIN SESSION LOOP
# ============================================================================


def main():
    args = sys.argv[1:]

    # ── --build flag ─────────────────────────────────────────────────────
    if "--build" in args:
        ok = build_sandbox_image()
        if not ok:
            sys.exit(1)
        args = [a for a in args if a != "--build"]

    # ── init sandbox ─────────────────────────────────────────────────────
    try:
        sandbox = SandboxClient()
    except RuntimeError as exc:
        print(f"\n{red('✗')} {exc}")
        print(
            f"  Run:  {cyan('python cli.py --build')}  to build the Docker image first.\n"
        )
        sys.exit(1)

    # ── init session state ───────────────────────────────────────────────
    history: ConversationHistory = ConversationHistory(max_messages=20)
    state: dict = {}

    _banner()
    print(green("  Docker sandbox ready.") + dim("  Start typing your question.\n"))

    # ── REPL ─────────────────────────────────────────────────────────────
    while True:
        try:
            raw = input(bold(cyan("  you › "))).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {dim('Goodbye.')}\n")
            sys.exit(0)

        if not raw:
            continue

        # ── built-in commands ─────────────────────────────────────────────
        if raw.startswith("/"):
            cmd = raw.lower().split()[0]

            if cmd == "/exit":
                print(f"\n  {dim('Goodbye.')}\n")
                sys.exit(0)

            elif cmd == "/clear":
                _banner()

            elif cmd == "/help":
                _print_help()

            elif cmd == "/reset":
                history = ConversationHistory(max_messages=20)
                state = {}
                _print_info("Session reset. History and sandbox state cleared.")

            elif cmd == "/history":
                cmd_history(history)

            elif cmd == "/state":
                cmd_state(state)

            else:
                _print_error(f"Unknown command: {cmd}  (type /help for list)")

            continue

        # ── agent query ───────────────────────────────────────────────────
        print()
        print(_hr())
        answer, state = run_interactive(
            query=raw,
            history=history,
            state=state,
            sandbox=sandbox,
        )

        if answer is None:
            _print_error("Agent did not produce a final answer.")


if __name__ == "__main__":
    main()
