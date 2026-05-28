#!/usr/bin/env python3
"""
cli.py — FinSight interactive CLI

Usage:
  python cli.py              # start session
  python cli.py --build      # build Docker image, then start

Session commands:
  /help               show commands
  /clear              clear screen
  /reset              wipe history + sandbox state
  /history            show conversation log
  /state              show sandbox variables
  /ingest TICKER YEAR [YEAR...]   ingest 10-K filings into RAG
  /sources            show what's been ingested
  /exit               quit
"""

from __future__ import annotations

import os, sys, time, textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.loop import SYSTEM_PROMPT, _extract, _build_observation
from api.apis import llm_request
from agent.history import ConversationHistory
from agent.sandbox_client import SandboxClient, build_sandbox_image, ExecResult

# ============================================================================
# COLOURS
# ============================================================================

_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(t, code):
    return f"\033[{code}m{t}\033[0m" if _TTY else t


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


def blue(t):
    return _c(t, "34")


WIDTH = 72


def _hr(ch="─"):
    return dim(ch * WIDTH)


# ============================================================================
# PRINTERS
# ============================================================================


def _banner():
    os.system("clear" if os.name != "nt" else "cls")
    print()
    print(bold(cyan("  ███████╗██╗███╗   ██╗███████╗██╗ ██████╗ ██╗  ██╗████████╗")))
    print(bold(cyan("  ██╔════╝██║████╗  ██║██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝")))
    print(bold(cyan("  █████╗  ██║██╔██╗ ██║███████╗██║██║  ███╗███████║   ██║   ")))
    print(bold(cyan("  ██╔══╝  ██║██║╚██╗██║╚════██║██║██║   ██║██╔══██║   ██║   ")))
    print(bold(cyan("  ██║     ██║██║ ╚████║███████║██║╚██████╔╝██║  ██║   ██║   ")))
    print(bold(cyan("  ╚═╝     ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝ ")))
    print()
    print(bold("  Financial Research Agent  ·  CodeAct + RAG + Docker Sandbox"))
    print(dim("  /help for commands  ·  /ingest AAPL 2023 to load filings"))
    print()
    print(_hr("═"))
    print()


def _print_thought(t):
    print(f"\n  {bold(yellow('🧠 THOUGHT'))}")
    for line in textwrap.wrap(t, WIDTH - 4):
        print(f"    {dim(line)}")


def _print_code(code):
    print(f"\n  {bold(blue('💻 CODE'))}")
    print(f"  {dim('┌' + '─'*(WIDTH-2))}")
    for line in code.splitlines():
        print(f"  {dim('│')} {cyan(line)}")
    print(f"  {dim('└' + '─'*(WIDTH-2))}")


def _print_exec(r: ExecResult, iteration: int, elapsed: float):
    ok = green("✓ ok") if r.ok else red("✗ error")
    print(
        f"\n  {bold('📤 EXECUTION')}  {dim(f'iter={iteration}  {elapsed*1000:.0f}ms')}  {ok}"
    )
    for line in r.stdout.rstrip().splitlines():
        print(f"    {line}")
    for line in r.stderr.rstrip().splitlines():
        print(f"    {red(line)}")
    if r.state:
        print(f"  {dim(f'vars: {list(r.state.keys())}')}")


def _print_answer(answer: str, elapsed: float, iters: int):
    print(f"\n{_hr()}")
    print(f"  {bold(green('✅ ANSWER'))}  {dim(f'{iters} iter · {elapsed:.1f}s')}")
    print(_hr())
    print()
    for para in answer.split("\n"):
        if para.strip():
            for line in textwrap.wrap(para, WIDTH - 2):
                print(f"  {line}")
        else:
            print()
    print(f"\n{_hr()}")


def _print_rag(chunks: list[dict]):
    print(f"\n  {bold(yellow('📚 RAG'))}  {dim(f'{len(chunks)} chunk(s) retrieved')}")
    for c in chunks:
        label = f"{c['ticker']} {c['year']} {c['section']}"
        score = f"score={c.get('score', 0):.4f}"
        snippet = c["text"][:120].replace("\n", " ")
        print(f"    {cyan(label)}  {dim(score)}")
        print(f"    {dim(snippet)}…")


def _print_error(msg):
    print(f"\n  {red('✗')} {msg}\n")


def _print_info(msg):
    print(f"\n  {dim(msg)}\n")


def _print_help():
    cmds = [
        ("/ingest TICKER YEAR [YEAR...]", "Download & index 10-K filings"),
        ("/sources", "Show ingested tickers/years"),
        ("/help", "Show this help"),
        ("/clear", "Clear screen"),
        ("/reset", "New session (wipes history + state)"),
        ("/history", "Print conversation log"),
        ("/state", "Show sandbox variables"),
        ("/exit", "Quit"),
    ]
    print()
    print(f"  {bold('Commands')}")
    for cmd, desc in cmds:
        print(f"    {cyan(cmd):<38} {dim(desc)}")
    print()
    print(f"  {bold('Example flow')}")
    for tip in [
        "/ingest AAPL 2022 2023 2024",
        "/ingest MSFT 2022 2023 2024",
        "Compare Apple and Microsoft R&D spend as % of revenue",
    ]:
        print(f"    {dim('›')} {tip}")
    print()


# ============================================================================
# RAG HELPERS
# ============================================================================


def _try_import_rag():
    """Return RAGPipeline class or None if rag/ deps not installed."""
    try:
        from rag import RAGPipeline

        return RAGPipeline
    except ImportError:
        return None


def _build_rag_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks as a context block injected into the system prompt.
    The agent can read these values directly or write code to manipulate them.
    """
    if not chunks:
        return ""
    lines = ["## Retrieved Filing Context (use this data in your code)\n"]
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] {c['ticker']} {c['year']} {c['section']} — {c['title']}\n"
            f"{c['text'][:800]}\n"
        )
    return "\n".join(lines)


def _augmented_system_prompt(rag_context: str) -> str:
    """Prepend RAG context to the base system prompt for this turn."""
    if not rag_context:
        return SYSTEM_PROMPT
    return (
        "## Filing Data Retrieved from SEC EDGAR\n"
        "The following sections were retrieved for your query. "
        "Extract numbers from them and use Python to compute the answer.\n\n"
        f"{rag_context}\n\n"
        "---\n\n" + SYSTEM_PROMPT
    )


# ============================================================================
# MAIN AGENT LOOP  (RAG-aware)
# ============================================================================


def run_interactive(
    query: str,
    history: ConversationHistory,
    state: dict,
    sandbox: SandboxClient,
    rag,  # RAGPipeline instance or None
    max_iterations: int = 10,
    timeout_s: float = 60.0,
) -> tuple[str | None, dict]:
    """
    One query through the CodeAct loop.
    If rag is not None, retrieves relevant chunks and injects them into
    the system prompt before the first LLM call.
    """
    history.add_user(query)
    answer = None
    t0 = time.monotonic()

    # ── RAG retrieval (once per query, before first LLM call) ────────────
    rag_context = ""
    if rag and len(rag) > 0:
        print(f"  {dim('[ retrieving from RAG... ]')}", end="", flush=True)
        chunks = rag.retrieve(query, top_k=4)
        print(f"\r{' '*50}\r", end="", flush=True)
        if chunks:
            _print_rag(chunks)
            rag_context = _build_rag_context(chunks)

    system_prompt = _augmented_system_prompt(rag_context)

    for iteration in range(1, max_iterations + 1):
        if time.monotonic() - t0 >= timeout_s:
            _print_error(f"Timeout. Stopping.")
            break

        print(
            f"\n  {dim(f'[ iter {iteration} · calling LLM... ]')}", end="", flush=True
        )
        response = llm_request(system_prompt=system_prompt, messages=history.as_dicts())
        print(f"\r{' '*50}\r", end="", flush=True)

        if response is None:
            _print_error("LLM call failed.")
            break

        history.add_assistant(response)

        thought = _extract("THOUGHT", response)
        code = _extract("CODE", response)
        final_answer = _extract("FINAL_ANSWER", response)

        if thought:
            _print_thought(thought)
        if final_answer:
            answer = final_answer
            _print_answer(answer, time.monotonic() - t0, iteration)
            break
        if not code:
            _print_error("No <CODE> block — stopping.")
            break

        _print_code(code)
        print(f"  {dim('[ executing in Docker sandbox... ]')}", end="", flush=True)
        t_exec = time.monotonic()
        result = sandbox.run(code, state)
        elapsed = time.monotonic() - t_exec
        print(f"\r{' '*50}\r", end="", flush=True)

        state = result.state
        _print_exec(result, iteration, elapsed)
        history.add_user(_build_observation(result))

    return answer, state


# ============================================================================
# COMMAND HANDLERS
# ============================================================================


def cmd_history(history: ConversationHistory):
    msgs = history.full_history()
    if not msgs:
        _print_info("No messages yet.")
        return
    print(f"\n  {bold('Conversation')}  {dim(f'({len(msgs)} messages)')}")
    print(_hr())
    for i, m in enumerate(msgs):
        role = bold(cyan("YOU")) if m["role"] == "user" else bold(yellow("AGENT"))
        snip = m["content"][:200].replace("\n", " ")
        tail = "…" if len(m["content"]) > 200 else ""
        print(f"  [{i:02d}] {role}: {dim(snip)}{tail}")
    print(_hr())
    print()


def cmd_state(state: dict):
    if not state:
        _print_info("Sandbox state is empty.")
        return
    print(f"\n  {bold('Sandbox state')}  {dim(f'({len(state)} var(s))')}")
    print(_hr())
    for k, v in state.items():
        s = repr(v)
        s = s[:77] + "…" if len(s) > 80 else s
        print(f"  {cyan(k)} = {s}")
    print(_hr())
    print()


def cmd_sources(rag):
    if rag is None:
        _print_info("RAG not available (install rag/ dependencies).")
        return
    if len(rag) == 0:
        _print_info("Nothing ingested yet. Use /ingest TICKER YEAR.")
        return
    # Pull source list from the store chunks
    seen: dict[str, set] = {}
    for c in rag.store._chunks:
        seen.setdefault(c["ticker"], set()).add(c["year"])
    print(f"\n  {bold('Ingested sources')}  {dim(f'({len(rag)} chunks total)')}")
    print(_hr())
    for ticker, years in sorted(seen.items()):
        print(f"  {cyan(ticker)}  {dim(sorted(years))}")
    print(_hr())
    print()


def cmd_ingest(parts: list[str], rag) -> bool:
    """
    /ingest TICKER YEAR [YEAR ...]
    Returns True if rag is available, False otherwise.
    """
    if rag is None:
        _print_error("RAG not available. Install: pip install -r requirements.txt")
        return False
    if len(parts) < 3:
        _print_error(
            "Usage: /ingest TICKER YEAR [YEAR ...]  e.g. /ingest AAPL 2022 2023"
        )
        return True

    ticker = parts[1].upper()
    try:
        years = [int(y) for y in parts[2:]]
    except ValueError:
        _print_error("Years must be integers, e.g. /ingest AAPL 2022 2023")
        return True

    print(f"\n  {bold('📥 INGEST')}  {cyan(ticker)}  {dim(str(years))}")
    print(_hr())
    try:
        n = rag.ingest(ticker, years=years)
        if n:
            print(f"\n  {green('✓')} Ingested {n} chunks for {ticker} {years}")
        else:
            _print_error(f"No chunks produced for {ticker} {years}.")
    except Exception as exc:
        _print_error(f"Ingest failed: {exc}")
    print()
    return True


# ============================================================================
# MAIN
# ============================================================================


def main():
    args = sys.argv[1:]

    if "--build" in args:
        if not build_sandbox_image():
            sys.exit(1)
        args = [a for a in args if a != "--build"]

    try:
        sandbox = SandboxClient()
    except RuntimeError as exc:
        print(f"\n{red('✗')} {exc}")
        print(f"  Run: {cyan('python cli.py --build')} first.\n")
        sys.exit(1)

    # RAG — optional; silently unavailable if deps not installed
    RAGPipeline = _try_import_rag()
    rag = RAGPipeline() if RAGPipeline else None

    history: ConversationHistory = ConversationHistory(max_messages=20)
    state: dict = {}

    _banner()
    status = green("  Docker sandbox ready.")
    rag_status = (
        dim("  RAG ready. Use /ingest TICKER YEAR to load filings.")
        if rag
        else dim("  RAG unavailable (pip install -r requirements.txt).")
    )
    print(status)
    print(rag_status)
    print()

    while True:
        try:
            raw = input(bold(cyan("  you › "))).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {dim('Goodbye.')}\n")
            sys.exit(0)

        if not raw:
            continue

        if raw.startswith("/"):
            parts = raw.split()
            cmd = parts[0].lower()

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
                _print_info("Session reset.")
            elif cmd == "/history":
                cmd_history(history)
            elif cmd == "/state":
                cmd_state(state)
            elif cmd == "/sources":
                cmd_sources(rag)
            elif cmd == "/ingest":
                cmd_ingest(parts, rag)
            else:
                _print_error(f"Unknown command: {cmd}  (type /help)")
            continue

        print(f"\n{_hr()}")
        answer, state = run_interactive(
            query=raw,
            history=history,
            state=state,
            sandbox=sandbox,
            rag=rag,
        )
        if answer is None:
            _print_error("Agent did not produce a final answer.")


if __name__ == "__main__":
    main()
