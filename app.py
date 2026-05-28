from __future__ import annotations

import os
import re
import sys
import time
import threading
import queue
from typing import Optional, Generator

import streamlit as st

# ── project root on path ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# PAGE CONFIG  (must be first Streamlit call)
# ============================================================================

st.set_page_config(
    page_title="FinSight",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(
    """
<style>
/* Chat bubbles */
.user-bubble {
    background: #1e3a5f;
    color: #e8f4fd;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 15px;
}
.agent-bubble {
    background: #1a1a2e;
    color: #e0e0e0;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 18px;
    margin: 8px 0;
    max-width: 90%;
    font-size: 15px;
}

/* Thought block */
.thought-block {
    background: #1c1c1c;
    border-left: 3px solid #f0c040;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 6px 0;
    color: #c8b96e;
    font-style: italic;
    font-size: 14px;
}

/* Code block */
.code-block {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 6px 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 13px;
    color: #79c0ff;
    overflow-x: auto;
    white-space: pre;
}

/* Execution output */
.exec-ok   { border-left: 3px solid #3fb950; background: #0d1f0d; border-radius: 4px; padding: 8px 14px; margin: 4px 0; color: #7ee787; font-family: monospace; font-size: 13px; }
.exec-err  { border-left: 3px solid #f85149; background: #1f0d0d; border-radius: 4px; padding: 8px 14px; margin: 4px 0; color: #ff7b72; font-family: monospace; font-size: 13px; }

/* RAG context */
.rag-block {
    background: #12232e;
    border-left: 3px solid #388bfd;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 13px;
    color: #79c0ff;
}

/* Final answer */
.answer-block {
    background: linear-gradient(135deg, #0d2137 0%, #0a2a1a 100%);
    border: 1px solid #3fb950;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
    color: #e6edf3;
    font-size: 15px;
    line-height: 1.6;
}

/* Sidebar */
.sidebar-section { margin-bottom: 20px; }

/* Status badge */
.badge-green { background: #196127; color: #3fb950; border-radius: 12px; padding: 2px 10px; font-size: 12px; }
.badge-blue  { background: #0d2137; color: #388bfd; border-radius: 12px; padding: 2px 10px; font-size: 12px; }
.badge-gray  { background: #21262d; color: #8b949e; border-radius: 12px; padding: 2px 10px; font-size: 12px; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# LAZY IMPORTS  (don't crash if deps missing)
# ============================================================================


@st.cache_resource(show_spinner="Initialising Docker sandbox...")
def _get_sandbox():
    from agent.sandbox_client import SandboxClient

    return SandboxClient()


@st.cache_resource(show_spinner="Initialising RAG pipeline...")
def _get_rag():
    try:
        from rag import RAGPipeline

        return RAGPipeline()
    except ImportError:
        return None


def _groq_chat(system_prompt, messages):
    from api.apis import llm_request

    return llm_request(system_prompt=system_prompt, messages=messages)


def _extract(tag, text):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


# ============================================================================
# SESSION STATE INIT
# ============================================================================


def _init_state():
    defaults = {
        "messages": [],  # list of display dicts (role, content, type)
        "history": [],  # raw LLM message dicts for groq_chat
        "sandbox_state": {},  # persisted Python variables
        "ingested": {},  # {ticker: [years]}
        "running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ============================================================================
# SYSTEM PROMPT (with optional RAG context)
# ============================================================================

_BASE_SYSTEM_PROMPT = """You are FinSight, an expert financial research agent.
You solve problems by writing and executing Python code.

## Response format — follow EXACTLY

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
- Available: numpy (np), pandas (pd), plus standard builtins
- Variables persist across iterations
- print() output is captured and returned to you
- Never fabricate numbers — always compute them
- Maximum 10 iterations, 60s wall-clock budget
"""


def _build_system_prompt(rag_chunks: list[dict]) -> str:
    if not rag_chunks:
        return _BASE_SYSTEM_PROMPT
    context_lines = ["## Retrieved SEC Filing Data\nUse this data in your code:\n"]
    for i, c in enumerate(rag_chunks, 1):
        context_lines.append(
            f"[{i}] {c['ticker']} {c['year']} {c['section']} — {c['title']}\n"
            f"{c['text'][:600]}\n"
        )
    return "\n".join(context_lines) + "\n---\n\n" + _BASE_SYSTEM_PROMPT


# ============================================================================
# DISPLAY HELPERS
# ============================================================================


def _add_message(role: str, content: str, msg_type: str = "text"):
    """Append to display message log."""
    st.session_state.messages.append(
        {"role": role, "content": content, "type": msg_type}
    )


def _render_message(msg: dict):
    """Render a single message dict to the chat area."""
    role = msg["role"]
    content = msg["content"]
    mtype = msg.get("type", "text")

    if role == "user":
        st.markdown(
            f'<div class="user-bubble">🧑 {content}</div>', unsafe_allow_html=True
        )

    elif mtype == "thought":
        st.markdown(
            f'<div class="thought-block">🧠 <b>Thought</b><br>{content}</div>',
            unsafe_allow_html=True,
        )

    elif mtype == "code":
        st.markdown(f'<div class="code-block">{content}</div>', unsafe_allow_html=True)

    elif mtype == "exec_ok":
        st.markdown(
            f'<div class="exec-ok">📤 <b>Output</b><br><pre>{content}</pre></div>',
            unsafe_allow_html=True,
        )

    elif mtype == "exec_err":
        st.markdown(
            f'<div class="exec-err">⚠️ <b>Error</b><br><pre>{content}</pre></div>',
            unsafe_allow_html=True,
        )

    elif mtype == "rag":
        st.markdown(
            f'<div class="rag-block">📚 <b>RAG Context</b><br>{content}</div>',
            unsafe_allow_html=True,
        )

    elif mtype == "answer":
        st.markdown(
            f'<div class="answer-block">✅ <b>Answer</b><br><br>{content}</div>',
            unsafe_allow_html=True,
        )

    elif mtype == "error":
        st.error(content)

    else:
        st.markdown(
            f'<div class="agent-bubble">{content}</div>', unsafe_allow_html=True
        )


def _render_all_messages():
    for msg in st.session_state.messages:
        _render_message(msg)


# ============================================================================
# TICKER DETECTION  (auto-ingest from user query)
# ============================================================================

# Map company names → tickers
_NAME_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "intel": "INTC",
    "amd": "AMD",
    "oracle": "ORCL",
}

_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")


def _detect_tickers(query: str) -> list[str]:
    """Extract ticker symbols and company names from the query."""
    found = set()
    q_lower = query.lower()

    # Match company names
    for name, ticker in _NAME_TO_TICKER.items():
        if name in q_lower:
            found.add(ticker)

    # Match explicit tickers (ALL-CAPS words 1–5 chars)
    for m in _TICKER_RE.finditer(query):
        t = m.group(1)
        if len(t) >= 2 and t not in {"I", "A", "THE", "AND", "OR", "OF", "IN"}:
            found.add(t)

    return list(found)


def _detect_years(query: str) -> list[int]:
    """Extract years mentioned in query, default to last 3 if none."""
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", query)]
    if not years:
        current = time.localtime().tm_year
        years = [current - 3, current - 2, current - 1]
    return sorted(set(years))


# ============================================================================
# CORE AGENT LOOP  (runs synchronously, yields display events)
# ============================================================================


def run_agent_streaming(
    query: str,
    sandbox,
    rag,
    max_iterations: int = 10,
    timeout_s: float = 90.0,
) -> Generator[dict, None, None]:
    """
    Generator that runs the CodeAct loop and yields display events.
    Each event is a dict: {"type": ..., "content": ...}
    Caller renders events as they arrive.
    """
    history = list(st.session_state.history)  # copy current LLM history
    state = dict(st.session_state.sandbox_state)
    t0 = time.monotonic()

    # ── auto-ingest based on intent ──────────────────────────────────────
    if rag is not None:
        tickers = _detect_tickers(query)
        years = _detect_years(query)

        for ticker in tickers:
            already = set(st.session_state.ingested.get(ticker, []))
            to_fetch = [y for y in years if y not in already]
            if to_fetch:
                yield {
                    "type": "status",
                    "content": f"📥 Auto-ingesting {ticker} {to_fetch}...",
                }
                try:
                    n = rag.ingest(ticker, years=to_fetch)
                    existing = st.session_state.ingested.get(ticker, [])
                    st.session_state.ingested[ticker] = sorted(set(existing + to_fetch))
                    yield {
                        "type": "status",
                        "content": f"✓ Ingested {n} chunks for {ticker}",
                    }
                except Exception as exc:
                    yield {
                        "type": "status",
                        "content": f"⚠ Ingest failed for {ticker}: {exc}",
                    }

        # ── RAG retrieval ────────────────────────────────────────────────
        if len(rag) > 0:
            chunks = rag.retrieve(query, top_k=4)
            if chunks:
                rag_summary = "\n".join(
                    f"• {c['ticker']} {c['year']} {c['section']}: "
                    f"{c['text'][:120].replace(chr(10), ' ')}…"
                    for c in chunks
                )
                yield {"type": "rag", "content": rag_summary}
                system_prompt = _build_system_prompt(chunks)
            else:
                system_prompt = _BASE_SYSTEM_PROMPT
        else:
            system_prompt = _BASE_SYSTEM_PROMPT
    else:
        system_prompt = _BASE_SYSTEM_PROMPT

    # ── CodeAct loop ─────────────────────────────────────────────────────
    history.append({"role": "user", "content": query})
    final_answer = None

    for iteration in range(1, max_iterations + 1):
        elapsed = time.monotonic() - t0
        if elapsed >= timeout_s:
            yield {"type": "error", "content": f"Timeout after {elapsed:.0f}s."}
            break

        yield {"type": "status", "content": f"⟳ Iteration {iteration} — calling LLM..."}

        response = _groq_chat(system_prompt, history)
        if response is None:
            yield {"type": "error", "content": "LLM call failed."}
            break

        history.append({"role": "assistant", "content": response})

        thought = _extract("THOUGHT", response)
        code = _extract("CODE", response)
        final_answer = _extract("FINAL_ANSWER", response)

        if thought:
            yield {"type": "thought", "content": thought}

        if final_answer:
            yield {"type": "answer", "content": final_answer}
            break

        if not code:
            yield {"type": "error", "content": "No <CODE> block in response."}
            break

        yield {"type": "code", "content": code}
        yield {"type": "status", "content": "⟳ Executing in Docker sandbox..."}

        result = sandbox.run(code, state)
        state = result.state

        if result.stdout.strip():
            yield {"type": "exec_ok", "content": result.stdout.rstrip()}
        if result.stderr.strip():
            yield {"type": "exec_err", "content": result.stderr.rstrip()}

        # Feed back observation
        obs_parts = [f"[Code Execution Result]\nSuccess: {result.ok}"]
        if result.stdout.strip():
            obs_parts.append(f"Stdout:\n{result.stdout.rstrip()}")
        if result.stderr.strip():
            obs_parts.append(f"Stderr:\n{result.stderr.rstrip()}")
        if state:
            obs_parts.append(f"Available variables: {list(state.keys())}")
        history.append({"role": "user", "content": "\n\n".join(obs_parts)})

    # ── persist session state ─────────────────────────────────────────────
    # Store only assistant messages to avoid duplicating observations
    st.session_state.history = [m for m in history if m["role"] != "user"][-20:]
    st.session_state.sandbox_state = state

    if not final_answer:
        yield {"type": "error", "content": "Agent did not produce a final answer."}


# ============================================================================
# SIDEBAR
# ============================================================================


def _render_sidebar(rag):
    with st.sidebar:
        st.markdown("## 📈 FinSight")
        st.markdown("*Financial Research Agent*")
        st.divider()

        # Status
        st.markdown("### System Status")
        try:
            _get_sandbox()
            st.markdown(
                '<span class="badge-green">● Docker Sandbox</span>',
                unsafe_allow_html=True,
            )
        except Exception:
            st.markdown(
                '<span class="badge-gray">○ Docker Sandbox offline</span>',
                unsafe_allow_html=True,
            )

        if rag is not None:
            st.markdown(
                '<span class="badge-green">● RAG Pipeline</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="badge-gray">○ RAG unavailable</span>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Ingested sources
        st.markdown("### 📚 Ingested Filings")
        if st.session_state.ingested:
            for ticker, years in sorted(st.session_state.ingested.items()):
                st.markdown(f"**{ticker}** — {', '.join(str(y) for y in years)}")
        else:
            st.caption(
                "No filings loaded yet.\nJust ask about a company and I'll fetch automatically."
            )

        st.divider()

        # Manual ingest
        st.markdown("### Manual Ingest")
        col1, col2 = st.columns([2, 1])
        with col1:
            ticker_input = st.text_input(
                "Ticker", placeholder="AAPL", label_visibility="collapsed"
            )
        with col2:
            year_input = st.text_input(
                "Years", placeholder="2023", label_visibility="collapsed"
            )

        if st.button("Ingest", use_container_width=True, disabled=rag is None):
            if ticker_input and year_input:
                try:
                    years = [
                        int(y.strip()) for y in year_input.replace(",", " ").split()
                    ]
                    with st.spinner(f"Ingesting {ticker_input.upper()} {years}..."):
                        n = rag.ingest(ticker_input.upper(), years=years)
                    existing = st.session_state.ingested.get(ticker_input.upper(), [])
                    st.session_state.ingested[ticker_input.upper()] = sorted(
                        set(existing + years)
                    )
                    st.success(f"✓ {n} chunks ingested")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed: {exc}")

        st.divider()

        # Sandbox state
        if st.session_state.sandbox_state:
            st.markdown("### 🔧 Sandbox Variables")
            for k, v in list(st.session_state.sandbox_state.items())[:8]:
                v_str = repr(v)
                v_str = v_str[:50] + "…" if len(v_str) > 50 else v_str
                st.caption(f"`{k}` = {v_str}")

        st.divider()

        # Reset
        if st.button("🗑 Reset Session", use_container_width=True):
            for k in ["messages", "history", "sandbox_state", "ingested"]:
                st.session_state[k] = {} if k in ("sandbox_state", "ingested") else []
            st.rerun()

        # Example questions
        st.divider()
        st.markdown("### 💡 Example Questions")
        examples = [
            "Compare Apple and Microsoft R&D spend 2021–2023",
            "Calculate CAGR of revenues [120, 145, 160, 178, 210]",
            "Build a DCF: FCF $50B, 15% growth 5yr, 3% terminal, 10% discount",
            "Monthly returns volatility: [142,150,138,155,162,158,170]",
        ]
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state["_prefill"] = ex
                st.rerun()


# ============================================================================
# MAIN
# ============================================================================


def main():
    # Load resources
    sandbox = _get_sandbox()
    rag = _get_rag()

    # Sidebar
    _render_sidebar(rag)

    # Header
    st.markdown("## 📈 FinSight Agent")
    st.caption(
        "Ask financial questions — I'll fetch SEC filings, write Python, and show my work."
    )
    st.divider()

    # Chat history
    chat_container = st.container()
    with chat_container:
        _render_all_messages()

    # Input
    prefill = st.session_state.pop("_prefill", "")
    query = st.chat_input(
        "Ask a financial question...",
        disabled=st.session_state.running,
    )

    # Handle example button prefill
    if prefill and not query:
        query = prefill

    if query and not st.session_state.running:
        st.session_state.running = True

        # Show user message immediately
        _add_message("user", query)

        # Stream agent events
        with chat_container:
            _render_message({"role": "user", "content": query, "type": "text"})

            status_placeholder = st.empty()

            for event in run_agent_streaming(query, sandbox, rag):
                etype = event["type"]
                content = event["content"]

                if etype == "status":
                    status_placeholder.caption(content)
                    continue

                # Clear status line once real content arrives
                status_placeholder.empty()

                # Persist to message log and render
                if etype in (
                    "thought",
                    "code",
                    "exec_ok",
                    "exec_err",
                    "rag",
                    "answer",
                    "error",
                ):
                    _add_message("agent", content, etype)
                    _render_message(
                        {"role": "agent", "content": content, "type": etype}
                    )

            status_placeholder.empty()

        st.session_state.running = False
        st.rerun()


if __name__ == "__main__":
    main()
