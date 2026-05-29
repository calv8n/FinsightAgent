"""
api/main.py — FastAPI backend with SSE streaming.

Endpoints:
    POST /query              Submit a question → returns {session_id}
    GET  /stream/{session_id} SSE stream of agent events
    GET  /history/{session_id} Full message log for a session
    POST /ingest             Ingest a ticker+years into RAG
    GET  /sources            List ingested tickers/years
    GET  /health             Health check

Run:
    pip install fastapi uvicorn[standard] sse-starlette
    uvicorn api.main:app --reload --port 8000

Streamlit connects to this instead of running the agent in-process.
This means Streamlit stays non-blocking — it just opens an SSE connection
and renders events as they arrive.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import uuid
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.apis import llm_request
from agent.sandbox_client import SandboxClient
from rag import RAGPipeline

# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(title="FinSight API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons (initialised once at startup) ──────────────────────────────────
_sandbox: Optional[SandboxClient] = None
_rag: Optional[RAGPipeline] = None


@app.on_event("startup")
async def _startup():
    global _sandbox, _rag
    _sandbox = SandboxClient()
    _rag = RAGPipeline()
    print("✅ Sandbox and RAG pipeline ready.")


# ── In-memory session store ──────────────────────────────────────────────────
# Maps session_id → {"history": [...], "state": {...}, "events": asyncio.Queue}
_sessions: dict[str, dict] = {}


# ============================================================================
# SYSTEM PROMPT (same as loop.py)
# ============================================================================

SYSTEM_PROMPT = """You are FinSight, an autonomous financial research agent.
You MUST solve every problem by writing and executing Python code.
You are NOT allowed to answer directly — you must always write code first.

## STRICT OUTPUT FORMAT

RULE: Every single response must start with <THOUGHT> and contain either <CODE> or <FINAL_ANSWER>.
RULE: Never write a prose answer without first computing it in code.
RULE: Even if the answer seems obvious, you must verify it by running code.

STRUCTURE A — when you need to compute:
<THOUGHT>
What you plan to compute and why.
</THOUGHT>
<CODE>
# Python code — use print() to output results
</CODE>

STRUCTURE B — ONLY after code has run and confirmed the answer:
<THOUGHT>
One sentence summary of what the code produced.
</THOUGHT>
<FINAL_ANSWER>
Complete answer with numbers taken directly from code output.
</FINAL_ANSWER>

## Execution environment
- numpy as `np`, pandas as `pd`
- print() is the only way to see values
- Variables persist across iterations
- Max 10 iterations, 90s budget
"""


# ============================================================================
# MODELS
# ============================================================================


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # provide to continue a session


class IngestRequest(BaseModel):
    ticker: str
    years: list[int]


class QueryResponse(BaseModel):
    session_id: str


# ============================================================================
# HELPERS
# ============================================================================


def _extract(tag: str, text: str) -> Optional[str]:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def _sse_event(event_type: str, data: dict) -> dict:
    return {"event": event_type, "data": json.dumps(data)}


def _build_system_prompt(rag_chunks: list[dict]) -> str:
    if not rag_chunks:
        return SYSTEM_PROMPT
    ctx = ["## Retrieved SEC Filing Data\n"]
    for i, c in enumerate(rag_chunks, 1):
        ctx.append(
            f"[{i}] {c['ticker']} {c['year']} {c['section']} — {c['title']}\n{c['text'][:600]}\n"
        )
    return "\n".join(ctx) + "\n---\n\n" + SYSTEM_PROMPT


# ============================================================================
# AGENT LOOP  (runs in a background thread via asyncio.to_thread)
# ============================================================================


async def _run_agent(session_id: str, query: str):
    """
    Run the full CodeAct loop, pushing SSE events onto the session queue.
    Runs in a thread pool so it doesn't block the event loop.
    """
    session = _sessions[session_id]
    q: asyncio.Queue = session["events"]
    history: list[dict] = session["history"]
    state: dict = session["state"]
    t0 = time.monotonic()

    async def push(event_type: str, **data):
        await q.put(_sse_event(event_type, {"session_id": session_id, **data}))

    # ── auto-detect tickers and retrieve RAG context ──────────────────────
    await push("status", message="Analysing query...")

    rag_chunks: list[dict] = []
    if _rag and len(_rag) > 0:
        await push("status", message="Retrieving from RAG...")
        # Run blocking retrieve() in thread pool
        rag_chunks = await asyncio.to_thread(_rag.retrieve, query, 5)
        if rag_chunks:
            await push(
                "rag",
                chunks=[
                    {
                        "ticker": c["ticker"],
                        "year": c["year"],
                        "section": c["section"],
                        "snippet": c["text"][:150],
                    }
                    for c in rag_chunks
                ],
            )

    system_prompt = _build_system_prompt(rag_chunks)
    history.append({"role": "user", "content": query})

    final_answer = None

    for iteration in range(1, 11):
        elapsed = time.monotonic() - t0
        if elapsed >= 90:
            await push("error", message=f"Timeout after {elapsed:.0f}s.")
            break

        await push("status", message=f"Iteration {iteration} — calling LLM...")

        # Blocking LLM call in thread pool
        response = await asyncio.to_thread(llm_request, system_prompt, history)

        if response is None:
            await push("error", message="LLM call failed.")
            break

        history.append({"role": "assistant", "content": response})

        thought = _extract("THOUGHT", response)
        code = _extract("CODE", response)
        final_answer = _extract("FINAL_ANSWER", response)

        if thought:
            await push("thought", content=thought)

        if final_answer:
            await push("answer", content=final_answer)
            break

        if not code:
            nudge = (
                "Your response did not contain a <CODE> block. "
                "Respond now with <THOUGHT> and <CODE>."
            )
            history.append({"role": "user", "content": nudge})
            await push("status", message="Re-prompting for code...")
            continue

        await push("code", content=code)
        await push("status", message="Executing in Docker sandbox...")

        # Blocking sandbox execution in thread pool
        result = await asyncio.to_thread(_sandbox.run, code, state)
        state = result.state

        if result.stdout.strip():
            await push("exec_ok", content=result.stdout.rstrip())
        if result.stderr.strip():
            await push("exec_err", content=result.stderr.rstrip())

        obs_parts = [f"[Code Execution Result]\nSuccess: {result.ok}"]
        if result.stdout.strip():
            obs_parts.append(f"Stdout:\n{result.stdout.rstrip()}")
        if result.stderr.strip():
            obs_parts.append(f"Stderr:\n{result.stderr.rstrip()}")
        if state:
            obs_parts.append(f"Available variables: {list(state.keys())}")
        history.append({"role": "user", "content": "\n\n".join(obs_parts)})

    # Persist state back
    session["state"] = state
    session["history"] = history
    await push("done", final_answer=final_answer or "")
    await q.put(None)  # sentinel — stream is complete


# ============================================================================
# ENDPOINTS
# ============================================================================


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sandbox": _sandbox is not None,
        "rag": len(_rag) if _rag else 0,
    }


@app.post("/query", response_model=QueryResponse)
async def submit_query(req: QueryRequest):
    """
    Submit a question. Returns a session_id immediately.
    Connect to GET /stream/{session_id} to receive events.
    """
    session_id = req.session_id or str(uuid.uuid4())

    if session_id not in _sessions:
        _sessions[session_id] = {
            "history": [],
            "state": {},
            "events": asyncio.Queue(),
        }
    else:
        # Reuse existing session — fresh queue for this query
        _sessions[session_id]["events"] = asyncio.Queue()

    # Fire agent loop as background task
    asyncio.create_task(_run_agent(session_id, req.query))

    return QueryResponse(session_id=session_id)


@app.get("/stream/{session_id}")
async def stream_events(session_id: str):
    """
    SSE endpoint. Client connects here after POST /query.
    Events stream until a 'done' event is received.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    q: asyncio.Queue = _sessions[session_id]["events"]

    async def generator() -> AsyncGenerator[dict, None]:
        while True:
            event = await q.get()
            if event is None:  # sentinel
                break
            yield event

    return EventSourceResponse(generator())


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_id": session_id, "history": _sessions[session_id]["history"]}


@app.post("/ingest")
async def ingest(req: IngestRequest):
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not available.")
    ticker = req.ticker.upper()
    n = await asyncio.to_thread(_rag.ingest, ticker, req.years)
    return {"ticker": ticker, "years": req.years, "chunks_added": n}


@app.get("/sources")
async def sources():
    if _rag is None:
        return {"sources": []}
    seen: dict[str, list[int]] = {}
    for c in _rag.store._cache:
        t = c.get("ticker", "?")
        y = c.get("year", 0)
        seen.setdefault(t, [])
        if y not in seen[t]:
            seen[t].append(y)
    return {
        "sources": {t: sorted(y) for t, y in seen.items()},
        "total_chunks": len(_rag),
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
