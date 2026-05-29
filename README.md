# FinSight Agent

An autonomous financial research agent that answers complex questions by retrieving SEC filings, writing Python code, and executing it in a hardened sandbox. Built from scratch — no LangChain, no LangGraph.

---

## What It Does

You ask a question in plain English. FinSight:

1. Detects the companies and years you're asking about
2. Fetches the relevant 10-K sections from SEC EDGAR
3. Writes Python code to compute the answer
4. Executes that code in an isolated Docker container
5. Iterates until it has a verified, computed answer
6. Streams its reasoning step by step back to you

```
you › Compare Apple and Microsoft R&D spend as % of revenue 2021–2023

📥 Auto-ingesting AAPL [2021, 2022, 2023]...
📥 Auto-ingesting MSFT [2021, 2022, 2023]...
📚 RAG — 4 chunks retrieved from SEC filings

🧠 THOUGHT  I'll build a DataFrame with R&D and revenue for both companies...

💻 CODE
  import pandas as pd
  data = {"AAPL": [6.5, 6.7, 7.8], "MSFT": [12.3, 12.0, 12.4]}
  ...

📤 OUTPUT
  AAPL R&D%: 6.5 → 6.7 → 7.8  (avg 7.0%)
  MSFT R&D%: 12.3 → 12.0 → 12.4  (avg 12.2%)

✅ ANSWER
  Apple R&D intensity grew from 6.5% to 7.8% — a meaningful acceleration.
  Microsoft held steady around 12%, spending nearly 2× more as a % of revenue.
```

---

## Architecture

```
                   ┌─────────────────────────────────┐
                   │       Streamlit / CLI            │
                   └──────────────┬──────────────────┘
                                  │ HTTP + SSE
                   ┌──────────────▼──────────────────┐
                   │         FastAPI Backend          │
                   │   POST /query  GET /stream       │
                   └──────────────┬──────────────────┘
                                  │
         ┌────────────────────────▼──────────────────────────┐
         │               CodeAct Agent Loop                   │
         │  Thought → write Python → execute → observe → ...  │
         └──────┬────────────────────────────┬───────────────┘
                │                            │
  ┌─────────────▼──────────┐  ┌─────────────▼────────────────┐
  │      Groq API          │  │      Docker Sandbox           │
  │  raw HTTP, no SDK      │  │  --network none               │
  │  llama-3.3-70b         │  │  --cap-drop ALL               │
  └────────────────────────┘  │  non-root, 512 MB cap         │
                              └──────────────────────────────┘
                │
  ┌─────────────▼──────────────────────────────────────────┐
  │                    RAG Pipeline                          │
  │                                                          │
  │  SEC EDGAR → HTML strip → Item-header chunking           │
  │       ↓                                                  │
  │  all-MiniLM-L6-v2 embeddings → Qdrant (HNSW)            │
  │       ↓                                                  │
  │  Dense ANN + BM25 → RRF fusion                           │
  │       ↓                                                  │
  │  cross-encoder/ms-marco-MiniLM-L-6-v2 reranker           │
  └──────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
finsight/
│
├── agent/
│   ├── __init__.py          # package exports
│   ├── groq_client.py       # raw HTTP Groq API, retry + rate-limit handling
│   ├── history.py           # rolling-window conversation history
│   ├── loop.py              # CodeAct agent loop + system prompt
│   └── sandbox_client.py    # Docker orchestrator (host side)
│
├── sandbox/
│   ├── Dockerfile           # python:3.13-slim, non-root user
│   ├── requirements.txt     # numpy, pandas (inside container)
│   └── runner.py            # executes inside the container
│
├── rag/
│   ├── __init__.py          # RAGPipeline orchestrator
│   ├── ingest.py            # SEC EDGAR download + HTML stripping
│   ├── chunker.py           # Item-header section splitting
│   ├── embedder.py          # all-MiniLM-L6-v2 + disk cache
│   ├── store.py             # Qdrant dense + BM25 + RRF
│   └── reranker.py          # cross-encoder reranker
│
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI + SSE streaming backend
│
├── eval/
│   ├── __init__.py
│   └── harness.py           # 20-question benchmark + LLM-as-judge
│
├── tests/
│   ├── test_sandbox.py      # 16 sandbox security + correctness tests
│   └── test_rag.py          # 11 RAG pipeline tests
│
├── app.py                   # Streamlit frontend
├── cli.py                   # Terminal CLI
├── main.py                  # Simple entry point
└── requirements.txt         # All host-side dependencies
```

---

## Setup

### Prerequisites

- Python 3.11+
- Docker Desktop (running)
- Free [Groq API key](https://console.groq.com/keys)

### 1. Clone and install

```bash
git clone https://github.com/yourname/finsight.git
cd finsight
pip install -r requirements.txt

# CPU-only PyTorch (saves ~2 GB):
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers==2.7.0
```

### 2. Set environment variables

```bash
export GROQ_API_KEY=gsk_...
```

### 3. Build the Docker sandbox

```bash
docker build -t finsight-sandbox ./sandbox
```

### 4. Start Qdrant

```bash
docker run -d -p 6333:6333 \
  -v $(pwd)/data/qdrant:/qdrant/storage \
  qdrant/qdrant
```

### 5. Verify everything works

```bash
python tests/test_sandbox.py   # 23 tests
python tests/test_rag.py       # 11 tests
```

---

## Running

### Streamlit web app

```bash
streamlit run app.py
# → http://localhost:8501
```

### FastAPI + Streamlit (production mode)

```bash
# Terminal 1
uvicorn api.main:app --reload --port 8000

# Terminal 2
streamlit run app.py
```

### CLI

```bash
python cli.py
```

Commands:

| Command                       | Description                     |
| ----------------------------- | ------------------------------- |
| `/ingest AAPL 2022 2023 2024` | Download and index 10-K filings |
| `/sources`                    | Show what's been ingested       |
| `/state`                      | Show current sandbox variables  |
| `/history`                    | Show the conversation log       |
| `/reset`                      | Start a fresh session           |
| `/exit`                       | Quit                            |

### Eval harness

```bash
# Quick (5 questions):
python eval/harness.py --quick

# Full benchmark (20 questions):
python eval/harness.py --out eval/results.json

# Single question:
python eval/harness.py --id B01
```

---

## Example Questions

**Arithmetic**

```
Calculate the CAGR of revenues [100, 130, 160, 200] over 3 years.

What is the Sharpe ratio for a portfolio returning 18% with 11% volatility
and a 4.5% risk-free rate?
```

**Financial modelling**

```
Build a DCF: FCF $50B, 15% growth for 5 years, 3% terminal growth,
10% WACC. Show year-by-year PV and total enterprise value.

Simulate dollar-cost averaging $500/month for 12 months into a stock
with these prices: [142,150,138,155,162,158,170,165,172,180,168,175].
Compare to lump-sum investing the full amount on day one.
```

**SEC filing grounded (auto-ingest on first ask)**

```
Compare Apple and Microsoft R&D spend as % of revenue from 2021 to 2023.

What risk factors did Apple highlight in its 2023 10-K?
```

**Complex multi-hop**

```
Model an LBO: $5B acquisition, 60% debt at 8%, EBITDA $800M growing 12%/yr,
exit at same multiple after 5 years. What is the equity IRR?

Build a DCF sensitivity table across WACC [8.5%, 9.5%, 10.5%]
and terminal growth [2%, 3%, 4%].

Run a 1000-path Monte Carlo for a $100K portfolio: monthly return mean=1.2%,
std=3.5%. Show median outcome and 5th percentile.
```

---

## Security Model

Code execution is hardened at four independent layers. All four must fail simultaneously for unsafe code to do anything.

| Layer               | Mechanism                          | Blocks                                      |
| ------------------- | ---------------------------------- | ------------------------------------------- |
| Docker network      | `--network none`                   | All inbound and outbound traffic            |
| Docker capabilities | `--cap-drop ALL`                   | Privilege escalation, raw sockets           |
| Docker user         | UID 10001, non-root                | Writing to `/app`, installing packages      |
| Docker resources    | `--memory 512m`, `--pids-limit 64` | OOM, fork bombs                             |
| Python builtins     | Custom `__builtins__` dict         | `open`, `eval`, `exec`, `__import__`        |
| sys.modules         | Dangerous modules set to `None`    | `os`, `socket`, `subprocess`, `pickle`      |
| Audit hook          | `sys.addaudithook`                 | Stray import/open attempts from exec'd code |

---

## RAG Pipeline Detail

```
SEC EDGAR (10-K filings)
        ↓
sec-edgar-downloader → raw HTML
        ↓
BeautifulSoup → plain text (strips XBRL, scripts, styles)
        ↓
Regex split on Item headers (Item 1, 1A, 1B, 2 ... 15)
        ↓
Sub-chunk sections > 6,000 chars on blank-line boundaries
        ↓
all-MiniLM-L6-v2 → 384-dim unit-norm embeddings
SHA-256 disk cache (re-runs skip re-embedding)
        ↓
Qdrant upsert (HNSW index, cosine distance, persistent to disk)
        ↓
At query time:
  Dense ANN (Qdrant, k=20) ──┐
                              ├→ RRF fusion → cross-encoder rerank → top 5
  BM25 sparse (k=20)     ────┘
        ↓
Top 5 chunks injected into system prompt as grounded context
```

---

## Stack

| Component      | Technology                                   |
| -------------- | -------------------------------------------- |
| LLM            | Groq API — `llama-3.3-70b-versatile`         |
| Agent pattern  | CodeAct (Thought → Code → Execute → Observe) |
| API calls      | Raw HTTP `requests` — no SDK                 |
| Code execution | Docker sandbox (hardened)                    |
| Embeddings     | `sentence-transformers/all-MiniLM-L6-v2`     |
| Vector DB      | Qdrant (self-hosted, HNSW)                   |
| Sparse search  | BM25 (implemented from scratch)              |
| Fusion         | Reciprocal Rank Fusion                       |
| Reranker       | `cross-encoder/ms-marco-MiniLM-L-6-v2`       |
| Filing data    | SEC EDGAR via `sec-edgar-downloader`         |
| Backend        | FastAPI + SSE (`sse-starlette`)              |
| Frontend       | Streamlit                                    |
| CLI            | Python with ANSI terminal output             |

---

## Eval Results

Run `python eval/harness.py --out eval/results.json` to generate your own numbers. The harness covers 20 questions across four categories scored by an LLM judge (0–3 per answer).

| Category            | IDs     | Focus                                               |
| ------------------- | ------- | --------------------------------------------------- |
| Arithmetic          | A01–A05 | CAGR, standard deviation, compound growth           |
| Financial modelling | B01–B05 | DCF, Sharpe ratio, VaR, dollar-cost averaging       |
| Statistical         | C01–C05 | Regression, Monte Carlo, correlation, rolling stats |
| Multi-hop pandas    | D01–D05 | Income statements, LBO, sensitivity tables          |

**Reported metrics:** average judge score (0–3), code execution success rate, average iterations per question, latency p50 and p95.

---

## Interview Talking Points

**Why CodeAct instead of fixed tool schemas?**
Tool schemas force you to pre-define every action the agent can take. CodeAct lets the agent write arbitrary Python — so it can build DataFrames, run regressions, simulate Monte Carlo paths, compute IRR — without me having to anticipate any of it in advance. The agent's action space is the full Python standard library plus numpy and pandas.

**Why raw HTTP instead of an LLM SDK?**
Full visibility into every request and response. The message history structure, retry logic, and rate-limit handling are explicit and debuggable. Nothing is hidden behind an abstraction, which matters both for correctness and for being able to explain exactly what's happening in an interview.

**Why Docker instead of restricted `exec()`?**
In-process `exec()` with a restricted `__builtins__` dict can be escaped via `ctypes`, `gc` object traversal, or `__class__.__mro__` attribute chains. Docker provides OS-level isolation — separate network namespace, separate PID namespace, kernel-enforced memory limits. Four independent security layers need to fail simultaneously for anything unsafe to execute.

**Why BM25 alongside dense vectors?**
Dense retrieval is strong at semantic similarity but misses exact keyword matches — ticker symbols, Item numbers, specific financial line items like "operating cash flow". BM25 catches those. RRF fusion consistently outperforms either retriever alone on financial text.

**Why a cross-encoder reranker on top of RRF?**
Cosine similarity scores each passage independently of the query. A cross-encoder sees the query and passage concatenated and scores them jointly — far more accurate at distinguishing topically related passages from passages that actually answer the question. Applied only to the top 10 RRF candidates so latency stays manageable.

**How does the conversation history rolling window work?**
The original user query is pinned at index 0. The first assistant response is pinned at index 1. When total messages exceed the window limit, the oldest middle pairs are dropped. The LLM always has the original question and the most recent context, regardless of session length.

---

## Roadmap

- [ ] Qdrant Cloud (replace local Docker)
- [ ] yfinance integration for real-time price and fundamentals data
- [ ] `get_financials()` and `get_filing()` functions exposed to sandbox
- [ ] Streaming token output from LLM
- [ ] PostgreSQL session storage
- [ ] RAGAS evaluation metrics (faithfulness, answer relevance, context precision)
- [ ] Fine-tuned system prompt based on eval results

---

## License

MIT
