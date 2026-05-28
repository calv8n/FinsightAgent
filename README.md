# FinSight Agent

FinSight is an agentic financial research project that answers multi-step financial questions by retrieving SEC filings, executing Python code inside a hardened Docker sandbox, and synthesizing results for the user.

## What it does

- Converts natural language financial queries into a CodeAct-style reasoning loop.
- Uses a Docker sandbox to safely execute Python code written by the agent.
- Retrieves and ingests SEC filing content for financial research using a RAG pipeline.
- Supports both a CLI demo (`main.py`) and a Streamlit frontend (`app.py`).

## Key Features

- CodeAct agent loop: thought → code → execution → observation → repeat
- Hardened Docker sandbox with isolated execution and persisted state across iterations
- Groq-based LLM integration via `api/apis.py`
- SEC filing ingestion and retrieval pipeline in `rag/`
- Streamlit UI for interactive question answering
- CLI entry point for quick demos and sandbox image build

## Repository Layout

- `app.py` — Streamlit frontend for interactive question answering
- `main.py` — CLI entrypoint and sandbox build helper
- `agent/` — core agent loop, sandbox client, and conversation history
- `api/` — Groq API integration and request handling
- `rag/` — retrieval-augmented generation ingestion and search
- `sandbox/` — Docker sandbox definition and runner
- `docs/finagent.md` — project design spec and architecture overview
- `requirements.txt` — Python dependencies

## Prerequisites

- Python 3.11+ (or compatible Python 3.x environment)
- Docker installed and running
- `GROQ_API_KEY` set in environment or `.env`
- Optional: `MODEL` environment variable to override the default Groq model

## Installation

1. Create and activate your Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Groq API key:

```bash
cat > .env <<'EOF'
GROQ_API_KEY=your_groq_api_key_here
EOF
```

4. Build the Docker sandbox image:

```bash
python main.py --build
```

## Usage

### Run the CLI demo

```bash
python main.py
```

### Ask a custom question

```bash
python main.py "Compare AAPL and MSFT revenue growth from 2021 to 2024."
```

### Launch the Streamlit app

```bash
streamlit run app.py
```

The Streamlit UI will show a conversational agent interface with code, execution output, and answers.

## Configuration

Environment variables:

- `GROQ_API_KEY` — required Groq API key for `api/apis.py`
- `MODEL` — optional model override, defaults to `llama-3.3-70b-versatile`

Other configuration is currently driven by the code in `agent/`, `api/`, and `rag/`.

## Development

- `agent/loop.py` implements the agent loop and execution flow
- `agent/sandbox_client.py` handles Docker sandbox interaction
- `api/apis.py` wraps Groq chat completions and retry logic
- `app.py` defines the Streamlit app and UI experience

If you add new retrieval or ingestion behavior, keep the sandbox and agent loop contract in mind:

- agent responses must include `<THOUGHT>` and either `<CODE>` or `<FINAL_ANSWER>`
- code is executed in the sandbox and output is fed back into the agent
- final answers are returned only after the agent produces `<FINAL_ANSWER>`

## Troubleshooting

- If Docker image is missing:

```bash
python main.py --build
```

- If `GROQ_API_KEY` is not set, set it in `.env` or your shell environment.
- If Streamlit cannot start, verify the virtual environment is activated and `streamlit` is installed.

## Notes

- The agent is designed as a research/demo system rather than a production product.
- `docs/finagent.md` contains the project design and architecture notes used during development.

## License

This repository does not currently include a formal license file. Add one if you plan to publish or share the project.
