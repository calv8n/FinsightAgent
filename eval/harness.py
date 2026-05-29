"""
eval/harness.py — 20-question financial Q&A eval with LLM-as-judge scoring.

Metrics:
    answer_accuracy   — LLM-as-judge 0-3 score: wrong/partial/correct/perfect
    code_success_rate — % of runs where code executed without error
    retrieval_hit     — % of runs where RAG returned relevant chunks
    avg_iterations    — mean CodeAct iterations per question
    latency_p50/p95   — response time percentiles

Run:
    python eval/harness.py                   # run all 20 questions
    python eval/harness.py --quick           # run first 5 only
    python eval/harness.py --out results.json

Output: prints a report + writes JSON with per-question details.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.apis import llm_request
from agent.sandbox_client import SandboxClient
from rag import RAGPipeline

# ============================================================================
# BENCHMARK QUESTIONS
# 20 questions across 4 categories: arithmetic, financial modelling,
# multi-hop retrieval, and statistical analysis.
# ============================================================================

QUESTIONS = [
    # ── Category A: Pure arithmetic (no RAG needed) ───────────────────────
    {
        "id": "A01",
        "category": "arithmetic",
        "question": "Calculate the CAGR of revenues [100, 130, 160, 200] over 3 years.",
        "expected_contains": ["26", "0.26"],  # 26.0%
        "requires_rag": False,
    },
    {
        "id": "A02",
        "category": "arithmetic",
        "question": "What is the median of [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]?",
        "expected_contains": ["4"],
        "requires_rag": False,
    },
    {
        "id": "A03",
        "category": "arithmetic",
        "question": "An investment of $10,000 grows at 8% annually for 10 years. What is the final value?",
        "expected_contains": ["21589", "21,589"],
        "requires_rag": False,
    },
    {
        "id": "A04",
        "category": "arithmetic",
        "question": "Calculate the standard deviation of annual returns [12%, -8%, 23%, 5%, -3%, 18%].",
        "expected_contains": ["10", "11"],  # ~10.6%
        "requires_rag": False,
    },
    {
        "id": "A05",
        "category": "arithmetic",
        "question": "I have revenues [480, 532, 601, 678] and costs [288, 314, 349, 392]. Calculate gross margin for each year and the average gross margin.",
        "expected_contains": ["40", "41"],  # ~40%
        "requires_rag": False,
    },
    # ── Category B: Financial modelling ──────────────────────────────────
    {
        "id": "B01",
        "category": "financial_model",
        "question": "Build a DCF: starting FCF $50B, growing at 15% for 5 years, then 3% terminal growth, 10% discount rate. What is the total enterprise value?",
        "expected_contains": ["billion", "B", "value"],
        "requires_rag": False,
    },
    {
        "id": "B02",
        "category": "financial_model",
        "question": "Calculate the Sharpe ratio for a portfolio with annual return 18%, volatility 11%, and risk-free rate 4.5%.",
        "expected_contains": ["1.2", "1.3"],  # ~1.23
        "requires_rag": False,
    },
    {
        "id": "B03",
        "category": "financial_model",
        "question": "A company has EPS of $3.50 growing at 12% annually. Using a P/E of 20x, what is the projected stock price in 5 years?",
        "expected_contains": ["123", "124"],  # ~$123.6
        "requires_rag": False,
    },
    {
        "id": "B04",
        "category": "financial_model",
        "question": "Dollar-cost average $500/month for 12 months into a stock with prices [142,150,138,155,162,158,170,165,172,180,168,175]. What is the average cost basis and total return vs buying all at month 1?",
        "expected_contains": ["161", "162", "163"],  # avg cost ~$161
        "requires_rag": False,
    },
    {
        "id": "B05",
        "category": "financial_model",
        "question": "Calculate Value at Risk (95% confidence) for a $1M position with daily returns: [0.8,-1.2,0.5,-0.3,1.1,-2.1,0.9,-0.7,1.4,-0.2,0.6,-1.8,1.2,0.4,-0.9,1.7,-0.5,0.8,-1.3,0.6]. Use historical simulation.",
        "expected_contains": ["VaR", "var", "1,", "$"],
        "requires_rag": False,
    },
    # ── Category C: Statistical / numpy-heavy ─────────────────────────────
    {
        "id": "C01",
        "category": "statistical",
        "question": "Fit a linear regression to revenue data points: year=[1,2,3,4,5], revenue=[320,378,445,521,589]. What is the projected revenue for year 6 and year 7?",
        "expected_contains": ["6", "7", "year"],
        "requires_rag": False,
    },
    {
        "id": "C02",
        "category": "statistical",
        "question": "Calculate the correlation between AAPL monthly returns [2.1,-1.5,3.2,-0.8,1.9,2.5,-1.1,0.7,2.8,-0.5,1.3,2.0] and MSFT monthly returns [1.8,-1.2,2.9,-0.6,1.5,2.1,-0.9,0.5,2.4,-0.3,1.0,1.7].",
        "expected_contains": ["0.99", "0.98"],  # very high correlation
        "requires_rag": False,
    },
    {
        "id": "C03",
        "category": "statistical",
        "question": "Run a Monte Carlo simulation with 1000 paths over 12 months. Monthly return mean=1.2%, std=3.5%. Starting value $100,000. What is the median final value and the 5th percentile (worst case)?",
        "expected_contains": ["115", "116", "median"],
        "requires_rag": False,
    },
    {
        "id": "C04",
        "category": "statistical",
        "question": "Calculate rolling 3-month average and volatility for stock prices [142,150,138,155,162,158,170,165,172,180,168,175].",
        "expected_contains": ["rolling", "average", "volatil"],
        "requires_rag": False,
    },
    {
        "id": "C05",
        "category": "statistical",
        "question": "Two portfolios: A returns [8%,12%,-3%,15%,7%] and B returns [5%,18%,-8%,22%,3%]. Compare Sharpe ratios (rf=4%), max drawdown, and which has better risk-adjusted returns.",
        "expected_contains": ["Sharpe", "sharpe", "drawdown"],
        "requires_rag": False,
    },
    # ── Category D: Multi-hop pandas ──────────────────────────────────────
    {
        "id": "D01",
        "category": "multi_hop",
        "question": "Build a pandas DataFrame with companies A and B. A: revenue=[100,115,132,148], COGS=[60,68,77,84]. B: revenue=[80,100,128,165], COGS=[52,63,79,99]. Calculate gross margins and identify which company improved more.",
        "expected_contains": ["gross margin", "A", "B"],
        "requires_rag": False,
    },
    {
        "id": "D02",
        "category": "multi_hop",
        "question": "Apple quarterly revenues 2023 ($B): Q1=117.2, Q2=81.8, Q3=81.8, Q4=89.5. Calculate TTM at each quarter end, seasonality index per quarter, and whether revenue is accelerating or decelerating.",
        "expected_contains": ["TTM", "seasonalit", "370", "370.3"],
        "requires_rag": False,
    },
    {
        "id": "D03",
        "category": "multi_hop",
        "question": "Given R&D as % of revenue: Apple=[6.3,6.3,6.5,6.7,7.8] and Microsoft=[13.4,13.3,12.3,12.0,12.4] for 2019-2023. Fit a linear trend for both and project to 2027. At what year would Apple match Microsoft's R&D intensity?",
        "expected_contains": ["2", "never", "year"],  # likely never or far future
        "requires_rag": False,
    },
    {
        "id": "D04",
        "category": "multi_hop",
        "question": "Model an LBO: purchase price $5B, 60% debt at 8% interest, 40% equity. EBITDA starts at $800M and grows 12%/yr for 5 years. Assuming exit at same 6.25x EBITDA multiple, calculate equity IRR.",
        "expected_contains": ["IRR", "irr", "%"],
        "requires_rag": False,
    },
    {
        "id": "D05",
        "category": "multi_hop",
        "question": "Calculate a sensitivity table for a DCF model. Base FCF=$60B, WACC=[8.5%,9.5%,10.5%], terminal growth=[2%,3%,4%]. Show the 9-cell enterprise value table.",
        "expected_contains": ["8.5", "9.5", "10.5", "2%", "3%", "4%"],
        "requires_rag": False,
    },
]


# ============================================================================
# LLM-AS-JUDGE
# ============================================================================

JUDGE_PROMPT = """You are an expert financial analyst evaluating an AI agent's answer.

Question: {question}

Agent's answer: {answer}

Expected answer should contain at least one of: {expected}

Score the answer on a 0-3 scale:
0 = Wrong or no answer
1 = Partially correct (right approach, wrong numbers)
2 = Correct answer with minor issues (rounding, formatting)
3 = Perfect answer (correct numbers, clear explanation)

Respond with ONLY a JSON object:
{{"score": <0-3>, "reason": "<one sentence>"}}
"""


def judge_answer(question: str, answer: str, expected_contains: list[str]) -> dict:
    """Use LLM to score an answer 0-3."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=answer[:1000],
        expected=", ".join(expected_contains),
    )
    response = llm_request(
        system_prompt="You are a precise evaluator. Respond only with valid JSON.",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
    )
    if not response:
        return {"score": 0, "reason": "Judge call failed"}
    try:
        # Strip markdown fences if present
        clean = response.strip().strip("```json").strip("```").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        # Try to extract score from raw text
        import re

        m = re.search(r'"score"\s*:\s*([0-3])', response)
        score = int(m.group(1)) if m else 0
        return {"score": score, "reason": response[:100]}


# ============================================================================
# SINGLE-QUESTION RUNNER
# ============================================================================


@dataclass
class EvalResult:
    question_id: str
    category: str
    question: str
    final_answer: str
    judge_score: int  # 0-3
    judge_reason: str
    code_success: bool  # True if all code blocks executed without error
    iterations: int
    latency_s: float
    rag_retrieved: bool


def run_one(q: dict, sandbox: SandboxClient, rag: RAGPipeline) -> EvalResult:
    from agent.loop import SYSTEM_PROMPT, _extract, _build_observation

    history: list[dict] = []
    state: dict = {}
    t0 = time.monotonic()

    rag_retrieved = False
    rag_chunks: list[dict] = []

    if q["requires_rag"] and len(rag) > 0:
        rag_chunks = rag.retrieve(q["question"], top_k=5)
        rag_retrieved = bool(rag_chunks)

    system = SYSTEM_PROMPT
    if rag_chunks:
        ctx = ["## Retrieved Context\n"]
        for c in rag_chunks:
            ctx.append(f"{c['ticker']} {c['year']} {c['section']}: {c['text'][:400]}\n")
        system = "\n".join(ctx) + "\n---\n\n" + system

    history.append({"role": "user", "content": q["question"]})

    final_answer = ""
    code_success = True
    iteration = 0

    for iteration in range(1, 11):
        if time.monotonic() - t0 > 90:
            break

        response = llm_request(system_prompt=system, messages=history)
        if not response:
            break

        history.append({"role": "assistant", "content": response})

        thought = _extract("THOUGHT", response)
        code = _extract("CODE", response)
        final_answer = _extract("FINAL_ANSWER", response) or final_answer

        if _extract("FINAL_ANSWER", response):
            break

        if not code:
            nudge = "Your response did not contain a <CODE> block. Respond with <THOUGHT> and <CODE>."
            history.append({"role": "user", "content": nudge})
            continue

        result = sandbox.run(code, state)
        state = result.state
        if not result.ok:
            code_success = False

        obs = _build_observation(result)
        history.append({"role": "user", "content": obs})

    latency = time.monotonic() - t0
    judgment = judge_answer(q["question"], final_answer, q["expected_contains"])

    return EvalResult(
        question_id=q["id"],
        category=q["category"],
        question=q["question"],
        final_answer=final_answer,
        judge_score=judgment.get("score", 0),
        judge_reason=judgment.get("reason", ""),
        code_success=code_success,
        iterations=iteration,
        latency_s=round(latency, 2),
        rag_retrieved=rag_retrieved,
    )


# ============================================================================
# REPORT
# ============================================================================


def print_report(results: list[EvalResult]):
    print("\n" + "═" * 72)
    print("  FinSight Eval Report")
    print("═" * 72)

    scores = [r.judge_score for r in results]
    latencies = [r.latency_s for r in results]

    print(f"\n  Questions:          {len(results)}")
    print(f"  Avg judge score:    {np.mean(scores):.2f} / 3.0")
    print(f"  Perfect (score=3):  {sum(s==3 for s in scores)}/{len(scores)}")
    print(f"  Correct  (score≥2): {sum(s>=2 for s in scores)}/{len(scores)}")
    print(
        f"  Code success rate:  {sum(r.code_success for r in results)/len(results):.0%}"
    )
    print(f"  Avg iterations:     {np.mean([r.iterations for r in results]):.1f}")
    print(f"  Latency p50:        {np.percentile(latencies, 50):.1f}s")
    print(f"  Latency p95:        {np.percentile(latencies, 95):.1f}s")

    print(f"\n  {'ID':<6} {'Cat':<18} {'Score':>5} {'Iter':>4} {'Time':>6}  Reason")
    print(f"  {'─'*6} {'─'*18} {'─'*5} {'─'*4} {'─'*6}  {'─'*30}")
    for r in results:
        flag = "✅" if r.judge_score >= 2 else ("⚠️ " if r.judge_score == 1 else "❌")
        print(
            f"  {r.question_id:<6} {r.category:<18} {flag}{r.judge_score:>2}/3 "
            f"{r.iterations:>4} {r.latency_s:>5.1f}s  {r.judge_reason[:40]}"
        )

    # Per-category breakdown
    cats: dict[str, list[int]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.judge_score)
    print(f"\n  Category Breakdown:")
    for cat, cat_scores in cats.items():
        print(f"    {cat:<20} avg={np.mean(cat_scores):.2f}  " f"n={len(cat_scores)}")

    print("\n" + "═" * 72 + "\n")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true", help="Run first 5 questions only"
    )
    parser.add_argument("--out", default="eval/results.json", help="Output JSON path")
    parser.add_argument("--id", help="Run a single question by ID, e.g. B01")
    args = parser.parse_args()

    questions = QUESTIONS
    if args.id:
        questions = [q for q in QUESTIONS if q["id"] == args.id]
        if not questions:
            print(f"Question {args.id} not found.")
            sys.exit(1)
    elif args.quick:
        questions = QUESTIONS[:5]

    print(f"\n[eval] Running {len(questions)} question(s)...")

    sandbox = SandboxClient()
    rag = RAGPipeline()

    results: list[EvalResult] = []
    for i, q in enumerate(questions, 1):
        print(f"\n[eval] {i}/{len(questions)} — {q['id']}: {q['question'][:60]}...")
        r = run_one(q, sandbox, rag)
        results.append(r)
        print(f"  → score={r.judge_score}/3  iters={r.iterations}  {r.latency_s:.1f}s")

    print_report(results)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"[eval] Results written to {args.out}")


if __name__ == "__main__":
    main()
