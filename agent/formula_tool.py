"""
agent/formula_tool.py — Financial formula reference tool.

The agent calls get_formula(topic) to fetch the correct formula,
its derivation, common mistakes, and a code template.
This is injected into the system prompt when a formula is needed.
"""

from __future__ import annotations

# ── Formula database ─────────────────────────────────────────────────────────
# Each entry has: formula (LaTeX-style), explanation, common_mistakes, code

FORMULAS: dict[str, dict] = {
    "cagr": {
        "name": "Compound Annual Growth Rate (CAGR)",
        "formula": "CAGR = (end_value / start_value)^(1/n) - 1",
        "where": "n = number of PERIODS (= len(values) - 1, NOT len(values))",
        "common_mistakes": [
            "Using len(values) instead of len(values)-1 for n",
            "Forgetting to subtract 1 at the end",
        ],
        "code": """
values = [100, 130, 160, 200]
n    = len(values) - 1          # number of periods, NOT number of data points
cagr = (values[-1] / values[0]) ** (1 / n) - 1
print(f"CAGR: {cagr:.2%}")
""",
    },
    "irr": {
        "name": "Internal Rate of Return (IRR)",
        "formula": "NPV = Σ CF_t / (1+IRR)^t = 0",
        "where": "CF_t = cash flow at time t, t starts at 0",
        "cash_flow_structure": {
            "lbo": "[-equity, 0, 0, ..., 0, exit_equity]  — zeros for hold period",
            "dcf": "[-investment, FCF_1, FCF_2, ..., FCF_n + terminal_value]",
            "bond": "[-price, coupon, coupon, ..., coupon + face_value]",
        },
        "common_mistakes": [
            "Mixing FCF and equity cash flows in an LBO — they are separate waterfalls",
            "Not having a sign change in cash flows (IRR requires at least one + and one -)",
            "Using infinite loop in bisection — always cap at max_iter",
            "Setting bounds too narrow — use lo=-0.999, hi=10.0",
        ],
        "code": """
def irr_bisect(cash_flows, lo=-0.999, hi=10.0, tol=1e-6, max_iter=10000):
    \"\"\"Bisection IRR. cash_flows[0] must be negative (outflow).\"\"\"
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        npv = sum(cf / (1 + mid)**t for t, cf in enumerate(cash_flows))
        if abs(npv) < tol:
            return mid
        if npv < 0:
            hi = mid
        else:
            lo = mid
    return mid  # best estimate

# LBO example:
cash_flows = [-2_000e6, 0, 0, 0, 0, 5_809e6]  # entry equity, hold, exit equity
print(f"IRR: {irr_bisect(cash_flows):.1%}")
""",
    },
    "lbo": {
        "name": "Leveraged Buyout (LBO) Model",
        "formula": "Entry EV = EBITDA × entry_multiple; Debt = EV × debt_%; Equity = EV × equity_%",
        "mechanics": [
            "1. Entry: compute debt and equity from EV",
            "2. Each year: grow EBITDA and revenue; compute FCF",
            "3. Debt service: interest = opening_debt × rate",
            "4. Principal = max(0, min(FCF - interest, remaining_debt))",
            "5. Exit: exit_EV = exit_EBITDA × exit_multiple",
            "6. Exit equity = exit_EV - remaining_debt",
            "7. IRR: cash_flows = [-initial_equity, 0×hold_years, exit_equity]",
        ],
        "common_mistakes": [
            "Allowing negative principal: must use max(0, FCF - interest)",
            "Overpaying debt: must use min(principal, remaining_debt)",
            "Including FCF in equity IRR cash flows — wrong waterfall",
            "Using closing debt instead of opening debt for interest calc",
        ],
        "code": """
# === LBO INPUTS ===
ev             = 5_000e6
entry_multiple = 6
initial_ebitda = ev / entry_multiple        # implied from EV/EBITDA
debt_pct       = 0.60
equity_pct     = 0.40
rate           = 0.08
growth         = 0.12
years          = 5
revenue_0      = 4_000e6
fcf_pct        = 0.01
exit_multiple  = 6

initial_debt   = ev * debt_pct
initial_equity = ev * equity_pct

# === DEBT PAYDOWN SCHEDULE ===
remaining_debt = initial_debt
for yr in range(1, years + 1):
    revenue        = revenue_0 * (1 + growth) ** yr
    fcf            = revenue * fcf_pct
    interest       = remaining_debt * rate              # on OPENING balance
    principal_paid = max(0, min(fcf - interest, remaining_debt))  # floor 0, cap at debt
    remaining_debt -= principal_paid
    print(f"Yr{yr}: FCF=${fcf/1e6:.1f}M Int=${interest/1e6:.1f}M "
          f"Prin=${principal_paid/1e6:.1f}M Debt=${remaining_debt/1e9:.2f}B")

# === EXIT ===
exit_ebitda = initial_ebitda * (1 + growth) ** years
exit_ev     = exit_ebitda * exit_multiple
exit_equity = exit_ev - remaining_debt
mom         = exit_equity / initial_equity
print(f"\\nExit EV=${exit_ev/1e9:.2f}B  Exit Equity=${exit_equity/1e9:.2f}B  MoM={mom:.2f}x")

# === IRR (equity cash flows ONLY) ===
cfs = [-initial_equity] + [0] * years + [exit_equity]
def irr_bisect(cfs, lo=-0.999, hi=10.0, tol=1e-6, max_iter=10000):
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        npv = sum(cf/(1+mid)**t for t, cf in enumerate(cfs))
        if abs(npv) < tol: return mid
        if npv < 0: hi = mid
        else:       lo = mid
    return mid
print(f"IRR: {irr_bisect(cfs):.1%}")
""",
    },
    "dcf": {
        "name": "Discounted Cash Flow (DCF)",
        "formula": "EV = Σ FCF_t/(1+WACC)^t  +  TV/(1+WACC)^n",
        "terminal_value": "TV = FCF_n × (1+g) / (WACC - g)   [Gordon Growth Model]",
        "where": "g = terminal growth rate, must be < WACC",
        "common_mistakes": [
            "Using FCF_n directly in TV instead of FCF_n × (1+g)",
            "Not discounting the terminal value back to present",
            "Using g >= WACC (causes division by zero or negative TV)",
            "Discounting year 0 FCF (it's already in present value)",
        ],
        "code": """
fcf_0   = 50e9       # base FCF
growth  = [0.18, 0.15, 0.12, 0.10, 0.08]   # year-by-year growth
wacc    = 0.095
g_term  = 0.03       # terminal growth (must be < wacc)
years   = len(growth)

# Project FCFs
fcfs = []
fcf = fcf_0
for g in growth:
    fcf = fcf * (1 + g)
    fcfs.append(fcf)

# PV of each FCF
pv_fcfs = [fcf / (1 + wacc)**t for t, fcf in enumerate(fcfs, start=1)]

# Terminal value (Gordon Growth on final year FCF)
tv    = fcfs[-1] * (1 + g_term) / (wacc - g_term)
pv_tv = tv / (1 + wacc)**years

ev = sum(pv_fcfs) + pv_tv
print(f"PV of FCFs: ${sum(pv_fcfs)/1e9:.2f}B")
print(f"Terminal Value: ${tv/1e9:.2f}B  →  PV: ${pv_tv/1e9:.2f}B")
print(f"Enterprise Value: ${ev/1e9:.2f}B")
""",
    },
    "sharpe": {
        "name": "Sharpe Ratio",
        "formula": "Sharpe = (R_p - R_f) / σ_p",
        "where": "R_p = portfolio return, R_f = risk-free rate, σ_p = std dev of returns",
        "annualisation": "Monthly returns: annualised_std = monthly_std × √12",
        "common_mistakes": [
            "Using total return instead of excess return (must subtract R_f)",
            "Forgetting to annualise if using monthly/daily returns",
            "Using population std instead of sample std (use ddof=1 with numpy)",
        ],
        "code": """
import numpy as np
returns = [0.12, -0.08, 0.23, 0.05, -0.03, 0.18]   # annual returns as decimals
rf      = 0.045   # risk-free rate

excess_returns = [r - rf for r in returns]
sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
print(f"Sharpe Ratio: {sharpe:.2f}")
""",
    },
    "var": {
        "name": "Value at Risk (VaR)",
        "formula": {
            "historical": "VaR = -percentile(returns, 1-confidence) × portfolio_value",
            "parametric": "VaR = -(μ - z × σ) × portfolio_value",
        },
        "where": "z = 1.645 for 95%, 2.326 for 99%; μ = mean return; σ = std dev",
        "common_mistakes": [
            "Using percentile(5) instead of percentile(1-confidence) × value",
            "Not negating — VaR is a positive number representing a loss",
        ],
        "code": """
import numpy as np
returns    = [0.8,-1.2,0.5,-0.3,1.1,-2.1,0.9,-0.7,1.4,-0.2,
              0.6,-1.8,1.2,0.4,-0.9,1.7,-0.5,0.8,-1.3,0.6]
returns    = [r/100 for r in returns]   # convert % to decimal
portfolio  = 1_000_000
confidence = 0.95

# Historical simulation
hist_var = -np.percentile(returns, (1-confidence)*100) * portfolio
print(f"Historical VaR (95%): ${hist_var:,.0f}")

# Parametric (normal distribution)
from scipy import stats
mu, sigma  = np.mean(returns), np.std(returns, ddof=1)
z          = stats.norm.ppf(1-confidence)
param_var  = -(mu + z * sigma) * portfolio
print(f"Parametric VaR (95%): ${param_var:,.0f}")

# Expected Shortfall (CVaR) — average loss beyond VaR
sorted_r   = sorted(returns)
cutoff_idx = int(len(sorted_r) * (1-confidence))
cvar       = -np.mean(sorted_r[:cutoff_idx+1]) * portfolio
print(f"CVaR (Expected Shortfall): ${cvar:,.0f}")
""",
    },
    "wacc": {
        "name": "Weighted Average Cost of Capital (WACC)",
        "formula": "WACC = (E/V) × Re + (D/V) × Rd × (1 - Tc)",
        "where": "E=equity, D=debt, V=E+D, Re=cost of equity, Rd=cost of debt, Tc=tax rate",
        "capm": "Re = Rf + β × (Rm - Rf)   [CAPM for cost of equity]",
        "code": """
equity    = 6_000e6
debt      = 4_000e6
v         = equity + debt
re        = 0.12    # cost of equity (from CAPM or analyst estimate)
rd        = 0.06    # cost of debt (yield on bonds)
tax_rate  = 0.21

wacc = (equity/v)*re + (debt/v)*rd*(1-tax_rate)
print(f"WACC: {wacc:.2%}")
""",
    },
    "gross_margin": {
        "name": "Gross / Operating / Net Margin",
        "formulas": {
            "gross": "(Revenue - COGS) / Revenue",
            "operating": "EBIT / Revenue",
            "net": "Net Income / Revenue",
            "ebitda": "EBITDA / Revenue",
        },
        "common_mistakes": [
            "Using absolute dollar amounts instead of ratios",
            "Confusing EBIT and EBITDA (EBITDA adds back D&A to EBIT)",
        ],
        "code": """
import pandas as pd
data = {
    "Revenue": [480, 532, 601, 678],
    "COGS":    [288, 314, 349, 392],
    "OpEx":    [ 96, 108, 122, 138],
}
df = pd.DataFrame(data, index=[2020,2021,2022,2023])
df["Gross Profit"]  = df["Revenue"] - df["COGS"]
df["EBIT"]          = df["Gross Profit"] - df["OpEx"]
df["Gross Margin"]  = df["Gross Profit"] / df["Revenue"]
df["Op Margin"]     = df["EBIT"] / df["Revenue"]
print(df[["Gross Margin","Op Margin"]].applymap(lambda x: f"{x:.1%}"))
""",
    },
    "monte_carlo": {
        "name": "Monte Carlo Portfolio Simulation",
        "formula": "S_t = S_0 × exp((μ - σ²/2)×t + σ×ε×√t)   [GBM]",
        "where": "ε ~ N(0,1), μ = expected return, σ = volatility",
        "common_mistakes": [
            "Using normal returns instead of log-normal (prices can't go negative)",
            "Not fixing random seed for reproducibility in reports",
            "Forgetting to annualise σ if using monthly steps",
        ],
        "code": """
import numpy as np
np.random.seed(42)

S0        = 100_000    # starting value
mu        = 0.012      # monthly mean return
sigma     = 0.035      # monthly std dev
months    = 12
n_paths   = 1000

# Simulate n_paths × months returns
returns   = np.random.normal(mu, sigma, (n_paths, months))
paths     = S0 * np.cumprod(1 + returns, axis=1)
final     = paths[:, -1]

print(f"Median outcome:      ${np.median(final):,.0f}")
print(f"5th percentile:      ${np.percentile(final, 5):,.0f}")
print(f"95th percentile:     ${np.percentile(final, 95):,.0f}")
print(f"Prob > starting val: {(final > S0).mean():.1%}")
""",
    },
}


# ── Public API ────────────────────────────────────────────────────────────────

_KEYWORDS: dict[str, str] = {
    "cagr": "cagr",
    "compound annual": "cagr",
    "irr": "irr",
    "internal rate": "irr",
    "lbo": "lbo",
    "leveraged buyout": "lbo",
    "buyout": "lbo",
    "dcf": "dcf",
    "discounted cash flow": "dcf",
    "enterprise value": "dcf",
    "sharpe": "sharpe",
    "risk-adjusted": "sharpe",
    "var": "var",
    "value at risk": "var",
    "cvar": "var",
    "wacc": "wacc",
    "cost of capital": "wacc",
    "gross margin": "gross_margin",
    "operating margin": "gross_margin",
    "net margin": "gross_margin",
    "ebitda margin": "gross_margin",
    "monte carlo": "monte_carlo",
    "simulation": "monte_carlo",
}


def get_formula(topic: str) -> str:
    """
    Return the correct formula, explanation, common mistakes, and code
    template for a given financial topic.

    Args:
        topic: e.g. "irr", "lbo", "dcf", "cagr", "sharpe", "var"

    Returns:
        Formatted string ready to inject into the system prompt.
    """
    key = topic.lower().strip()

    # Direct match
    if key not in FORMULAS:
        # Keyword match
        for kw, formula_key in _KEYWORDS.items():
            if kw in key:
                key = formula_key
                break

    if key not in FORMULAS:
        return f"[Formula tool] No formula found for '{topic}'. Available: {list(FORMULAS.keys())}"

    f = FORMULAS[key]
    lines = [
        f"## ✅ Correct Formula: {f['name']}",
        "",
    ]

    if isinstance(f.get("formula"), str):
        lines += [f"**Formula:** `{f['formula']}`", ""]
    elif isinstance(f.get("formula"), dict):
        lines.append("**Formulas:**")
        for label, formula in f["formula"].items():
            lines.append(f"  - {label}: `{formula}`")
        lines.append("")

    for key_name in ("where", "terminal_value", "capm", "annualisation"):
        if key_name in f:
            lines += [f"**{key_name.replace('_',' ').title()}:** {f[key_name]}", ""]

    if "mechanics" in f:
        lines.append("**Mechanics (in order):**")
        for step in f["mechanics"]:
            lines.append(f"  {step}")
        lines.append("")

    if "cash_flow_structure" in f:
        lines.append("**Cash flow structures:**")
        for ctx, structure in f["cash_flow_structure"].items():
            lines.append(f"  - {ctx}: `{structure}`")
        lines.append("")

    if "common_mistakes" in f:
        lines.append("**⚠ Common mistakes to avoid:**")
        for m in f["common_mistakes"]:
            lines.append(f"  ❌ {m}")
        lines.append("")

    if "code" in f:
        lines += ["**Correct code template:**", "```python", f["code"].strip(), "```"]

    return "\n".join(lines)


def detect_formulas_needed(query: str) -> list[str]:
    """Detect which formulas are needed based on the query."""
    q = query.lower()
    found = []
    for kw, formula_key in _KEYWORDS.items():
        if kw in q and formula_key not in found:
            found.append(formula_key)
    return found


def get_all_formulas_for_query(query: str) -> str:
    """Return all relevant formula blocks for a query."""
    keys = detect_formulas_needed(query)
    if not keys:
        return ""
    blocks = [get_formula(k) for k in keys]
    return "\n\n---\n\n".join(blocks)
