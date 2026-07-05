import streamlit as st
import yfinance as yf
import pandas as pd

# -----------------------------
# KPI Scoring Functions
# -----------------------------

def score_roic(roic):
    if roic is None: return 0
    if roic > 20: return 10
    if roic > 15: return 8
    if roic > 10: return 6
    if roic > 5: return 4
    return 2

def score_peg(peg):
    if peg is None: return 0
    if peg < 1: return 10
    if peg < 1.5: return 8
    if peg < 2: return 6
    return 3

def score_operating_margin(m):
    if m is None: return 0
    if m > 30: return 10
    if m > 20: return 8
    if m > 10: return 6
    return 3

def score_forward_pe(fpe):
    if fpe is None: return 0
    if fpe < 12: return 10
    if fpe < 18: return 8
    if fpe < 25: return 6
    return 3

def score_pe(pe):
    if pe is None: return 0
    if pe < 15: return 10
    if pe < 22: return 8
    if pe < 30: return 6
    return 3

def score_eps_growth(g):
    if g is None: return 0
    if g > 20: return 10
    if g > 10: return 8
    if g > 5: return 6
    return 3

def score_revenue_growth(g):
    if g is None: return 0
    if g > 15: return 10
    if g > 8: return 8
    if g > 3: return 6
    return 3

def score_profit_margin(m):
    if m is None: return 0
    if m > 25: return 10
    if m > 15: return 8
    if m > 8: return 6
    return 3

def score_debt_to_equity(d):
    if d is None: return 0
    if d < 0.3: return 10
    if d < 0.6: return 8
    if d < 1.0: return 6
    return 3

def score_cash_to_debt(c):
    if c is None: return 0
    if c > 1.5: return 10
    if c > 1.0: return 8
    if c > 0.5: return 6
    return 3

def score_fcf_yield(y):
    if y is None: return 0
    if y > 6: return 10
    if y > 4: return 8
    if y > 2: return 6
    return 3

# Gewichtung
WEIGHTS = {
    "roic": 0.15,
    "peg": 0.12,
    "operating_margin": 0.10,
    "forward_pe": 0.10,
    "pe": 0.08,
    "eps_growth": 0.10,
    "revenue_growth": 0.08,
    "profit_margin": 0.08,
    "debt_to_equity": 0.05,
    "cash_to_debt": 0.04,
    "fcf_yield": 0.10,
}

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("📈 Aktienanalyse Agent – Multi‑Ticker KPI & Score")

tickers_input = st.text_input("Gib mehrere Ticker ein (z.B. MSFT, NVDA, GOOG):")

if tickers_input:
    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    results = []

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info

        pe = info.get("trailingPE")
        fpe = info.get("forwardPE")
        peg = info.get("pegRatio")
        profit_margin = info.get("profitMargins")
        operating_margin = info.get("operatingMargins")
        roe = info.get("returnOnEquity")
        revenue_growth = info.get("revenueGrowth")
        eps_growth = info.get("earningsQuarterlyGrowth")
        debt_to_equity = info.get("debtToEquity")
        cash_to_debt = info.get("totalCash", 0) / info.get("totalDebt", 1)

        if profit_margin: profit_margin *= 100
        if operating_margin: operating_margin *= 100
        if roe: roe *= 100
        if revenue_growth: revenue_growth *= 100
        if eps_growth: eps_growth *= 100

        score = (
            score_roic(roe) * WEIGHTS["roic"] +
            score_peg(peg) * WEIGHTS["peg"] +
            score_operating_margin(operating_margin) * WEIGHTS["operating_margin"] +
            score_forward_pe(fpe) * WEIGHTS["forward_pe"] +
            score_pe(pe) * WEIGHTS["pe"] +
            score_eps_growth(eps_growth) * WEIGHTS["eps_growth"] +
            score_revenue_growth(revenue_growth) * WEIGHTS["revenue_growth"] +
            score_profit_margin(profit_margin) * WEIGHTS["profit_margin"] +
            score_debt_to_equity(debt_to_equity) * WEIGHTS["debt_to_equity"] +
            score_cash_to_debt(cash_to_debt) * WEIGHTS["cash_to_debt"]
        )

        results.append({
            "Ticker": ticker,
            "Score": round(score * 10, 2),
            "P/E": pe,
            "Forward P/E": fpe,
            "PEG": peg,
            "Profit Margin": profit_margin,
            "Operating Margin": operating_margin,
            "ROE": roe,
            "EPS Growth": eps_growth,
            "Revenue Growth": revenue_growth,
            "Debt/Equity": debt_to_equity,
            "Cash/Debt": cash_to_debt
        })

    df = pd.DataFrame(results)
    st.subheader("📊 Ergebnisse aller Aktien")
    st.dataframe(df)
