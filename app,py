import streamlit as st
import yfinance as yf
import pandas as pd

# -----------------------------
# KPI Scoring Functions
# -----------------------------

def score_roic(roic):
    if roic > 20: return 10
    if roic > 15: return 8
    if roic > 10: return 6
    if roic > 5: return 4
    return 2

def score_peg(peg):
    if peg < 1: return 10
    if peg < 1.5: return 8
    if peg < 2: return 6
    return 3

def score_operating_margin(m):
    if m > 30: return 10
    if m > 20: return 8
    if m > 10: return 6
    return 3

def score_forward_pe(fpe):
    if fpe < 12: return 10
    if fpe < 18: return 8
    if fpe < 25: return 6
    return 3

def score_pe(pe):
    if pe < 15: return 10
    if pe < 22: return 8
    if pe < 30: return 6
    return 3

def score_eps_growth(g):
    if g > 20: return 10
    if g > 10: return 8
    if g > 5: return 6
    return 3

def score_revenue_growth(g):
    if g > 15: return 10
    if g > 8: return 8
    if g > 3: return 6
    return 3

def score_profit_margin(m):
    if m > 25: return 10
    if m > 15: return 8
    if m > 8: return 6
    return 3

def score_debt_to_equity(d):
    if d < 0.3: return 10
    if d < 0.6: return 8
    if d < 1.0: return 6
    return 3

def score_cash_to_debt(c):
    if c > 1.5: return 10
    if c > 1.0: return 8
    if c > 0.5: return 6
    return 3

def score_fcf_yield(y):
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

st.title("📈 Aktienanalyse Agent – KPI & Score")

ticker = st.text_input("Gib einen Ticker ein (z.B. MSFT, AAPL, TSLA):")

if ticker:
    stock = yf.Ticker(ticker)
    info = stock.info

    # KPIs aus Yahoo Finance
    pe = info.get("trailingPE", None)
    fpe = info.get("forwardPE", None)
    peg = info.get("pegRatio", None)
    profit_margin = info.get("profitMargins", None) * 100 if info.get("profitMargins") else None
    operating_margin = info.get("operatingMargins", None) * 100 if info.get("operatingMargins") else None
    roe = info.get("returnOnEquity", None) * 100 if info.get("returnOnEquity") else None
    revenue_growth = info.get("revenueGrowth", None) * 100 if info.get("revenueGrowth") else None
    eps_growth = info.get("earningsQuarterlyGrowth", None) * 100 if info.get("earningsQuarterlyGrowth") else None
    debt_to_equity = info.get("debtToEquity", None)
    cash_to_debt = info.get("totalCash", 0) / info.get("totalDebt", 1)
    fcf_yield = None  # Yahoo Finance liefert FCF nicht direkt

    # Score berechnen
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

    st.subheader("📊 Ergebnis")
    st.metric("Gesamt‑Score", f"{round(score * 10, 2)} / 100")

    st.subheader("🔍 Einzel‑KPIs")
    df = pd.DataFrame({
        "KPI": ["P/E", "Forward P/E", "PEG", "Profit Margin", "Operating Margin", "ROE", "EPS Growth", "Revenue Growth", "Debt/Equity", "Cash/Debt"],
        "Wert": [pe, fpe, peg, profit_margin, operating_margin, roe, eps_growth, revenue_growth, debt_to_equity, cash_to_debt]
    })

    st.dataframe(df)
