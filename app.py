import streamlit as st
import yfinance as yf
import pandas as pd

# -----------------------------
# 1. STANDARDSCORE (Alex-KPI) FUNCTIONS
# -----------------------------
def score_roe_standard(roe):
    if roe is None: return 0
    if roe > 20: return 10
    if roe > 15: return 8
    if roe > 10: return 6
    if roe > 5: return 4
    return 2

def score_peg_standard(peg):
    if peg is None: return 0
    if peg < 1: return 10
    if peg < 1.5: return 8
    if peg < 2: return 6
    return 3

def score_operating_margin_standard(m):
    if m is None: return 0
    if m > 30: return 10
    if m > 20: return 8
    if m > 10: return 6
    return 3

def score_forward_pe_standard(fpe):
    if fpe is None: return 0
    if fpe < 12: return 10
    if fpe < 18: return 8
    if fpe < 25: return 6
    return 3

def score_pe_standard(pe):
    if pe is None: return 0
    if pe < 15: return 10
    if pe < 22: return 8
    if pe < 30: return 6
    return 3

def score_eps_growth_standard(g):
    if g is None: return 0
    if g > 20: return 10
    if g > 10: return 8
    if g > 5: return 6
    return 3

def score_revenue_growth_standard(g):
    if g is None: return 0
    if g > 15: return 10
    if g > 8: return 8
    if g > 3: return 6
    return 3

def score_profit_margin_standard(m):
    if m is None: return 0
    if m > 25: return 10
    if m > 15: return 8
    if m > 8: return 6
    return 3

def score_debt_to_equity_standard(d):
    if d is None: return 0
    if d < 0.3: return 10
    if d < 0.6: return 8
    if d < 1.0: return 6
    return 3

def score_cash_to_debt_standard(c):
    if c is None: return 0
    if c > 1.5: return 10
    if c > 1.0: return 8
    if c > 0.5: return 6
    return 3

def score_fcf_yield_standard(y):
    if y is None: return 0
    if y > 6: return 10
    if y > 4: return 8
    if y > 2: return 6
    return 3

# -----------------------------
# 2. DEFENSIVSCORE FUNCTIONS
# -----------------------------
def score_roe_defensive(roe):
    if roe is None: return 0
    if roe > 18: return 10
    if roe > 12: return 8
    if roe > 8: return 6
    if roe > 4: return 4
    return 1

def score_operating_margin_defensive(m):
    if m is None: return 0
    if m > 25: return 10
    if m > 18: return 8
    if m > 10: return 6
    return 2

def score_forward_pe_defensive(fpe):
    if fpe is None: return 0
    if fpe < 12: return 10
    if fpe < 16: return 8
    if fpe < 22: return 5
    return 1

def score_pe_defensive(pe):
    if pe is None: return 0
    if pe < 14: return 10
    if pe < 20: return 8
    if pe < 26: return 5
    return 1

def score_eps_growth_defensive(g):
    if g is None: return 0
    if g > 12: return 10
    if g > 7: return 8
    if g > 3: return 6
    return 2

def score_revenue_growth_defensive(g):
    if g is None: return 0
    if g > 10: return 10
    if g > 6: return 8
    if g > 2: return 6
    return 2

def score_profit_margin_defensive(m):
    if m is None: return 0
    if m > 20: return 10
    if m > 12: return 8
    if m > 6: return 6
    return 2

def score_debt_to_equity_defensive(d):
    if d is None: return 0
    if d < 0.25: return 10
    if d < 0.50: return 8
    if d < 0.80: return 5
    return 1

def score_cash_to_debt_defensive(c):
    if c is None: return 0
    if c > 2.0: return 10
    if c > 1.2: return 8
    if c > 0.6: return 6
    return 1

def score_fcf_yield_defensive(y):
    if y is None: return 0
    if y > 7: return 10
    if y > 5: return 8
    if y > 3: return 6
    return 1

# -----------------------------
# 3. OFFENSIV/AGGRESSIV FUNCTIONS
# -----------------------------
def score_revenue_growth_offensive(g):
    if g is None: return 0
    if g > 25: return 10
    if g > 15: return 8
    if g > 10: return 6
    return 2

def score_eps_growth_offensive(g):
    if g is None: return 0
    if g > 30: return 10
    if g > 18: return 8
    if g > 10: return 6
    return 2

def score_peg_offensive(peg):
    if peg is None: return 0
    if peg < 1.0: return 10
    if peg < 1.6: return 8
    if peg < 2.2: return 6
    return 3

def score_roe_offensive(roe):
    if roe is None: return 0
    if roe > 25: return 10
    if roe > 15: return 8
    if roe > 8: return 5
    return 1

def score_operating_margin_offensive(m):
    if m is None: return 0
    if m > 20: return 10
    if m > 12: return 8
    if m > 5: return 5
    return 1

def score_fcf_yield_offensive(y):
    if y is None: return 0
    if y > 5: return 10
    if y > 3: return 8
    if y > 1: return 5
    return 1

# -----------------------------
# GEWICHTUNGS-DICTIONARIES
# -----------------------------
STANDARD_WEIGHTS = {
    "roe": 0.15, "peg": 0.12, "operating_margin": 0.10, "forward_pe": 0.10,
    "pe": 0.08, "eps_growth": 0.10, "revenue_growth": 0.08, "profit_margin": 0.08,
    "debt_to_equity": 0.05, "cash_to_debt": 0.04, "fcf_yield": 0.10
}

DEFENSIVE_WEIGHTS = {
    "operating_margin": 0.15, "roe": 0.13, "forward_pe": 0.13, "debt_to_equity": 0.12,
    "fcf_yield": 0.12, "pe": 0.10, "cash_to_debt": 0.10, "eps_growth": 0.05,
    "revenue_growth": 0.05, "profit_margin": 0.05
}

OFFENSIVE_WEIGHTS = {
    "revenue_growth": 0.20, "eps_growth": 0.20, "peg": 0.15, "roe": 0.15,
    "operating_margin": 0.15, "fcf_yield": 0.15
}

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Multi-Score Kriterien Agent", layout="wide")

st.title("📊 Multi-Score KPI Analyse Agent")
st.write("Vergleiche Aktien über das gesamte Risikospektrum: **Defensiv** (Stabilität), **Alex-KPI** (Ausgewogen) und **Offensiv** (Wachstum).")

# Standardwert hinzugefügt, um leere Seiten beim Start zu verhindern
tickers_input = st.text_input("Gib mehrere Ticker ein (getrennt durch Komma):", value="MSFT, JNJ, NVDA")

if tickers_input:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    results = []

    if tickers:
        with st.spinner("Rufe Echtzeit-Marktdaten ab..."):
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    if not info or "symbol" not in info:
                        continue

                    # Rohdaten ziehen
                    pe = info.get("trailingPE")
                    fpe = info.get("forwardPE")
                    peg = info.get("pegRatio")
                    profit_margin = info.get("profitMargins")
                    operating_margin = info.get("operatingMargins")
                    roe = info.get("returnOnEquity")
                    revenue_growth = info.get("revenueGrowth")
                    eps_growth = info.get("earningsQuarterlyGrowth")
                    debt_to_equity = info.get("debtToEquity")
                    
                    total_cash = info.get("totalCash", 0)
                    total_debt = info.get("totalDebt", 1)
                    cash_to_debt = total_cash / total_debt if total_debt and total_debt > 0 else 1.5

                    fcf = info.get("freeCashflow")
                    market_cap = info.get("marketCap")
                    fcf_yield = (fcf / market_cap) * 100 if fcf and market_cap else None

                    # Skalierung
                    pm_scaled = profit_margin * 100 if profit_margin else None
                    om_scaled = operating_margin * 100 if operating_margin else None
                    roe_scaled = roe * 100 if roe else None
                    rev_scaled = revenue_growth * 100 if revenue_growth else None
                    eps_scaled = eps_growth * 100 if eps_growth else None

                    # 1. Standard (Alex-KPI)
                    score_standard = (
                        score_roe_standard(roe_scaled) * STANDARD_WEIGHTS["roe"] +
                        score_peg_standard(peg) * STANDARD_WEIGHTS["peg"] +
                        score_operating_margin_standard(om_scaled) * STANDARD_WEIGHTS["operating_margin"] +
                        score_forward_pe_standard(fpe) * STANDARD_WEIGHTS["forward_pe"] +
                        score_pe_standard(pe) * STANDARD_WEIGHTS["pe"] +
                        score_eps_growth_standard(eps_scaled) * STANDARD_WEIGHTS["eps_growth"] +
                        score_revenue_growth_standard(rev_scaled) * STANDARD_WEIGHTS["revenue_growth"] +
                        score_profit_margin_standard(pm_scaled) * STANDARD_WEIGHTS["profit_margin"] +
                        score_debt_to_equity_standard(debt_to_equity) * STANDARD_WEIGHTS["debt_to_equity"] +
                        score_cash_to_debt_standard(cash_to_debt) * STANDARD_WEIGHTS["cash_to_debt"] +
                        score_fcf_yield_standard(fcf_yield) * STANDARD_WEIGHTS["fcf_yield"]
                    )

                    # 2. Defensiv
                    score_defensive = (
                        score_operating_margin_defensive(om_scaled) * DEFENSIVE_WEIGHTS["operating_margin"] +
                        score_roe_defensive(roe_scaled) * DEFENSIVE_WEIGHTS["roe"] +
                        score_forward_pe_defensive(fpe) * DEFENSIVE_WEIGHTS["forward_pe"] +
                        score_debt_to_equity_defensive(debt_to_equity) * DEFENSIVE_WEIGHTS["debt_to_equity"] +
                        score_fcf_yield_defensive(fcf_yield) * DEFENSIVE_WEIGHTS["fcf_yield"] +
                        score_pe_defensive(pe) * DEFENSIVE_WEIGHTS["pe"] +
                        score_cash_to_debt_defensive(cash_to_debt) * DEFENSIVE_WEIGHTS["cash_to_debt"] +
                        score_eps_growth_defensive(eps_scaled) * DEFENSIVE_WEIGHTS["eps_growth"] +
                        score_revenue_growth_defensive(rev_scaled) * DEFENSIVE_WEIGHTS["revenue_growth"] +
                        score_profit_margin_defensive(pm_scaled) * DEFENSIVE_WEIGHTS["profit_margin"]
                    )

                    # 3. Offensiv / Aggressiv
                    score_offensive = (
                        score_revenue_growth_offensive(rev_scaled) * OFFENSIVE_WEIGHTS["revenue_growth"] +
                        score_eps_growth_offensive(eps_scaled) * OFFENSIVE_WEIGHTS["eps_growth"] +
                        score_peg_offensive(peg) * OFFENSIVE_WEIGHTS["peg"] +
                        score_roe_offensive(roe_scaled) * OFFENSIVE_WEIGHTS["roe"] +
                        score_operating_margin_offensive(om_scaled) * OFFENSIVE_WEIGHTS["operating_margin"] +
                        score_fcf_yield_offensive(fcf_yield) * OFFENSIVE_WEIGHTS["fcf_yield"]
                    )

                    results.append({
                        "Ticker": ticker,
                        "Defensiv-Score": round(score_defensive * 10, 2),
                        "Alex-KPI Score": round(score_standard * 10, 2),
                        "Offensiv-Score": round(score_offensive * 10, 2),
                        "P/E (KGV)": round(pe, 2) if pe else None,
                        "Forward P/E": round(fpe, 2) if fpe else None,
                        "PEG Ratio": round(peg, 2) if peg else None,
                        "Revenue Growth (%)": round(rev_scaled, 2) if rev_scaled else None,
                        "EPS Growth (%)": round(eps_scaled, 2) if eps_scaled else None,
                        "Operating Margin (%)": round(om_scaled, 2) if om_scaled else None,
                        "FCF Yield (%)": round(fcf_yield, 2) if fcf_yield else None
                    })
                except Exception as e:
                    st.error(f"Fehler bei {ticker}: {str(e)}")

        if results:
            df = pd.DataFrame(results)
            st.subheader("📊 Ergebnis-Matrix")
            df = df.sort_values(by="Alex-KPI Score", ascending=False)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Es konnten keine Daten für die angegebenen Ticker geladen werden.")
