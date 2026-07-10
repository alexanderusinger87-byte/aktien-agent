import streamlit as st
import yfinance as yf
import pandas as pd

# -----------------------------
# KPI Scoring Functions (Punkte von 0 bis 10)
# -----------------------------

def score_roe(roe):
    if roe is None: return 0
    if roe > 20: return 10
    if roe > 15: return 8
    if roe > 10: return 6
    if roe > 5: return 4
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

# Mathematisch saubere Gewichtung (Summe = 1.0)
WEIGHTS = {
    "roe": 0.15,
    "peg": 0.12,
    "operating_margin": 0.10,
    "forward_pe": 0.10,
    "pe": 0.08,
    "eps_growth": 0.10,
    "revenue_growth": 0.08,
    "profit_margin": 0.08,
    "debt_to_equity": 0.05,
    "cash_to_debt": 0.04,
    "fcf_yield": 0.10,  # Integriert mit 10%
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Aktienanalyse Agent", layout="wide")

st.title("📈 Aktienanalyse Agent – Multi‑Ticker KPI & Score")
st.write("Dieses Tool berechnet einen Qualitätsscore basierend auf fundamentalen Kennzahlen von Yahoo Finance.")

tickers_input = st.text_input("Gib mehrere Ticker ein (getrennt durch Komma, z.B. MSFT, NVDA, GOOG):")

if tickers_input:
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    results = []

    with st.spinner("Lade Daten von Yahoo Finance..."):
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                if not info or "symbol" not in info:
                    st.warning(f"Keine Daten für Ticker gefunden: {ticker}")
                    continue

                # Rohdaten auslesen
                pe = info.get("trailingPE")
                fpe = info.get("forwardPE")
                peg = info.get("pegRatio")
                profit_margin = info.get("profitMargins")
                operating_margin = info.get("operatingMargins")
                roe = info.get("returnOnEquity")
                revenue_growth = info.get("revenueGrowth")
                eps_growth = info.get("earningsQuarterlyGrowth")
                debt_to_equity = info.get("debtToEquity")
                
                # Cash/Debt sicher berechnen
                total_cash = info.get("totalCash", 0)
                total_debt = info.get("totalDebt", 1)
                cash_to_debt = total_cash / total_debt if total_debt and total_debt > 0 else 1.5

                # Free Cashflow Yield berechnen (Proxy via Free Cashflow / Market Cap)
                fcf = info.get("freeCashflow")
                market_cap = info.get("marketCap")
                fcf_yield = (fcf / market_cap) * 100 if fcf and market_cap else None

                # Skalierung in Prozentwerte für die Scoring-Funktionen
                if profit_margin: profit_margin *= 100
                if operating_margin: operating_margin *= 100
                if roe: roe *= 100
                if revenue_growth: revenue_growth *= 100
                if eps_growth: eps_growth *= 100

                # Score-Berechnung (Inklusive FCF Yield)
                score = (
                    score_roe(roe) * WEIGHTS["roe"] +
                    score_peg(peg) * WEIGHTS["peg"] +
                    score_operating_margin(operating_margin) * WEIGHTS["operating_margin"] +
                    score_forward_pe(fpe) * WEIGHTS["forward_pe"] +
                    score_pe(pe) * WEIGHTS["pe"] +
                    score_eps_growth(eps_growth) * WEIGHTS["eps_growth"] +
                    score_revenue_growth(revenue_growth) * WEIGHTS["revenue_growth"] +
                    score_profit_margin(profit_margin) * WEIGHTS["profit_margin"] +
                    score_debt_to_equity(debt_to_equity) * WEIGHTS["debt_to_equity"] +
                    score_cash_to_debt(cash_to_debt) * WEIGHTS["cash_to_debt"] +
                    score_fcf_yield(fcf_yield) * WEIGHTS["fcf_yield"]
                )

                # Ergebnisse für die Tabelle aufbereiten
                results.append({
                    "Ticker": ticker,
                    "Alex-KPI Score": round(score * 10, 2),  # Skala bis max 100
                    "ROE (%)": round(roe, 2) if roe else None,
                    "P/E (KGV)": round(pe, 2) if pe else None,
                    "Forward P/E": round(fpe, 2) if fpe else None,
                    "PEG Ratio": round(peg, 2) if peg else None,
                    "Operating Margin (%)": round(operating_margin, 2) if operating_margin else None,
                    "Profit Margin (%)": round(profit_margin, 2) if profit_margin else None,
                    "EPS Growth (%)": round(eps_growth, 2) if eps_growth else None,
                    "Revenue Growth (%)": round(revenue_growth, 2) if revenue_growth else None,
                    "Debt/Equity": round(debt_to_equity, 2) if debt_to_equity else None,
                    "Cash/Debt": round(cash_to_debt, 2),
                    "FCF Yield (%)": round(fcf_yield, 2) if fcf_yield else None
                })
            except Exception as e:
                st.error(f"Fehler beim Verarbeiten von {ticker}: {str(e)}")

    if results:
        df = pd.DataFrame(results)
        st.subheader("📊 Ergebnisse aller analysierten Aktien")
        
        # Sortieren nach dem höchsten Score
        df = df.sort_values(by="Alex-KPI Score", ascending=False)
        
        # Anzeige im interaktiven Streamlit-Datiframe
        st.dataframe(df, use_container_width=True)
