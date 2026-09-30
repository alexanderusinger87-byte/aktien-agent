import streamlit as st
import yfinance as yf
import pandas as pd

# -----------------------------
# GENERISCHE SCORING-FUNKTION
# -----------------------------
def get_score(value, thresholds, reverse=False):
    """
    Berechnet den Score anhand flexibler Schwellenwerte.
    `thresholds` ist eine Liste von Tupeln: [(Schwelle, Score), ...]
    `reverse=True` bedeutet: Niedrigerer Wert ist besser (z.B. KGV, Debt/Equity).
    """
    if value is None or pd.isna(value):
        return None

    if not reverse:
        # Höher ist besser
        for limit, score in thresholds:
            if value > limit:
                return score
        return thresholds[-1][1] if thresholds else 1
    else:
        # Niedriger ist besser
        for limit, score in thresholds:
            if value < limit:
                return score
        return thresholds[-1][1] if thresholds else 1

# -----------------------------
# SCHWELLENWERTE (CONFIG)
# -----------------------------
THRESHOLDS = {
    "STANDARD": {
        "roe": [(20, 10), (15, 8), (10, 6), (5, 4)],
        "peg": [(1.0, 10), (1.5, 8), (2.0, 6)],
        "operating_margin": [(30, 10), (20, 8), (10, 6)],
        "forward_pe": [(12, 10), (18, 8), (25, 6)],
        "pe": [(15, 10), (22, 8), (30, 6)],
        "eps_growth": [(20, 10), (10, 8), (5, 6)],
        "revenue_growth": [(15, 10), (8, 8), (3, 6)],
        "profit_margin": [(25, 10), (15, 8), (8, 6)],
        "debt_to_equity": [(30, 10), (60, 8), (100, 6)],  # Gefixt: Skala von yfinance (Prozent)
        "cash_to_debt": [(1.5, 10), (1.0, 8), (0.5, 6)],
        "fcf_yield": [(6, 10), (4, 8), (2, 6)]
    },
    "DEFENSIVE": {
        "roe": [(18, 10), (12, 8), (8, 6), (4, 4)],
        "operating_margin": [(25, 10), (18, 8), (10, 6)],
        "forward_pe": [(12, 10), (16, 8), (22, 5)],
        "pe": [(14, 10), (20, 8), (26, 5)],
        "eps_growth": [(12, 10), (7, 8), (3, 6)],
        "revenue_growth": [(10, 10), (6, 8), (2, 6)],
        "profit_margin": [(20, 10), (12, 8), (6, 6)],
        "debt_to_equity": [(25, 10), (50, 8), (80, 5)],  # Gefixt
        "cash_to_debt": [(2.0, 10), (1.2, 8), (0.6, 6)],
        "fcf_yield": [(7, 10), (5, 8), (3, 6)]
    },
    "OFFENSIVE": {
        "revenue_growth": [(25, 10), (15, 8), (10, 6)],
        "eps_growth": [(30, 10), (18, 8), (10, 6)],
        "peg": [(1.0, 10), (1.6, 8), (2.2, 6)],
        "roe": [(25, 10), (15, 8), (8, 5)],
        "operating_margin": [(20, 10), (12, 8), (5, 5)],
        "fcf_yield": [(5, 10), (3, 8), (1, 5)]
    }
}

REVERSE_METRICS = ["peg", "forward_pe", "pe", "debt_to_equity"]

WEIGHTS = {
    "STANDARD": {
        "roe": 0.15, "peg": 0.12, "operating_margin": 0.10, "forward_pe": 0.10,
        "pe": 0.08, "eps_growth": 0.10, "revenue_growth": 0.08, "profit_margin": 0.08,
        "debt_to_equity": 0.05, "cash_to_debt": 0.04, "fcf_yield": 0.10
    },
    "DEFENSIVE": {
        "operating_margin": 0.15, "roe": 0.13, "forward_pe": 0.13, "debt_to_equity": 0.12,
        "fcf_yield": 0.12, "pe": 0.10, "cash_to_debt": 0.10, "eps_growth": 0.05,
        "revenue_growth": 0.05, "profit_margin": 0.05
    },
    "OFFENSIVE": {
        "revenue_growth": 0.20, "eps_growth": 0.20, "peg": 0.15, "roe": 0.15,
        "operating_margin": 0.15, "fcf_yield": 0.15
    }
}

def calculate_category_score(data, category):
    """Rechnet den gewichteten Score für ein Profil aus und gleicht fehlende Werte dynamisch aus."""
    total_score = 0
    total_weight = 0

    weights = WEIGHTS[category]
    thresholds = THRESHOLDS[category]

    for metric, weight in weights.items():
        val = data.get(metric)
        is_reverse = metric in REVERSE_METRICS
        score = get_score(val, thresholds.get(metric, []), reverse=is_reverse)

        if score is not None:
            total_score += score * weight
            total_weight += weight

    if total_weight == 0:
        return None
    
    # Skalierung auf 0–100 % (bzw. 0–10 Punkte) basierend auf den vorhandenen Daten
    return round((total_score / total_weight) * 10, 2)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Multi-Score Kriterien Agent", layout="wide")
st.title("📊 Multi-Score KPI Analyse Agent")
st.write("Vergleiche Aktien über das gesamte Risikospektrum: **Defensiv**, **Alex-KPI (Standard)** und **Offensiv**.")

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

                    # Rohdaten auslesen
                    fcf = info.get("freeCashflow")
                    market_cap = info.get("marketCap")
                    fcf_yield = (fcf / market_cap) * 100 if fcf and market_cap else None

                    total_cash = info.get("totalCash", 0)
                    total_debt = info.get("totalDebt", 1)
                    cash_to_debt = (total_cash / total_debt) if total_debt and total_debt > 0 else None

                    metrics = {
                        "pe": info.get("trailingPE"),
                        "forward_pe": info.get("forwardPE"),
                        "peg": info.get("pegRatio"),
                        "profit_margin": info.get("profitMargins") * 100 if info.get("profitMargins") else None,
                        "operating_margin": info.get("operatingMargins") * 100 if info.get("operatingMargins") else None,
                        "roe": info.get("returnOnEquity") * 100 if info.get("returnOnEquity") else None,
                        "revenue_growth": info.get("revenueGrowth") * 100 if info.get("revenueGrowth") else None,
                        "eps_growth": info.get("earningsGrowth") * 100 if info.get("earningsGrowth") else None,  # Gefixt auf Vorjahresvergleich
                        "debt_to_equity": info.get("debtToEquity"),  # Liefert bereits Prozentwerte von yfinance
                        "cash_to_debt": cash_to_debt,
                        "fcf_yield": fcf_yield
                    }

                    results.append({
                        "Ticker": ticker,
                        "Defensiv-Score": calculate_category_score(metrics, "DEFENSIVE"),
                        "Alex-KPI Score": calculate_category_score(metrics, "STANDARD"),
                        "Offensiv-Score": calculate_category_score(metrics, "OFFENSIVE"),
                        "P/E (KGV)": round(metrics["pe"], 2) if metrics["pe"] else None,
                        "Forward P/E": round(metrics["forward_pe"], 2) if metrics["forward_pe"] else None,
                        "PEG Ratio": round(metrics["peg"], 2) if metrics["peg"] else None,
                        "Revenue Growth (%)": round(metrics["revenue_growth"], 2) if metrics["revenue_growth"] else None,
                        "EPS Growth (%)": round(metrics["eps_growth"], 2) if metrics["eps_growth"] else None,
                        "Operating Margin (%)": round(metrics["operating_margin"], 2) if metrics["operating_margin"] else None,
                        "FCF Yield (%)": round(metrics["fcf_yield"], 2) if metrics["fcf_yield"] else None
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
