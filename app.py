import streamlit as st
import yfinance as yf
import pandas as pd

# -----------------------------
# 0. HELPER FÜR SICHERE PROZENTUMRECHNUNG
# -----------------------------
def safe_pct(val):
    """Rechnet Dezimalwerte von yfinance sicher in Prozent um (hält 0.0 aufrecht)."""
    if val is None or pd.isna(val):
        return None
    try:
        return float(val) * 100.0
    except (ValueError, TypeError):
        return None

# -----------------------------
# 1. INTERPOLATIONSSCORE-FUNKTION
# -----------------------------
def get_score_interpolated(value, min_val, max_val, reverse=False):
    """
    Berechnet einen stufenlosen Score zwischen 0 und 10 Punkten.
    reverse=True bedeutet: Kleinere Werte sind BESSER (z.B. KGV, Beta, Schulden).
    """
    if value is None or pd.isna(value):
        return None

    if reverse and value < 0:
        return 0.0  # Negativwerte bei Reverse-Metriken = Höchstes Risiko / Verlust

    lower = min(min_val, max_val)
    upper = max(min_val, max_val)
    clamped_val = max(min(value, upper), lower)

    if upper == lower:
        normalized = 1.0
    else:
        normalized = (clamped_val - lower) / (upper - lower)

    if reverse:
        normalized = 1.0 - normalized

    return normalized * 10

# -----------------------------
# 2. VISUELLE INDIKATOREN & EMPFEHLUNG
# -----------------------------
def get_star_rating(score):
    if score is None:
        return "N/A"
    stars_count = round((score / 10) * 5)
    stars_count = max(0, min(5, stars_count))
    filled = "★" * stars_count
    empty = "☆" * (5 - stars_count)
    return f"{filled}{empty} ({round(score / 2, 1)})"

def get_recommendation(score):
    if score is None:
        return "Keine Daten"
    if score >= 8.0:
        return "Strong Buy 🟢"
    elif score >= 6.5:
        return "Buy 🟢"
    elif score >= 5.0:
        return "Hold 🟡"
    elif score >= 3.5:
        return "Sell 🔴"
    else:
        return "Strong Sell 🔴"

def get_risk_label(risk_score):
    if risk_score is None:
        return "N/A"
    if risk_score >= 7.5:
        return "Niedrig 🛡️"
    elif risk_score >= 5.0:
        return "Moderat ⚖️"
    elif risk_score >= 3.0:
        return "Erhöht ⚠️"
    else:
        return "Hoch 🚨"

# -----------------------------
# 3. SEKTORSPEZIFISCHE BOUNDS
# -----------------------------
SECTOR_BOUNDS = {
    "Technology": {
        "roe": (-10, 35), "operating_margin": (-10, 40), "forward_pe": (15, 45),
        "pe": (15, 50), "debt_to_equity": (0, 100), "revenue_growth": (-5, 30),
        "peg": (0.8, 3.0), "eps_growth": (-5, 30), "profit_margin": (-10, 30),
        "cash_to_debt": (0.5, 3.0), "fcf_yield": (0, 6), "beta": (0.8, 1.8)
    },
    "Healthcare": {
        "roe": (-10, 25), "operating_margin": (-5, 30), "forward_pe": (12, 30),
        "pe": (15, 35), "debt_to_equity": (0, 120), "revenue_growth": (-5, 15),
        "peg": (0.8, 2.5), "eps_growth": (-5, 20), "profit_margin": (-5, 25),
        "cash_to_debt": (0.3, 2.0), "fcf_yield": (0, 8), "beta": (0.4, 1.2)
    },
    "Consumer Cyclical": {
        "roe": (-10, 20), "operating_margin": (-5, 20), "forward_pe": (10, 25),
        "pe": (12, 30), "debt_to_equity": (0, 200), "revenue_growth": (-5, 15),
        "peg": (0.6, 2.2), "eps_growth": (-5, 18), "profit_margin": (-5, 15),
        "cash_to_debt": (0.2, 1.5), "fcf_yield": (0, 9), "beta": (0.7, 1.6)
    },
    "Financial Services": {
        "roe": (0, 18), "operating_margin": (0, 25), "forward_pe": (6, 18),
        "pe": (8, 20), "debt_to_equity": (0, 800), "revenue_growth": (-5, 12),
        "peg": (0.5, 2.0), "eps_growth": (-5, 15), "profit_margin": (0, 25),
        "cash_to_debt": (0.0, 1.0), "fcf_yield": (0, 10), "beta": (0.5, 1.4)
    },
    "DEFAULT": {
        "roe": (-10, 25), "operating_margin": (-10, 35), "forward_pe": (10, 30),
        "pe": (10, 35), "debt_to_equity": (0, 150), "revenue_growth": (-5, 20),
        "peg": (0.5, 2.5), "eps_growth": (-5, 25), "profit_margin": (-10, 30),
        "cash_to_debt": (0, 2.0), "fcf_yield": (0, 8), "beta": (0.5, 1.5)
    }
}

REVERSE_METRICS = ["peg", "forward_pe", "pe", "debt_to_equity", "beta"]

# -----------------------------
# 4. TEIL-GEWICHTUNGEN
# -----------------------------
QUALITY_WEIGHTS = {
    "roe": 0.30, "operating_margin": 0.25, "eps_growth": 0.20,
    "revenue_growth": 0.15, "profit_margin": 0.10
}

VALUATION_WEIGHTS = {
    "peg": 0.40, "forward_pe": 0.30, "pe": 0.15, "fcf_yield": 0.15
}

RISK_WEIGHTS = {
    "debt_to_equity": 0.40,
    "cash_to_debt": 0.30,
    "beta": 0.30
}

def calculate_sub_score(data, weights, sector=None):
    total_score = 0.0
    total_weight = 0.0
    bounds = SECTOR_BOUNDS.get(sector, SECTOR_BOUNDS["DEFAULT"])

    for metric, weight in weights.items():
        val = data.get(metric)
        is_reverse = metric in REVERSE_METRICS

        if metric in bounds:
            min_v, max_v = bounds[metric]
            score = get_score_interpolated(val, min_v, max_v, reverse=is_reverse)

            if score is not None:
                total_score += score * weight
                total_weight += weight

    if total_weight == 0:
        return None

    return round(total_score / total_weight, 2)

# -----------------------------
# 5. STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Alex Overall KPI Agent", layout="wide")
st.title("📊 Alex-KPI Gesamt-Analyse")
st.write("Gewichtung: **Qualität (45%)** | **Bewertung/Preis (40%)** | **Risiko (15%)**")

tickers_input = st.text_input("Gib mehrere Ticker ein (getrennt durch Komma):", value="MSFT, JNJ, NVDA, AAPL")

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

                    company_name = info.get("longName") or info.get("shortName") or ticker
                    sector = info.get("sector", "Unbekannt")
                    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
                    currency = info.get("currency", "USD")

                    fcf = info.get("freeCashflow")
                    market_cap = info.get("marketCap")
                    fcf_yield = (fcf / market_cap) * 100 if fcf and market_cap else None

                    total_cash = info.get("totalCash", 0) or 0
                    total_debt = info.get("totalDebt", 0) or 0

                    if total_debt == 0 and total_cash > 0:
                        cash_to_debt = 10.0
                    elif total_debt > 0:
                        cash_to_debt = total_cash / total_debt
                    else:
                        cash_to_debt = None

                    pe_val = info.get("trailingPE")
                    trailing_eps = info.get("trailingEps")
                    if pe_val is None and trailing_eps is not None and trailing_eps < 0:
                        pe_val = -1

                    metrics = {
                        "pe": pe_val,
                        "forward_pe": info.get("forwardPE"),
                        "peg": info.get("pegRatio"),
                        "profit_margin": safe_pct(info.get("profitMargins")),
                        "operating_margin": safe_pct(info.get("operatingMargins")),
                        "roe": safe_pct(info.get("returnOnEquity")),
                        "revenue_growth": safe_pct(info.get("revenueGrowth")),
                        "eps_growth": safe_pct(info.get("earningsGrowth")),
                        "debt_to_equity": info.get("debtToEquity"),
                        "cash_to_debt": cash_to_debt,
                        "fcf_yield": fcf_yield,
                        "beta": info.get("beta")
                    }

                    quality_score = calculate_sub_score(metrics, QUALITY_WEIGHTS, sector=sector)
                    valuation_score = calculate_sub_score(metrics, VALUATION_WEIGHTS, sector=sector)
                    risk_safety_score = calculate_sub_score(metrics, RISK_WEIGHTS, sector=sector)

                    scores_weighted = []
                    weights_sum = 0.0

                    if quality_score is not None:
                        scores_weighted.append(quality_score * 0.45)
                        weights_sum += 0.45
                    if valuation_score is not None:
                        scores_weighted.append(valuation_score * 0.40)
                        weights_sum += 0.40
                    if risk_safety_score is not None:
                        scores_weighted.append(risk_safety_score * 0.15)
                        weights_sum += 0.15

                    overall_score = round(sum(scores_weighted) / weights_sum, 2) if weights_sum > 0 else None
                    display_pe = "N/A (Verlust)" if pe_val == -1 else (round(pe_val, 2) if pe_val and pe_val > 0 else "N/A")

                    results.append({
                        "Ticker": ticker,
                        "Unternehmen": company_name,
                        "🏆 Alex Gesamtscore": overall_score,
                        "Empfehlung": get_recommendation(overall_score),
                        "Rating": get_star_rating(overall_score),
                        "Aktueller Kurs": f"{current_price:.2f} {currency}" if current_price else "N/A",
                        "Qualitäts Score": quality_score,
                        "Bewertungs Score": valuation_score,
                        "Sicherheits Score": risk_safety_score,
                        "Risiko": get_risk_label(risk_safety_score),
                        "Beta": round(metrics["beta"], 2) if metrics["beta"] else "N/A",
                        "P/E (KGV)": display_pe,
                        "PEG Ratio": round(metrics["peg"], 2) if metrics["peg"] else None,
                        "Sektor": sector
                    })
                except Exception as e:
                    st.error(f"Fehler bei {ticker}: {str(e)}")

        if results:
            df = pd.DataFrame(results)
            df = df.sort_values(by="🏆 Alex Gesamtscore", ascending=False)

            st.subheader("📊 Ergebnis-Matrix")

            styled_df = df.style.background_gradient(
                cmap="YlGn", 
                subset=["🏆 Alex Gesamtscore"],
                vmin=0.0, 
                vmax=10.0
            ).format(
                {"🏆 Alex Gesamtscore": "{:.2f}"}, 
                na_rep="N/A"
            )

            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("Es konnten keine Daten für die angegebenen Ticker geladen werden.")
