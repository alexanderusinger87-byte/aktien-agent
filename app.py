import streamlit as st
import yfinance as yf
import pandas as pd

# -----------------------------
# 1. INTERPOLATIONSSCORE-FUNKTION
# -----------------------------
def get_score_interpolated(value, min_val, max_val, reverse=False):
    """
    Berechnet einen stufenlosen Score zwischen 0 und 10 Punkten
    mittels linearer Interpolation.
    """
    if value is None or pd.isna(value):
        return None

    # Schutz vor negativen KGVs/PEGs (Verlustunternehmen erhalten 0 Punkte)
    if reverse and value < 0:
        return 0.0

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
# 2. STERNE-RATING & EMPFEHLUNG (Basiert auf Price Score)
# -----------------------------
def get_star_rating(score):
    """Wandelt einen 0-10 Score in ein visuelles 0-5 Sterne-Rating um."""
    if score is None:
        return "N/A"
    
    stars_count = round((score / 10) * 5)
    stars_count = max(0, min(5, stars_count))
    
    filled = "★" * stars_count
    empty = "☆" * (5 - stars_count)
    return f"{filled}{empty} ({round(score / 2, 1)})"

def get_recommendation(score):
    """Leitet eine klare Handlungsempfehlung ab."""
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

# -----------------------------
# 3. SEKTORSPEZIFISCHE BOUNDS
# -----------------------------
SECTOR_BOUNDS = {
    "Technology": {
        "roe": (10, 35), "operating_margin": (10, 40), "forward_pe": (15, 45),
        "pe": (15, 50), "debt_to_equity": (0, 100), "revenue_growth": (5, 30),
        "peg": (0.8, 3.0), "eps_growth": (5, 30), "profit_margin": (10, 30),
        "cash_to_debt": (0.5, 3.0), "fcf_yield": (0, 6)
    },
    "Financial Services": {
        "roe": (5, 18), "operating_margin": (5, 25), "forward_pe": (6, 18),
        "pe": (8, 20), "debt_to_equity": (0, 800), "revenue_growth": (0, 12),
        "peg": (0.5, 2.0), "eps_growth": (0, 15), "profit_margin": (5, 25),
        "cash_to_debt": (0.0, 1.0), "fcf_yield": (0, 10)
    },
    "DEFAULT": {
        "roe": (0, 25), "operating_margin": (0, 35), "forward_pe": (10, 30),
        "pe": (10, 35), "debt_to_equity": (0, 150), "revenue_growth": (0, 20),
        "peg": (0.5, 2.5), "eps_growth": (0, 25), "profit_margin": (0, 30),
        "cash_to_debt": (0, 2.0), "fcf_yield": (0, 8)
    }
}

REVERSE_METRICS = ["peg", "forward_pe", "pe", "debt_to_equity"]

# -----------------------------
# 4. GEWICHTUNG MENTALE METHODIK
# -----------------------------
# Qualitativ / Operativ (Unternehmensstärke)
QUALITY_WEIGHTS = {
    "roe": 0.25, "operating_margin": 0.20, "eps_growth": 0.15,
    "revenue_growth": 0.15, "profit_margin": 0.10, "debt_to_equity": 0.08,
    "cash_to_debt": 0.07
}

# Preis & Bewertung (Wie teuer ist die Aktie am Markt?)
VALUATION_WEIGHTS = {
    "peg": 0.40, "forward_pe": 0.30, "pe": 0.15, "fcf_yield": 0.15
}

def calculate_sub_score(data, weights, sector=None):
    """Berechnet einen gewichteten Teil-Score (Qualität oder Bewertung)."""
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
st.set_page_config(page_title="Alex Price Score Analyser", layout="wide")
st.title("📊 Alex-KPI Aktien & Valuation Agent")
st.write("Kombinierte Bewertung aus **Qualität (Fundamentaldaten)** und **Aktuellem Preis (Bewertung)**.")

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

                    # Kennzahlen berechnen
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
                        "eps_growth": info.get("earningsGrowth") * 100 if info.get("earningsGrowth") else None,
                        "debt_to_equity": info.get("debtToEquity"),
                        "cash_to_debt": cash_to_debt,
                        "fcf_yield": fcf_yield
                    }

                    # Teil-Scores berechnen
                    quality_score = calculate_sub_score(metrics, QUALITY_WEIGHTS, sector=sector)
                    valuation_score = calculate_sub_score(metrics, VALUATION_WEIGHTS, sector=sector)

                    # PRICE SCORE (Gesamtscore): 60% Qualität, 40% Aktuelle Bewertung/Preis
                    if quality_score is not None and valuation_score is not None:
                        price_score = round((quality_score * 0.60) + (valuation_score * 0.40), 2)
                    elif quality_score is not None:
                        price_score = quality_score
                    else:
                        price_score = valuation_score

                    results.append({
                        "Ticker": ticker,
                        "Unternehmen": company_name,
                        "Aktueller Kurs": f"{current_price:.2f} {currency}" if current_price else "N/A",
                        "Empfehlung": get_recommendation(price_score),
                        "Rating": get_star_rating(price_score),
                        "Price Score": price_score,
                        "Qualitäts Score": quality_score,
                        "Bewertungs Score": valuation_score,
                        "Sektor": sector,
                        "P/E (KGV)": round(metrics["pe"], 2) if metrics["pe"] else None,
                        "Forward P/E": round(metrics["forward_pe"], 2) if metrics["forward_pe"] else None,
                        "PEG Ratio": round(metrics["peg"], 2) if metrics["peg"] else None,
                        "Revenue Growth (%)": round(metrics["revenue_growth"], 2) if metrics["revenue_growth"] else None,
                        "Operating Margin (%)": round(metrics["operating_margin"], 2) if metrics["operating_margin"] else None,
                        "FCF Yield (%)": round(metrics["fcf_yield"], 2) if metrics["fcf_yield"] else None
                    })
                except Exception as e:
                    st.error(f"Fehler bei {ticker}: {str(e)}")

        if results:
            df = pd.DataFrame(results)
            st.subheader("📊 Ergebnis-Matrix")
            # Sortierung nach dem neuen Price Score!
            df = df.sort_values(by="Price Score", ascending=False)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Es konnten keine Daten für die angegebenen Ticker geladen werden.")
