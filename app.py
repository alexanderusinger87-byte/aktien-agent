import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ==============================================================================
# 1. KONFIGURATION & SEKTOR-REFERENZKURVEN (V4)
# ==============================================================================

CATEGORY_WEIGHTS = {
    "valuation": 0.30,
    "quality": 0.35,
    "growth": 0.15,
    "dip": 0.10,
    "momentum": 0.10,
}

SECTOR_REFERENCES = {
    "Technology": {
        "pe": [60, 45, 30, 20, 12],
        "peg": [3.0, 2.2, 1.5, 1.0, 0.5],
        "ev_ebitda": [30, 22, 16, 11, 6],
        "fcf_yield": [0.0, 0.02, 0.04, 0.06, 0.09],
        "gross_margin": [0.20, 0.35, 0.50, 0.65, 0.80],
        "operating_margin": [0.05, 0.12, 0.20, 0.28, 0.38],
        "roe": [0.05, 0.12, 0.20, 0.30, 0.45],
        "fcf_conversion": [0.30, 0.50, 0.70, 0.90, 1.10],
    },
    "Consumer Cyclical": {
        "pe": [45, 35, 24, 16, 10],
        "peg": [2.8, 2.0, 1.4, 0.9, 0.5],
        "ev_ebitda": [22, 16, 12, 8, 5],
        "fcf_yield": [0.0, 0.02, 0.045, 0.07, 0.10],
        "gross_margin": [0.15, 0.28, 0.40, 0.52, 0.65],
        "operating_margin": [0.03, 0.07, 0.12, 0.18, 0.25],
        "roe": [0.04, 0.10, 0.16, 0.24, 0.35],
        "fcf_conversion": [0.25, 0.45, 0.65, 0.85, 1.05],
    },
    "Default": {
        "pe": [50, 38, 25, 17, 10],
        "peg": [2.8, 2.0, 1.4, 0.9, 0.5],
        "ev_ebitda": [25, 18, 13, 9, 5],
        "fcf_yield": [0.0, 0.02, 0.04, 0.065, 0.09],
        "gross_margin": [0.15, 0.30, 0.45, 0.60, 0.75],
        "operating_margin": [0.04, 0.09, 0.15, 0.22, 0.30],
        "roe": [0.05, 0.10, 0.18, 0.25, 0.35],
        "fcf_conversion": [0.30, 0.50, 0.70, 0.85, 1.00],
    },
}

PEER_GROUPS = {
    "Semiconductors": ["NVDA", "TSM", "AVGO", "AMD", "INTC", "QCOM", "MU", "ASML", "AMAT", "LRCX"],
    "BigTech": ["MSFT", "AAPL", "GOOGL", "AMZN", "META"],
    "Beverages": ["KO", "PEP"],
    "Pharma": ["PFE", "NVO"],
}

# ==============================================================================
# 2. HELPER & FORMATIERUNGS-FUNKTIONEN (AMPELN, STERNE, RATINGS)
# ==============================================================================

def get_sector_ref(sector: str) -> dict:
    return SECTOR_REFERENCES.get(sector, SECTOR_REFERENCES["Default"])

def interpolate_score(value: float, points: list, reverse: bool = False) -> float:
    if value is None or np.isnan(value):
        return np.nan
    xp = points
    fp = [0.0, 25.0, 50.0, 75.0, 100.0]
    if reverse:
        xp = list(reversed(points))
        fp = list(reversed(fp))
    return float(np.interp(value, xp, fp))

def compute_percentile_score(series: pd.Series, reverse: bool = False) -> pd.Series:
    valid = series.dropna()
    if len(valid) < 2:
        return pd.Series(50.0, index=series.index)
    ranks = valid.rank(pct=True) * 100.0
    if reverse:
        ranks = 100.0 - ranks
    res = pd.Series(np.nan, index=series.index)
    res.update(ranks)
    return res

def get_rating_and_ampel(score: float):
    """Generiert Empfehlungen, Sternchen und Ampel-Symbole basierend auf dem Score."""
    if np.isnan(score):
        return "⚪ Keine Daten", "N/A", "☆☆☆☆☆"
    if score >= 80:
        return "🟢 STARKKAUF", "Top Quality", "⭐⭐⭐⭐⭐"
    elif score >= 68:
        return "🟢 KAUFEN", "Akkumulieren", "⭐⭐⭐⭐"
    elif score >= 55:
        return "🟡 HALTEN", "Watchlist", "⭐⭐⭐"
    elif score >= 40:
        return "🟠 VERKAUFEN", "Untergewichten", "⭐⭐"
    else:
        return "🔴 STARK VERKAUFEN", "Meiden", "⭐"

# ==============================================================================
# 3. YFINANCE DATEN-ABRUF (ECHTE MOMENTUM & FCF CONVERSION BERECHNUNG)
# ==============================================================================

def fetch_stock_data(tickers: list) -> pd.DataFrame:
    data_list = []
    
    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            info = ticker_obj.info
            
            # Echte SMA200 & Momentum-Berechnung aus Kurshistorie
            hist = ticker_obj.history(period="1y")
            if not hist.empty and len(hist) >= 200:
                current_price = hist["Close"].iloc[-1]
                sma200 = hist["Close"].tail(200).mean()
                dist_sma200 = (current_price - sma200) / sma200
                
                high_52w = hist["High"].max()
                dist_52w_high = (current_price - high_52w) / high_52w
            else:
                current_price = info.get("currentPrice", np.nan)
                dist_sma200 = np.nan
                high_52w = info.get("fiftyTwoWeekHigh", np.nan)
                dist_52w_high = (current_price - high_52w) / high_52w if high_52w else np.nan

            # Echte FCF Conversion Berechnung
            fcf = info.get("freeCashflow", np.nan)
            net_income = info.get("netIncomeToCommon", np.nan)
            fcf_conv = (fcf / net_income) if (fcf and net_income and net_income > 0) else np.nan

            market_cap = info.get("marketCap", np.nan)
            fcf_yield = (fcf / market_cap) if (fcf and market_cap) else np.nan

            data_list.append({
                "ticker": t,
                "name": info.get("shortName", t),
                "sector": info.get("sector", "Default"),
                "pe": info.get("trailingPE", np.nan),
                "peg": info.get("pegRatio", np.nan),
                "ev_ebitda": info.get("enterpriseToEbitda", np.nan),
                "fcf_yield": fcf_yield,
                "gross_margin": info.get("grossMargins", np.nan),
                "operating_margin": info.get("operatingMargins", np.nan),
                "roe": info.get("returnOnEquity", np.nan),
                "fcf_conversion": fcf_conv,
                "revenue_growth": info.get("revenueGrowth", np.nan),
                "eps_growth": info.get("earningsGrowth", np.nan),
                "dist_52w_high": dist_52w_high,
                "dist_sma200": dist_sma200,
                "analyst_count": info.get("numberOfAnalystOpinions", 10),
            })
        except Exception as e:
            st.warning(f"Fehler beim Laden von {t}: {e}")
            
    return pd.DataFrame(data_list)

# ==============================================================================
# 4. KERN-ALGORITHMUS (ALEX KPI STOCK SCORE V4)
# ==============================================================================

def calculate_v4_score(df: pd.DataFrame) -> pd.DataFrame:
    scores_df = df.copy()

    abs_val_list, abs_qual_list, abs_gro_list, abs_dip_list, abs_mom_list = [], [], [], [], []

    for idx, row in df.iterrows():
        sec_ref = get_sector_ref(row.get("sector", "Default"))

        # Valuation
        s_pe = interpolate_score(row.get("pe"), sec_ref["pe"], reverse=True)
        s_peg = interpolate_score(row.get("peg"), sec_ref["peg"], reverse=True)
        s_ev = interpolate_score(row.get("ev_ebitda"), sec_ref["ev_ebitda"], reverse=True)
        s_fcf_y = interpolate_score(row.get("fcf_yield"), sec_ref["fcf_yield"], reverse=False)
        abs_val_list.append(np.nanmean([s_pe, s_peg, s_ev, s_fcf_y]))

        # Quality
        s_gm = interpolate_score(row.get("gross_margin"), sec_ref["gross_margin"], reverse=False)
        s_om = interpolate_score(row.get("operating_margin"), sec_ref["operating_margin"], reverse=False)
        s_roe = interpolate_score(row.get("roe"), sec_ref["roe"], reverse=False)
        s_fcf_c = interpolate_score(row.get("fcf_conversion"), sec_ref["fcf_conversion"], reverse=False)
        abs_qual_list.append(np.nanmean([s_gm, s_om, s_roe, s_fcf_c]))

        # Growth & Growth Quality
        rev_g = row.get("revenue_growth", np.nan)
        eps_g = row.get("eps_growth", np.nan)
        s_rev = interpolate_score(rev_g, [-0.05, 0.02, 0.08, 0.15, 0.25], reverse=False)
        s_eps = interpolate_score(eps_g, [-0.05, 0.03, 0.10, 0.18, 0.30], reverse=False)
        
        growth_diff = eps_g - rev_g if not (np.isnan(eps_g) or np.isnan(rev_g)) else 0
        s_g_qual = interpolate_score(growth_diff, [-0.10, -0.03, 0.0, 0.05, 0.12], reverse=False)
        abs_gro_list.append(np.nanmean([s_rev, s_eps, s_g_qual]))

        # Dip
        s_dip = interpolate_score(row.get("dist_52w_high"), [-0.40, -0.25, -0.15, -0.08, 0.0], reverse=False)
        abs_dip_list.append(s_dip)

        # Momentum (Echte Berechnung über SMA200 Distanz)
        s_mom = interpolate_score(row.get("dist_sma200"), [-0.20, -0.05, 0.05, 0.15, 0.30], reverse=False)
        abs_mom_list.append(s_mom)

    scores_df["abs_valuation"] = abs_val_list
    scores_df["abs_quality"] = abs_qual_list
    scores_df["abs_growth"] = abs_gro_list
    scores_df["abs_dip"] = abs_dip_list
    scores_df["abs_momentum"] = abs_mom_list

    # Relative Peer Group Perzentile
    rel_val = pd.Series(np.nan, index=df.index)
    rel_qual = pd.Series(np.nan, index=df.index)
    rel_gro = pd.Series(np.nan, index=df.index)

    for peer_name, tickers in PEER_GROUPS.items():
        mask = scores_df["ticker"].isin(tickers)
        if mask.sum() > 1:
            sub = scores_df[mask]
            rel_val.update(pd.concat([
                compute_percentile_score(sub["pe"], reverse=True),
                compute_percentile_score(sub["peg"], reverse=True),
                compute_percentile_score(sub["fcf_yield"], reverse=False)
            ], axis=1).mean(axis=1))
            
            rel_qual.update(pd.concat([
                compute_percentile_score(sub["gross_margin"], reverse=False),
                compute_percentile_score(sub["operating_margin"], reverse=False),
                compute_percentile_score(sub["roe"], reverse=False)
            ], axis=1).mean(axis=1))

            rel_gro.update(pd.concat([
                compute_percentile_score(sub["revenue_growth"], reverse=False),
                compute_percentile_score(sub["eps_growth"], reverse=False)
            ], axis=1).mean(axis=1))

    scores_df["rel_valuation"] = rel_val.fillna(scores_df["abs_valuation"])
    scores_df["rel_quality"] = rel_qual.fillna(scores_df["abs_quality"])
    scores_df["rel_growth"] = rel_gro.fillna(scores_df["abs_growth"])

    # Hybride Verrechnung (60% Absolut / 40% Relativ)
    scores_df["cat_valuation"] = np.round(0.60 * scores_df["abs_valuation"] + 0.40 * scores_df["rel_valuation"], 1)
    scores_df["cat_quality"] = np.round(0.60 * scores_df["abs_quality"] + 0.40 * scores_df["rel_quality"], 1)
    scores_df["cat_growth"] = np.round(0.60 * scores_df["abs_growth"] + 0.40 * scores_df["rel_growth"], 1)
    scores_df["cat_dip"] = np.round(scores_df["abs_dip"], 1)
    scores_df["cat_momentum"] = np.round(scores_df["abs_momentum"], 1)

    # Gesamter Score & Vertrauensfaktor
    raw_total = (
        scores_df["cat_valuation"] * CATEGORY_WEIGHTS["valuation"]
        + scores_df["cat_quality"] * CATEGORY_WEIGHTS["quality"]
        + scores_df["cat_growth"] * CATEGORY_WEIGHTS["growth"]
        + scores_df["cat_dip"] * CATEGORY_WEIGHTS["dip"]
        + scores_df["cat_momentum"] * CATEGORY_WEIGHTS["momentum"]
    )

    analysts = scores_df.get("analyst_count", pd.Series(10, index=df.index))
    conf_factor = np.where(analysts < 3, 0.85, np.where(analysts < 8, 0.93, 1.0))

    scores_df["Score_V4"] = np.round(raw_total * conf_factor, 1)

    # Sterne, Ampeln und Empfehlungen zuweisen
    ratings, statuses, stars = [], [], []
    for s in scores_df["Score_V4"]:
        r, st_val, star_val = get_rating_and_ampel(s)
        ratings.append(r)
        statuses.append(st_val)
        stars.append(star_val)

    scores_df["Empfehlung"] = ratings
    scores_df["Sterne"] = stars
    scores_df["Status"] = statuses

    return scores_df

# ==============================================================================
# 5. STREAMLIT FRONTEND
# ==============================================================================

def main():
    st.set_page_config(page_title="Alex KPI Stock Score V4", layout="wide")
    st.title("🏆 Alex KPI Stock Score V4 Screener")

    sample_tickers = ["NVDA", "TSM", "MSFT", "AAPL", "GOOGL", "AMZN", "KO", "PEP", "PFE", "NVO", "ASML", "AMD"]
    
    selected = st.sidebar.multiselect("Aktien auswählen:", sample_tickers, default=["NVDA", "MSFT", "AAPL", "KO", "PFE"])

    if st.button("🚀 Analyse starten"):
        with st.spinner("Lade Live-Marktdaten & berechneIndikatoren..."):
            raw_data = fetch_stock_data(selected)
            if not raw_data.empty:
                df_result = calculate_v4_score(raw_data)

                # Ausgabe-Tabelle konfigurieren
                display_df = df_result[[
                    "ticker", "name", "Score_V4", "Sterne", "Empfehlung",
                    "cat_valuation", "cat_quality", "cat_growth", "cat_dip", "cat_momentum"
                ]].sort_values(by="Score_V4", ascending=False)

                st.subheader("🎯 Kaufempfehlungen & Gesamtranking")
                
                # Streamlit Dataframe mit visueller Formatierung
                st.dataframe(
                    display_df.style.background_gradient(subset=["Score_V4"], cmap="RdYlGn")
                                    .format({"Score_V4": "{:.1f}"}),
                    use_container_width=True
                )
            else:
                st.error("Keine Daten geladen.")

if __name__ == "__main__":
    main()
