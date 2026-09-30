import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ==============================================================================
# 1. KONFIGURATION & SEKTOR-REFERENZKURVEN (Absolutes Scoring)
# ==============================================================================

# Gewichtung der Hauptkategorien (Summe = 1.0)
CATEGORY_WEIGHTS = {
    "valuation": 0.30,
    "quality": 0.35,
    "growth": 0.15,
    "dip": 0.10,
    "momentum": 0.10,
}

# Sektorbezogene 5-Punkt-Skalierung für absolute Kennzahlen
# Format: [Skalierungswerte für Scores 0, 25, 50, 75, 100]
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

# Zuordnung manueller Peer-Groups für die relative Bewertung
PEER_GROUPS = {
    "Semiconductors": ["NVDA", "TSM", "AVGO", "AMD", "INTC", "QCOM", "MU", "ASML", "AMAT", "LRCX"],
    "BigTech": ["MSFT", "AAPL", "GOOGL", "AMZN", "META"],
    "Beverages": ["KO", "PEP"],
    "Pharma": ["PFE", "NVO"],
}

# ==============================================================================
# 2. HILFSFUNKTIONEN UND SCORING-LOGIK
# ==============================================================================

def get_sector_ref(sector: str) -> dict:
    """Gibt das passende Sektor-Referenzprofil zurück."""
    return SECTOR_REFERENCES.get(sector, SECTOR_REFERENCES["Default"])


def interpolate_score(value: float, points: list, reverse: bool = False) -> float:
    """
    Interpoliert stufenlos einen Wert auf eine Skala von 0 bis 100 Punkte.
    reverse=True -> Kleinerer Wert = Höherer Score (z. B. bei KGV, PEG)
    """
    if value is None or np.isnan(value):
        return np.nan

    xp = points
    fp = [0.0, 25.0, 50.0, 75.0, 100.0]

    if reverse:
        xp = list(reversed(points))
        fp = list(reversed(fp))

    return float(np.interp(value, xp, fp))


def compute_percentile_score(series: pd.Series, reverse: bool = False) -> pd.Series:
    """Berechnet das Perzentil (0-100) innerhalb einer Peer-Group."""
    valid = series.dropna()
    if len(valid) < 2:
        return pd.Series(50.0, index=series.index)

    ranks = valid.rank(pct=True) * 100.0
    if reverse:
        ranks = 100.0 - ranks

    res = pd.Series(np.nan, index=series.index)
    res.update(ranks)
    return res


# ==============================================================================
# 3. KERN-ALGORITHMUS (ALEX KPI STOCK SCORE V4)
# ==============================================================================

def calculate_v4_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet den hybriden Alex KPI Stock Score V4 (60% Absolut + 40% Relativ).
    """
    scores_df = df.copy()

    # --------------------------------------------------------------------------
    # A. ABSOLUTES SCORING (60% Gewichtung)
    # --------------------------------------------------------------------------
    abs_valuation = []
    abs_quality = []
    abs_growth = []
    abs_dip = []
    abs_momentum = []

    for idx, row in df.iterrows():
        sec_ref = get_sector_ref(row.get("sector", "Default"))

        # Valuation
        s_pe = interpolate_score(row.get("pe"), sec_ref["pe"], reverse=True)
        s_peg = interpolate_score(row.get("peg"), sec_ref["peg"], reverse=True)
        s_ev = interpolate_score(row.get("ev_ebitda"), sec_ref["ev_ebitda"], reverse=True)
        s_fcf_y = interpolate_score(row.get("fcf_yield"), sec_ref["fcf_yield"], reverse=False)
        abs_val = np.nanmean([s_pe, s_peg, s_ev, s_fcf_y])

        # Quality
        s_gm = interpolate_score(row.get("gross_margin"), sec_ref["gross_margin"], reverse=False)
        s_om = interpolate_score(row.get("operating_margin"), sec_ref["operating_margin"], reverse=False)
        s_roe = interpolate_score(row.get("roe"), sec_ref["roe"], reverse=False)
        s_fcf_c = interpolate_score(row.get("fcf_conversion"), sec_ref["fcf_conversion"], reverse=False)
        abs_qual = np.nanmean([s_gm, s_om, s_roe, s_fcf_c])

        # Growth & Growth Quality
        rev_g = row.get("revenue_growth", np.nan)
        eps_g = row.get("eps_growth", np.nan)
        
        s_rev = interpolate_score(rev_g, [-0.05, 0.02, 0.08, 0.15, 0.25], reverse=False)
        s_eps = interpolate_score(eps_g, [-0.05, 0.03, 0.10, 0.18, 0.30], reverse=False)
        
        # Growth Quality Bonus/Malus (EPS Growth - Revenue Growth)
        growth_diff = eps_g - rev_g if not (np.isnan(eps_g) or np.isnan(rev_g)) else 0
        s_g_qual = interpolate_score(growth_diff, [-0.10, -0.03, 0.0, 0.05, 0.12], reverse=False)
        abs_gro = np.nanmean([s_rev, s_eps, s_g_qual])

        # Dip (Rücksetzer vom 52W-Hoch)
        s_dip = interpolate_score(row.get("dist_52w_high"), [-0.40, -0.25, -0.15, -0.08, 0.0], reverse=False)
        
        # Momentum (Performance vs. Moving Averages)
        dist_sma200 = row.get("dist_sma200", np.nan)
        s_mom = interpolate_score(dist_sma200, [-0.20, -0.05, 0.05, 0.15, 0.30], reverse=False)

        # Zusammenfassung
        abs_valuation.append(abs_val)
        abs_quality.append(abs_qual)
        abs_growth.append(abs_gro)
        abs_dip.append(s_dip)
        abs_momentum.append(s_mom)

    scores_df["abs_valuation"] = abs_valuation
    scores_df["abs_quality"] = abs_quality
    scores_df["abs_growth"] = abs_growth
    scores_df["abs_dip"] = abs_dip
    scores_df["abs_momentum"] = abs_momentum

    # --------------------------------------------------------------------------
    # B. RELATIYES SCORING (40% Gewichtung via Peer Group Perzentile)
    # --------------------------------------------------------------------------
    rel_valuation = pd.Series(np.nan, index=df.index)
    rel_quality = pd.Series(np.nan, index=df.index)
    rel_growth = pd.Series(np.nan, index=df.index)

    # Identifikation von Peer Groups
    for peer_name, tickers in PEER_GROUPS.items():
        mask = scores_df["ticker"].isin(tickers)
        if mask.sum() > 1:
            sub = scores_df[mask]

            p_pe = compute_percentile_score(sub["pe"], reverse=True)
            p_peg = compute_percentile_score(sub["peg"], reverse=True)
            p_fcf_y = compute_percentile_score(sub["fcf_yield"], reverse=False)
            rel_val_sub = pd.concat([p_pe, p_peg, p_fcf_y], axis=1).mean(axis=1)

            p_gm = compute_percentile_score(sub["gross_margin"], reverse=False)
            p_om = compute_percentile_score(sub["operating_margin"], reverse=False)
            p_roe = compute_percentile_score(sub["roe"], reverse=False)
            rel_qual_sub = pd.concat([p_gm, p_om, p_roe], axis=1).mean(axis=1)

            p_rev_g = compute_percentile_score(sub["revenue_growth"], reverse=False)
            p_eps_g = compute_percentile_score(sub["eps_growth"], reverse=False)
            rel_gro_sub = pd.concat([p_rev_g, p_eps_g], axis=1).mean(axis=1)

            rel_valuation.update(rel_val_sub)
            rel_quality.update(rel_qual_sub)
            rel_growth.update(rel_gro_sub)

    # Fülle fehlende Peer-Werte mit Absolut-Scores auf
    scores_df["rel_valuation"] = rel_valuation.fillna(scores_df["abs_valuation"])
    scores_df["rel_quality"] = rel_quality.fillna(scores_df["abs_quality"])
    scores_df["rel_growth"] = rel_growth.fillna(scores_df["abs_growth"])

    # --------------------------------------------------------------------------
    # C. HYBRIDE KATEGORIE-SCORES (60% Absolut / 40% Relativ)
    # --------------------------------------------------------------------------
    scores_df["cat_valuation"] = 0.60 * scores_df["abs_valuation"] + 0.40 * scores_df["rel_valuation"]
    scores_df["cat_quality"] = 0.60 * scores_df["abs_quality"] + 0.40 * scores_df["rel_quality"]
    scores_df["cat_growth"] = 0.60 * scores_df["abs_growth"] + 0.40 * scores_df["rel_growth"]
    scores_df["cat_dip"] = scores_df["abs_dip"]
    scores_df["cat_momentum"] = scores_df["abs_momentum"]

    # --------------------------------------------------------------------------
    # D. GESAMTSCORE BERECHNUNG & ANALYSTEN-GEWICHTUNG
    # --------------------------------------------------------------------------
    raw_total_score = (
        scores_df["cat_valuation"] * CATEGORY_WEIGHTS["valuation"]
        + scores_df["cat_quality"] * CATEGORY_WEIGHTS["quality"]
        + scores_df["cat_growth"] * CATEGORY_WEIGHTS["growth"]
        + scores_df["cat_dip"] * CATEGORY_WEIGHTS["dip"]
        + scores_df["cat_momentum"] * CATEGORY_WEIGHTS["momentum"]
    )

    # Berücksichtigung von Analysten-Abdeckung
    analyst_counts = scores_df.get("analyst_count", pd.Series(10, index=df.index))
    confidence_factor = np.where(analyst_counts < 3, 0.85, np.where(analyst_counts < 8, 0.93, 1.0))

    scores_df["final_v4_score"] = np.round(raw_total_score * confidence_factor, 2)

    return scores_df


# ==============================================================================
# 4. STREAMLIT BENUTZEROBERFLÄCHE
# ==============================================================================

def main():
    st.set_page_config(page_title="Alex KPI Stock Score V4", layout="wide")
    st.title("📈 Alex KPI Stock Score V4 Screener")
    st.caption("Hybrider Aktien-Screener (60% Absolut / 40% Peer Group Relativ)")

    # Test-Datenset definieren
    sample_tickers = ["NVDA", "TSM", "MSFT", "AAPL", "GOOGL", "AMZN", "KO", "PEP", "PFE", "NVO"]
    
    st.sidebar.header("Optionen")
    selected_tickers = st.sidebar.multiselect("Aktien auswählen:", sample_tickers, default=sample_tickers)

    if st.button("Score V4 berechnen"):
        with st.spinner("Lade Marktdaten..."):
            # Dummy-Struktur zur Demonstration (kann mit yfinance-Dataframe erweitert werden)
            data_list = []
            for t in selected_tickers:
                ticker_obj = yf.Ticker(t)
                info = ticker_obj.info
                
                # Sektor bestimmen
                sector = info.get("sector", "Default")
                
                # Daten extrahieren mit Fallbacks
                data_list.append({
                    "ticker": t,
                    "name": info.get("shortName", t),
                    "sector": sector,
                    "pe": info.get("trailingPE", np.nan),
                    "peg": info.get("pegRatio", np.nan),
                    "ev_ebitda": info.get("enterpriseToEbitda", np.nan),
                    "fcf_yield": (info.get("freeCashflow", 0) / info.get("marketCap", 1)) if info.get("marketCap") else np.nan,
                    "gross_margin": info.get("grossMargins", np.nan),
                    "operating_margin": info.get("operatingMargins", np.nan),
                    "roe": info.get("returnOnEquity", np.nan),
                    "fcf_conversion": 0.8,  # Platzhalter-Wert für Demo
                    "revenue_growth": info.get("revenueGrowth", np.nan),
                    "eps_growth": info.get("earningsGrowth", np.nan),
                    "dist_52w_high": (info.get("currentPrice", 0) - info.get("fiftyTwoWeekHigh", 1)) / info.get("fiftyTwoWeekHigh", 1) if info.get("fiftyTwoWeekHigh") else np.nan,
                    "dist_sma200": 0.05,  # Platzhalter
                    "analyst_count": info.get("numberOfAnalystOpinions", 10),
                })

            df_input = pd.DataFrame(data_list)
            df_scored = calculate_v4_score(df_input)

            # Ergebnistabelle formatieren
            display_cols = [
                "ticker",
                "name",
                "sector",
                "final_v4_score",
                "cat_valuation",
                "cat_quality",
                "cat_growth",
                "cat_dip",
                "cat_momentum",
            ]
            
            res_df = df_scored[display_cols].sort_values(by="final_v4_score", ascending=False)
            
            st.subheader("📊 Ranking Ergebnisse")
            st.dataframe(
                res_df.style.background_gradient(subset=["final_v4_score"], cmap="Greens"),
                use_container_width=True
            )

if __name__ == "__main__":
    main()
