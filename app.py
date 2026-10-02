import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Quant-Aktien-Screener V11.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 1. DATENBESCHAFFUNG & INDIKATOREN-BERECHNUNG
# ==============================================================================


@st.cache_data(ttl=900)
def fetch_stock_data(ticker_symbol):
    """Holt Fundamentaldaten und technische Indikatoren via yfinance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info or {}
        hist = ticker.history(period="1y")

        if hist.empty or len(hist) < 50:
            return None

        # --- Technische Indikatoren ---
        close = hist["Close"]
        current_price = close.iloc[-1]

        # SMA 50 & 200
        sma50 = close.rolling(window=50).mean().iloc[-1]
        sma200 = (
            close.rolling(window=200).mean().iloc[-1]
            if len(close) >= 200
            else sma50
        )

        sma200_diff = (
            ((current_price - sma200) / sma200) * 100 if sma200 > 0 else 0
        )

        # RSI (14 Tage)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50

        # 12M Momentum
        momentum_12m = (
            ((current_price - close.iloc[0]) / close.iloc[0]) * 100
            if len(close) > 0
            else 0
        )

        # --- Fundamentaldaten ---
        sector = info.get("sector", "Unbekannt")
        is_financial = sector in ["Financial Services", "Financials"]

        # Margen & Renditen
        roic = info.get("returnOnCapital") or info.get("returnOnAssets") or 0.0
        roe = info.get("returnOnEquity") or 0.0
        gross_margin = info.get("grossMargins") or 0.0
        operating_margin = info.get("operatingMargins") or 0.0

        fcf = info.get("freeCashflow") or 0
        total_revenue = info.get("totalRevenue") or 1
        fcf_margin = fcf / total_revenue if total_revenue > 0 else 0.0

        # Bewertung & Lynch Metriken
        forward_pe = info.get("forwardPE") or info.get("trailingPE") or 25.0
        trailing_pe = info.get("trailingPE") or forward_pe
        peg_ratio = info.get("pegRatio") or 2.0
        price_to_fcf = (
            info.get("marketCap", 0) / fcf
            if fcf > 0
            else (info.get("priceToBook") or 30.0)
        )
        ev_ebitda = info.get("enterpriseToEbitda") or 20.0

        # Risikokennzahlen
        net_debt_ebitda = (
            (info.get("totalDebt", 0) - info.get("totalCash", 0))
            / (info.get("ebitda") or 1)
            if info.get("ebitda")
            else 2.0
        )
        current_ratio = info.get("currentRatio") or 1.0

        # Zinsdeckung (Interest Coverage)
        ebit = info.get("operatingCashflow") or info.get("ebitda") or 0
        interest_exp = info.get("interestExpense") or 1
        interest_coverage = (
            abs(ebit / interest_exp) if interest_exp != 0 else 5.0
        )

        # --- Gecapped DCF Fair Value Modell ---
        fcf_yield = fcf / info.get("marketCap", 1) if info.get("marketCap") else 0.04
        hist_growth = info.get("earningsGrowth") or info.get("revenueGrowth") or 0.08
        
        # Wachstum realistisch korrigieren (Capping zwischen 1% und 18%)
        proj_growth = max(0.01, min(hist_growth, 0.18))
        discount_rate = 0.09
        terminal_growth = 0.025

        # 5-Jahre DCF Projektion
        if current_price > 0 and fcf_yield > 0:
            future_fcf_sum = sum([(1 + proj_growth)**i / (1 + discount_rate)**i for i in range(1, 6)])
            terminal_val = ((1 + proj_growth)**5 * (1 + terminal_growth)) / (discount_rate - terminal_growth)
            terminal_disc = terminal_val / (1 + discount_rate)**5
            
            raw_dcf_price = current_price * fcf_yield * (future_fcf_sum + terminal_disc)
            # 15% Sicherheitsabschlag (Margin of Safety)
            dcf_fair_value = raw_dcf_price * 0.85
        else:
            dcf_fair_value = current_price

        dcf_upside = ((dcf_fair_value - current_price) / current_price) * 100 if current_price > 0 else 0

        # Analysten Kursziel
        target_mean = info.get("targetMeanPrice") or current_price
        analyst_upside = ((target_mean - current_price) / current_price) * 100 if current_price > 0 else 0

        return {
            "Ticker": ticker_symbol,
            "Name": info.get("shortName", ticker_symbol),
            "Sector": sector,
            "Price": current_price,
            "ROIC": roic,
            "ROE": roe,
            "Gross_Margin": gross_margin,
            "FCF_Margin": fcf_margin,
            "PEG_Ratio": peg_ratio,
            "Price_to_FCF": price_to_fcf,
            "Forward_PE": forward_pe,
            "EV_EBITDA": ev_ebitda,
            "Net_Debt_EBITDA": net_debt_ebitda,
            "Current_Ratio": current_ratio,
            "Interest_Coverage": interest_coverage,
            "DCF_Fair_Value": dcf_fair_value,
            "DCF_Upside_%": dcf_upside,
            "Target_Price": target_mean,
            "Analyst_Upside_%": analyst_upside,
            "SMA200_Diff_%": sma200_diff,
            "Momentum_12M_%": momentum_12m,
            "RSI": rsi,
            "Is_Financial": is_financial
        }
    except Exception as e:
        return None

# ==============================================================================
# 2. OPTIMIERTES QUANT-SCORING MODELL 2.0
# ==============================================================================

def calculate_quant_score_v2(row):
    """Berechnet das gewichtete 5-Dimensionen Quantitative Scoring inklusive Penalty."""
    
    # --- 1. QUALITY SCORE (30%) ---
    s_roic = np.interp(row.get('ROIC', 0), [0, 0.08, 0.15, 0.25], [20, 50, 85, 100])
    s_fcf_m = np.interp(row.get('FCF_Margin', 0), [0, 0.05, 0.15, 0.25], [10, 40, 80, 100])
    s_gross = np.interp(row.get('Gross_Margin', 0), [0.1, 0.3, 0.5, 0.7], [20, 50, 80, 100])
    s_roe = np.interp(row.get('ROE', 0), [0, 0.10, 0.20, 0.35], [20, 50, 85, 100])
    quality_score = 0.35 * s_roic + 0.30 * s_fcf_m + 0.20 * s_gross + 0.15 * s_roe

    # --- 2. VALUATION SCORE (25%) ---
    s_peg = np.interp(row.get('PEG_Ratio', 2.0), [0.3, 1.0, 1.8, 3.0], [100, 85, 45, 10])
    s_pfcf = np.interp(row.get('Price_to_FCF', 30), [8, 15, 25, 45], [100, 80, 45, 10])
    s_fwd_pe = np.interp(row.get('Forward_PE', 25), [8, 15, 25, 40], [100, 80, 45, 10])
    s_ev = np.interp(row.get('EV_EBITDA', 20), [6, 11, 18, 30], [100, 80, 45, 10])
    valuation_score = 0.35 * s_peg + 0.30 * s_pfcf + 0.20 * s_fwd_pe + 0.15 * s_ev

    # --- 3. FAIR VALUE & MARGIN OF SAFETY (20%) ---
    s_dcf = np.interp(row.get('DCF_Upside_%', 0), [-30, 0, 20, 50], [10, 45, 80, 100])
    s_analyst = np.interp(row.get('Analyst_Upside_%', 0), [-20, 0, 15, 35], [15, 45, 75, 100])
    fair_value_score = 0.60 * s_dcf + 0.40 * s_analyst

    # --- 4. RISK & FINANCIAL HEALTH SCORE (15%) ---
    s_nd_ebitda = np.interp(row.get('Net_Debt_EBITDA', 2.0), [-1.0, 1.0, 3.0, 5.0], [100, 85, 40, 0])
    s_curr = np.interp(row.get('Current_Ratio', 1.0), [0.7, 1.2, 2.0, 3.5], [10, 55, 85, 100])
    s_int_cov = np.interp(row.get('Interest_Coverage', 5.0), [1.0, 3.0, 8.0, 15.0], [0, 40, 80, 100])
    risk_score = 0.45 * s_nd_ebitda + 0.30 * s_int_cov + 0.25 * s_curr

    # --- 5. TECHNICALS & MOMENTUM (10%) ---
    s_sma = np.interp(row.get('SMA200_Diff_%', 0), [-20, 0, 15, 35], [20, 50, 80, 100])
    s_mom = np.interp(row.get('Momentum_12M_%', 0), [-25, 0, 20, 50], [10, 45, 80, 100])
    s_rsi = np.interp(row.get('RSI', 50), [25, 40, 60, 75], [70, 90, 70, 30])
    tech_score = 0.40 * s_sma + 0.40 * s_mom + 0.20 * s_rsi

    # --- GESAMTSCORE-BERECHNUNG MIT PENALTY ---
    raw_total_score = (
        0.30 * quality_score +
        0.25 * valuation_score +
        0.20 * fair_value_score +
        0.15 * risk_score +
        0.10 * tech_score
    )
    
    # Malus für kritische Bilanzrisiken (Financial Distress Protection)
    penalty = 0
    if row.get('Net_Debt_EBITDA', 0) > 4.2:
        penalty += 10
    if row.get('Interest_Coverage', 5) < 2.0:
        penalty += 8
    if row.get('FCF_Margin', 0) < -0.05:
        penalty += 7
        
    final_score = max(0, min(100, raw_total_score - penalty))
    
    return {
        "Total_Score": round(final_score, 1),
        "Quality_Score": round(quality_score, 1),
        "Valuation_Score": round(valuation_score, 1),
        "FairValue_Score": round(fair_value_score, 1),
        "Risk_Score": round(risk_score, 1),
        "Tech_Score": round(tech_score, 1)
    }

# ==============================================================================
# 3. ADVANCED RECOMMENDATION ENGINE
# ==============================================================================

def get_advanced_recommendation(row):
    """Ermittelt die Handlungsempfehlung basierend auf Score, Upside und Turnaround-Regeln."""
    score = row.get('Total_Score', 0)
    dcf_upside = row.get('DCF_Upside_%', 0)
    quality_score = row.get('Quality_Score', 0)
    risk_score = row.get('Risk_Score', 0)
    rsi = row.get('RSI', 50)
    sma200_diff = row.get('SMA200_Diff_%', 0)

    # Turnaround / Rebound Spezialfall
    is_fundamentally_sound = (quality_score >= 60) and (risk_score >= 55)
    is_oversold = (rsi < 42) or (sma200_diff < -15)
    has_deep_value = dcf_upside >= 20.0

    if is_fundamentally_sound and is_oversold and has_deep_value:
        return "🔥 Strong Turnaround Buy"
    
    if score >= 75 and dcf_upside >= 5.0 and risk_score >= 60:
        return "🟢 Top Quality Buy"
    
    if score >= 68 and (dcf_upside >= 0 or quality_score >= 70):
        return "🟢 Buy"
    
    if is_oversold and has_deep_value and score >= 50:
        return "🟡 Speculative Turnaround"
    
    if 50 <= score < 68:
        return "🟡 Hold / Fairly Valued"
    
    if score < 50 and dcf_upside < -10:
        return "🔴 Strong Avoid"
    
    return "🔴 Avoid"

# ==============================================================================
# 4. BENUTZEROBERFLÄCHE (STREAMLIT APP)
# ==============================================================================

st.title("📈 Quant-Aktien-Screener V11.0")
st.markdown("Multi-Faktor Quant Modell mit Peter Lynch Metriken, gepuffertem DCF & Turnaround-Erkennung.")

# Sidebar Steuerungsbereich
st.sidebar.header("⚙️ Screener Einstellungen")
default_tickers = "MSFT, AAPL, NVDA, GOOGL, AMZN, TSMC, ASML, JNJ, PFE, KO, PG, NKE, V, MA, BAC"
user_input = st.sidebar.text_area("Aktien Ticker (kommagetrennt):", default_tickers, height=120)
min_score_filter = st.sidebar.slider("Mindest-Gesamtscore:", 0, 90, 50, step=5)

if st.sidebar.button("🔄 Daten aktualisieren"):
    st.cache_data.clear()

# Datenverarbeitung
tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]

if tickers:
    data_list = []
    progress_bar = st.progress(0)
    
    for idx, t in enumerate(tickers):
        raw_data = fetch_stock_data(t)
        if raw_data:
            scores = calculate_quant_score_v2(raw_data)
            raw_data.update(scores)
            raw_data["Recommendation"] = get_advanced_recommendation(raw_data)
            data_list.append(raw_data)
        progress_bar.progress((idx + 1) / len(tickers))
    
    progress_bar.empty()

    if data_list:
        df = pd.DataFrame(data_list)
        df_filtered = df[df["Total_Score"] >= min_score_filter].sort_values(by="Total_Score", ascending=False)

        # Tabellen-Vorbereitung
        display_cols = [
            "Ticker", "Name", "Total_Score", "Recommendation", "Price", 
            "DCF_Fair_Value", "DCF_Upside_%", "PEG_Ratio", "ROIC", "RSI"
        ]
        df_display = df_filtered[display_cols].copy()

        st.subheader(f"📊 Screening-Ergebnisse ({len(df_display)} Treffer)")

        # DYNAMISCHE HÖHENBERECHNUNG (Kein vertikaler Scrollbalken mehr)
        calc_height = int((len(df_display) + 1) * 35.5 + 38)

        st.dataframe(
            df_display,
            use_container_width=True,
            height=calc_height,
            column_config={
                "Total_Score": st.column_config.ProgressColumn(
                    "Gesamt-Score",
                    help="Quant-Wert von 0 bis 100",
                    format="%d",
                    min_value=0,
                    max_value=100,
                ),
                "Price": st.column_config.NumberColumn("Kurs", format="$%.2f"),
                "DCF_Fair_Value": st.column_config.NumberColumn("Fair Value (DCF)", format="$%.2f"),
                "DCF_Upside_%": st.column_config.NumberColumn("DCF Upside", format="%.1f%%"),
                "PEG_Ratio": st.column_config.NumberColumn("PEG Ratio", format="%.2f"),
                "ROIC": st.column_config.NumberColumn("ROIC", format="%.1f%%"),
                "RSI": st.column_config.NumberColumn("RSI (14)", format="%.0f"),
            }
        )

        # ==============================================================================
        # 5. DETAILANALYSE FÜR EINZELNE AKTIEN
        # ==============================================================================
        st.divider()
        st.subheader("🔍 Einzelwert-Detailanalyse")

        selected_ticker = st.selectbox("Aktie zur Detailanalyse auswählen:", df_filtered["Ticker"].tolist())
        
        if selected_ticker:
            stock = df_filtered[df_filtered["Ticker"] == selected_ticker].iloc[0]

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"### {stock['Name']} ({stock['Ticker']})")
                st.metric("Gesamt Quant-Score", f"{stock['Total_Score']} / 100", delta=stock['Recommendation'])
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Aktueller Kurs", f"${stock['Price']:.2f}")
                m2.metric("DCF Fair Value", f"${stock['DCF_Fair_Value']:.2f}", f"{stock['DCF_Upside_%']:.1f}% Upside")
                m3.metric("Analysten-Ziel", f"${stock['Target_Price']:.2f}", f"{stock['Analyst_Upside_%']:.1f}% Upside")

                m4, m5, m6 = st.columns(3)
                m4.metric("PEG Ratio", f"{stock['PEG_Ratio']:.2f}")
                m5.metric("ROIC", f"{stock['ROIC']*100:.1f}%")
                m6.metric("Net Debt / EBITDA", f"{stock['Net_Debt_EBITDA']:.2f}x")

            with col2:
                # Radar-Chart für Sub-Scores
                categories = ['Qualität', 'Bewertung', 'Fair Value', 'Gesundheit', 'Technik']
                values = [
                    stock['Quality_Score'],
                    stock['Valuation_Score'],
                    stock['FairValue_Score'],
                    stock['Risk_Score'],
                    stock['Tech_Score']
                ]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=stock['Ticker'],
                    line_color='#00CC96'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                    title="Quant Sub-Score Profil (0-100)",
                    height=300,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Keine Aktien gefunden oder Laden fehlgeschlagen.")
