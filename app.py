import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ==============================================================================
# STREAMLIT PAGE CONFIG & SECTOR PROFILES
# ==============================================================================

st.set_page_config(
    page_title="Aktien-Screener V9.5 (Robust Blend)",
    page_icon="📈",
    layout="wide"
)

SECTOR_WEIGHTS = {
    'Financial Services': {'valuation': 0.35, 'quality': 0.35, 'risk': 0.20, 'tech': 0.10},
    'Technology': {'valuation': 0.20, 'quality': 0.40, 'risk': 0.15, 'tech': 0.25},
    'Healthcare': {'valuation': 0.25, 'quality': 0.40, 'risk': 0.20, 'tech': 0.15},
    'Consumer Cyclical': {'valuation': 0.30, 'quality': 0.30, 'risk': 0.20, 'tech': 0.20},
    'Consumer Defensive': {'valuation': 0.25, 'quality': 0.40, 'risk': 0.25, 'tech': 0.10},
    'Default': {'valuation': 0.25, 'quality': 0.35, 'risk': 0.20, 'tech': 0.20}
}

# ==============================================================================
# HELPER FUNCTIONS & DATA EXTRACTION
# ==============================================================================

def safe_get(dictionary, key, default=np.nan):
    if not isinstance(dictionary, dict):
        return default
    val = dictionary.get(key, default)
    return default if val is None else val

def clean_percentage(val):
    if pd.isna(val):
        return np.nan
    if abs(val) > 2.0:
        return val / 100.0
    return val

def get_turnaround_status(metrics):
    dist_high = metrics.get('dist_52w_high', 0)
    perf_1m = metrics.get('perf_1m', 0)
    pe = metrics.get('pe_effective', np.nan)

    is_beaten_down = dist_high <= -0.20
    is_rebounding = perf_1m >= 0.02
    is_reasonably_priced = pd.isna(pe) or pe < 25

    if is_beaten_down and is_rebounding and is_reasonably_priced:
        return "🚀 Aktiver Turnaround"
    elif is_beaten_down and not is_rebounding:
        return "⚠️ Fallendes Messer"
    elif is_beaten_down and (-0.02 <= perf_1m < 0.02):
        return "⏱️️ Bodenbildung"
    else:
        return "🛡️ Trend / Normal"

def get_combined_recommendation(score, turnaround_status):
    if "Fallendes Messer" in turnaround_status:
        if score >= 60:
            return "🟠 Halten (Vorsicht: Fallendes Messer)"
        else:
            return "🔴 Verkaufen / Fallendes Messer"
            
    elif "Aktiver Turnaround" in turnaround_status:
        if score >= 60:
            return "🟢 Starker Turnaround-Kauf"
        elif score >= 45:
            return "🟡 Spekulativer Turnaround"
        else:
            return "🟠 Turnaround-Spekulation (Risiko)"
            
    elif "Bodenbildung" in turnaround_status:
        if score >= 60:
            return "🟢 Kauf (Bodenbildung)"
        elif score >= 45:
            return "🟡 Basisbildung abwarten"
        else:
            return "🟠 Halten"
            
    else:
        if score >= 75:
            return "🟢 Starker Kauf"
        elif score >= 60:
            return "🟡 Kauf"
        elif score >= 45:
            return "🟠 Halten"
        else:
            return "🔴 Verkaufen"

@st.cache_data(ttl=3600*12)
def fetch_stock_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < 50:
            return None

        return {
            'info': info,
            'hist': hist,
            'financials': ticker.financials,
            'balance_sheet': ticker.balance_sheet,
            'cashflow': ticker.cashflow
        }
    except Exception:
        return None

# ==============================================================================
# CORE FINANCIAL METRICS CALCULATOR
# ==============================================================================

def calculate_advanced_metrics(data, symbol):
    info = data['info']
    hist = data['hist']
    fin = data['financials']
    bs = data['balance_sheet']
    cf = data['cashflow']

    metrics = {}
    sector = safe_get(info, 'sector', 'Default')
    if not sector or pd.isna(sector):
        sector = 'Default'
        
    metrics['sector'] = sector
    
    # Robuste Namensermittlung
    name = safe_get(info, 'shortName')
    if pd.isna(name):
        name = safe_get(info, 'longName', symbol)
    metrics['name'] = name

    close = hist['Close']
    current_price = close.iloc[-1]
    metrics['current_price'] = current_price
    
    max_52w = close.max()
    metrics['dist_52w_high'] = (current_price - max_52w) / max_52w if max_52w > 0 else 0

    cutoff_1m = datetime.now() - timedelta(days=30)
    hist_1m = close[close.index >= cutoff_1m.strftime('%Y-%m-%d')]
    price_1m_ago = hist_1m.iloc[0] if not hist_1m.empty else close.iloc[0]
    metrics['perf_1m'] = (current_price - price_1m_ago) / price_1m_ago

    cutoff_6m = datetime.now() - timedelta(days=180)
    hist_6m = close[close.index >= cutoff_6m.strftime('%Y-%m-%d')]
    price_6m_ago = hist_6m.iloc[0] if not hist_6m.empty else close.iloc[0]
    
    metrics['perf_6m'] = (current_price - price_6m_ago) / price_6m_ago
    metrics['sma_200'] = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else close.mean()
    metrics['above_sma200'] = current_price > metrics['sma_200']

    # KGV Robustheits-Fallback (Forward -> Trailing -> aus Preis & EPS schätzen)
    pe_fwd = safe_get(info, 'forwardPE')
    pe_trail = safe_get(info, 'trailingPE')
    eps = safe_get(info, 'trailingEps')
    
    pe_effective = pe_fwd
    if pd.isna(pe_effective) or pe_effective <= 0:
        pe_effective = pe_trail
    if (pd.isna(pe_effective) or pe_effective <= 0) and pd.notna(eps) and eps > 0:
        pe_effective = current_price / eps

    metrics['pe_forward'] = pe_effective
    metrics['pe_effective'] = pe_effective
    metrics['peg_ratio'] = safe_get(info, 'pegRatio')
    metrics['pb_ratio'] = safe_get(info, 'priceToBook')
    
    fcf = np.nan
    market_cap = safe_get(info, 'marketCap')
    if not cf.empty and 'Operating Cash Flow' in cf.index:
        try:
            op_cf = cf.loc['Operating Cash Flow'].iloc[0]
            capex = cf.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cf.index else 0
            fcf = op_cf + capex
        except Exception:
            fcf = np.nan
    if pd.isna(fcf):
        fcf = safe_get(info, 'freeCashflow')
        
    metrics['fcf_yield'] = (fcf / market_cap) if (pd.notna(fcf) and pd.notna(market_cap) and market_cap > 0) else np.nan

    metrics['roe'] = clean_percentage(safe_get(info, 'returnOnEquity'))
    metrics['gross_margin'] = safe_get(info, 'grossMargins')

    # ROIC Berechnung
    if sector != 'Financial Services' and not fin.empty and not bs.empty:
        try:
            ebit_key = 'EBIT' if 'EBIT' in fin.index else ('Operating Income' if 'Operating Income' in fin.index else None)
            if ebit_key:
                ebit = fin.loc[ebit_key].iloc[0]
                inc_tax = fin.loc['Tax Provision'].iloc[0] if 'Tax Provision' in fin.index else 0
                pre_tax = fin.loc['Pretax Income'].iloc[0] if 'Pretax Income' in fin.index else 1
                tax_rate = max(0.15, min(inc_tax / pre_tax if pre_tax > 0 else 0.21, 0.35))
                nopat = ebit * (1 - tax_rate)
                
                total_assets = bs.loc['Total Assets'].iloc[0] if 'Total Assets' in bs.index else np.nan
                curr_liab = bs.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in bs.index else 0
                cash = bs.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in bs.index else 0
                
                if pd.notna(total_assets):
                    invested_capital = total_assets - curr_liab - cash
                    metrics['roic'] = nopat / invested_capital if invested_capital > 0 else np.nan
                else:
                    metrics['roic'] = np.nan
            else:
                metrics['roic'] = np.nan
        except Exception:
            metrics['roic'] = np.nan
    else:
        metrics['roic'] = np.nan

    # Net Debt / EBITDA
    if sector != 'Financial Services' and not fin.empty and not bs.empty:
        try:
            tot_debt = bs.loc['Total Debt'].iloc[0] if 'Total Debt' in bs.index else 0
            cash_eq = bs.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in bs.index else 0
            net_debt = tot_debt - cash_eq
            ebitda_key = 'Normalized EBITDA' if 'Normalized EBITDA' in fin.index else ('EBITDA' if 'EBITDA' in fin.index else None)
            if ebitda_key:
                ebitda = fin.loc[ebitda_key].iloc[0]
                metrics['net_debt_ebitda'] = (net_debt / ebitda) if ebitda > 0 else np.nan
            else:
                metrics['net_debt_ebitda'] = np.nan
        except Exception:
            metrics['net_debt_ebitda'] = np.nan
    else:
        metrics['net_debt_ebitda'] = np.nan

    metrics['current_ratio'] = safe_get(info, 'currentRatio')
    metrics['turnaround_status'] = get_turnaround_status(metrics)

    return metrics

# ==============================================================================
# SCORING ENGINE (ROBUST)
# ==============================================================================

def score_stock_v9(metrics):
    if not metrics:
        return 0, {'valuation': 50, 'quality': 50, 'risk': 50, 'tech': 50}

    sector = metrics.get('sector', 'Default')
    weights = SECTOR_WEIGHTS.get(sector, SECTOR_WEIGHTS['Default'])
    scores = {}

    # Valuation Score
    v_scores = []
    pe = metrics.get('pe_effective')
    if pd.notna(pe) and pe > 0:
        v_scores.append(np.interp(pe, [3, 10, 18, 30, 60], [100, 90, 75, 30, 0]))
    peg = metrics.get('peg_ratio')
    if pd.notna(peg) and peg > 0:
        v_scores.append(np.interp(peg, [0.3, 0.8, 1.3, 2.5], [100, 85, 50, 0]))
    
    scores['valuation'] = np.mean(v_scores) if v_scores else 55.0

    # Quality Score
    q_scores = []
    if sector == 'Financial Services':
        roe = metrics.get('roe')
        if pd.notna(roe):
            q_scores.append(np.interp(roe, [0.03, 0.08, 0.15, 0.22], [20, 60, 85, 100]))
    else:
        roic = metrics.get('roic')
        if pd.notna(roic):
            q_scores.append(np.interp(roic, [0.03, 0.10, 0.18, 0.30], [20, 60, 85, 100]))
        gm = metrics.get('gross_margin')
        if pd.notna(gm):
            q_scores.append(np.interp(gm, [0.10, 0.30, 0.50, 0.70], [20, 50, 80, 100]))
            
    fcf_y = metrics.get('fcf_yield')
    if pd.notna(fcf_y):
        q_scores.append(np.interp(fcf_y, [-0.02, 0.02, 0.05, 0.09], [10, 45, 80, 100]))
        
    scores['quality'] = np.mean(q_scores) if q_scores else 55.0

    # Risk Score
    r_scores = []
    if sector == 'Financial Services':
        pb = metrics.get('pb_ratio')
        if pd.notna(pb):
            r_scores.append(np.interp(pb, [0.5, 0.9, 1.5, 2.4], [100, 85, 50, 0]))
    else:
        nd_ebitda = metrics.get('net_debt_ebitda')
        if pd.notna(nd_ebitda):
            r_scores.append(np.interp(nd_ebitda, [-1.5, 0.0, 2.0, 4.0], [100, 90, 60, 10]))
        cr = metrics.get('current_ratio')
        if pd.notna(cr):
            r_scores.append(np.interp(cr, [0.5, 1.0, 1.6, 2.5], [10, 60, 95, 80]))
            
    scores['risk'] = np.mean(r_scores) if r_scores else 55.0

    # Tech / Momentum Score
    t_scores = [80 if metrics.get('above_sma200', False) else 30]
    perf_6m = metrics.get('perf_6m')
    if pd.notna(perf_6m):
        t_scores.append(np.interp(perf_6m, [-0.30, 0.0, 0.15, 0.40], [0, 45, 75, 100]))
        
    scores['tech'] = np.mean(t_scores) if t_scores else 55.0

    total_score = (
        scores['valuation'] * weights['valuation'] +
        scores['quality'] * weights['quality'] +
        scores['risk'] * weights['risk'] +
        scores['tech'] * weights['tech']
    )
    return round(total_score, 1), scores

# ==============================================================================
# STREAMLIT UI & DASHBOARD
# ==============================================================================

def main():
    st.title("📊 Quant-Aktien-Screener V9.5")
    st.caption("Sektor-adaptives Quant-Scoring mit robuster Datenextraktion")

    st.sidebar.header("⚙️️ Konfiguration")
    default_tickers = "BMW.DE, NVDA, MSFT, AAPL, GOOGL, AMZN, TTE.PA, ING, PFE, KO, NKE"
    ticker_input = st.sidebar.text_area(
        "Aktien Ticker (kommagetrennt):", 
        value=default_tickers, 
        height=100
    )
    
    min_score = st.sidebar.slider("Mindest-Gesamtscore", 0, 100, 0)

    if st.sidebar.button("🔄 Live-Daten neu laden"):
        st.cache_data.clear()
        st.rerun()

    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    if st.sidebar.button("🚀 Screening starten", type="primary") or "results_v9" not in st.session_state:
        results = []
        progress_bar = st.progress(0)
        
        for idx, symbol in enumerate(tickers):
            data = fetch_stock_data(symbol)
            if data:
                metrics = calculate_advanced_metrics(data, symbol)
                total_score, sub_scores = score_stock_v9(metrics)
                turnaround_status = metrics['turnaround_status']
                combined_rec = get_combined_recommendation(total_score, turnaround_status)
                
                results.append({
                    'Ticker': symbol,
                    'Name': metrics['name'],
                    'Sektor': metrics['sector'],
                    'Kurs': metrics['current_price'],
                    'Gesamtscore': total_score,
                    'Empfehlung': combined_rec,
                    'Turnaround Status': turnaround_status,
                    'Valuation': round(sub_scores['valuation'], 1),
                    'Quality': round(sub_scores['quality'], 1),
                    'Risk': round(sub_scores['risk'], 1),
                    'Tech': round(sub_scores['tech'], 1),
                    'KGV (Eff)': metrics['pe_effective'],
                    'sub_scores': sub_scores
                })
            progress_bar.progress((idx + 1) / len(tickers))
        
        progress_bar.empty()
        st.session_state["results_v9"] = pd.DataFrame(results)

    df_results = st.session_state.get("results_v9", pd.DataFrame())

    if not df_results.empty:
        filtered_df = df_results[df_results['Gesamtscore'] >= min_score].sort_values(by="Gesamtscore", ascending=False)

        st.subheader("🏆 Screener Ergebnisse")
        
        display_columns = ['Ticker', 'Name', 'Sektor', 'Gesamtscore', 'Empfehlung', 'Turnaround Status', 'Valuation', 'Quality', 'Risk', 'Tech', 'KGV (Eff)']
        
        st.dataframe(
            filtered_df[display_columns],
            column_config={
                "Gesamtscore": st.column_config.NumberColumn(format="%.1f"),
                "Empfehlung": st.column_config.TextColumn("Kombinierte Empfehlung"),
                "Turnaround Status": st.column_config.TextColumn("🔄 Turnaround Status"),
                "Valuation": st.column_config.NumberColumn(format="%.1f"),
                "Quality": st.column_config.NumberColumn(format="%.1f"),
                "Risk": st.column_config.NumberColumn(format="%.1f"),
                "Tech": st.column_config.NumberColumn(format="%.1f"),
                "KGV (Eff)": st.column_config.NumberColumn(format="%.2f"),
            },
            hide_index=True,
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("🔍 Einzelwert-Analyse")
        
        selected_ticker = st.selectbox("Wähle eine Aktie für das Radar-Profil:", filtered_df['Ticker'].tolist())
        
        if selected_ticker:
            stock_data = filtered_df[filtered_df['Ticker'] == selected_ticker].iloc[0]
            sub = stock_data['sub_scores']

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"### **{stock_data['Name']} ({stock_data['Ticker']})**")
                st.write(f"**Sektor:** {stock_data['Sektor']}")
                st.write(f"**Aktueller Kurs:** {stock_data['Kurs']:.2f}")
                st.write(f"**Empfehlung:** {stock_data['Empfehlung']}")
                st.write(f"**Turnaround Status:** {stock_data['Turnaround Status']}")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Gesamtscore", f"{stock_data['Gesamtscore']} / 100")
                m2.metric("KGV (Eff)", f"{stock_data['KGV (Eff)']:.2f}" if pd.notna(stock_data['KGV (Eff)']) else "N/A")
                m3.metric("Valuation-Teilnote", f"{stock_data['Valuation']:.1f}")

            with col2:
                categories = ['Valuation', 'Quality', 'Risk', 'Momentum / Tech']
                values = [sub['valuation'], sub['quality'], sub['risk'], sub['tech']]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=stock_data['Ticker']
                ))

                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                    margin=dict(l=40, r=40, t=40, b=40),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
