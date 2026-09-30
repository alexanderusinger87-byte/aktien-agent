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
    page_title="Aktien-Screener V9",
    page_icon="📈",
    layout="wide"
)

SECTOR_WEIGHTS = {
    'Financial Services': {
        'valuation': 0.35,
        'quality': 0.35,
        'risk': 0.20,
        'tech': 0.10
    },
    'Technology': {
        'valuation': 0.20,
        'quality': 0.40,
        'risk': 0.15,
        'tech': 0.25
    },
    'Default': {
        'valuation': 0.25,
        'quality': 0.35,
        'risk': 0.20,
        'tech': 0.20
    }
}

# ==============================================================================
# HELPER FUNCTIONS & DATA EXTRACTION
# ==============================================================================

def safe_get(dictionary, key, default=np.nan):
    val = dictionary.get(key, default)
    return default if val is None else val

def clean_percentage(val):
    if pd.isna(val):
        return np.nan
    if abs(val) > 2.0:
        return val / 100.0
    return val

def get_recommendation(score):
    """Ermittelt die Kaufempfehlung anhand des Gesamtscores."""
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
        if hist.empty or len(hist) < 100:
            return None

        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow

        return {
            'info': info,
            'hist': hist,
            'financials': financials,
            'balance_sheet': balance_sheet,
            'cashflow': cashflow
        }
    except Exception:
        return None

# ==============================================================================
# CORE FINANCIAL METRICS CALCULATOR (V9)
# ==============================================================================

def calculate_advanced_metrics(data):
    info = data['info']
    hist = data['hist']
    fin = data['financials']
    bs = data['balance_sheet']
    cf = data['cashflow']

    metrics = {}
    sector = safe_get(info, 'sector', 'Default')
    metrics['sector'] = sector
    metrics['name'] = safe_get(info, 'shortName', safe_get(info, 'symbol'))

    # 1. Preis & Momentum (Exakte 6M)
    close = hist['Close']
    current_price = close.iloc[-1]
    metrics['current_price'] = current_price
    
    cutoff_6m = datetime.now() - timedelta(days=180)
    hist_6m = close[close.index >= cutoff_6m.strftime('%Y-%m-%d')]
    price_6m_ago = hist_6m.iloc[0] if not hist_6m.empty else close.iloc[0]
    
    metrics['perf_6m'] = (current_price - price_6m_ago) / price_6m_ago
    metrics['sma_200'] = close.rolling(200).mean().iloc[-1]
    metrics['above_sma200'] = current_price > metrics['sma_200']
    metrics['dist_52w_high'] = (current_price - close.max()) / close.max()

    # 2. Bewertungskennzahlen
    metrics['pe_forward'] = safe_get(info, 'forwardPE')
    metrics['peg_ratio'] = safe_get(info, 'pegRatio')
    metrics['pb_ratio'] = safe_get(info, 'priceToBook')
    
    fcf = safe_get(info, 'freeCashflow')
    market_cap = safe_get(info, 'marketCap')
    if pd.isna(fcf) and not cf.empty:
        try:
            op_cf = cf.loc['Operating Cash Flow'].iloc[0]
            capex = cf.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cf.index else 0
            fcf = op_cf + capex
        except Exception:
            fcf = np.nan
            
    metrics['fcf_yield'] = (fcf / market_cap) if (fcf and market_cap) else np.nan

    # 3. Qualität & Rentabilität
    metrics['roe'] = clean_percentage(safe_get(info, 'returnOnEquity'))
    metrics['eps_growth_5y'] = clean_percentage(safe_get(info, 'earningsGrowth'))
    metrics['payout_ratio'] = clean_percentage(safe_get(info, 'payoutRatio'))

    try:
        ebit = fin.loc['EBIT'].iloc[0] if 'EBIT' in fin.index else fin.loc['Operating Income'].iloc[0]
        inc_tax = fin.loc['Tax Provision'].iloc[0] if 'Tax Provision' in fin.index else 0
        pre_tax = fin.loc['Pretax Income'].iloc[0] if 'Pretax Income' in fin.index else 1
        
        tax_rate = inc_tax / pre_tax if pre_tax > 0 else 0.21
        tax_rate = max(0.15, min(tax_rate, 0.35))
        
        nopat = ebit * (1 - tax_rate)
        
        total_assets = bs.loc['Total Assets'].iloc[0]
        curr_liab = bs.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in bs.index else 0
        cash = bs.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in bs.index else 0
        
        invested_capital = total_assets - curr_liab - cash
        metrics['roic'] = nopat / invested_capital if invested_capital > 0 else np.nan
    except Exception:
        metrics['roic'] = np.nan

    try:
        gross_profit = fin.loc['Gross Profit'].iloc[0]
        total_rev = fin.loc['Total Revenue'].iloc[0]
        metrics['gross_margin'] = gross_profit / total_rev
    except Exception:
        metrics['gross_margin'] = safe_get(info, 'grossMargins')

    # 4. Risikokennzahlen
    try:
        tot_debt = bs.loc['Total Debt'].iloc[0] if 'Total Debt' in bs.index else 0
        cash_eq = bs.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in bs.index else 0
        net_debt = tot_debt - cash_eq
        
        ebitda = fin.loc['Normalized EBITDA'].iloc[0] if 'Normalized EBITDA' in fin.index else fin.loc['EBITDA'].iloc[0]
        metrics['net_debt_ebitda'] = (net_debt / ebitda) if ebitda > 0 else 99.0
    except Exception:
        metrics['net_debt_ebitda'] = np.nan

    metrics['current_ratio'] = safe_get(info, 'currentRatio')

    return metrics

# ==============================================================================
# SCORING ENGINE (SECTOR-AWARE V9)
# ==============================================================================

def score_stock_v9(metrics):
    if not metrics:
        return 0, {}

    sector = metrics.get('sector', 'Default')
    weights = SECTOR_WEIGHTS.get(sector, SECTOR_WEIGHTS['Default'])

    scores = {}

    # 1. Valuation
    v_scores = []
    pe = metrics.get('pe_forward')
    if pd.notna(pe) and pe > 0:
        v_scores.append(np.interp(pe, [8, 15, 25, 40], [100, 80, 40, 0]))
        
    peg = metrics.get('peg_ratio')
    if pd.notna(peg) and peg > 0:
        v_scores.append(np.interp(peg, [0.5, 1.0, 1.5, 2.5], [100, 85, 50, 0]))

    scores['valuation'] = np.mean(v_scores) if v_scores else 50.0

    # 2. Quality
    q_scores = []
    roic = metrics.get('roic')
    if pd.notna(roic):
        q_scores.append(np.interp(roic, [0.05, 0.12, 0.20, 0.35], [20, 60, 90, 100]))

    gm = metrics.get('gross_margin')
    if pd.notna(gm):
        q_scores.append(np.interp(gm, [0.15, 0.35, 0.55, 0.75], [20, 50, 80, 100]))

    fcf_y = metrics.get('fcf_yield')
    if pd.notna(fcf_y):
        q_scores.append(np.interp(fcf_y, [0.01, 0.04, 0.07, 0.12], [20, 60, 90, 100]))

    scores['quality'] = np.mean(q_scores) if q_scores else 50.0

    # 3. Risk
    r_scores = []
    if sector != 'Financial Services':
        nd_ebitda = metrics.get('net_debt_ebitda')
        if pd.notna(nd_ebitda):
            r_scores.append(np.interp(nd_ebitda, [0.0, 1.5, 3.0, 5.0], [100, 80, 40, 0]))
            
        cr = metrics.get('current_ratio')
        if pd.notna(cr):
            r_scores.append(np.interp(cr, [0.8, 1.2, 2.0, 3.5], [20, 70, 100, 80]))
    else:
        pb = metrics.get('pb_ratio')
        if pd.notna(pb):
            r_scores.append(np.interp(pb, [0.7, 1.0, 1.5, 2.5], [100, 80, 50, 0]))

    scores['risk'] = np.mean(r_scores) if r_scores else 50.0

    # 4. Technical
    t_scores = []
    if metrics.get('above_sma200', False):
        t_scores.append(80)
    else:
        t_scores.append(20)

    perf_6m = metrics.get('perf_6m')
    if pd.notna(perf_6m):
        t_scores.append(np.interp(perf_6m, [-0.20, 0.0, 0.15, 0.40], [10, 40, 75, 100]))

    scores['tech'] = np.mean(t_scores) if t_scores else 50.0

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
    st.title("📊 Quant-Aktien-Screener V9")
    st.caption("Sektor-adaptives Quant-Scoring basierend auf Valuation, Quality, Risk & Momentum")

    # Sidebar: Ticker-Eingabe & Filter
    st.sidebar.header("⚙️ Konfiguration")
    default_tickers = "NVDA, MSFT, AAPL, GOOGL, AMZN, TTE.PA, ING, PFE, KO, NKE"
    ticker_input = st.sidebar.text_area(
        "Aktien Ticker (kommagetrennt):", 
        value=default_tickers, 
        height=100
    )
    
    min_score = st.sidebar.slider("Mindest-Gesamtscore", 0, 100, 50)

    # Button zum Leeren des Caches (Erzwingt frische Live-Daten)
    if st.sidebar.button("🔄 Live-Daten neu laden"):
        st.cache_data.clear()
        st.rerun()

    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    if st.sidebar.button("🚀 Screening starten", type="primary") or "results" not in st.session_state:
        results = []
        progress_bar = st.progress(0)
        
        for idx, symbol in enumerate(tickers):
            data = fetch_stock_data(symbol)
            if data:
                metrics = calculate_advanced_metrics(data)
                total_score, sub_scores = score_stock_v9(metrics)
                
                results.append({
                    'Ticker': symbol,
                    'Name': metrics['name'],
                    'Sektor': metrics['sector'],
                    'Kurs': metrics['current_price'],
                    'Gesamtscore': total_score,
                    'Empfehlung': get_recommendation(total_score),
                    'Valuation': round(sub_scores['valuation'], 1),
                    'Quality': round(sub_scores['quality'], 1),
                    'Risk': round(sub_scores['risk'], 1),
                    'Tech': round(sub_scores['tech'], 1),
                    'KGV (Fwd)': metrics['pe_forward'],
                    'PEG': metrics['peg_ratio'],
                    'ROIC': metrics['roic'],
                    'Gross Margin': metrics['gross_margin'],
                    'metrics_raw': metrics,
                    'sub_scores': sub_scores
                })
            progress_bar.progress((idx + 1) / len(tickers))
        
        progress_bar.empty()
        st.session_state["results"] = pd.DataFrame(results)

    df_results = st.session_state.get("results", pd.DataFrame())

    if not df_results.empty:
        filtered_df = df_results[df_results['Gesamtscore'] >= min_score].sort_values(by="Gesamtscore", ascending=False)

        st.subheader("🏆 Screener Ergebnisse")
        
        display_columns = ['Ticker', 'Name', 'Sektor', 'Gesamtscore', 'Empfehlung', 'Valuation', 'Quality', 'Risk', 'Tech', 'KGV (Fwd)', 'PEG']
        
        # Farbskala (Heatmap) auf den Gesamtscore anwenden
        styled_df = filtered_df[display_columns].style.background_gradient(
            subset=['Gesamtscore'],
            cmap='RdYlGn',
            vmin=0,
            vmax=100
        )

        st.dataframe(
            styled_df,
            column_config={
                "Gesamtscore": st.column_config.NumberColumn(format="%.1f"),
                "Empfehlung": st.column_config.TextColumn("Kaufempfehlung"),
                "Valuation": st.column_config.NumberColumn(format="%.1f"),
                "Quality": st.column_config.NumberColumn(format="%.1f"),
                "Risk": st.column_config.NumberColumn(format="%.1f"),
                "Tech": st.column_config.NumberColumn(format="%.1f"),
                "KGV (Fwd)": st.column_config.NumberColumn(format="%.2f"),
                "PEG": st.column_config.NumberColumn(format="%.2f"),
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
                st.write(f"**Aktueller Kurs:** {stock_data['Kurs']:.2f} $")
                st.write(f"**Einstufung:** {stock_data['Empfehlung']}")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Gesamtscore", f"{stock_data['Gesamtscore']} / 100")
                m2.metric("KGV (Fwd)", f"{stock_data['KGV (Fwd)']:.2f}" if pd.notna(stock_data['KGV (Fwd)']) else "N/A")
                m3.metric("PEG", f"{stock_data['PEG']:.2f}" if pd.notna(stock_data['PEG']) else "N/A")

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
