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
    page_title="Aktien-Screener V10.0 (Cache Fix)",
    page_icon="📈",
    layout="wide"
)

SECTOR_WEIGHTS = {
    'Financial Services': {'valuation': 0.35, 'quality': 0.35, 'risk': 0.20, 'tech': 0.10},
    'Technology': {'valuation': 0.20, 'quality': 0.40, 'risk': 0.15, 'tech': 0.25},
    'Healthcare': {'valuation': 0.25, 'quality': 0.40, 'risk': 0.20, 'tech': 0.15},
    'Consumer Cyclical': {'valuation': 0.30, 'quality': 0.30, 'risk': 0.20, 'tech': 0.20},
    'Consumer Defensive': {'valuation': 0.25, 'quality': 0.40, 'risk': 0.25, 'tech': 0.10},
    'Energy': {'valuation': 0.30, 'quality': 0.30, 'risk': 0.25, 'tech': 0.15},
    'Default': {'valuation': 0.25, 'quality': 0.35, 'risk': 0.20, 'tech': 0.20}
}

FALLBACK_SECTORS = {
    'NVDA': 'Technology', 'MSFT': 'Technology', 'AAPL': 'Technology',
    'GOOGL': 'Technology', 'AMZN': 'Consumer Cyclical', 'ASML': 'Technology',
    'TSM': 'Technology', 'BMW.DE': 'Consumer Cyclical', 'TTE.PA': 'Energy',
    'ING': 'Financial Services', 'PFE': 'Healthcare', 'KO': 'Consumer Defensive',
    'NKE': 'Consumer Cyclical', 'MRVL': 'Technology', 'MU': 'Technology',
    '000660.KS': 'Technology'
}

# ==============================================================================
# ROBUST HELPER FUNCTIONS
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

def get_row(df, possible_keys):
    if df is None or df.empty:
        return None
    for key in possible_keys:
        if key in df.index:
            try:
                val = df.loc[key].iloc[0]
                if pd.notna(val):
                    return val
            except Exception:
                continue
    return None

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
        return "⏱ Bodenbildung"
    else:
        return "🛡️️ Trend / Normal"

def get_combined_recommendation(score, turnaround_status):
    if pd.isna(score):
        return "🔴 Daten unvollständig"
    if "Fallendes Messer" in turnaround_status:
        return "🟠 Halten (Vorsicht)" if score >= 60 else "🔴 Verkaufen / Fallendes Messer"
    elif "Aktiver Turnaround" in turnaround_status:
        return "🟢 Starker Turnaround-Kauf" if score >= 60 else "🟡 Spekulativer Turnaround"
    elif "Bodenbildung" in turnaround_status:
        return "🟢 Kauf (Bodenbildung)" if score >= 60 else "🟠 Halten"
    else:
        if score >= 75:
            return "🟢 Starker Kauf"
        elif score >= 60:
            return "🟡 Kauf"
        elif score >= 45:
            return "🟠 Halten"
        else:
            return "🔴 Verkaufen"

# SICHERER DOWNLOAD OHNE STREAMLIT-CACHE-FEHLER AUF OBJEKTEN
def fetch_stock_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        try:
            info = dict(ticker.info)
        except Exception:
            info = {}
            
        try:
            fast_info = dict(ticker.fast_info)
        except Exception:
            fast_info = {}

        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < 30:
            return None
            
        return {
            'info': info,
            'fast_info': fast_info,
            'hist': hist,
            'financials': ticker.financials,
            'balance_sheet': ticker.balance_sheet,
            'cashflow': ticker.cashflow
        }
    except Exception:
        return None

# ==============================================================================
# ADVANCED METRICS & DATA QUALITY TRACKER
# ==============================================================================

def calculate_advanced_metrics(data, symbol):
    info = data['info']
    fast_info = data['fast_info']
    hist = data['hist']
    fin = data['financials']
    bs = data['balance_sheet']
    cf = data['cashflow']

    metrics = {}
    
    sector = safe_get(info, 'sector')
    if pd.isna(sector) or sector == 'Default':
        sector = FALLBACK_SECTORS.get(symbol, 'Default')
    metrics['sector'] = sector

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

    market_cap = np.nan
    try:
        market_cap = fast_info.get('market_cap', np.nan)
    except Exception:
        pass
    if pd.isna(market_cap):
        market_cap = safe_get(info, 'marketCap')
    shares = safe_get(info, 'sharesOutstanding')
    if pd.isna(market_cap) and pd.notna(shares) and shares > 0:
        market_cap = current_price * shares
    metrics['market_cap'] = market_cap

    net_income = get_row(fin, ['Net Income', 'Net Income Common Stockholders', 'NetIncome', 'Net Income From Continuing Operation'])
    revenue = get_row(fin, ['Total Revenue', 'Operating Revenue', 'Revenue'])
    equity = get_row(bs, ['Stockholders Equity', 'Total Stockholder Equity', 'Total Equity Gross Minority Interest', 'Common Stock Equity'])

    pe_effective = safe_get(info, 'forwardPE')
    if pd.isna(pe_effective) or pe_effective <= 0:
        pe_effective = safe_get(info, 'trailingPE')
    if (pd.isna(pe_effective) or pe_effective <= 0) and pd.notna(market_cap) and pd.notna(net_income) and net_income > 0:
        pe_effective = market_cap / net_income
    metrics['pe_effective'] = pe_effective

    ps_ratio = safe_get(info, 'priceToSalesTrailing12Months')
    if (pd.isna(ps_ratio) or ps_ratio <= 0) and pd.notna(market_cap) and pd.notna(revenue) and revenue > 0:
        ps_ratio = market_cap / revenue
    metrics['ps_ratio'] = ps_ratio

    pb_ratio = safe_get(info, 'priceToBook')
    if (pd.isna(pb_ratio) or pb_ratio <= 0) and pd.notna(market_cap) and pd.notna(equity) and equity > 0:
        pb_ratio = market_cap / equity
    metrics['pb_ratio'] = pb_ratio

    metrics['peg_ratio'] = safe_get(info, 'pegRatio')
    
    op_cf = get_row(cf, ['Operating Cash Flow', 'Total Cash From Operating Activities', 'OperatingCashFlow'])
    capex = get_row(cf, ['Capital Expenditure', 'CapEx', 'CapitalExpenditures', 'Purchase Of Property And Equipment'])
    fcf = np.nan
    if pd.notna(op_cf):
        fcf = op_cf + capex if pd.notna(capex) and capex < 0 else (op_cf - capex if pd.notna(capex) else op_cf * 0.8)
    if pd.isna(fcf):
        fcf = safe_get(info, 'freeCashflow')
    metrics['fcf_yield'] = (fcf / market_cap) if (pd.notna(fcf) and pd.notna(market_cap) and market_cap > 0) else np.nan

    metrics['roe'] = clean_percentage(safe_get(info, 'returnOnEquity'))
    if pd.isna(metrics['roe']) and pd.notna(net_income) and pd.notna(equity) and equity > 0:
        metrics['roe'] = net_income / equity

    metrics['gross_margin'] = safe_get(info, 'grossMargins')

    if sector != 'Financial Services':
        ebit = get_row(fin, ['EBIT', 'Operating Income', 'OperatingIncome', 'Normalized Income'])
        inc_tax = get_row(fin, ['Tax Provision', 'Income Tax Expense'])
        pre_tax = get_row(fin, ['Pretax Income', 'Income Before Tax'])
        total_assets = get_row(bs, ['Total Assets', 'TotalAssets'])
        curr_liab = get_row(bs, ['Current Liabilities', 'Total Current Liabilities', 'CurrentLiabilities'])
        cash = get_row(bs, ['Cash And Cash Equivalents', 'Cash Financial', 'CashAndCashEquivalents'])
        
        if pd.notna(ebit) and pd.notna(total_assets):
            tax_rate = 0.21
            if pd.notna(inc_tax) and pd.notna(pre_tax) and pre_tax > 0:
                tax_rate = max(0.10, min(inc_tax / pre_tax, 0.40))
            nopat = ebit * (1 - tax_rate)
            cl = curr_liab if pd.notna(curr_liab) else 0
            c = cash if pd.notna(cash) else 0
            invested_capital = total_assets - cl - c
            metrics['roic'] = nopat / invested_capital if invested_capital > 0 else np.nan
        else:
            metrics['roic'] = np.nan
    else:
        metrics['roic'] = np.nan

    if sector != 'Financial Services':
        tot_debt = get_row(bs, ['Total Debt', 'TotalDebt', 'Long Term Debt'])
        cash_eq = get_row(bs, ['Cash And Cash Equivalents', 'Cash Financial', 'CashAndCashEquivalents'])
        ebitda = get_row(fin, ['Normalized EBITDA', 'EBITDA', 'Operating Income'])
        td = tot_debt if pd.notna(tot_debt) else 0
        ce = cash_eq if pd.notna(cash_eq) else 0
        net_debt = td - ce
        if pd.notna(ebitda) and ebitda > 0:
            metrics['net_debt_ebitda'] = net_debt / ebitda
        else:
            metrics['net_debt_ebitda'] = np.nan
    else:
        metrics['net_debt_ebitda'] = np.nan

    metrics['current_ratio'] = safe_get(info, 'currentRatio')
    metrics['turnaround_status'] = get_turnaround_status(metrics)

    core_fields = [
        metrics['pe_effective'], metrics['ps_ratio'], metrics['fcf_yield'], 
        metrics['gross_margin'], metrics['market_cap'], metrics['current_ratio']
    ]
    if sector != 'Financial Services':
        core_fields.extend([metrics['roic'], metrics['net_debt_ebitda']])
    else:
        core_fields.extend([metrics['roe'], metrics['pb_ratio']])
        
    valid_count = sum([1 for f in core_fields if pd.notna(f)])
    metrics['data_quality'] = round((valid_count / len(core_fields)) * 100)

    return metrics

# ==============================================================================
# STRICT SCORING ENGINE
# ==============================================================================

def score_stock_v9(metrics):
    if not metrics:
        return np.nan, {'valuation': np.nan, 'quality': np.nan, 'risk': np.nan, 'tech': np.nan}

    sector = metrics.get('sector', 'Default')
    weights = SECTOR_WEIGHTS.get(sector, SECTOR_WEIGHTS['Default'])
    scores = {}

    v_scores = []
    pe = metrics.get('pe_effective')
    if pd.notna(pe) and pe > 0:
        v_scores.append(np.interp(pe, [5, 12, 20, 35, 70], [100, 90, 70, 25, 0]))
    ps = metrics.get('ps_ratio')
    if pd.notna(ps) and ps > 0:
        v_scores.append(np.interp(ps, [0.5, 1.5, 3.5, 7.0], [100, 85, 45, 0]))
    peg = metrics.get('peg_ratio')
    if pd.notna(peg) and peg > 0:
        v_scores.append(np.interp(peg, [0.4, 1.0, 1.8, 3.0], [100, 80, 40, 0]))
    scores['valuation'] = np.mean(v_scores) if v_scores else np.nan

    q_scores = []
    if sector == 'Financial Services':
        roe = metrics.get('roe')
        if pd.notna(roe):
            q_scores.append(np.interp(roe, [0.04, 0.10, 0.18, 0.25], [20, 60, 90, 100]))
    else:
        roic = metrics.get('roic')
        if pd.notna(roic):
            q_scores.append(np.interp(roic, [0.04, 0.10, 0.20, 0.35], [20, 60, 90, 100]))
        gm = metrics.get('gross_margin')
        if pd.notna(gm):
            q_scores.append(np.interp(gm, [0.15, 0.35, 0.55, 0.75], [20, 50, 80, 100]))
    fcf_y = metrics.get('fcf_yield')
    if pd.notna(fcf_y):
        q_scores.append(np.interp(fcf_y, [-0.01, 0.03, 0.06, 0.10], [10, 50, 85, 100]))
    scores['quality'] = np.mean(q_scores) if q_scores else np.nan

    r_scores = []
    if sector == 'Financial Services':
        pb = metrics.get('pb_ratio')
        if pd.notna(pb):
            r_scores.append(np.interp(pb, [0.6, 1.0, 1.6, 2.5], [100, 85, 50, 0]))
    else:
        nd_ebitda = metrics.get('net_debt_ebitda')
        if pd.notna(nd_ebitda):
            r_scores.append(np.interp(nd_ebitda, [-1.0, 0.5, 2.5, 5.0], [100, 90, 55, 10]))
        cr = metrics.get('current_ratio')
        if pd.notna(cr):
            r_scores.append(np.interp(cr, [0.6, 1.0, 1.5, 2.5], [15, 60, 90, 75]))
    scores['risk'] = np.mean(r_scores) if r_scores else np.nan

    t_scores = [85 if metrics.get('above_sma200', False) else 30]
    perf_6m = metrics.get('perf_6m')
    if pd.notna(perf_6m):
        t_scores.append(np.interp(perf_6m, [-0.25, 0.0, 0.15, 0.40], [0, 45, 75, 100]))
    scores['tech'] = np.mean(t_scores) if t_scores else np.nan

    active_weights = weights.copy()
    valid_scores = {}
    for cat in ['valuation', 'quality', 'risk', 'tech']:
        if pd.notna(scores.get(cat)):
            valid_scores[cat] = scores[cat]
        else:
            active_weights[cat] = 0.0

    total_w = sum(active_weights.values())
    if total_w <= 0:
        return np.nan, scores

    norm_weights = {k: v / total_w for k, v in active_weights.items()}
    total_score = sum(valid_scores[cat] * norm_weights[cat] for cat in valid_scores)
    return round(total_score, 1), scores

# ==============================================================================
# STREAMLIT UI & DASHBOARD
# ==============================================================================

def main():
    st.title("📊 Quant-Aktien-Screener V10.0")
    st.caption("Stabiler Live-Screener ohne Serialisierungsfehler")

    st.sidebar.header("⚙️ Konfiguration")
    default_tickers = "BMW.DE, NVDA, MSFT, AAPL, GOOGL, AMZN, TTE.PA, ING, PFE, KO, NKE"
    ticker_input = st.sidebar.text_area(
        "Aktien Ticker (kommagetrennt):", 
        value=default_tickers, 
        height=100
    )
    
    min_score = st.sidebar.slider("Mindest-Gesamtscore", 0, 100, 0)

    if st.sidebar.button("🔄 Cache leeren & neu laden"):
        st.cache_data.clear()
        st.rerun()

    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    if st.sidebar.button("🚀 Screening starten", type="primary") or "results_v10" not in st.session_state:
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
                    'Datenqualität': f"{metrics['data_quality']}%",
                    'Empfehlung': combined_rec,
                    'Turnaround Status': turnaround_status,
                    'Valuation': round(sub_scores['valuation'], 1) if pd.notna(sub_scores['valuation']) else np.nan,
                    'Quality': round(sub_scores['quality'], 1) if pd.notna(sub_scores['quality']) else np.nan,
                    'Risk': round(sub_scores['risk'], 1) if pd.notna(sub_scores['risk']) else np.nan,
                    'Tech': round(sub_scores['tech'], 1) if pd.notna(sub_scores['tech']) else np.nan,
                    'KGV (Eff)': metrics['pe_effective'],
                    'sub_scores': sub_scores
                })
            progress_bar.progress((idx + 1) / len(tickers))
        
        progress_bar.empty()
        st.session_state["results_v10"] = pd.DataFrame(results)

    df_results = st.session_state.get("results_v10", pd.DataFrame())

    if not df_results.empty:
        filtered_df = df_results[(df_results['Gesamtscore'] >= min_score) | (df_results['Gesamtscore'].isna())].sort_values(by="Gesamtscore", ascending=False, na_position='last')

        st.subheader("🏆 Screener Ergebnisse & Datenqualität")
        
        display_columns = ['Ticker', 'Name', 'Sektor', 'Gesamtscore', 'Datenqualität', 'Empfehlung', 'Turnaround Status', 'Valuation', 'Quality', 'Risk', 'Tech', 'KGV (Eff)']
        
        def highlight_missing(row):
            if pd.isna(row['Gesamtscore']) or int(str(row['Datenqualität']).replace('%','')) < 70:
                return ['background-color: rgba(255, 75, 75, 0.15)'] * len(row)
            return [''] * len(row)

        st.dataframe(
            filtered_df[display_columns].style.apply(highlight_missing, axis=1),
            column_config={
                "Gesamtscore": st.column_config.NumberColumn(format="%.1f"),
                "Datenqualität": st.column_config.TextColumn("📊 Datenqualität"),
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
        st.subheader("🔍 Einzelwert-Analyse & Radar")
        
        selected_ticker = st.selectbox("Wähle eine Aktie für das Radar-Profil:", filtered_df['Ticker'].tolist())
        
        if selected_ticker:
            stock_data = filtered_df[filtered_df['Ticker'] == selected_ticker].iloc[0]
            sub = stock_data['sub_scores']

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"### **{stock_data['Name']} ({stock_data['Ticker']})**")
                st.write(f"**Sektor:** {stock_data['Sektor']}")
                st.write(f"**Datenqualität:** {stock_data['Datenqualität']}")
                st.write(f"**Aktueller Kurs:** {stock_data['Kurs']:.2f}")
                st.write(f"**Empfehlung:** {stock_data['Empfehlung']}")
                st.write(f"**Turnaround Status:** {stock_data['Turnaround Status']}")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Gesamtscore", f"{stock_data['Gesamtscore']} / 100" if pd.notna(stock_data['Gesamtscore']) else "N/A")
                m2.metric("KGV (Eff)", f"{stock_data['KGV (Eff)']:.2f}" if pd.notna(stock_data['KGV (Eff)']) else "Fehlt")
                m3.metric("Valuation-Note", f"{stock_data['Valuation']:.1f}" if pd.notna(stock_data['Valuation']) else "Fehlt")

            with col2:
                categories = ['Valuation', 'Quality', 'Risk', 'Momentum / Tech']
                values = [
                    sub.get('valuation', 0) if pd.notna(sub.get('valuation')) else 0,
                    sub.get('quality', 0) if pd.notna(sub.get('quality')) else 0,
                    sub.get('risk', 0) if pd.notna(sub.get('risk')) else 0,
                    sub.get('tech', 0) if pd.notna(sub.get('tech')) else 0
                ]

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
