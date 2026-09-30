import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==============================================================================
# CONFIG & SECTOR PROFILES
# ==============================================================================

# Sektorspezifische Gewichtungen (Summe = 1.0)
SECTOR_WEIGHTS = {
    'Financial Services': {
        'valuation': 0.35,  # Fokus auf KGV / KBV
        'quality': 0.35,    # ROE, Margen, Payout
        'risk': 0.20,       # Eigenkapitalquote (statt Net Debt/EBITDA)
        'tech': 0.10
    },
    'Technology': {
        'valuation': 0.20,  # KGV/PEG weniger strikt
        'quality': 0.40,    # Hohes Umsatz-/EPS-Wachstum, Bruttomarge
        'risk': 0.15,
        'tech': 0.25        # Trend & Momentum wichtiger
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
    """Sicheres Auslesen aus Dictionaries mit Fallback."""
    val = dictionary.get(key, default)
    return default if val is None else val

def clean_percentage(val):
    """Bringt Prozentwerte auf einheitliches Dezimalformat (z.B. 0.15 für 15%)."""
    if pd.isna(val):
        return np.nan
    if abs(val) > 2.0:  # Wert liegt vermutlich als 15.0 statt 0.15 vor
        return val / 100.0
    return val

@st.cache_data(ttl=3600*12)
def fetch_stock_data(ticker_symbol):
    """Treibender Data Fetcher für Yahoo Finance mit robuster Fehlerbehandlung."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Hist. Kurse (1 Jahr für SMA200 & exakte 6M-Performance)
        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < 100:
            return None

        # Finanzberichte
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
    except Exception as e:
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

    # --- 1. Preis & Momentum (Exakte 6M) ---
    close = hist['Close']
    current_price = close.iloc[-1]
    
    cutoff_6m = datetime.now() - timedelta(days=180)
    hist_6m = close[close.index >= cutoff_6m.strftime('%Y-%m-%d')]
    price_6m_ago = hist_6m.iloc[0] if not hist_6m.empty else close.iloc[0]
    
    metrics['perf_6m'] = (current_price - price_6m_ago) / price_6m_ago
    metrics['sma_200'] = close.rolling(200).mean().iloc[-1]
    metrics['above_sma200'] = current_price > metrics['sma_200']
    metrics['dist_52w_high'] = (current_price - close.max()) / close.max()

    # --- 2. Bewertungskennzahlen (Valuation) ---
    metrics['pe_forward'] = safe_get(info, 'forwardPE')
    metrics['peg_ratio'] = safe_get(info, 'pegRatio')
    metrics['pb_ratio'] = safe_get(info, 'priceToBook')
    
    # FCF Yield (Ohne Dopplung zu P/FCF im Valuation Score)
    fcf = safe_get(info, 'freeCashflow')
    market_cap = safe_get(info, 'marketCap')
    if pd.isna(fcf) and not cf.empty:
        try:
            op_cf = cf.loc['Operating Cash Flow'].iloc[0]
            capex = cf.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cf.index else 0
            fcf = op_cf + capex  # CapEx ist meist negativ
        except:
            fcf = np.nan
            
    metrics['fcf_yield'] = (fcf / market_cap) if (fcf and market_cap) else np.nan

    # --- 3. Qualität & Rentabilität (Quality) ---
    metrics['roe'] = clean_percentage(safe_get(info, 'returnOnEquity'))
    metrics['eps_growth_5y'] = clean_percentage(safe_get(info, 'earningsGrowth'))
    metrics['payout_ratio'] = clean_percentage(safe_get(info, 'payoutRatio'))

    # Dynamische ROIC-Berechnung mit effektivem Steuersatz
    try:
        ebit = fin.loc['EBIT'].iloc[0] if 'EBIT' in fin.index else fin.loc['Operating Income'].iloc[0]
        inc_tax = fin.loc['Tax Provision'].iloc[0] if 'Tax Provision' in fin.index else 0
        pre_tax = fin.loc['Pretax Income'].iloc[0] if 'Pretax Income' in fin.index else 1
        
        # Effektiver Steuersatz (Gedeckelt zwischen 15% und 35%)
        tax_rate = inc_tax / pre_tax if pre_tax > 0 else 0.21
        tax_rate = max(0.15, min(tax_rate, 0.35))
        
        nopat = ebit * (1 - tax_rate)
        
        total_assets = bs.loc['Total Assets'].iloc[0]
        curr_liab = bs.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in bs.index else 0
        cash = bs.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in bs.index else 0
        
        invested_capital = total_assets - curr_liab - cash
        metrics['roic'] = nopat / invested_capital if invested_capital > 0 else np.nan
    except:
        metrics['roic'] = np.nan

    # Bruttomarge (Gross Margin)
    try:
        gross_profit = fin.loc['Gross Profit'].iloc[0]
        total_rev = fin.loc['Total Revenue'].iloc[0]
        metrics['gross_margin'] = gross_profit / total_rev
    except:
        metrics['gross_margin'] = safe_get(info, 'grossMargins')

    # --- 4. Risikokennzahlen (Risk) ---
    try:
        tot_debt = bs.loc['Total Debt'].iloc[0] if 'Total Debt' in bs.index else 0
        cash_eq = bs.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in bs.index else 0
        net_debt = tot_debt - cash_eq
        
        ebitda = fin.loc['Normalized EBITDA'].iloc[0] if 'Normalized EBITDA' in fin.index else fin.loc['EBITDA'].iloc[0]
        
        # Negatives EBITDA abfangen
        metrics['net_debt_ebitda'] = (net_debt / ebitda) if ebitda > 0 else 99.0
    except:
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

    # 1. Valuation Score (0 - 100)
    v_scores = []
    pe = metrics.get('pe_forward')
    if pd.notna(pe) and pe > 0:
        v_scores.append(np.interp(pe, [8, 15, 25, 40], [100, 80, 40, 0]))
        
    peg = metrics.get('peg_ratio')
    if pd.notna(peg) and peg > 0:
        v_scores.append(np.interp(peg, [0.5, 1.0, 1.5, 2.5], [100, 85, 50, 0]))

    scores['valuation'] = np.mean(v_scores) if v_scores else 50.0

    # 2. Quality Score (0 - 100)
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

    # 3. Risk Score (0 - 100)
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

    # 4. Technical / Momentum Score (0 - 100)
    t_scores = []
    if metrics.get('above_sma200', False):
        t_scores.append(80)
    else:
        t_scores.append(20)

    perf_6m = metrics.get('perf_6m')
    if pd.notna(perf_6m):
        t_scores.append(np.interp(perf_6m, [-0.20, 0.0, 0.15, 0.40], [10, 40, 75, 100]))

    scores['tech'] = np.mean(t_scores) if t_scores else 50.0

    # Gesamter Sektor-gewichteter Score
    total_score = (
        scores['valuation'] * weights['valuation'] +
        scores['quality'] * weights['quality'] +
        scores['risk'] * weights['risk'] +
        scores['tech'] * weights['tech']
    )

    return round(total_score, 1), scores
