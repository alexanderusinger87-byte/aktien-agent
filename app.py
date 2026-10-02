import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aktien-Screener V10.1",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# SECTOR WEIGHTS
# ============================================================

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
    'Healthcare': {
        'valuation': 0.25,
        'quality': 0.40,
        'risk': 0.20,
        'tech': 0.15
    },
    'Consumer Cyclical': {
        'valuation': 0.30,
        'quality': 0.30,
        'risk': 0.20,
        'tech': 0.20
    },
    'Consumer Defensive': {
        'valuation': 0.25,
        'quality': 0.40,
        'risk': 0.25,
        'tech': 0.10
    },
    'Energy': {
        'valuation': 0.30,
        'quality': 0.30,
        'risk': 0.25,
        'tech': 0.15
    },
    'Default': {
        'valuation': 0.25,
        'quality': 0.35,
        'risk': 0.20,
        'tech': 0.20
    }
}


# ============================================================
# FALLBACK SECTORS
# ============================================================

FALLBACK_SECTORS = {
    'NVDA': 'Technology',
    'MSFT': 'Technology',
    'AAPL': 'Technology',
    'GOOGL': 'Technology',
    'AMZN': 'Consumer Cyclical',
    'ASML': 'Technology',
    'TSM': 'Technology',
    'BMW.DE': 'Consumer Cyclical',
    'TTE.PA': 'Energy',
    'ING': 'Financial Services',
    'PFE': 'Healthcare',
    'KO': 'Consumer Defensive',
    'NKE': 'Consumer Cyclical',
    'MRVL': 'Technology',
    'MU': 'Technology',
    '000660.KS': 'Technology'
}


# ============================================================
# HELPERS
# ============================================================

def safe_get(dictionary, key, default=np.nan):
    if not isinstance(dictionary, dict):
        return default

    try:
        val = dictionary.get(key, default)
    except Exception:
        return default

    if val is None:
        return default

    return val


def to_number(value):
    try:
        if value is None:
            return np.nan

        if isinstance(value, (pd.Series, pd.DataFrame)):
            if value.empty:
                return np.nan
            value = value.iloc[0]

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return value

    except Exception:
        return np.nan


def clean_percentage(val):
    val = to_number(val)

    if pd.isna(val):
        return np.nan

    # Yahoo occasionally delivers percentages as 25 instead of 0.25
    if abs(val) > 2.0:
        return val / 100.0

    return val


def clean_positive(val):
    val = to_number(val)

    if pd.isna(val) or val <= 0:
        return np.nan

    return val


# ============================================================
# ROBUST FINANCIAL STATEMENT READER
# ============================================================

def get_row(df, possible_keys):
    """
    Robustly retrieves the latest available value from a
    yfinance financial statement.

    Yahoo/yfinance can use slightly different row names.
    """

    if df is None:
        return np.nan

    if not isinstance(df, pd.DataFrame):
        return np.nan

    if df.empty:
        return np.nan

    for key in possible_keys:

        if key not in df.index:
            continue

        try:
            row = df.loc[key]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            # Convert to numeric
            row = pd.to_numeric(row, errors="coerce")

            row = row.dropna()

            if row.empty:
                continue

            # yfinance normally delivers newest period first.
            # We deliberately DO NOT sort the columns because
            # some Yahoo datasets use non-standard column labels.
            return to_number(row.iloc[0])

        except Exception:
            continue

    return np.nan


# ============================================================
# FINANCIAL STATEMENT ROW ALIASES
# ============================================================

REVENUE_KEYS = [
    'Total Revenue',
    'Operating Revenue',
    'Revenue',
    'TotalRevenue'
]

NET_INCOME_KEYS = [
    'Net Income',
    'Net Income Common Stockholders',
    'Net Income Including Noncontrolling Interests',
    'Net Income From Continuing Operation Net Minority Interest',
    'Net Income From Continuing Operation',
    'NetIncome'
]

EBIT_KEYS = [
    'EBIT',
    'Operating Income',
    'OperatingIncome'
]

EBITDA_KEYS = [
    'Normalized EBITDA',
    'EBITDA',
    'NormalizedEBITDA'
]

PRETAX_KEYS = [
    'Pretax Income',
    'Income Before Tax',
    'PretaxIncome'
]

TAX_KEYS = [
    'Tax Provision',
    'Income Tax Expense',
    'TaxProvision'
]

TOTAL_ASSETS_KEYS = [
    'Total Assets',
    'TotalAssets'
]

CURRENT_LIABILITY_KEYS = [
    'Current Liabilities',
    'Total Current Liabilities',
    'CurrentLiabilities'
]

CASH_KEYS = [
    'Cash And Cash Equivalents',
    'Cash Financial',
    'CashAndCashEquivalents',
    'Cash Cash Equivalents And Short Term Investments'
]

DEBT_KEYS = [
    'Total Debt',
    'TotalDebt',
    'Long Term Debt',
    'Long Term Debt And Capital Lease Obligation'
]

OPERATING_CF_KEYS = [
    'Operating Cash Flow',
    'Total Cash From Operating Activities',
    'OperatingCashFlow'
]

CAPEX_KEYS = [
    'Capital Expenditure',
    'Capital Expenditures',
    'CapEx',
    'CapitalExpenditures',
    'Purchase Of Property And Equipment'
]


# ============================================================
# DATA FETCH
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_stock_data(ticker_symbol):

    try:

        ticker = yf.Ticker(ticker_symbol)

        # ----------------------------------------------------
        # INFO
        # ----------------------------------------------------

        try:
            info = dict(ticker.info)
        except Exception:
            info = {}

        # ----------------------------------------------------
        # FAST INFO
        # ----------------------------------------------------

        try:
            fast_info = dict(ticker.fast_info)
        except Exception:
            fast_info = {}

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        try:
            hist = ticker.history(
                period="2y",
                auto_adjust=False
            )
        except Exception:
            hist = pd.DataFrame()

        if hist.empty or len(hist) < 30:
            return None

        # ----------------------------------------------------
        # FINANCIAL STATEMENTS
        # ----------------------------------------------------

        try:
            financials = ticker.get_income_stmt(
                freq="yearly"
            )
        except Exception:
            try:
                financials = ticker.financials
            except Exception:
                financials = pd.DataFrame()

        try:
            balance_sheet = ticker.get_balance_sheet(
                freq="yearly"
            )
        except Exception:
            try:
                balance_sheet = ticker.balance_sheet
            except Exception:
                balance_sheet = pd.DataFrame()

        try:
            cashflow = ticker.get_cashflow(
                freq="yearly"
            )
        except Exception:
            try:
                cashflow = ticker.cashflow
            except Exception:
                cashflow = pd.DataFrame()

        return {
            'ticker': ticker_symbol,
            'info': info,
            'fast_info': fast_info,
            'hist': hist,
            'financials': financials,
            'balance_sheet': balance_sheet,
            'cashflow': cashflow
        }

    except Exception:
        return None


# ============================================================
# MARKET CAP
# ============================================================

def get_market_cap(info, fast_info):

    market_cap = safe_get(info, 'marketCap')

    market_cap = to_number(market_cap)

    if pd.notna(market_cap) and market_cap > 0:
        return market_cap

    for key in ['market_cap', 'marketCap']:

        value = safe_get(fast_info, key)

        value = to_number(value)

        if pd.notna(value) and value > 0:
            return value

    return np.nan


# ============================================================
# CURRENT PRICE
# ============================================================

def get_current_price(info, fast_info, hist):

    for key in [
        'currentPrice',
        'regularMarketPrice',
        'previousClose'
    ]:

        value = to_number(safe_get(info, key))

        if pd.notna(value) and value > 0:
            return value

    for key in [
        'last_price',
        'regularMarketPrice'
    ]:

        value = to_number(safe_get(fast_info, key))

        if pd.notna(value) and value > 0:
            return value

    if hist is not None and not hist.empty:

        try:
            value = to_number(hist['Close'].iloc[-1])

            if pd.notna(value) and value > 0:
                return value

        except Exception:
            pass

    return np.nan


# ============================================================
# SECTOR
# ============================================================

def get_sector(ticker_symbol, info):

    sector = safe_get(info, 'sector', None)

    if isinstance(sector, str) and sector.strip():
        return sector

    return FALLBACK_SECTORS.get(
        ticker_symbol.upper(),
        'Default'
    )


# ============================================================
# VALUATION METRICS
# ============================================================

def calculate_pe(
    info,
    market_cap,
    net_income,
    current_price
):

    # --------------------------------------------------------
    # 1. Yahoo forward P/E
    # --------------------------------------------------------

    forward_pe = to_number(
        safe_get(info, 'forwardPE')
    )

    if pd.notna(forward_pe) and forward_pe > 0:
        return forward_pe

    # --------------------------------------------------------
    # 2. Yahoo trailing P/E
    # --------------------------------------------------------

    trailing_pe = to_number(
        safe_get(info, 'trailingPE')
    )

    if pd.notna(trailing_pe) and trailing_pe > 0:
        return trailing_pe

    # --------------------------------------------------------
    # 3. Price / trailing EPS
    # --------------------------------------------------------

    trailing_eps = to_number(
        safe_get(info, 'trailingEps')
    )

    if (
        pd.notna(current_price)
        and current_price > 0
        and pd.notna(trailing_eps)
        and trailing_eps > 0
    ):
        return current_price / trailing_eps

    # --------------------------------------------------------
    # 4. Market cap / net income
    # --------------------------------------------------------

    if (
        pd.notna(market_cap)
        and market_cap > 0
        and pd.notna(net_income)
        and net_income > 0
    ):
        return market_cap / net_income

    return np.nan


def calculate_ps(
    info,
    market_cap,
    revenue
):

    ps = to_number(
        safe_get(
            info,
            'priceToSalesTrailing12Months'
        )
    )

    if pd.notna(ps) and ps > 0:
        return ps

    if (
        pd.notna(market_cap)
        and market_cap > 0
        and pd.notna(revenue)
        and revenue > 0
    ):
        return market_cap / revenue

    return np.nan


def calculate_pb(
    info,
    market_cap,
    equity
):

    pb = to_number(
        safe_get(info, 'priceToBook')
    )

    if pd.notna(pb) and pb > 0:
        return pb

    if (
        pd.notna(market_cap)
        and market_cap > 0
        and pd.notna(equity)
        and equity > 0
    ):
        return market_cap / equity

    return np.nan


# ============================================================
# MAIN METRIC EXTRACTION
# ============================================================

def extract_metrics(ticker_symbol, data):

    info = data['info']
    fast_info = data['fast_info']
    hist = data['hist']
    fin = data['financials']
    bs = data['balance_sheet']
    cf = data['cashflow']

    sector = get_sector(
        ticker_symbol,
        info
    )

    metrics = {}

    # --------------------------------------------------------
    # BASIC
    # --------------------------------------------------------

    metrics['name'] = (
        safe_get(info, 'longName')
        or safe_get(info, 'shortName')
        or ticker_symbol
    )

    metrics['sector'] = sector

    metrics['price'] = get_current_price(
        info,
        fast_info,
        hist
    )

    metrics['market_cap'] = get_market_cap(
        info,
        fast_info
    )

    # --------------------------------------------------------
    # INCOME STATEMENT
    # --------------------------------------------------------

    revenue = get_row(
        fin,
        REVENUE_KEYS
    )

    net_income = get_row(
        fin,
        NET_INCOME_KEYS
    )

    ebit = get_row(
        fin,
        EBIT_KEYS
    )

    ebitda = get_row(
        fin,
        EBITDA_KEYS
    )

    pretax_income = get_row(
        fin,
        PRETAX_KEYS
    )

    tax_provision = get_row(
        fin,
        TAX_KEYS
    )

    metrics['revenue'] = revenue
    metrics['net_income'] = net_income
    metrics['ebit'] = ebit
    metrics['ebitda'] = ebitda

    # --------------------------------------------------------
    # BALANCE SHEET
    # --------------------------------------------------------

    total_assets = get_row(
        bs,
        TOTAL_ASSETS_KEYS
    )

    current_liabilities = get_row(
        bs,
        CURRENT_LIABILITY_KEYS
    )

    cash = get_row(
        bs,
        CASH_KEYS
    )

    total_debt = get_row(
        bs,
        DEBT_KEYS
    )

    equity = get_row(
        bs,
        [
            'Stockholders Equity',
            'Stockholders Equity Including Minority Interest',
            'Total Equity Gross Minority Interest',
            'Common Stock Equity'
        ]
    )

    metrics['total_assets'] = total_assets
    metrics['current_liabilities'] = current_liabilities
    metrics['cash'] = cash
    metrics['total_debt'] = total_debt
    metrics['equity'] = equity

    # --------------------------------------------------------
    # PE
    # --------------------------------------------------------

    metrics['pe_effective'] = calculate_pe(
        info,
        metrics['market_cap'],
        net_income,
        metrics['price']
    )

    # --------------------------------------------------------
    # P/S
    # --------------------------------------------------------

    metrics['ps_ratio'] = calculate_ps(
        info,
        metrics['market_cap'],
        revenue
    )

    # --------------------------------------------------------
    # P/B
    # --------------------------------------------------------

    metrics['pb_ratio'] = calculate_pb(
        info,
        metrics['market_cap'],
        equity
    )

    # --------------------------------------------------------
    # PEG
    # --------------------------------------------------------

    metrics['peg_ratio'] = clean_positive(
        safe_get(info, 'pegRatio')
    )

    # --------------------------------------------------------
    # GROSS MARGIN
    # --------------------------------------------------------

    gross_margin = clean_percentage(
        safe_get(info, 'grossMargins')
    )

    if pd.isna(gross_margin):

        gross_profit = get_row(
            fin,
            [
                'Gross Profit',
                'GrossProfit'
            ]
        )

        if (
            pd.notna(gross_profit)
            and pd.notna(revenue)
            and revenue > 0
        ):
            gross_margin = (
                gross_profit / revenue
            )

    metrics['gross_margin'] = gross_margin

    # --------------------------------------------------------
    # ROE
    # --------------------------------------------------------

    roe = clean_percentage(
        safe_get(info, 'returnOnEquity')
    )

    if pd.isna(roe):

        if (
            pd.notna(net_income)
            and pd.notna(equity)
            and equity > 0
        ):
            roe = net_income / equity

    metrics['roe'] = roe

    # --------------------------------------------------------
    # ROIC
    # --------------------------------------------------------

    if sector != 'Financial Services':

        if (
            pd.notna(ebit)
            and pd.notna(total_assets)
        ):

            tax_rate = 0.21

            if (
                pd.notna(tax_provision)
                and pd.notna(pretax_income)
                and pretax_income > 0
            ):

                calculated_tax_rate = (
                    tax_provision /
                    pretax_income
                )

                # Guard against absurd values
                if 0 <= calculated_tax_rate <= 0.60:
                    tax_rate = max(
                        0.10,
                        min(
                            calculated_tax_rate,
                            0.40
                        )
                    )

            nopat = ebit * (1 - tax_rate)

            cl = (
                current_liabilities
                if pd.notna(current_liabilities)
                else 0
            )

            c = (
                cash
                if pd.notna(cash)
                else 0
            )

            invested_capital = (
                total_assets - cl - c
            )

            if invested_capital > 0:

                roic = (
                    nopat /
                    invested_capital
                )

                # Prevent pathological outliers
                if -1 <= roic <= 3:
                    metrics['roic'] = roic
                else:
                    metrics['roic'] = np.nan

            else:
                metrics['roic'] = np.nan

        else:
            metrics['roic'] = np.nan

    else:

        metrics['roic'] = np.nan

    # --------------------------------------------------------
    # OPERATING CASH FLOW
    # --------------------------------------------------------

    op_cf = get_row(
        cf,
        OPERATING_CF_KEYS
    )

    capex = get_row(
        cf,
        CAPEX_KEYS
    )

    # --------------------------------------------------------
    # FREE CASH FLOW
    # --------------------------------------------------------

    fcf = np.nan

    if (
        pd.notna(op_cf)
        and pd.notna(capex)
    ):

        # Yahoo usually reports CapEx as negative.
        if capex < 0:
            fcf = op_cf + capex
        else:
            fcf = op_cf - capex

    # Optional Yahoo fallback
    if pd.isna(fcf):

        yahoo_fcf = to_number(
            safe_get(
                info,
                'freeCashflow'
            )
        )

        if pd.notna(yahoo_fcf):
            fcf = yahoo_fcf

    metrics['fcf'] = fcf

    if (
        pd.notna(fcf)
        and pd.notna(metrics['market_cap'])
        and metrics['market_cap'] > 0
    ):

        metrics['fcf_yield'] = (
            fcf /
            metrics['market_cap']
        )

    else:
        metrics['fcf_yield'] = np.nan

    # --------------------------------------------------------
    # NET DEBT / EBITDA
    # --------------------------------------------------------

    if sector != 'Financial Services':

        if pd.notna(total_debt):
            debt_value = total_debt
        else:
            debt_value = 0

        if pd.notna(cash):
            cash_value = cash
        else:
            cash_value = 0

        net_debt = (
            debt_value -
            cash_value
        )

        if (
            pd.notna(ebitda)
            and ebitda > 0
        ):

            metrics['net_debt_ebitda'] = (
                net_debt /
                ebitda
            )

        else:

            # No EBITDA substitute.
            metrics['net_debt_ebitda'] = np.nan

    else:

        metrics['net_debt_ebitda'] = np.nan

    # --------------------------------------------------------
    # CURRENT RATIO
    # --------------------------------------------------------

    current_ratio = to_number(
        safe_get(info, 'currentRatio')
    )

    if pd.isna(current_ratio):

        current_assets = get_row(
            bs,
            [
                'Current Assets',
                'Total Current Assets',
                'CurrentAssets'
            ]
        )

        if (
            pd.notna(current_assets)
            and pd.notna(current_liabilities)
            and current_liabilities > 0
        ):

            current_ratio = (
                current_assets /
                current_liabilities
            )

    metrics['current_ratio'] = current_ratio

    # --------------------------------------------------------
    # TECHNICAL DATA
    # --------------------------------------------------------

    close = pd.to_numeric(
        hist['Close'],
        errors='coerce'
    ).dropna()

    if len(close) >= 200:

        sma200 = (
            close
            .rolling(200)
            .mean()
            .iloc[-1]
        )

        metrics['above_sma200'] = (
            metrics['price'] > sma200
            if pd.notna(metrics['price'])
            and pd.notna(sma200)
            else False
        )

        metrics['sma200'] = sma200

    else:

        metrics['above_sma200'] = np.nan
        metrics['sma200'] = np.nan

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    metrics['perf_1m'] = np.nan
    metrics['perf_6m'] = np.nan

    if len(close) >= 2:

        # 1 month ≈ 21 trading days
        if len(close) > 21:

            old = close.iloc[-22]
            new = close.iloc[-1]

            if old > 0:
                metrics['perf_1m'] = (
                    new / old - 1
                )

        # 6 months ≈ 126 trading days
        if len(close) > 126:

            old = close.iloc[-127]
            new = close.iloc[-1]

            if old > 0:
                metrics['perf_6m'] = (
                    new / old - 1
                )

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    core_fields = [
        metrics['pe_effective'],
        metrics['ps_ratio'],
        metrics['fcf_yield'],
        metrics['gross_margin'],
        metrics['market_cap'],
        metrics['current_ratio']
    ]

    if sector != 'Financial Services':

        core_fields.extend([
            metrics['roic'],
            metrics['net_debt_ebitda']
        ])

    else:

        core_fields.extend([
            metrics['roe'],
            metrics['pb_ratio']
        ])

    valid_count = sum(
        1 for x in core_fields
        if pd.notna(x)
    )

    metrics['data_quality'] = round(
        valid_count /
        len(core_fields) *
        100
    )

    return metrics


# ============================================================
# SCORING
# ============================================================

def calculate_scores(metrics):

    sector = metrics.get(
        'sector',
        'Default'
    )

    weights = SECTOR_WEIGHTS.get(
        sector,
        SECTOR_WEIGHTS['Default']
    )

    scores = {}

    # ========================================================
    # VALUATION
    # ========================================================

    v_scores = []

    pe = metrics.get(
        'pe_effective'
    )

    if (
        pd.notna(pe)
        and pe > 0
    ):

        v_scores.append(
            np.interp(
                pe,
                [5, 12, 20, 35, 70],
                [100, 90, 70, 25, 0]
            )
        )

    ps = metrics.get(
        'ps_ratio'
    )

    if (
        pd.notna(ps)
        and ps > 0
    ):

        v_scores.append(
            np.interp(
                ps,
                [0.5, 1.5, 3.5, 7.0],
                [100, 85, 45, 0]
            )
        )

    peg = metrics.get(
        'peg_ratio'
    )

    if (
        pd.notna(peg)
        and peg > 0
    ):

        v_scores.append(
            np.interp(
                peg,
                [0.4, 1.0, 1.8, 3.0],
                [100, 80, 40, 0]
            )
        )

    scores['valuation'] = (
        np.mean(v_scores)
        if v_scores
        else np.nan
    )

    # ========================================================
    # QUALITY
    # ========================================================

    q_scores = []

    if sector == 'Financial Services':

        roe = metrics.get('roe')

        if pd.notna(roe):

            q_scores.append(
                np.interp(
                    roe,
                    [0.04, 0.10, 0.18, 0.25],
                    [20, 60, 90, 100]
                )
            )

    else:

        roic = metrics.get('roic')

        if pd.notna(roic):

            q_scores.append(
                np.interp(
                    roic,
                    [0.04, 0.10, 0.20, 0.35],
                    [20, 60, 90, 100]
                )
            )

        gm = metrics.get(
            'gross_margin'
        )

        if pd.notna(gm):

            q_scores.append(
                np.interp(
                    gm,
                    [0.15, 0.35, 0.55, 0.75],
                    [20, 50, 80, 100]
                )
            )

    fcf_y = metrics.get(
        'fcf_yield'
    )

    if pd.notna(fcf_y):

        q_scores.append(
            np.interp(
                fcf_y,
                [-0.01, 0.03, 0.06, 0.10],
                [10, 50, 85, 100]
            )
        )

    scores['quality'] = (
        np.mean(q_scores)
        if q_scores
        else np.nan
    )

    # ========================================================
    # RISK
    # ========================================================

    r_scores = []

    if sector == 'Financial Services':

        pb = metrics.get(
            'pb_ratio'
        )

        if pd.notna(pb):

            r_scores.append(
                np.interp(
                    pb,
                    [0.6, 1.0, 1.6, 2.5],
                    [100, 85, 50, 0]
                )
            )

    else:

        nd_ebitda = metrics.get(
            'net_debt_ebitda'
        )

        if pd.notna(nd_ebitda):

            r_scores.append(
                np.interp(
                    nd_ebitda,
                    [-1.0, 0.5, 2.5, 5.0],
                    [100, 90, 55, 10]
                )
            )

        cr = metrics.get(
            'current_ratio'
        )

        if pd.notna(cr):

            r_scores.append(
                np.interp(
                    cr,
                    [0.6, 1.0, 1.5, 2.5],
                    [15, 60, 90, 75]
                )
            )

    scores['risk'] = (
        np.mean(r_scores)
        if r_scores
        else np.nan
    )

    # ========================================================
    # TECH
    # ========================================================

    t_scores = []

    above_sma200 = metrics.get(
        'above_sma200'
    )

    if pd.notna(above_sma200):

        t_scores.append(
            85 if above_sma200
            else 30
        )

    perf_6m = metrics.get(
        'perf_6m'
    )

    if pd.notna(perf_6m):

        t_scores.append(
            np.interp(
                perf_6m,
                [-0.25, 0.0, 0.15, 0.40],
                [0, 45, 75, 100]
            )
        )

    scores['tech'] = (
        np.mean(t_scores)
        if t_scores
        else np.nan
    )

    # ========================================================
    # FINAL SCORE
    # ========================================================

    active_weights = weights.copy()
    valid_scores = {}

    for category in [
        'valuation',
        'quality',
        'risk',
        'tech'
    ]:

        value = scores.get(
            category,
            np.nan
        )

        if pd.notna(value):

            valid_scores[category] = value

        else:

            active_weights[category] = 0.0

    total_weight = sum(
        active_weights.values()
    )

    if (
        total_weight <= 0
        or not valid_scores
    ):

        return np.nan, scores

    normalized_weights = {
        key: value / total_weight
        for key, value
        in active_weights.items()
    }

    total_score = sum(
        valid_scores[key] *
        normalized_weights[key]
        for key in valid_scores
    )

    return round(
        total_score,
        1
    ), scores


# ============================================================
# RECOMMENDATION
# ============================================================

def get_recommendation(score):

    if pd.isna(score):
        return "Keine Bewertung"

    if score > 82.5:
        return "Strong Buy"

    if score >= 77.5:
        return "Buy"

    if score >= 70:
        return "Hold"

    if score >= 55:
        return "Reduce / Watch"

    return "Avoid"


# ============================================================
# TURNAROUND
# ============================================================

def get_turnaround_status(metrics):

    perf_1m = metrics.get(
        'perf_1m'
    )

    perf_6m = metrics.get(
        'perf_6m'
    )

    if (
        pd.notna(perf_1m)
        and pd.notna(perf_6m)
    ):

        if (
            perf_1m > 0
            and perf_6m < 0
        ):
            return "Turnaround?"

        if (
            perf_1m > 0.05
            and perf_6m > 0
        ):
            return "Momentum"

        if (
            perf_1m < -0.05
            and perf_6m < 0
        ):
            return "Abwärtstrend"

    return "Neutral"


# ============================================================
# RADAR
# ============================================================

def create_radar(scores):

    categories = [
        'Valuation',
        'Quality',
        'Risk',
        'Tech'
    ]

    values = [
        scores.get(
            'valuation',
            np.nan
        ),
        scores.get(
            'quality',
            np.nan
        ),
        scores.get(
            'risk',
            np.nan
        ),
        scores.get(
            'tech',
            np.nan
        )
    ]

    # Close radar polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            name='Score'
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        margin=dict(
            l=30,
            r=30,
            t=30,
            b=30
        )
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.title("📊 Quant-Aktien-Screener V10.1")

st.caption(
    "Fundamentaler + technischer Aktien-Screener "
    "mit sektorabhängiger Gewichtung"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Einstellungen")

ticker_input = st.sidebar.text_input(
    "Ticker",
    value=(
        "BMW.DE, NVDA, MSFT, AAPL, "
        "GOOGL, AMZN, TTE.PA, ING, "
        "PFE, KO, NKE"
    )
)

min_score = st.sidebar.slider(
    "Minimaler Score",
    min_value=0,
    max_value=100,
    value=0,
    step=1
)

if st.sidebar.button(
    "🗑️ Cache löschen"
):

    st.cache_data.clear()

    st.rerun()


# ============================================================
# SCREENING
# ============================================================

if st.button(
    "🚀 Aktien analysieren",
    type="primary"
):

    tickers = [
        x.strip().upper()
        for x in ticker_input.split(',')
        if x.strip()
    ]

    results = []
    metric_store = {}

    progress = st.progress(0)

    for i, ticker_symbol in enumerate(tickers):

        progress.progress(
            (i + 1) / len(tickers)
        )

        data = fetch_stock_data(
            ticker_symbol
        )

        if data is None:
            continue

        try:

            metrics = extract_metrics(
                ticker_symbol,
                data
            )

            score, scores = calculate_scores(
                metrics
            )

            if (
                pd.notna(score)
                and score < min_score
            ):
                continue

            recommendation = (
                get_recommendation(score)
            )

            turnaround = (
                get_turnaround_status(
                    metrics
                )
            )

            results.append({

                'Ticker':
                    ticker_symbol,

                'Name':
                    metrics.get(
                        'name',
                        ticker_symbol
                    ),

                'Sektor':
                    metrics.get(
                        'sector',
                        'Default'
                    ),

                'Gesamtscore':
                    score,

                'Datenqualität':
                    metrics.get(
                        'data_quality',
                        np.nan
                    ),

                'Empfehlung':
                    recommendation,

                'Turnaround Status':
                    turnaround,

                'Valuation':
                    scores.get(
                        'valuation',
                        np.nan
                    ),

                'Quality':
                    scores.get(
                        'quality',
                        np.nan
                    ),

                'Risk':
                    scores.get(
                        'risk',
                        np.nan
                    ),

                'Tech':
                    scores.get(
                        'tech',
                        np.nan
                    ),

                'KGV (Eff)':
                    metrics.get(
                        'pe_effective',
                        np.nan
                    )
            })

            metric_store[
                ticker_symbol
            ] = (
                metrics,
                scores
            )

        except Exception:
            continue

    progress.empty()

    # ========================================================
    # RESULTS
    # ========================================================

    if results:

        results_df = pd.DataFrame(
            results
        )

        results_df = results_df.sort_values(
            'Gesamtscore',
            ascending=False,
            na_position='last'
        )

        st.subheader(
            "📋 Screening-Ergebnis"
        )

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # INDIVIDUAL ANALYSIS
        # ====================================================

        st.divider()

        st.subheader(
            "🔎 Einzelanalyse"
        )

        selected_ticker = st.selectbox(
            "Aktie auswählen",
            list(metric_store.keys())
        )

        metrics, scores = metric_store[
            selected_ticker
        ]

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Gesamtscore",
                (
                    f"{calculate_scores(metrics)[0]:.1f}"
                    if pd.notna(
                        calculate_scores(metrics)[0]
                    )
                    else "n/a"
                )
            )

        with col2:

            st.metric(
                "KGV effektiv",
                (
                    f"{metrics['pe_effective']:.1f}"
                    if pd.notna(
                        metrics['pe_effective']
                    )
                    else "n/a"
                )
            )

        with col3:

            st.metric(
                "Datenqualität",
                f"{metrics['data_quality']}%"
            )

        with col4:

            st.metric(
                "Kurs",
                (
                    f"{metrics['price']:.2f}"
                    if pd.notna(
                        metrics['price']
                    )
                    else "n/a"
                )
            )

        # ====================================================
        # RADAR
        # ====================================================

        col_left, col_right = st.columns(
            [1, 1]
        )

        with col_left:

            st.plotly_chart(
                create_radar(scores),
                use_container_width=True
            )

        with col_right:

            detail_df = pd.DataFrame({
                'Kennzahl': [
                    'KGV',
                    'P/S',
                    'P/B',
                    'PEG',
                    'Gross Margin',
                    'ROIC',
                    'ROE',
                    'FCF Yield',
                    'Net Debt / EBITDA',
                    'Current Ratio',
                    '1M Performance',
                    '6M Performance',
                    'SMA200'
                ],

                'Wert': [

                    metrics.get(
                        'pe_effective'
                    ),

                    metrics.get(
                        'ps_ratio'
                    ),

                    metrics.get(
                        'pb_ratio'
                    ),

                    metrics.get(
                        'peg_ratio'
                    ),

                    metrics.get(
                        'gross_margin'
                    ),

                    metrics.get(
                        'roic'
                    ),

                    metrics.get(
                        'roe'
                    ),

                    metrics.get(
                        'fcf_yield'
                    ),

                    metrics.get(
                        'net_debt_ebitda'
                    ),

                    metrics.get(
                        'current_ratio'
                    ),

                    metrics.get(
                        'perf_1m'
                    ),

                    metrics.get(
                        'perf_6m'
                    ),

                    metrics.get(
                        'sma200'
                    )
                ]
            })

            st.dataframe(
                detail_df,
                use_container_width=True,
                hide_index=True
            )

    else:

        st.warning(
            "Keine Aktien mit den aktuellen "
            "Einstellungen gefunden."
        )
