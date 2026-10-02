import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aktien-Screener V10.2",
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
# FINANCIAL STATEMENT ALIASES
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
# HELPER FUNCTIONS
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

    if abs(val) > 2.0:
        return val / 100.0

    return val


def clean_positive(val):
    val = to_number(val)

    if pd.isna(val) or val <= 0:
        return np.nan

    return val


def get_row(df, possible_keys):
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

            row = pd.to_numeric(row, errors="coerce")
            row = row.dropna()

            if row.empty:
                continue

            return to_number(row.iloc[0])

        except Exception:
            continue

    return np.nan


# ============================================================
# STOCK NAME
# ============================================================

def get_stock_name(info, ticker_symbol):

    for key in [
        'longName',
        'shortName',
        'displayName'
    ]:

        value = safe_get(info, key, None)

        if isinstance(value, str):

            value = value.strip()

            if value and value.lower() != 'nan':
                return value

    return ticker_symbol


# ============================================================
# DATA FETCHING
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
        # INCOME STATEMENT
        # ----------------------------------------------------

        try:
            financials = ticker.get_income_stmt(freq="yearly")
        except Exception:

            try:
                financials = ticker.financials
            except Exception:
                financials = pd.DataFrame()

        # ----------------------------------------------------
        # BALANCE SHEET
        # ----------------------------------------------------

        try:
            balance_sheet = ticker.get_balance_sheet(freq="yearly")
        except Exception:

            try:
                balance_sheet = ticker.balance_sheet
            except Exception:
                balance_sheet = pd.DataFrame()

        # ----------------------------------------------------
        # CASH FLOW
        # ----------------------------------------------------

        try:
            cashflow = ticker.get_cashflow(freq="yearly")
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
# MARKET DATA
# ============================================================

def get_market_cap(info, fast_info):

    market_cap = to_number(
        safe_get(info, 'marketCap')
    )

    if pd.notna(market_cap) and market_cap > 0:
        return market_cap

    for key in [
        'market_cap',
        'marketCap'
    ]:

        value = to_number(
            safe_get(fast_info, key)
        )

        if pd.notna(value) and value > 0:
            return value

    return np.nan


def get_current_price(info, fast_info, hist):

    for key in [
        'currentPrice',
        'regularMarketPrice',
        'previousClose'
    ]:

        value = to_number(
            safe_get(info, key)
        )

        if pd.notna(value) and value > 0:
            return value

    for key in [
        'last_price',
        'regularMarketPrice'
    ]:

        value = to_number(
            safe_get(fast_info, key)
        )

        if pd.notna(value) and value > 0:
            return value

    if hist is not None and not hist.empty:

        try:

            value = to_number(
                hist['Close'].iloc[-1]
            )

            if pd.notna(value) and value > 0:
                return value

        except Exception:
            pass

    return np.nan


# ============================================================
# SECTOR
# ============================================================

def get_sector(ticker_symbol, info):

    sector = safe_get(
        info,
        'sector',
        None
    )

    if isinstance(sector, str) and sector.strip():
        return sector

    return FALLBACK_SECTORS.get(
        ticker_symbol.upper(),
        'Default'
    )


# ============================================================
# VALUATION
# ============================================================

def calculate_pe(
    info,
    market_cap,
    net_income,
    current_price
):

    forward_pe = to_number(
        safe_get(info, 'forwardPE')
    )

    if pd.notna(forward_pe) and forward_pe > 0:
        return forward_pe

    trailing_pe = to_number(
        safe_get(info, 'trailingPE')
    )

    if pd.notna(trailing_pe) and trailing_pe > 0:
        return trailing_pe

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
# METRIC EXTRACTION
# ============================================================

def extract_metrics(data):

    info = data['info']
    fast_info = data['fast_info']
    hist = data['hist']
    financials = data['financials']
    balance_sheet = data['balance_sheet']
    cashflow = data['cashflow']

    ticker_symbol = data['ticker']

    # --------------------------------------------------------
    # BASIC DATA
    # --------------------------------------------------------

    market_cap = get_market_cap(
        info,
        fast_info
    )

    current_price = get_current_price(
        info,
        fast_info,
        hist
    )

    sector = get_sector(
        ticker_symbol,
        info
    )

    name = get_stock_name(
        info,
        ticker_symbol
    )

    # --------------------------------------------------------
    # INCOME STATEMENT
    # --------------------------------------------------------

    revenue = get_row(
        financials,
        REVENUE_KEYS
    )

    net_income = get_row(
        financials,
        NET_INCOME_KEYS
    )

    ebit = get_row(
        financials,
        EBIT_KEYS
    )

    ebitda = get_row(
        financials,
        EBITDA_KEYS
    )

    pretax_income = get_row(
        financials,
        PRETAX_KEYS
    )

    tax_provision = get_row(
        financials,
        TAX_KEYS
    )

    # --------------------------------------------------------
    # BALANCE SHEET
    # --------------------------------------------------------

    total_assets = get_row(
        balance_sheet,
        TOTAL_ASSETS_KEYS
    )

    current_liabilities = get_row(
        balance_sheet,
        CURRENT_LIABILITY_KEYS
    )

    cash = get_row(
        balance_sheet,
        CASH_KEYS
    )

    total_debt = get_row(
        balance_sheet,
        DEBT_KEYS
    )

    # Equity
    equity = np.nan

    equity_keys = [
        'Stockholders Equity',
        'Total Equity Gross Minority Interest',
        'Common Stock Equity',
        'StockholdersEquity'
    ]

    equity = get_row(
        balance_sheet,
        equity_keys
    )

    # --------------------------------------------------------
    # VALUATION
    # --------------------------------------------------------

    pe_effective = calculate_pe(
        info,
        market_cap,
        net_income,
        current_price
    )

    ps_ratio = calculate_ps(
        info,
        market_cap,
        revenue
    )

    pb_ratio = calculate_pb(
        info,
        market_cap,
        equity
    )

    peg_ratio = clean_positive(
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
            financials,
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
            gross_margin = gross_profit / revenue

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

    # --------------------------------------------------------
    # ROIC
    # --------------------------------------------------------

    roic = np.nan

    if sector != 'Financial Services':

        if (
            pd.notna(ebit)
            and pd.notna(total_assets)
            and pd.notna(current_liabilities)
            and pd.notna(cash)
        ):

            tax_rate = 0.21

            if (
                pd.notna(tax_provision)
                and pd.notna(pretax_income)
                and pretax_income > 0
            ):

                calculated_tax_rate = (
                    tax_provision / pretax_income
                )

                if (
                    calculated_tax_rate >= 0
                    and calculated_tax_rate <= 0.60
                ):
                    tax_rate = np.clip(
                        calculated_tax_rate,
                        0.10,
                        0.40
                    )

            nopat = ebit * (1 - tax_rate)

            invested_capital = (
                total_assets
                - current_liabilities
                - cash
            )

            if invested_capital > 0:

                calculated_roic = (
                    nopat / invested_capital
                )

                if (
                    calculated_roic >= -1
                    and calculated_roic <= 3
                ):
                    roic = calculated_roic

    # --------------------------------------------------------
    # FREE CASH FLOW
    # --------------------------------------------------------

    operating_cf = get_row(
        cashflow,
        OPERATING_CF_KEYS
    )

    capex = get_row(
        cashflow,
        CAPEX_KEYS
    )

    fcf = np.nan

    if (
        pd.notna(operating_cf)
        and pd.notna(capex)
    ):

        if capex < 0:
            fcf = operating_cf + capex
        else:
            fcf = operating_cf - capex

    # Fallback to Yahoo's direct free cash flow
    if pd.isna(fcf):

        yahoo_fcf = to_number(
            safe_get(
                info,
                'freeCashflow'
            )
        )

        if pd.notna(yahoo_fcf):
            fcf = yahoo_fcf

    fcf_yield = np.nan

    if (
        pd.notna(fcf)
        and pd.notna(market_cap)
        and market_cap > 0
    ):
        fcf_yield = fcf / market_cap

    # --------------------------------------------------------
    # NET DEBT / EBITDA
    # --------------------------------------------------------

    net_debt_ebitda = np.nan

    if sector != 'Financial Services':

        if (
            pd.notna(total_debt)
            and pd.notna(cash)
            and pd.notna(ebitda)
            and ebitda > 0
        ):

            net_debt = total_debt - cash

            net_debt_ebitda = (
                net_debt / ebitda
            )

    # --------------------------------------------------------
    # CURRENT RATIO
    # --------------------------------------------------------

    current_ratio = clean_positive(
        safe_get(info, 'currentRatio')
    )

    if pd.isna(current_ratio):

        current_assets = get_row(
            balance_sheet,
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
                current_assets
                / current_liabilities
            )

    # --------------------------------------------------------
    # TECHNICAL DATA
    # --------------------------------------------------------

    above_sma200 = np.nan
    perf_1m = np.nan
    perf_6m = np.nan

    try:

        close = pd.to_numeric(
            hist['Close'],
            errors='coerce'
        ).dropna()

        # ----------------------------------------------------
        # SMA 200
        # ----------------------------------------------------

        if len(close) >= 200:

            sma200 = close.rolling(
                200
            ).mean().iloc[-1]

            if pd.notna(sma200):

                above_sma200 = (
                    close.iloc[-1] > sma200
                )

        # ----------------------------------------------------
        # 1 MONTH
        # ----------------------------------------------------

        if len(close) >= 22:

            old_price_1m = close.iloc[-22]
            latest_price = close.iloc[-1]

            if (
                pd.notna(old_price_1m)
                and old_price_1m > 0
            ):

                perf_1m = (
                    latest_price
                    / old_price_1m
                    - 1
                )

        # ----------------------------------------------------
        # 6 MONTHS
        # ----------------------------------------------------

        if len(close) >= 127:

            old_price_6m = close.iloc[-127]
            latest_price = close.iloc[-1]

            if (
                pd.notna(old_price_6m)
                and old_price_6m > 0
            ):

                perf_6m = (
                    latest_price
                    / old_price_6m
                    - 1
                )

    except Exception:
        pass

    # --------------------------------------------------------
    # DATA COMPLETENESS
    # --------------------------------------------------------
    #
    # Measures availability of the core inputs used by the
    # scoring model, not every possible Yahoo field.
    #
    # This makes the percentage directly relevant to the
    # actual score calculation.
    # --------------------------------------------------------

    core_fields = [
        pe_effective,
        ps_ratio,
        fcf_yield,
        gross_margin,
        market_cap,
        current_ratio
    ]

    if sector != 'Financial Services':

        core_fields.extend([
            roic,
            net_debt_ebitda
        ])

    else:

        core_fields.extend([
            roe,
            pb_ratio
        ])

    valid_count = sum(
        1
        for x in core_fields
        if pd.notna(x)
    )

    data_completeness = round(
        valid_count
        / len(core_fields)
        * 100
    )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        # Basic
        'ticker': ticker_symbol,
        'name': name,
        'sector': sector,
        'price': current_price,
        'market_cap': market_cap,

        # Financials
        'revenue': revenue,
        'net_income': net_income,
        'ebit': ebit,
        'ebitda': ebitda,
        'pretax_income': pretax_income,
        'tax_provision': tax_provision,

        # Balance sheet
        'total_assets': total_assets,
        'current_liabilities': current_liabilities,
        'cash': cash,
        'total_debt': total_debt,
        'equity': equity,

        # Valuation
        'pe_effective': pe_effective,
        'ps_ratio': ps_ratio,
        'pb_ratio': pb_ratio,
        'peg_ratio': peg_ratio,

        # Quality
        'gross_margin': gross_margin,
        'roe': roe,
        'roic': roic,
        'fcf': fcf,
        'fcf_yield': fcf_yield,

        # Risk
        'net_debt_ebitda': net_debt_ebitda,
        'current_ratio': current_ratio,

        # Technical
        'above_sma200': above_sma200,
        'perf_1m': perf_1m,
        'perf_6m': perf_6m,

        # Data completeness
        'data_completeness': data_completeness
    }


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

    scores = {
        'valuation': np.nan,
        'quality': np.nan,
        'risk': np.nan,
        'tech': np.nan
    }

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

    if v_scores:
        scores['valuation'] = np.mean(
            v_scores
        )

    # ========================================================
    # QUALITY
    # ========================================================

    q_scores = []

    if sector == 'Financial Services':

        roe = metrics.get(
            'roe'
        )

        if pd.notna(roe):

            q_scores.append(
                np.interp(
                    roe,
                    [0.04, 0.10, 0.18, 0.25],
                    [20, 60, 90, 100]
                )
            )

    else:

        roic = metrics.get(
            'roic'
        )

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

    if q_scores:

        scores['quality'] = np.mean(
            q_scores
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

    if r_scores:

        scores['risk'] = np.mean(
            r_scores
        )

    # ========================================================
    # TECHNICAL
    # ========================================================

    t_scores = []

    above_sma200 = metrics.get(
        'above_sma200'
    )

    if pd.notna(above_sma200):

        t_scores.append(
            85 if above_sma200 else 30
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

    if t_scores:

        scores['tech'] = np.mean(
            t_scores
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
        for key, value in active_weights.items()
    }

    total_score = sum(
        valid_scores[key]
        * normalized_weights[key]
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
# RADAR CHART
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

    categories_closed = (
        categories
        + [categories[0]]
    )

    values_closed = (
        values
        + [values[0]]
    )

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

st.title(
    "📊 Quant-Aktien-Screener V10.2"
)

st.caption(
    "Fundamentaler und technischer Aktien-Score "
    "mit sektorabhängiger Gewichtung"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Einstellungen"
)

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
    0,
    100,
    0,
    1
)

if st.sidebar.button(
    "🗑️ Cache löschen"
):

    st.cache_data.clear()
    st.rerun()


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🚀 Aktien analysieren"
):

    # --------------------------------------------------------
    # TICKERS
    # --------------------------------------------------------

    tickers = [
        x.strip().upper()
        for x in ticker_input.split(',')
        if x.strip()
    ]

    if not tickers:

        st.warning(
            "Bitte mindestens einen Ticker eingeben."
        )

        st.stop()

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    progress_bar = st.progress(0)

    status_text = st.empty()

    results = []

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    for i, ticker_symbol in enumerate(tickers):

        status_text.write(
            f"Analysiere {ticker_symbol} ..."
        )

        data = fetch_stock_data(
            ticker_symbol
        )

        if data is None:

            progress_bar.progress(
                (i + 1) / len(tickers)
            )

            continue

        metrics = extract_metrics(
            data
        )

        score, scores = calculate_scores(
            metrics
        )

        # ----------------------------------------------------
        # MIN SCORE FILTER
        # ----------------------------------------------------

        if (
            pd.notna(score)
            and score < min_score
        ):

            progress_bar.progress(
                (i + 1) / len(tickers)
            )

            continue

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        results.append({

            'Ticker': metrics['ticker'],

            'Name': metrics['name'],

            'Sektor': metrics['sector'],

            'Gesamtscore': score,

            'Datenvollständigkeit (%)':
                metrics['data_completeness'],

            'Empfehlung':
                get_recommendation(score),

            'Turnaround Status':
                get_turnaround_status(metrics),

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

        progress_bar.progress(
            (i + 1) / len(tickers)
        )

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    progress_bar.empty()
    status_text.empty()

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if not results:

        st.warning(
            "Keine Aktien konnten mit den "
            "aktuellen Einstellungen bewertet werden."
        )

        st.stop()

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # NUMERIC CLEANUP
    # --------------------------------------------------------

    numeric_columns = [
        'Valuation',
        'Quality',
        'Risk',
        'Tech',
        'Gesamtscore',
        'KGV (Eff)',
        'Datenvollständigkeit (%)'
    ]

    for col in numeric_columns:

        results_df[col] = pd.to_numeric(
            results_df[col],
            errors='coerce'
        )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    results_df = results_df.sort_values(
        'Gesamtscore',
        ascending=False,
        na_position='last'
    )

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    st.subheader(
        "📋 Ergebnisse"
    )

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        column_config={

            'Gesamtscore':
                st.column_config.NumberColumn(
                    'Gesamtscore',
                    format='%.1f'
                ),

            'Datenvollständigkeit (%)':
                st.column_config.NumberColumn(
                    'Datenvollständigkeit (%)',
                    format='%.0f%%'
                ),

            'Valuation':
                st.column_config.NumberColumn(
                    'Valuation',
                    format='%.0f'
                ),

            'Quality':
                st.column_config.NumberColumn(
                    'Quality',
                    format='%.0f'
                ),

            'Risk':
                st.column_config.NumberColumn(
                    'Risk',
                    format='%.0f'
                ),

            'Tech':
                st.column_config.NumberColumn(
                    'Tech',
                    format='%.0f'
                ),

            'KGV (Eff)':
                st.column_config.NumberColumn(
                    'KGV (Eff)',
                    format='%.1f'
                )
        }
    )

    # ========================================================
    # DETAIL ANALYSIS
    # ========================================================

    st.subheader(
        "🔎 Detailanalyse"
    )

    selected_ticker = st.selectbox(
        "Aktie auswählen",
        results_df['Ticker'].tolist()
    )

    # --------------------------------------------------------
    # GET ORIGINAL DATA
    # --------------------------------------------------------

    detail_data = fetch_stock_data(
        selected_ticker
    )

    if detail_data is not None:

        detail_metrics = extract_metrics(
            detail_data
        )

        detail_score, detail_scores = (
            calculate_scores(
                detail_metrics
            )
        )

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        st.markdown(
            f"### {detail_metrics['name']} "
            f"({selected_ticker})"
        )

        st.caption(
            f"Sektor: {detail_metrics['sector']}"
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:

            st.metric(
                "Gesamtscore",
                (
                    f"{detail_score:.1f}"
                    if pd.notna(detail_score)
                    else "–"
                )
            )

        with col2:

            st.metric(
                "Valuation",
                (
                    f"{detail_scores['valuation']:.0f}"
                    if pd.notna(
                        detail_scores['valuation']
                    )
                    else "–"
                )
            )

        with col3:

            st.metric(
                "Quality",
                (
                    f"{detail_scores['quality']:.0f}"
                    if pd.notna(
                        detail_scores['quality']
                    )
                    else "–"
                )
            )

        with col4:

            st.metric(
                "Risk",
                (
                    f"{detail_scores['risk']:.0f}"
                    if pd.notna(
                        detail_scores['risk']
                    )
                    else "–"
                )
            )

        with col5:

            st.metric(
                "Tech",
                (
                    f"{detail_scores['tech']:.0f}"
                    if pd.notna(
                        detail_scores['tech']
                    )
                    else "–"
                )
            )

        # ----------------------------------------------------
        # DATA COMPLETENESS
        # ----------------------------------------------------

        st.metric(
            "Datenvollständigkeit",
            f"{detail_metrics['data_completeness']}%"
        )

        # ----------------------------------------------------
        # RADAR
        # ----------------------------------------------------

        radar_col, detail_col = st.columns(
            [1, 1]
        )

        with radar_col:

            st.plotly_chart(
                create_radar(
                    detail_scores
                ),
                use_container_width=True
            )

        # ----------------------------------------------------
        # DETAIL TABLE
        # ----------------------------------------------------

        with detail_col:

            detail_df = pd.DataFrame({

                'Kennzahl': [

                    'Aktueller Kurs',
                    'Marktkapitalisierung',
                    'KGV (Eff)',
                    'KUV',
                    'KBV',
                    'PEG',
                    'Bruttomarge',
                    'ROE',
                    'ROIC',
                    'FCF',
                    'FCF-Rendite',
                    'Net Debt / EBITDA',
                    'Current Ratio',
                    '1M Performance',
                    '6M Performance',
                    'Über SMA200',
                    'Datenvollständigkeit'
                ],

                'Wert': [

                    detail_metrics['price'],
                    detail_metrics['market_cap'],
                    detail_metrics['pe_effective'],
                    detail_metrics['ps_ratio'],
                    detail_metrics['pb_ratio'],
                    detail_metrics['peg_ratio'],
                    detail_metrics['gross_margin'],
                    detail_metrics['roe'],
                    detail_metrics['roic'],
                    detail_metrics['fcf'],
                    detail_metrics['fcf_yield'],
                    detail_metrics['net_debt_ebitda'],
                    detail_metrics['current_ratio'],
                    detail_metrics['perf_1m'],
                    detail_metrics['perf_6m'],
                    (
                        "Ja"
                        if detail_metrics['above_sma200'] is True
                        else (
                            "Nein"
                            if detail_metrics['above_sma200'] is False
                            else "–"
                        )
                    ),
                    f"{detail_metrics['data_completeness']}%"
                ]
            })

            # ------------------------------------------------
            # FORMAT DETAIL VALUES
            # ------------------------------------------------

            percentage_metrics = {
                'Bruttomarge',
                'ROE',
                'ROIC',
                'FCF-Rendite',
                '1M Performance',
                '6M Performance'
            }

            formatted_values = []

            for _, row in detail_df.iterrows():

                metric = row['Kennzahl']
                value = row['Wert']

                if metric in percentage_metrics:

                    if pd.notna(value):

                        formatted_values.append(
                            f"{value * 100:.1f}%"
                        )

                    else:

                        formatted_values.append("–")

                elif metric == 'Marktkapitalisierung':

                    if pd.notna(value):

                        formatted_values.append(
                            f"{value / 1_000_000_000:.2f} Mrd."
                        )

                    else:

                        formatted_values.append("–")

                elif metric == 'FCF':

                    if pd.notna(value):

                        formatted_values.append(
                            f"{value / 1_000_000:.0f} Mio."
                        )

                    else:

                        formatted_values.append("–")

                elif metric in [
                    'KGV (Eff)',
                    'KUV',
                    'KBV',
                    'PEG',
                    'Net Debt / EBITDA',
                    'Current Ratio'
                ]:

                    if pd.notna(value):

                        formatted_values.append(
                            f"{value:.2f}"
                        )

                    else:

                        formatted_values.append("–")

                elif metric == 'Aktueller Kurs':

                    if pd.notna(value):

                        formatted_values.append(
                            f"{value:.2f}"
                        )

                    else:

                        formatted_values.append("–")

                else:

                    formatted_values.append(
                        str(value)
                    )

            detail_df['Wert'] = formatted_values

            st.dataframe(
                detail_df,
                use_container_width=True,
                hide_index=True
            )

        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        st.markdown(
            f"**Empfehlung:** "
            f"{get_recommendation(detail_score)}"
        )

        st.markdown(
            f"**Turnaround Status:** "
            f"{get_turnaround_status(detail_metrics)}"
        )

else:

    st.info(
        "Ticker eingeben und "
        "„🚀 Aktien analysieren“ klicken."
    )
