import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aktien-Screener V10.8",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# FIXED SCORE WEIGHTS
# ============================================================

SCORE_WEIGHTS = {
    'fair_value': 0.25,
    'valuation': 0.20,
    'quality': 0.30,
    'risk': 0.15,
    'technical': 0.10
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
    'Cash Flow From Continuing Operating Activities',
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
        value = dictionary.get(key, default)
    except Exception:
        return default

    if value is None:
        return default

    return value


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


def clean_percentage(value):

    value = to_number(value)

    if pd.isna(value):
        return np.nan

    if abs(value) > 2.0:
        return value / 100.0

    return value


def clean_positive(value):

    value = to_number(value)

    if pd.isna(value) or value <= 0:
        return np.nan

    return value


def get_row(df, possible_keys):

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return np.nan

    for key in possible_keys:

        if key not in df.index:
            continue

        try:

            row = df.loc[key]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            row = pd.to_numeric(
                row,
                errors="coerce"
            ).dropna()

            if row.empty:
                continue

            return to_number(
                row.iloc[0]
            )

        except Exception:
            continue

    return np.nan


def get_first_valid_row(df, possible_keys):

    """
    Liefert den aktuellsten verfügbaren Wert einer der
    angegebenen Zeilen aus einem Yahoo-Financial-Statement.
    """

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return np.nan

    for key in possible_keys:

        if key not in df.index:
            continue

        try:

            row = df.loc[key]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            values = pd.to_numeric(
                row,
                errors="coerce"
            ).dropna()

            if not values.empty:

                return to_number(
                    values.iloc[0]
                )

        except Exception:
            continue

    return np.nan


def get_numeric_series(df, possible_keys):

    """
    Liefert die komplette numerische Zeitreihe der ersten
    gefundenen Finanzzeile. Die Reihenfolge wird chronologisch
    sortiert, damit daraus Jahreswachstum berechnet werden kann.
    """

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return pd.Series(dtype=float)

    for key in possible_keys:

        if key not in df.index:
            continue

        try:

            row = df.loc[key]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            values = pd.to_numeric(
                row,
                errors="coerce"
            ).dropna()

            if values.empty:
                continue

            try:
                values.index = pd.to_datetime(
                    values.index
                )
                values = values.sort_index()
            except Exception:
                pass

            return values.astype(float)

        except Exception:
            continue

    return pd.Series(dtype=float)


def calculate_growth_volatility(series):

    if (
        series is None
        or not isinstance(series, pd.Series)
        or len(series) < 3
    ):
        return np.nan

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(values) < 3:
        return np.nan

    growth_rates = []

    for previous, current in zip(
        values.iloc[:-1],
        values.iloc[1:]
    ):

        if (
            pd.notna(previous)
            and pd.notna(current)
            and previous > 0
            and current > 0
        ):
            growth_rates.append(
                current / previous - 1
            )

    if len(growth_rates) < 2:
        return np.nan

    return float(
        pd.Series(growth_rates).std(ddof=1)
    )


def get_estimate_value(df, row_name, column='avg'):

    if df is None:
        return np.nan

    if not isinstance(df, pd.DataFrame):
        return np.nan

    if df.empty:
        return np.nan

    try:

        if row_name not in df.index:
            return np.nan

        if column not in df.columns:
            return np.nan

        return to_number(
            df.loc[row_name, column]
        )

    except Exception:

        return np.nan


def safe_mean(values):

    values = [
        x for x in values
        if pd.notna(x)
    ]

    if not values:
        return np.nan

    return float(np.mean(values))


def extract_analyst_target(targets, key):

    """
    Robustly extracts an analyst price target from the different
    structures yfinance may return (dict, Series or DataFrame).
    """

    value = np.nan

    if isinstance(targets, dict):
        value = targets.get(key, np.nan)

    elif isinstance(targets, pd.Series):
        try:
            if key in targets.index:
                value = targets.loc[key]
        except Exception:
            pass

    elif isinstance(targets, pd.DataFrame):
        try:
            if key in targets.index:
                value = targets.loc[key].iloc[0]
            elif key in targets.columns:
                value = targets[key].iloc[0]
        except Exception:
            pass

    return clean_positive(value)


# ============================================================
# COMPANY NAME
# ============================================================

def get_stock_name(info, search_quotes, ticker_symbol):

    for key in [
        'longName',
        'shortName',
        'displayName'
    ]:

        value = safe_get(
            info,
            key,
            None
        )

        if isinstance(value, str):

            value = value.strip()

            if (
                value
                and value.lower() != 'nan'
                and value.upper() != ticker_symbol.upper()
            ):
                return value

    if isinstance(search_quotes, list):

        ticker_upper = ticker_symbol.upper()

        for quote in search_quotes:

            if not isinstance(quote, dict):
                continue

            symbol = str(
                quote.get(
                    'symbol',
                    ''
                )
            ).upper()

            if symbol != ticker_upper:
                continue

            for key in [
                'longname',
                'longName',
                'shortname',
                'shortName',
                'displayName'
            ]:

                value = quote.get(key)

                if (
                    isinstance(value, str)
                    and value.strip()
                    and value.strip().upper()
                    != ticker_upper
                ):
                    return value.strip()

        for quote in search_quotes:

            if not isinstance(quote, dict):
                continue

            for key in [
                'longname',
                'longName',
                'shortname',
                'shortName',
                'displayName'
            ]:

                value = quote.get(key)

                if (
                    isinstance(value, str)
                    and value.strip()
                ):
                    return value.strip()

    return ticker_symbol


# ============================================================
# DATA FETCHING
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def fetch_stock_data(ticker_symbol):

    try:

        ticker = yf.Ticker(
            ticker_symbol
        )

        # ----------------------------------------------------
        # INFO
        # ----------------------------------------------------

        try:
            info = dict(
                ticker.info
            )
        except Exception:
            info = {}

        # ----------------------------------------------------
        # FAST INFO
        # ----------------------------------------------------

        try:
            fast_info = dict(
                ticker.fast_info
            )
        except Exception:
            fast_info = {}

        # ----------------------------------------------------
        # COMPANY NAME SEARCH
        # ----------------------------------------------------

        search_quotes = []

        try:

            search_result = yf.Search(
                ticker_symbol,
                max_results=5,
                news_count=0,
                lists_count=0,
                include_nav_links=False,
                include_research=False
            )

            search_quotes = search_result.quotes

            if search_quotes is None:
                search_quotes = []

        except Exception:

            search_quotes = []

        # ----------------------------------------------------
        # ANALYST PRICE TARGETS
        # ----------------------------------------------------

        try:

            analyst_targets = dict(
                ticker.get_analyst_price_targets()
            )

        except Exception:

            try:
                analyst_targets = dict(
                    ticker.analyst_price_targets
                )
            except Exception:
                analyst_targets = {}

        # ----------------------------------------------------
        # EARNINGS ESTIMATES
        # ----------------------------------------------------

        try:

            earnings_estimate = (
                ticker.get_earnings_estimate()
            )

        except Exception:

            try:
                earnings_estimate = (
                    ticker.earnings_estimate
                )
            except Exception:
                earnings_estimate = pd.DataFrame()

        # ----------------------------------------------------
        # REVENUE ESTIMATES
        # ----------------------------------------------------

        try:

            revenue_estimate = (
                ticker.get_revenue_estimate()
            )

        except Exception:

            try:
                revenue_estimate = (
                    ticker.revenue_estimate
                )
            except Exception:
                revenue_estimate = pd.DataFrame()

        # ----------------------------------------------------
        # GROWTH ESTIMATES
        # ----------------------------------------------------

        try:

            growth_estimates = (
                ticker.get_growth_estimates()
            )

        except Exception:

            try:
                growth_estimates = (
                    ticker.growth_estimates
                )
            except Exception:
                growth_estimates = pd.DataFrame()

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

        if (
            hist.empty
            or len(hist) < 30
        ):
            return None

        # ----------------------------------------------------
        # INCOME STATEMENT
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

        # ----------------------------------------------------
        # BALANCE SHEET
        # ----------------------------------------------------

        try:

            balance_sheet = ticker.get_balance_sheet(
                freq="yearly"
            )

        except Exception:

            try:
                balance_sheet = ticker.balance_sheet
            except Exception:
                balance_sheet = pd.DataFrame()

        # ----------------------------------------------------
        # CASH FLOW
        # ----------------------------------------------------

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
            'search_quotes': search_quotes,
            'analyst_targets': analyst_targets,
            'earnings_estimate': earnings_estimate,
            'revenue_estimate': revenue_estimate,
            'growth_estimates': growth_estimates,
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
        safe_get(
            info,
            'marketCap'
        )
    )

    if (
        pd.notna(market_cap)
        and market_cap > 0
    ):
        return market_cap

    for key in [
        'market_cap',
        'marketCap'
    ]:

        value = to_number(
            safe_get(
                fast_info,
                key
            )
        )

        if (
            pd.notna(value)
            and value > 0
        ):
            return value

    return np.nan


def get_current_price(
    info,
    fast_info,
    hist
):

    for key in [
        'currentPrice',
        'regularMarketPrice',
        'previousClose'
    ]:

        value = to_number(
            safe_get(
                info,
                key
            )
        )

        if (
            pd.notna(value)
            and value > 0
        ):
            return value

    for key in [
        'last_price',
        'regularMarketPrice'
    ]:

        value = to_number(
            safe_get(
                fast_info,
                key
            )
        )

        if (
            pd.notna(value)
            and value > 0
        ):
            return value

    if (
        hist is not None
        and not hist.empty
    ):

        try:

            value = to_number(
                hist['Close'].iloc[-1]
            )

            if (
                pd.notna(value)
                and value > 0
            ):
                return value

        except Exception:
            pass

    return np.nan


def get_shares_outstanding(
    info,
    fast_info,
    market_cap,
    current_price
):

    # --------------------------------------------------------
    # 1. Direkte Yahoo-Angabe
    # --------------------------------------------------------

    for key in [
        'sharesOutstanding',
        'impliedSharesOutstanding'
    ]:

        value = to_number(
            safe_get(
                info,
                key
            )
        )

        if (
            pd.notna(value)
            and value > 0
        ):
            return value

    # --------------------------------------------------------
    # 2. Fast Info
    # --------------------------------------------------------

    for key in [
        'shares',
        'sharesOutstanding'
    ]:

        value = to_number(
            safe_get(
                fast_info,
                key
            )
        )

        if (
            pd.notna(value)
            and value > 0
        ):
            return value

    # --------------------------------------------------------
    # 3. Fallback über Marktkapitalisierung / Kurs
    # --------------------------------------------------------

    if (
        pd.notna(market_cap)
        and market_cap > 0
        and pd.notna(current_price)
        and current_price > 0
    ):

        shares = (
            market_cap
            / current_price
        )

        if (
            pd.notna(shares)
            and shares > 0
        ):
            return shares

    return np.nan


# ============================================================
# SECTOR
# ============================================================

def get_sector(
    ticker_symbol,
    info
):

    sector = safe_get(
        info,
        'sector',
        None
    )

    if (
        isinstance(sector, str)
        and sector.strip()
    ):
        return sector

    return FALLBACK_SECTORS.get(
        ticker_symbol.upper(),
        'Default'
    )


# ============================================================
# VALUATION HELPERS
# ============================================================

def calculate_trailing_pe(
    info,
    market_cap,
    net_income,
    current_price
):

    trailing_pe = to_number(
        safe_get(
            info,
            'trailingPE'
        )
    )

    if (
        pd.notna(trailing_pe)
        and trailing_pe > 0
    ):
        return trailing_pe

    trailing_eps = to_number(
        safe_get(
            info,
            'trailingEps'
        )
    )

    if (
        pd.notna(current_price)
        and current_price > 0
        and pd.notna(trailing_eps)
        and trailing_eps > 0
    ):

        return (
            current_price
            / trailing_eps
        )

    if (
        pd.notna(market_cap)
        and market_cap > 0
        and pd.notna(net_income)
        and net_income > 0
    ):

        return (
            market_cap
            / net_income
        )

    return np.nan


def calculate_forward_pe(info):

    value = to_number(
        safe_get(
            info,
            'forwardPE'
        )
    )

    if (
        pd.notna(value)
        and value > 0
    ):
        return value

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

    if (
        pd.notna(ps)
        and ps > 0
    ):
        return ps

    if (
        pd.notna(market_cap)
        and market_cap > 0
        and pd.notna(revenue)
        and revenue > 0
    ):

        return (
            market_cap
            / revenue
        )

    return np.nan


def calculate_pb(
    info,
    market_cap,
    equity
):

    pb = to_number(
        safe_get(
            info,
            'priceToBook'
        )
    )

    if (
        pd.notna(pb)
        and pb > 0
    ):
        return pb

    if (
        pd.notna(market_cap)
        and market_cap > 0
        and pd.notna(equity)
        and equity > 0
    ):

        return (
            market_cap
            / equity
        )

    return np.nan


# ============================================================
# FREE CASH FLOW
# ============================================================

def get_free_cash_flow(cashflow):

    if (
        cashflow is None
        or not isinstance(cashflow, pd.DataFrame)
        or cashflow.empty
    ):
        return np.nan

    # --------------------------------------------------------
    # 1. Direkte Free-Cash-Flow-Zeile aus dem Jahresabschluss
    # --------------------------------------------------------

    direct_statement_fcf = get_first_valid_row(
        cashflow,
        [
            'Free Cash Flow',
            'FreeCashFlow',
            'Free Cashflow',
            'FreeCashflow'
        ]
    )

    if pd.notna(direct_statement_fcf):
        return direct_statement_fcf

    # --------------------------------------------------------
    # 2. Operating CF - CapEx
    # --------------------------------------------------------

    operating_cf = get_first_valid_row(
        cashflow,
        [
            'Operating Cash Flow',
            'Total Cash From Operating Activities',
            'Cash Flow From Continuing Operating Activities',
            'OperatingCashFlow'
        ]
    )

    capex = get_first_valid_row(
        cashflow,
        [
            'Capital Expenditure',
            'Capital Expenditures',
            'CapitalExpenditures',
            'CapEx',
            'Purchase Of Property And Equipment'
        ]
    )

    if (
        pd.notna(operating_cf)
        and pd.notna(capex)
    ):

        if capex < 0:
            return operating_cf + capex

        return operating_cf - capex

    return np.nan


# ============================================================
# FAIR VALUE MODEL
# ============================================================

def calculate_fcf_fair_value(
    fcf,
    market_cap,
    current_price,
    growth_rate
):

    if (
        pd.isna(fcf)
        or fcf <= 0
        or pd.isna(market_cap)
        or market_cap <= 0
        or pd.isna(current_price)
        or current_price <= 0
    ):
        return np.nan

    # --------------------------------------------------------
    # FCF YIELD
    # --------------------------------------------------------

    fcf_yield = (
        fcf / market_cap
    )

    if (
        pd.isna(fcf_yield)
        or fcf_yield <= 0
        or fcf_yield > 0.50
    ):
        return np.nan

    fcf_per_share = (
        fcf_yield
        * current_price
    )

    if (
        pd.isna(fcf_per_share)
        or fcf_per_share <= 0
    ):
        return np.nan

    # --------------------------------------------------------
    # GROWTH
    # --------------------------------------------------------

    if pd.isna(growth_rate):
        growth_rate = 0.05

    growth_rate = np.clip(
        growth_rate,
        -0.03,
        0.12
    )

    # --------------------------------------------------------
    # DCF ASSUMPTIONS
    #
    # 9% discount rate and 2.5% terminal growth remain
    # deliberately unchanged. The key correction is that the
    # DCF is now treated as a cross-check, not as an unlimited
    # source of upside.
    # --------------------------------------------------------

    discount_rate = 0.09
    terminal_growth = 0.025

    if terminal_growth >= discount_rate:
        return np.nan

    # --------------------------------------------------------
    # 5-YEAR FCF PROJECTION
    # --------------------------------------------------------

    pv_fcf = 0.0
    projected_fcf = fcf_per_share

    for year in range(1, 6):

        year_growth = (
            growth_rate
            + (
                terminal_growth
                - growth_rate
            )
            * (year - 1)
            / 4
        )

        projected_fcf *= (
            1 + year_growth
        )

        pv_fcf += (
            projected_fcf
            / (
                (1 + discount_rate)
                ** year
            )
        )

    # --------------------------------------------------------
    # TERMINAL VALUE
    # --------------------------------------------------------

    terminal_value = (
        projected_fcf
        * (1 + terminal_growth)
        / (
            discount_rate
            - terminal_growth
        )
    )

    pv_terminal = (
        terminal_value
        / (
            (1 + discount_rate)
            ** 5
        )
    )

    fair_value = (
        pv_fcf
        + pv_terminal
    )

    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    if (
        pd.isna(fair_value)
        or fair_value <= 0
        or fair_value > current_price * 10
        or fair_value < current_price * 0.05
    ):
        return np.nan

    # --------------------------------------------------------
    # DCF SANITY RANGE
    #
    # IMPORTANT: Do NOT clip a weak DCF to exactly 60% of the
    # current price. That artificial floor creates exactly -40%
    # upside and therefore exactly 0 Fair-Value points.
    #
    # Instead, an implausibly low/high DCF is treated as an
    # unusable cross-check (NaN). This is much more honest:
    # missing/unreliable DCF data must not become a fake score.
    # --------------------------------------------------------

    dcf_lower_bound = current_price * 0.60
    dcf_upper_bound = current_price * 1.75

    if (
        fair_value < dcf_lower_bound
        or fair_value > dcf_upper_bound
    ):
        return np.nan

    return fair_value


def calculate_fair_value_score(
    current_price,
    analyst_target,
    fcf_fair_value
):

    fair_values = []

    if (
        pd.notna(analyst_target)
        and analyst_target > 0
    ):
        fair_values.append(
            ('analyst', analyst_target)
        )

    if (
        pd.notna(fcf_fair_value)
        and fcf_fair_value > 0
    ):
        fair_values.append(
            ('fcf', fcf_fair_value)
        )

    if not fair_values:
        return np.nan, np.nan

    # --------------------------------------------------------
    # COMBINATION
    #
    # Analyst consensus is the primary external reference.
    # DCF is deliberately a secondary cross-check.
    # --------------------------------------------------------

    if len(fair_values) == 2:

        analyst_value = next(
            value
            for source, value in fair_values
            if source == 'analyst'
        )

        fcf_value = next(
            value
            for source, value in fair_values
            if source == 'fcf'
        )

        fair_value_price = (
            0.75 * analyst_value
            + 0.25 * fcf_value
        )

    else:

        fair_value_price = fair_values[0][1]

    if (
        pd.isna(current_price)
        or current_price <= 0
    ):
        return fair_value_price, np.nan

    upside = (
        fair_value_price
        / current_price
        - 1
    )

    # --------------------------------------------------------
    # FAIR VALUE SCORE
    #
    # More conservative around 0% and moderate upside.
    # 100 points require a very substantial valuation buffer.
    # --------------------------------------------------------

    fair_value_score = np.interp(
        upside,
        [
            -0.40,
            -0.20,
            0.00,
            0.10,
            0.25,
            0.50,
            0.75
        ],
        [
            0,
            15,
            40,
            58,
            75,
            90,
            100
        ]
    )

    return (
        fair_value_price,
        fair_value_score
    )


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(close, period=14):

    if close is None or len(close) < period + 2:
        return np.nan

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .rolling(period)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(period)
        .mean()
    )

    latest_gain = avg_gain.iloc[-1]
    latest_loss = avg_loss.iloc[-1]

    if pd.isna(latest_gain) or pd.isna(latest_loss):
        return np.nan

    if latest_loss == 0:

        if latest_gain > 0:
            return 100.0

        return 50.0

    rs = (
        latest_gain
        / latest_loss
    )

    return (
        100
        - 100 / (1 + rs)
    )


def calculate_macd(close):

    if close is None or len(close) < 35:
        return (
            np.nan,
            np.nan,
            np.nan
        )

    ema12 = (
        close
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        close
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    macd = (
        ema12
        - ema26
    )

    signal = (
        macd
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    histogram = (
        macd
        - signal
    )

    return (
        macd.iloc[-1],
        signal.iloc[-1],
        histogram.iloc[-1]
    )


# ============================================================
# METRIC EXTRACTION
# ============================================================

def extract_metrics(data):

    info = data['info']
    fast_info = data['fast_info']
    search_quotes = data['search_quotes']

    analyst_targets = data['analyst_targets']
    earnings_estimate = data['earnings_estimate']
    revenue_estimate = data['revenue_estimate']
    growth_estimates = data['growth_estimates']

    hist = data['hist']
    financials = data['financials']
    balance_sheet = data['balance_sheet']
    cashflow = data['cashflow']

    ticker_symbol = data['ticker']

    # --------------------------------------------------------
    # BASIC
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

    shares_outstanding = get_shares_outstanding(
        info,
        fast_info,
        market_cap,
        current_price
    )

    sector = get_sector(
        ticker_symbol,
        info
    )

    name = get_stock_name(
        info,
        search_quotes,
        ticker_symbol
    )

    # --------------------------------------------------------
    # INCOME
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

    equity = get_row(
        balance_sheet,
        [
            'Stockholders Equity',
            'Total Equity Gross Minority Interest',
            'Common Stock Equity',
            'StockholdersEquity'
        ]
    )

    current_assets = get_row(
        balance_sheet,
        [
            'Current Assets',
            'Total Current Assets',
            'CurrentAssets'
        ]
    )

    # --------------------------------------------------------
    # EPS / GROWTH ESTIMATES
    # --------------------------------------------------------

    trailing_eps = to_number(
        safe_get(
            info,
            'trailingEps'
        )
    )

    forward_eps = to_number(
        safe_get(
            info,
            'forwardEps'
        )
    )

    earnings_growth = clean_percentage(
        safe_get(
            info,
            'earningsGrowth'
        )
    )

    revenue_growth = clean_percentage(
        safe_get(
            info,
            'revenueGrowth'
        )
    )

    # Analyst EPS growth fallback
    estimate_growth = get_estimate_value(
        earnings_estimate,
        '+1y',
        'growth'
    )

    if pd.notna(estimate_growth):
        estimate_growth = clean_percentage(
            estimate_growth
        )

    if pd.isna(earnings_growth):
        earnings_growth = estimate_growth

    # Analyst revenue growth fallback
    estimate_revenue_growth = get_estimate_value(
        revenue_estimate,
        '+1y',
        'growth'
    )

    if pd.notna(estimate_revenue_growth):
        estimate_revenue_growth = clean_percentage(
            estimate_revenue_growth
        )

    if pd.isna(revenue_growth):
        revenue_growth = estimate_revenue_growth

    # Longer-term growth estimate fallback
    long_term_growth = np.nan

    try:

        if (
            isinstance(
                growth_estimates,
                pd.DataFrame
            )
            and not growth_estimates.empty
            and '+5y' in growth_estimates.index
            and 'stock' in growth_estimates.columns
        ):

            long_term_growth = clean_percentage(
                growth_estimates.loc[
                    '+5y',
                    'stock'
                ]
            )

    except Exception:
        pass

    if pd.isna(long_term_growth):
        long_term_growth = earnings_growth

    # --------------------------------------------------------
    # VALUATION
    # --------------------------------------------------------

    trailing_pe = calculate_trailing_pe(
        info,
        market_cap,
        net_income,
        current_price
    )

    forward_pe = calculate_forward_pe(
        info
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

    # Forward PEG calculated from forward PE
    # and expected EPS growth.
    forward_peg = np.nan

    if (
        pd.notna(forward_pe)
        and forward_pe > 0
        and pd.notna(earnings_growth)
        and earnings_growth > 0
    ):

        growth_percent = (
            earnings_growth * 100
        )

        if growth_percent > 0:

            forward_peg = (
                forward_pe
                / growth_percent
            )

    # Yahoo PEG fallback
    if pd.isna(forward_peg):

        yahoo_peg = clean_positive(
            safe_get(
                info,
                'pegRatio'
            )
        )

        if pd.notna(yahoo_peg):
            forward_peg = yahoo_peg

    # --------------------------------------------------------
    # GROSS MARGIN
    # --------------------------------------------------------

    gross_margin = clean_percentage(
        safe_get(
            info,
            'grossMargins'
        )
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

            gross_margin = (
                gross_profit
                / revenue
            )

    # --------------------------------------------------------
    # NET MARGIN
    # --------------------------------------------------------

    net_margin = np.nan

    if (
        pd.notna(net_income)
        and pd.notna(revenue)
        and revenue > 0
    ):

        net_margin = (
            net_income
            / revenue
        )

    # --------------------------------------------------------
    # ROE
    # --------------------------------------------------------

    roe = clean_percentage(
        safe_get(
            info,
            'returnOnEquity'
        )
    )

    if pd.isna(roe):

        if (
            pd.notna(net_income)
            and pd.notna(equity)
            and equity > 0
        ):

            roe = (
                net_income
                / equity
            )

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
                    tax_provision
                    / pretax_income
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

            nopat = (
                ebit
                * (1 - tax_rate)
            )

            invested_capital = (
                total_assets
                - current_liabilities
                - cash
            )

            if invested_capital > 0:

                calculated_roic = (
                    nopat
                    / invested_capital
                )

                if (
                    calculated_roic >= -1
                    and calculated_roic <= 3
                ):

                    roic = calculated_roic

    # --------------------------------------------------------
    # CASH FLOW
    # --------------------------------------------------------

    operating_cf = get_first_valid_row(
        cashflow,
        [
            'Operating Cash Flow',
            'Total Cash From Operating Activities',
            'Cash Flow From Continuing Operating Activities',
            'OperatingCashFlow'
        ]
    )

    capex = get_first_valid_row(
        cashflow,
        [
            'Capital Expenditure',
            'Capital Expenditures',
            'CapitalExpenditures',
            'CapEx',
            'Purchase Of Property And Equipment'
        ]
    )

    # --------------------------------------------------------
    # FREE CASH FLOW
    # --------------------------------------------------------

    fcf = get_free_cash_flow(
        cashflow
    )

    fcf_yield = np.nan
    fcf_margin = np.nan

    if (
        pd.notna(fcf)
        and pd.notna(market_cap)
        and market_cap > 0
    ):

        fcf_yield = (
            fcf
            / market_cap
        )

    if (
        pd.notna(fcf)
        and pd.notna(revenue)
        and revenue > 0
    ):

        fcf_margin = (
            fcf
            / revenue
        )

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

            net_debt = (
                total_debt
                - cash
            )

            net_debt_ebitda = (
                net_debt
                / ebitda
            )

    # --------------------------------------------------------
    # DEBT / EQUITY
    # --------------------------------------------------------

    debt_to_equity = clean_positive(
        safe_get(
            info,
            'debtToEquity'
        )
    )

    if pd.isna(debt_to_equity):

        if (
            pd.notna(total_debt)
            and pd.notna(equity)
            and equity > 0
        ):

            debt_to_equity = (
                total_debt
                / equity
                * 100
            )

    # --------------------------------------------------------
    # CASH / DEBT
    # --------------------------------------------------------

    cash_to_debt = np.nan

    if (
        pd.notna(cash)
        and pd.notna(total_debt)
        and total_debt > 0
    ):

        cash_to_debt = (
            cash
            / total_debt
        )

    # --------------------------------------------------------
    # CURRENT RATIO
    # --------------------------------------------------------

    current_ratio = clean_positive(
        safe_get(
            info,
            'currentRatio'
        )
    )

    if pd.isna(current_ratio):

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
    # RISK MARKET / BUSINESS STABILITY
    # --------------------------------------------------------

    annualized_volatility = np.nan
    max_drawdown = np.nan

    try:

        close_for_risk = pd.to_numeric(
            hist['Close'],
            errors='coerce'
        ).dropna()

        if len(close_for_risk) >= 30:

            daily_returns = (
                close_for_risk
                .pct_change()
                .dropna()
            )

            if len(daily_returns) >= 20:

                annualized_volatility = (
                    daily_returns.std(ddof=1)
                    * np.sqrt(252)
                )

            running_max = (
                close_for_risk
                .cummax()
            )

            drawdown_series = (
                close_for_risk
                / running_max
                - 1.0
            )

            if not drawdown_series.empty:
                max_drawdown = float(
                    drawdown_series.min()
                )

    except Exception:
        pass

    historical_revenue = get_numeric_series(
        financials,
        REVENUE_KEYS
    )

    historical_net_income = get_numeric_series(
        financials,
        NET_INCOME_KEYS
    )

    revenue_growth_volatility = calculate_growth_volatility(
        historical_revenue
    )

    earnings_growth_volatility = calculate_growth_volatility(
        historical_net_income
    )

    # --------------------------------------------------------
    # ANALYST TARGET
    # --------------------------------------------------------

    analyst_target_mean = np.nan
    analyst_target_median = np.nan
    analyst_target_low = np.nan
    analyst_target_high = np.nan

    analyst_target_mean = extract_analyst_target(
        analyst_targets,
        'mean'
    )

    analyst_target_median = extract_analyst_target(
        analyst_targets,
        'median'
    )

    analyst_target_low = extract_analyst_target(
        analyst_targets,
        'low'
    )

    analyst_target_high = extract_analyst_target(
        analyst_targets,
        'high'
    )

    # --------------------------------------------------------
    # FALLBACKS FROM INFO
    # --------------------------------------------------------

    if pd.isna(analyst_target_mean):
        analyst_target_mean = clean_positive(
            safe_get(
                info,
                'targetMeanPrice'
            )
        )

    if pd.isna(analyst_target_median):
        analyst_target_median = clean_positive(
            safe_get(
                info,
                'targetMedianPrice'
            )
        )

    if pd.isna(analyst_target_low):
        analyst_target_low = clean_positive(
            safe_get(
                info,
                'targetLowPrice'
            )
        )

    if pd.isna(analyst_target_high):
        analyst_target_high = clean_positive(
            safe_get(
                info,
                'targetHighPrice'
            )
        )

    # --------------------------------------------------------
    # FCF FAIR VALUE
    # --------------------------------------------------------

    # --------------------------------------------------------
    # DCF GROWTH
    #
    # Revenue growth gets slightly more weight than earnings
    # growth because EPS/net-income growth can be strongly
    # distorted by buybacks, margins and one-off effects.
    # Growth is intentionally capped more conservatively than
    # the previous model.
    # --------------------------------------------------------

    growth_inputs = []

    if pd.notna(revenue_growth):
        growth_inputs.append(
            (revenue_growth, 0.60)
        )

    if pd.notna(earnings_growth):
        growth_inputs.append(
            (earnings_growth, 0.40)
        )

    if growth_inputs:
        weighted_growth = (
            sum(
                value * weight
                for value, weight
                in growth_inputs
            )
            / sum(
                weight
                for _, weight
                in growth_inputs
            )
        )

        dcf_growth = np.clip(
            weighted_growth,
            -0.03,
            0.12
        )

    else:
        dcf_growth = 0.05

    fcf_fair_value = np.nan

    if sector != 'Financial Services':

        fcf_fair_value = calculate_fcf_fair_value(
            fcf,
            market_cap,
            current_price,
            dcf_growth
        )

    # --------------------------------------------------------
    # COMBINED FAIR VALUE
    # --------------------------------------------------------

    fair_value_price, fair_value_score = (
        calculate_fair_value_score(
            current_price,
            analyst_target_mean,
            fcf_fair_value
        )
    )

    analyst_target_upside = np.nan

    if (
        pd.notna(analyst_target_mean)
        and pd.notna(current_price)
        and current_price > 0
    ):

        analyst_target_upside = (
            analyst_target_mean
            / current_price
            - 1
        )

    fair_value_upside = np.nan

    if (
        pd.notna(fair_value_price)
        and pd.notna(current_price)
        and current_price > 0
    ):

        fair_value_upside = (
            fair_value_price
            / current_price
            - 1
        )

    # --------------------------------------------------------
    # TECHNICAL
    # --------------------------------------------------------

    above_sma200 = np.nan
    above_sma50 = np.nan

    perf_1m = np.nan
    perf_6m = np.nan
    perf_12m = np.nan

    sma50 = np.nan
    sma200 = np.nan

    rsi14 = np.nan
    macd = np.nan
    macd_signal = np.nan
    macd_hist = np.nan

    try:

        close = pd.to_numeric(
            hist['Close'],
            errors='coerce'
        ).dropna()

        latest_price = close.iloc[-1]

        if len(close) >= 50:

            sma50 = (
                close
                .rolling(50)
                .mean()
                .iloc[-1]
            )

            if pd.notna(sma50):

                above_sma50 = (
                    latest_price > sma50
                )

        if len(close) >= 200:

            sma200 = (
                close
                .rolling(200)
                .mean()
                .iloc[-1]
            )

            if pd.notna(sma200):

                above_sma200 = (
                    latest_price > sma200
                )

        if len(close) >= 22:

            old_price_1m = close.iloc[-22]

            if (
                pd.notna(old_price_1m)
                and old_price_1m > 0
            ):

                perf_1m = (
                    latest_price
                    / old_price_1m
                    - 1
                )

        if len(close) >= 127:

            old_price_6m = close.iloc[-127]

            if (
                pd.notna(old_price_6m)
                and old_price_6m > 0
            ):

                perf_6m = (
                    latest_price
                    / old_price_6m
                    - 1
                )

        if len(close) >= 253:

            old_price_12m = close.iloc[-253]

            if (
                pd.notna(old_price_12m)
                and old_price_12m > 0
            ):

                perf_12m = (
                    latest_price
                    / old_price_12m
                    - 1
                )

        rsi14 = calculate_rsi(
            close,
            14
        )

        (
            macd,
            macd_signal,
            macd_hist
        ) = calculate_macd(
            close
        )

    except Exception:
        pass

    # ========================================================
    # DATA COMPLETENESS
    # ========================================================

    completeness_blocks = []

    completeness_blocks.append(
        pd.notna(fair_value_score)
    )

    completeness_blocks.append(
        (
            pd.notna(forward_pe)
            or pd.notna(trailing_pe)
            or pd.notna(ps_ratio)
            or pd.notna(pb_ratio)
        )
    )

    if sector == 'Financial Services':

        quality_available = (
            pd.notna(roe)
            or pd.notna(net_margin)
            or pd.notna(earnings_growth)
        )

    else:

        quality_available = (
            pd.notna(roic)
            or pd.notna(gross_margin)
            or pd.notna(fcf_margin)
            or pd.notna(earnings_growth)
        )

    completeness_blocks.append(
        quality_available
    )

    risk_available = (
        pd.notna(annualized_volatility)
        or pd.notna(max_drawdown)
        or pd.notna(revenue_growth_volatility)
        or pd.notna(earnings_growth_volatility)
        or pd.notna(net_debt_ebitda)
        or pd.notna(current_ratio)
        or pd.notna(debt_to_equity)
        or pd.notna(cash_to_debt)
    )

    completeness_blocks.append(
        risk_available
    )

    technical_available = (
        pd.notna(rsi14)
        or pd.notna(macd)
        or pd.notna(perf_6m)
        or pd.notna(above_sma200)
    )

    completeness_blocks.append(
        technical_available
    )

    data_completeness = round(
        sum(completeness_blocks)
        / len(completeness_blocks)
        * 100
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        'ticker': ticker_symbol,
        'name': name,
        'sector': sector,

        'price': current_price,
        'market_cap': market_cap,
        'shares_outstanding': shares_outstanding,

        'revenue': revenue,
        'net_income': net_income,
        'ebit': ebit,
        'ebitda': ebitda,
        'pretax_income': pretax_income,
        'tax_provision': tax_provision,

        'total_assets': total_assets,
        'current_liabilities': current_liabilities,
        'current_assets': current_assets,
        'cash': cash,
        'total_debt': total_debt,
        'equity': equity,

        'trailing_eps': trailing_eps,
        'forward_eps': forward_eps,

        'earnings_growth': earnings_growth,
        'revenue_growth': revenue_growth,
        'long_term_growth': long_term_growth,

        'trailing_pe': trailing_pe,
        'forward_pe': forward_pe,
        'ps_ratio': ps_ratio,
        'pb_ratio': pb_ratio,
        'forward_peg': forward_peg,

        'gross_margin': gross_margin,
        'net_margin': net_margin,
        'roe': roe,
        'roic': roic,

        'operating_cf': operating_cf,
        'fcf': fcf,
        'fcf_yield': fcf_yield,
        'fcf_margin': fcf_margin,

        'net_debt_ebitda': net_debt_ebitda,
        'debt_to_equity': debt_to_equity,
        'cash_to_debt': cash_to_debt,
        'current_ratio': current_ratio,

        'annualized_volatility': annualized_volatility,
        'max_drawdown': max_drawdown,
        'revenue_growth_volatility': revenue_growth_volatility,
        'earnings_growth_volatility': earnings_growth_volatility,

        'analyst_target_mean': analyst_target_mean,
        'analyst_target_median': analyst_target_median,
        'analyst_target_low': analyst_target_low,
        'analyst_target_high': analyst_target_high,
        'analyst_target_upside': analyst_target_upside,

        'fcf_fair_value': fcf_fair_value,
        'fair_value_price': fair_value_price,
        'fair_value_upside': fair_value_upside,
        'fair_value_score': fair_value_score,

        'sma50': sma50,
        'sma200': sma200,
        'above_sma50': above_sma50,
        'above_sma200': above_sma200,

        'perf_1m': perf_1m,
        'perf_6m': perf_6m,
        'perf_12m': perf_12m,

        'rsi14': rsi14,
        'macd': macd,
        'macd_signal': macd_signal,
        'macd_hist': macd_hist,

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

    scores = {
        'fair_value': np.nan,
        'valuation': np.nan,
        'quality': np.nan,
        'risk': np.nan,
        'technical': np.nan
    }

    # ========================================================
    # FAIR VALUE
    # ========================================================

    fair_value_score = metrics.get(
        'fair_value_score'
    )

    if pd.notna(fair_value_score):

        scores['fair_value'] = (
            fair_value_score
        )

    # ========================================================
    # VALUATION
    # ========================================================

    valuation_components = []

    # Forward PE
    forward_pe = metrics.get(
        'forward_pe'
    )

    if (
        pd.notna(forward_pe)
        and forward_pe > 0
    ):

        valuation_components.append(
            (
                np.interp(
                    forward_pe,
                    [5, 10, 15, 22, 35, 60],
                    [100, 92, 80, 65, 25, 0]
                ),
                0.40
            )
        )

    # Trailing PE
    trailing_pe = metrics.get(
        'trailing_pe'
    )

    if (
        pd.notna(trailing_pe)
        and trailing_pe > 0
    ):

        valuation_components.append(
            (
                np.interp(
                    trailing_pe,
                    [5, 10, 15, 22, 35, 60],
                    [100, 92, 80, 65, 25, 0]
                ),
                0.20
            )
        )

    # Forward PEG
    forward_peg = metrics.get(
        'forward_peg'
    )

    if (
        pd.notna(forward_peg)
        and forward_peg > 0
    ):

        valuation_components.append(
            (
                np.interp(
                    forward_peg,
                    [0.4, 0.8, 1.0, 1.5, 2.5, 4.0],
                    [100, 92, 80, 60, 20, 0]
                ),
                0.20
            )
        )

    if sector == 'Financial Services':

        pb = metrics.get(
            'pb_ratio'
        )

        if (
            pd.notna(pb)
            and pb > 0
        ):

            valuation_components.append(
                (
                    np.interp(
                        pb,
                        [0.5, 0.8, 1.1, 1.6, 2.5, 4.0],
                        [100, 92, 80, 55, 20, 0]
                    ),
                    0.20
                )
            )

    else:

        ps = metrics.get(
            'ps_ratio'
        )

        if (
            pd.notna(ps)
            and ps > 0
        ):

            valuation_components.append(
                (
                    np.interp(
                        ps,
                        [0.5, 1.5, 3.0, 5.0, 8.0],
                        [100, 85, 60, 30, 0]
                    ),
                    0.20
                )
            )

    if valuation_components:

        weighted_sum = sum(
            score * weight
            for score, weight
            in valuation_components
        )

        weight_sum = sum(
            weight
            for _, weight
            in valuation_components
        )

        if weight_sum > 0:

            scores['valuation'] = (
                weighted_sum
                / weight_sum
            )

    # ========================================================
    # QUALITY
    # ========================================================

    quality_components = []

    if sector == 'Financial Services':

        # ROE
        roe = metrics.get(
            'roe'
        )

        if pd.notna(roe):

            quality_components.append(
                (
                    np.interp(
                        roe,
                        [0.04, 0.08, 0.12, 0.18, 0.25],
                        [20, 45, 70, 90, 100]
                    ),
                    0.35
                )
            )

        # Net margin
        net_margin = metrics.get(
            'net_margin'
        )

        if pd.notna(net_margin):

            quality_components.append(
                (
                    np.interp(
                        net_margin,
                        [0.05, 0.10, 0.20, 0.30, 0.40],
                        [20, 45, 70, 90, 100]
                    ),
                    0.25
                )
            )

    else:

        # ROIC
        roic = metrics.get(
            'roic'
        )

        if pd.notna(roic):

            quality_components.append(
                (
                    np.interp(
                        roic,
                        [0.04, 0.08, 0.12, 0.20, 0.35],
                        [20, 45, 65, 90, 100]
                    ),
                    0.30
                )
            )

        # Gross margin
        gross_margin = metrics.get(
            'gross_margin'
        )

        if pd.notna(gross_margin):

            quality_components.append(
                (
                    np.interp(
                        gross_margin,
                        [0.15, 0.30, 0.45, 0.60, 0.75],
                        [20, 40, 65, 85, 100]
                    ),
                    0.15
                )
            )

        # FCF margin
        fcf_margin = metrics.get(
            'fcf_margin'
        )

        if pd.notna(fcf_margin):

            quality_components.append(
                (
                    np.interp(
                        fcf_margin,
                        [-0.05, 0.00, 0.05, 0.10, 0.20],
                        [0, 30, 65, 85, 100]
                    ),
                    0.20
                )
            )

    # Expected earnings growth
    earnings_growth = metrics.get(
        'earnings_growth'
    )

    if pd.notna(earnings_growth):

        quality_components.append(
            (
                np.interp(
                    earnings_growth,
                    [-0.10, 0.00, 0.05, 0.15, 0.30, 0.50],
                    [10, 35, 55, 75, 90, 100]
                ),
                0.20
            )
        )

    # Expected revenue growth
    revenue_growth = metrics.get(
        'revenue_growth'
    )

    if pd.notna(revenue_growth):

        quality_components.append(
            (
                np.interp(
                    revenue_growth,
                    [-0.10, 0.00, 0.05, 0.15, 0.30, 0.50],
                    [10, 35, 55, 75, 90, 100]
                ),
                0.10
            )
        )

    if quality_components:

        weighted_sum = sum(
            score * weight
            for score, weight
            in quality_components
        )

        weight_sum = sum(
            weight
            for _, weight
            in quality_components
        )

        if weight_sum > 0:

            scores['quality'] = (
                weighted_sum
                / weight_sum
            )

    # ========================================================
    # RISK
    # ========================================================
    #
    # Risk = financial strength + market risk + drawdown +
    # business stability. 100 = lower risk, 0 = higher risk.
    # Valuation is deliberately excluded because it is already
    # represented by Fair Value / Valuation.

    risk_components = []

    # --------------------------------------------------------
    # 1. FINANCIAL STRENGTH (40%)
    # --------------------------------------------------------

    financial_components = []

    if sector == 'Financial Services':

        debt_to_equity = metrics.get(
            'debt_to_equity'
        )

        if (
            pd.notna(debt_to_equity)
            and debt_to_equity >= 0
        ):

            financial_components.append(
                (
                    np.interp(
                        debt_to_equity,
                        [20, 50, 100, 200, 400],
                        [100, 90, 70, 40, 0]
                    ),
                    0.60
                )
            )

        cash_to_debt = metrics.get(
            'cash_to_debt'
        )

        if (
            pd.notna(cash_to_debt)
            and cash_to_debt >= 0
        ):

            financial_components.append(
                (
                    np.interp(
                        cash_to_debt,
                        [0.05, 0.20, 0.40, 0.75, 1.50],
                        [10, 35, 60, 85, 100]
                    ),
                    0.40
                )
            )

    else:

        net_debt_ebitda = metrics.get(
            'net_debt_ebitda'
        )

        if pd.notna(net_debt_ebitda):

            financial_components.append(
                (
                    np.interp(
                        net_debt_ebitda,
                        [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        [100, 100, 85, 70, 50, 30, 10, 0]
                    ),
                    0.45
                )
            )

        current_ratio = metrics.get(
            'current_ratio'
        )

        if pd.notna(current_ratio):

            financial_components.append(
                (
                    np.interp(
                        current_ratio,
                        [0.5, 0.8, 1.0, 1.5, 2.0, 2.5],
                        [10, 35, 55, 80, 95, 100]
                    ),
                    0.20
                )
            )

        debt_to_equity = metrics.get(
            'debt_to_equity'
        )

        if (
            pd.notna(debt_to_equity)
            and debt_to_equity >= 0
        ):

            financial_components.append(
                (
                    np.interp(
                        debt_to_equity,
                        [10, 30, 60, 100, 150, 250, 400],
                        [100, 90, 75, 55, 40, 15, 0]
                    ),
                    0.35
                )
            )

    if financial_components:

        financial_weighted_sum = sum(
            score * weight
            for score, weight
            in financial_components
        )

        financial_weight_sum = sum(
            weight
            for _, weight
            in financial_components
        )

        if financial_weight_sum > 0:

            financial_score = (
                financial_weighted_sum
                / financial_weight_sum
            )

            risk_components.append(
                (
                    financial_score,
                    0.40
                )
            )

    # --------------------------------------------------------
    # 2. HISTORICAL VOLATILITY (15%)
    # --------------------------------------------------------

    annualized_volatility = metrics.get(
        'annualized_volatility'
    )

    if pd.notna(annualized_volatility):

        volatility_score = np.interp(
            annualized_volatility,
            [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
            [100, 90, 80, 65, 50, 25, 0]
        )

        risk_components.append(
            (
                volatility_score,
                0.15
            )
        )

    # --------------------------------------------------------
    # 3. MAXIMUM DRAWDOWN (15%)
    # --------------------------------------------------------

    max_drawdown = metrics.get(
        'max_drawdown'
    )

    if pd.notna(max_drawdown):

        drawdown_score = np.interp(
            abs(max_drawdown),
            [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70],
            [100, 90, 80, 60, 40, 20, 0]
        )

        risk_components.append(
            (
                drawdown_score,
                0.15
            )
        )

    # --------------------------------------------------------
    # 4. BUSINESS STABILITY (30%)
    # --------------------------------------------------------

    stability_components = []

    revenue_growth_volatility = metrics.get(
        'revenue_growth_volatility'
    )

    if pd.notna(revenue_growth_volatility):

        revenue_stability_score = np.interp(
            revenue_growth_volatility,
            [0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
            [100, 85, 70, 50, 25, 0]
        )

        stability_components.append(
            (
                revenue_stability_score,
                0.40
            )
        )

    earnings_growth_volatility = metrics.get(
        'earnings_growth_volatility'
    )

    if pd.notna(earnings_growth_volatility):

        earnings_stability_score = np.interp(
            earnings_growth_volatility,
            [0.05, 0.10, 0.20, 0.30, 0.50, 0.70],
            [100, 85, 65, 45, 20, 0]
        )

        stability_components.append(
            (
                earnings_stability_score,
                0.60
            )
        )

    if stability_components:

        stability_weighted_sum = sum(
            score * weight
            for score, weight
            in stability_components
        )

        stability_weight_sum = sum(
            weight
            for _, weight
            in stability_components
        )

        if stability_weight_sum > 0:

            stability_score = (
                stability_weighted_sum
                / stability_weight_sum
            )

            risk_components.append(
                (
                    stability_score,
                    0.30
                )
            )

    if risk_components:

        weighted_sum = sum(
            score * weight
            for score, weight
            in risk_components
        )

        weight_sum = sum(
            weight
            for _, weight
            in risk_components
        )

        if weight_sum > 0:

            scores['risk'] = (
                weighted_sum
                / weight_sum
            )

    # ========================================================
    # TECHNICAL
    # ========================================================

    technical_components = []

    # --------------------------------------------------------
    # SMA200 / long-term trend
    # --------------------------------------------------------

    above_sma200 = metrics.get(
        'above_sma200'
    )

    if pd.notna(above_sma200):

        technical_components.append(
            (
                80 if above_sma200 else 30,
                0.25
            )
        )

    # --------------------------------------------------------
    # 6M momentum
    # --------------------------------------------------------

    perf_6m = metrics.get(
        'perf_6m'
    )

    if pd.notna(perf_6m):

        technical_components.append(
            (
                np.interp(
                    perf_6m,
                    [-0.40, -0.20, 0.0, 0.15, 0.35, 0.60],
                    [0, 20, 45, 70, 90, 100]
                ),
                0.25
            )
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi = metrics.get(
        'rsi14'
    )

    if pd.notna(rsi):

        rsi_score = np.interp(
            rsi,
            [20, 30, 40, 50, 60, 70, 80, 90],
            [20, 35, 55, 68, 78, 85, 70, 50]
        )

        technical_components.append(
            (
                rsi_score,
                0.20
            )
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd = metrics.get(
        'macd'
    )

    macd_signal = metrics.get(
        'macd_signal'
    )

    macd_hist = metrics.get(
        'macd_hist'
    )

    if (
        pd.notna(macd)
        and pd.notna(macd_signal)
        and pd.notna(macd_hist)
    ):

        if macd > macd_signal:

            macd_score = 70

            if macd_hist > 0:
                macd_score += 15

        else:

            macd_score = 35

            if macd_hist < 0:
                macd_score -= 10

        macd_score = np.clip(
            macd_score,
            0,
            100
        )

        technical_components.append(
            (
                macd_score,
                0.20
            )
        )

    # --------------------------------------------------------
    # SMA50 vs SMA200
    # --------------------------------------------------------

    sma50 = metrics.get(
        'sma50'
    )

    sma200 = metrics.get(
        'sma200'
    )

    if (
        pd.notna(sma50)
        and pd.notna(sma200)
    ):

        technical_components.append(
            (
                80 if sma50 > sma200 else 35,
                0.10
            )
        )

    if technical_components:

        weighted_sum = sum(
            score * weight
            for score, weight
            in technical_components
        )

        weight_sum = sum(
            weight
            for _, weight
            in technical_components
        )

        if weight_sum > 0:

            scores['technical'] = (
                weighted_sum
                / weight_sum
            )

    # ========================================================
    # FINAL SCORE
    # ========================================================

    available_weight = 0.0
    weighted_score = 0.0

    for category, weight in SCORE_WEIGHTS.items():

        value = scores.get(
            category,
            np.nan
        )

        if pd.notna(value):

            weighted_score += (
                value
                * weight
            )

            available_weight += weight

    missing_weight = (
        1.0
        - available_weight
    )

    if available_weight <= 0:

        return np.nan, scores

    total_score = (
        weighted_score
        + 50.0 * missing_weight
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

    rsi = metrics.get(
        'rsi14'
    )

    macd = metrics.get(
        'macd'
    )

    macd_signal = metrics.get(
        'macd_signal'
    )

    if (
        pd.notna(perf_1m)
        and pd.notna(perf_6m)
    ):

        if (
            perf_1m > 0
            and perf_6m < 0
        ):

            if (
                pd.notna(rsi)
                and rsi < 50
            ):

                return "Turnaround?"

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

    if (
        pd.notna(macd)
        and pd.notna(macd_signal)
        and macd > macd_signal
        and pd.notna(rsi)
        and rsi < 50
    ):

        return "Turnaround?"

    return "Neutral"


# ============================================================
# RADAR
# ============================================================

def create_radar(scores):

    categories = [
        'Fair Value',
        'Valuation',
        'Quality',
        'Risk',
        'Technical'
    ]

    values = [
        scores.get(
            'fair_value',
            np.nan
        ),
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
            'technical',
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
# TABLE COLORING
# ============================================================

def style_results_table(df):

    score_columns = [
        'Gesamtscore',
        'Fair Value',
        'Valuation',
        'Quality',
        'Risk',
        'Technical'
    ]

    completeness_column = [
        'Datenvollständigkeit (%)'
    ]

    styler = (
        df.style
        .background_gradient(
            cmap='RdYlGn',
            subset=score_columns,
            vmin=0,
            vmax=100
        )
        .background_gradient(
            cmap='RdYlGn',
            subset=completeness_column,
            vmin=0,
            vmax=100
        )
        .format({
            'Gesamtscore': '{:.1f}',
            'Fair Value': '{:.0f}',
            'Valuation': '{:.0f}',
            'Quality': '{:.0f}',
            'Risk': '{:.0f}',
            'Technical': '{:.0f}',
            'Forward-KGV': '{:.1f}',
            'Datenvollständigkeit (%)': '{:.0f}%'
        })
    )

    return styler


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 Quant-Aktien-Screener V10.8"
)

st.caption(
    "Fundamentaler und technischer Aktien-Score "
    "mit Fair Value, Bewertung, Qualität, Risiko und Markttechnik"
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
# ANALYSE
# ============================================================

if st.button(
    "🚀 Aktien analysieren"
):

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

    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []

    # --------------------------------------------------------
    # FETCH + SCORE
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

        if (
            pd.notna(score)
            and score < min_score
        ):

            progress_bar.progress(
                (i + 1) / len(tickers)
            )

            continue

        results.append({

            'Ticker':
                metrics['ticker'],

            'Name':
                metrics['name'],

            'Sektor':
                metrics['sector'],

            'Gesamtscore':
                score,

            'Datenvollständigkeit (%)':
                metrics['data_completeness'],

            'Empfehlung':
                get_recommendation(score),

            'Turnaround Status':
                get_turnaround_status(metrics),

            'Fair Value':
                scores.get(
                    'fair_value',
                    np.nan
                ),

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

            'Technical':
                scores.get(
                    'technical',
                    np.nan
                ),

            'Forward-KGV':
                metrics.get(
                    'forward_pe',
                    np.nan
                )
        })

        progress_bar.progress(
            (i + 1) / len(tickers)
        )

    progress_bar.empty()
    status_text.empty()

    # ========================================================
    # RESULTS
    # ========================================================

    if not results:

        st.warning(
            "Keine Aktien konnten mit den "
            "aktuellen Einstellungen bewertet werden."
        )

        st.stop()

    results_df = pd.DataFrame(
        results
    )

    numeric_columns = [
        'Fair Value',
        'Valuation',
        'Quality',
        'Risk',
        'Technical',
        'Gesamtscore',
        'Forward-KGV',
        'Datenvollständigkeit (%)'
    ]

    for col in numeric_columns:

        results_df[col] = pd.to_numeric(
            results_df[col],
            errors='coerce'
        )

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
        style_results_table(
            results_df
        ),
        use_container_width=True,
        hide_index=True
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
        # COMPANY NAME
        # ----------------------------------------------------

        st.markdown(
            f"### {detail_metrics['name']} "
            f"({selected_ticker})"
        )

        st.caption(
            f"Sektor: {detail_metrics['sector']}"
        )

        # ----------------------------------------------------
        # SCORE CARDS
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
                "Fair Value",
                (
                    f"{detail_scores['fair_value']:.0f}"
                    if pd.notna(
                        detail_scores['fair_value']
                    )
                    else "–"
                )
            )

        with col3:

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

        with col4:

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

        with col5:

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

        # Technical remains visible below the main cards
        st.metric(
            "Technical",
            (
                f"{detail_scores['technical']:.0f}"
                if pd.notna(
                    detail_scores['technical']
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
        # RADAR + DETAILS
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

        with detail_col:

            detail_df = pd.DataFrame({

                'Kennzahl': [

                    'Aktueller Kurs',
                    'Marktkapitalisierung',

                    'Fair Value',
                    'Fair Value Potenzial',
                    'Analysten-Kursziel',
                    'Analysten-Potenzial',
                    'FCF Fair Value',

                    'Forward-KGV',
                    'Trailing-KGV',
                    'KUV',
                    'KBV',
                    'Forward-PEG',

                    'Forward EPS',
                    'Trailing EPS',
                    'Gewinnwachstum',
                    'Umsatzwachstum',

                    'Bruttomarge',
                    'Nettomarge',
                    'ROE',
                    'ROIC',

                    'Operativer Cashflow',
                    'FCF',
                    'FCF-Marge',
                    'FCF-Rendite',

                    'Net Debt / EBITDA',
                    'Debt / Equity',
                    'Cash / Debt',
                    'Current Ratio',

                    'RSI (14)',
                    'MACD',
                    'MACD Signal',
                    'MACD Histogramm',

                    'SMA50',
                    'SMA200',
                    '1M Performance',
                    '6M Performance',
                    '12M Performance',
                    'Über SMA200',

                    'Datenvollständigkeit'
                ],

                'Wert': [

                    detail_metrics['price'],
                    detail_metrics['market_cap'],

                    detail_metrics['fair_value_price'],
                    detail_metrics['fair_value_upside'],
                    detail_metrics['analyst_target_mean'],
                    detail_metrics['analyst_target_upside'],
                    detail_metrics['fcf_fair_value'],

                    detail_metrics['forward_pe'],
                    detail_metrics['trailing_pe'],
                    detail_metrics['ps_ratio'],
                    detail_metrics['pb_ratio'],
                    detail_metrics['forward_peg'],

                    detail_metrics['forward_eps'],
                    detail_metrics['trailing_eps'],
                    detail_metrics['earnings_growth'],
                    detail_metrics['revenue_growth'],

                    detail_metrics['gross_margin'],
                    detail_metrics['net_margin'],
                    detail_metrics['roe'],
                    detail_metrics['roic'],

                    detail_metrics['operating_cf'],
                    detail_metrics['fcf'],
                    detail_metrics['fcf_margin'],
                    detail_metrics['fcf_yield'],

                    detail_metrics['net_debt_ebitda'],
                    detail_metrics['debt_to_equity'],
                    detail_metrics['cash_to_debt'],
                    detail_metrics['current_ratio'],

                    detail_metrics['rsi14'],
                    detail_metrics['macd'],
                    detail_metrics['macd_signal'],
                    detail_metrics['macd_hist'],

                    detail_metrics['sma50'],
                    detail_metrics['sma200'],
                    detail_metrics['perf_1m'],
                    detail_metrics['perf_6m'],
                    detail_metrics['perf_12m'],

                    (
                        "Ja"
                        if detail_metrics[
                            'above_sma200'
                        ] is True

                        else (
                            "Nein"
                            if detail_metrics[
                                'above_sma200'
                            ] is False
                            else "–"
                        )
                    ),

                    f"{detail_metrics['data_completeness']}%"
                ]
            })

            # ------------------------------------------------
            # FORMAT DETAILS
            # ------------------------------------------------

            percentage_metrics = {
                'Fair Value Potenzial',
                'Analysten-Potenzial',
                'Gewinnwachstum',
                'Umsatzwachstum',
                'Bruttomarge',
                'Nettomarge',
                'ROE',
                'ROIC',
                'FCF-Marge',
                'FCF-Rendite',
                '1M Performance',
                '6M Performance',
                '12M Performance'
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

                        formatted_values.append(
                            "–"
                        )

                elif metric == 'Marktkapitalisierung':

                    if pd.notna(value):

                        formatted_values.append(
                            f"{value / 1_000_000_000:.2f} Mrd."
                        )

                    else:

                        formatted_values.append(
                            "–"
                        )

                elif metric in [
                    'Operativer Cashflow',
                    'FCF'
                ]:

                    if pd.notna(value):

                        formatted_values.append(
                            f"{value / 1_000_000:.0f} Mio."
                        )

                    else:

                        formatted_values.append(
                            "–"
                        )

                elif metric in [
                    'Forward-KGV',
                    'Trailing-KGV',
                    'KUV',
                    'KBV',
                    'Forward-PEG',
                    'Net Debt / EBITDA',
                    'Debt / Equity',
                    'Cash / Debt',
                    'Current Ratio',
                    'RSI (14)',
                    'MACD',
                    'MACD Signal',
                    'MACD Histogramm'
                ]:

                    if pd.notna(value):

                        formatted_values.append(
                            f"{value:.2f}"
                        )

                    else:

                        formatted_values.append(
                            "–"
                        )

                elif metric in [
                    'Aktueller Kurs',
                    'Fair Value',
                    'Analysten-Kursziel',
                    'FCF Fair Value',
                    'Forward EPS',
                    'Trailing EPS',
                    'SMA50',
                    'SMA200'
                ]:

                    if pd.notna(value):

                        formatted_values.append(
                            f"{value:.2f}"
                        )

                    else:

                        formatted_values.append(
                            "–"
                        )

                else:

                    formatted_values.append(
                        str(value)
                    )

            detail_df['Wert'] = (
                formatted_values
            )

            st.dataframe(
                detail_df,
                use_container_width=True,
                hide_index=True
            )

        # ----------------------------------------------------
        # STATUS
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
