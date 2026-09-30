import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Aktien-Screener V7",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Aktien-Screener V7 – Live Ranking")


# ============================================================
# TICKER-MAPPING
# ============================================================
# Yahoo Finance benötigt bei manchen europäischen Aktien
# einen Börsenplatz-Suffix.

TICKER_MAP = {
    "SAP": "SAP.DE",
    "SIE": "SIE.DE",
    "MC": "MC.PA",
    "OR": "OR.PA",
    "AIR": "AIR.PA",
    "SU": "SU.PA",
    "TTE": "TTE.PA",

    # bereits börsenplatzspezifisch
    "IRE.MI": "IRE.MI",
    "SMTOF": "SMTOF",
    "SSUN.F": "SSUN.F",
    "3CP.F": "3CP.F",

    # ADR / US-Ticker
    "NVO": "NVO",
    "BYDDY": "BYDDY",
    "NOK": "NOK",
    "TSM": "TSM",
    "ASML": "ASML",
    "MU": "MU",
    "VRT": "VRT",
    "VST": "VST",
    "PEP": "PEP",
    "PFE": "PFE",
    "AMZN": "AMZN",
    "MRVL": "MRVL",
    "NKE": "NKE",
    "AAPL": "AAPL",
    "AVGO": "AVGO",
    "META": "META",
    "TSLA": "TSLA",
    "MSFT": "MSFT",
    "GOOGL": "GOOGL",
    "NVDA": "NVDA",
}


# ============================================================
# SCORE-SPALTEN
# ============================================================

REQUIRED_COLUMNS = [
    "current_price",
    "pe_ratio",
    "peg_ratio",
    "pfcf_ratio",
    "eps_forward",
    "eps_growth_5y",
    "roe",
    "roic",
    "fcf_margin",
    "revenue_growth_3y",
    "fcf_growth_3y",
    "net_debt_ebitda",
    "interest_coverage",
    "fcf_to_debt",
    "beta",
    "dist_52w_high",
    "sma_200",
    "performance_6m",
    "analyst_upside",
]


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def num(value):
    """Sichere Umwandlung in float."""

    try:

        if value is None:
            return np.nan

        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return np.nan
            value = value[0]

        value = float(value)

        if np.isfinite(value):
            return value

        return np.nan

    except Exception:
        return np.nan


def first_valid(*values):

    for value in values:

        value = num(value)

        if pd.notna(value):
            return value

    return np.nan


def info_value(info, *keys):

    for key in keys:

        if key in info:

            value = num(info.get(key))

            if pd.notna(value):
                return value

    return np.nan


def safe_divide(a, b):

    a = num(a)
    b = num(b)

    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan

    return a / b


def statement_value(df, possible_names, latest=True):

    if df is None or df.empty:
        return np.nan

    for name in possible_names:

        if name in df.index:

            try:

                values = (
                    pd.to_numeric(
                        df.loc[name],
                        errors="coerce",
                    )
                    .dropna()
                )

                if values.empty:
                    continue

                if latest:
                    return num(values.iloc[0])

                return values

            except Exception:
                continue

    return np.nan


def statement_series(df, possible_names):

    if df is None or df.empty:
        return pd.Series(dtype=float)

    for name in possible_names:

        if name in df.index:

            try:

                series = pd.to_numeric(
                    df.loc[name],
                    errors="coerce",
                ).dropna()

                if not series.empty:
                    return series.sort_index()

            except Exception:
                pass

    return pd.Series(dtype=float)


def cagr_from_series(series, years=3):

    if series is None or len(series) < years + 1:
        return np.nan

    try:

        series = (
            pd.to_numeric(
                series,
                errors="coerce",
            )
            .dropna()
            .sort_index()
        )

        if len(series) < years + 1:
            return np.nan

        old_value = num(series.iloc[-(years + 1)])
        new_value = num(series.iloc[-1])

        if (
            pd.isna(old_value)
            or pd.isna(new_value)
            or old_value <= 0
            or new_value <= 0
        ):
            return np.nan

        return (
            (new_value / old_value) ** (1 / years) - 1
        ) * 100

    except Exception:
        return np.nan


# ============================================================
# SCORING-HILFSFUNKTIONEN
# ============================================================

def ratio_score(series, excellent, good, weak):

    s = pd.to_numeric(series, errors="coerce")

    # Negative / 0 bei z.B. P/E, PEG oder P/FCF
    # ist nicht automatisch gut -> neutral behandeln.
    valid = s.where(s > 0)

    result = np.select(
        [
            valid <= excellent,
            (valid > excellent) & (valid <= good),
            (valid > good) & (valid <= weak),
            valid > weak,
        ],
        [
            100,
            100 - 30 * (valid - excellent) / (good - excellent),
            70 - 40 * (valid - good) / (weak - good),
            10,
        ],
        default=50,
    )

    result = np.asarray(result, dtype=float)

    result[pd.isna(valid)] = 50

    return result


def growth_score(series, excellent, good, weak):

    s = pd.to_numeric(series, errors="coerce")

    result = np.select(
        [
            s >= excellent,
            (s >= good) & (s < excellent),
            (s >= weak) & (s < good),
            s < weak,
        ],
        [
            100,
            70 + 30 * (s - good) / (excellent - good),
            30 + 40 * (s - weak) / (good - weak),
            10,
        ],
        default=50,
    )

    result = np.asarray(result, dtype=float)

    result[pd.isna(s)] = 50

    return result


def bounded_score(series, excellent, good, weak, bad):

    s = pd.to_numeric(series, errors="coerce")

    result = np.select(
        [
            s <= excellent,
            (s > excellent) & (s <= good),
            (s > good) & (s <= weak),
            s > weak,
        ],
        [
            100,
            100 - 30 * (s - excellent) / (good - excellent),
            70 - 40 * (s - good) / (weak - good),
            bad,
        ],
        default=50,
    )

    result = np.asarray(result, dtype=float)
    result[pd.isna(s)] = 50

    return result


# ============================================================
# ROIC BERECHNUNG
# ============================================================

def calculate_roic(info, income, balance):

    # Operating Income
    operating_income = first_valid(
        info_value(
            info,
            "operatingIncome",
        ),
        statement_value(
            income,
            [
                "Operating Income",
                "OperatingIncome",
                "EBIT",
            ],
        ),
    )

    if pd.isna(operating_income):
        return np.nan

    # Eigenkapital
    equity = first_valid(
        info_value(
            info,
            "stockholdersEquity",
            "totalStockholderEquity",
        ),
        statement_value(
            balance,
            [
                "Stockholders Equity",
                "StockholdersEquity",
                "Total Equity Gross Minority Interest",
                "Common Stock Equity",
            ],
        ),
    )

    # Schulden
    debt = first_valid(
        info_value(
            info,
            "totalDebt",
        ),
        statement_value(
            balance,
            [
                "Total Debt",
                "TotalDebt",
            ],
        ),
    )

    # Cash
    cash = first_valid(
        info_value(
            info,
            "totalCash",
        ),
        statement_value(
            balance,
            [
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Cash Equivalents",
                "Cash",
            ],
        ),
    )

    if (
        pd.isna(equity)
        or pd.isna(debt)
        or pd.isna(cash)
    ):
        return np.nan

    invested_capital = (
        equity
        + debt
        - cash
    )

    if invested_capital <= 0:
        return np.nan

    # Steuerquote
    tax_rate = np.nan

    tax_provision = statement_value(
        income,
        [
            "Tax Provision",
            "TaxProvision",
        ],
    )

    pretax_income = statement_value(
        income,
        [
            "Pretax Income",
            "PretaxIncome",
        ],
    )

    if (
        pd.notna(tax_provision)
        and pd.notna(pretax_income)
        and pretax_income > 0
    ):

        tax_rate = (
            tax_provision
            / pretax_income
        )

        tax_rate = np.clip(
            tax_rate,
            0,
            0.35,
        )

    if pd.isna(tax_rate):

        tax_rate = 0.21

    nopat = (
        operating_income
        * (1 - tax_rate)
    )

    return safe_divide(
        nopat,
        invested_capital,
    )


# ============================================================
# LIVE-DATEN VON YAHOO FINANCE
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def fetch_stock_data(user_ticker):

    user_ticker = user_ticker.upper().strip()

    yahoo_ticker = TICKER_MAP.get(
        user_ticker,
        user_ticker,
    )

    result = {
        "symbol": user_ticker,
        "yahoo_symbol": yahoo_ticker,
        "name": user_ticker,
        "currency": "",
        "error": "",
        "data_quality": 0,
    }

    # Alle Score-Spalten von Anfang an erzeugen.
    # Damit kann später KEIN KeyError entstehen.
    for column in REQUIRED_COLUMNS:
        result[column] = np.nan

    try:

        ticker = yf.Ticker(yahoo_ticker)

        # ====================================================
        # INFO
        # ====================================================

        try:
            info = ticker.get_info()
        except Exception:
            info = {}

        if not info:
            result["error"] = (
                "Yahoo Finance lieferte keine Fundamentaldaten."
            )
            return result

        result["name"] = (
            info.get("longName")
            or info.get("shortName")
            or user_ticker
        )

        result["currency"] = (
            info.get("currency")
            or ""
        )

        # ====================================================
        # KURS
        # ====================================================

        current_price = first_valid(
            info.get("currentPrice"),
            info.get("regularMarketPrice"),
            info.get("previousClose"),
        )

        # Fallback über fast_info
        if pd.isna(current_price):

            try:

                current_price = num(
                    ticker.fast_info.get(
                        "lastPrice"
                    )
                )

            except Exception:
                pass

        result["current_price"] = current_price

        # ====================================================
        # VALUATION
        # ====================================================

        result["pe_ratio"] = first_valid(
            info.get("forwardPE"),
            info.get("trailingPE"),
        )

        result["peg_ratio"] = info_value(
            info,
            "pegRatio",
        )

        result["pfcf_ratio"] = info_value(
            info,
            "priceToFreeCashflow",
        )

        result["eps_forward"] = first_valid(
            info.get("forwardEps"),
            info.get("trailingEps"),
        )

        # ====================================================
        # ANALYSTENZIEL
        # ====================================================

        try:

            targets = ticker.get_analyst_price_targets()

            target_mean = first_valid(
                targets.get("mean"),
                targets.get("median"),
            )

            if (
                pd.notna(target_mean)
                and pd.notna(current_price)
                and current_price > 0
            ):

                result["analyst_upside"] = (
                    target_mean
                    / current_price
                    - 1
                )

        except Exception:
            pass

        # ====================================================
        # GROWTH ESTIMATES
        # ====================================================

        try:

            growth = ticker.get_growth_estimates()

            if (
                growth is not None
                and not growth.empty
                and "+5y" in growth.index
            ):

                row = growth.loc["+5y"]

                if isinstance(row, pd.Series):

                    if "stock" in row.index:
                        value = row["stock"]
                    else:
                        value = row.iloc[0]

                else:
                    value = row

                value = num(value)

                if pd.notna(value):

                    # Yahoo meist 0.15 = 15 %
                    if abs(value) < 1:
                        value *= 100

                    result["eps_growth_5y"] = value

        except Exception:
            pass

        # Fallback
        if pd.isna(result["eps_growth_5y"]):

            fallback = info_value(
                info,
                "earningsGrowth",
            )

            if pd.notna(fallback):

                if abs(fallback) < 1:
                    fallback *= 100

                result["eps_growth_5y"] = fallback

        # ====================================================
        # FINANCIAL STATEMENTS
        # ====================================================

        income = pd.DataFrame()
        balance = pd.DataFrame()
        cashflow = pd.DataFrame()

        try:
            income = ticker.get_income_stmt(
                freq="yearly"
            )
        except Exception:
            pass

        try:
            balance = ticker.get_balance_sheet(
                freq="yearly"
            )
        except Exception:
            pass

        try:
            cashflow = ticker.get_cash_flow(
                freq="yearly"
            )
        except Exception:
            pass

        # ====================================================
        # REVENUE
        # ====================================================

        revenue_series = statement_series(
            income,
            [
                "Total Revenue",
                "TotalRevenue",
            ],
        )

        revenue = np.nan

        if not revenue_series.empty:

            revenue = num(
                revenue_series.iloc[-1]
            )

            result["revenue_growth_3y"] = (
                cagr_from_series(
                    revenue_series,
                    years=3,
                )
            )

        else:

            revenue = info_value(
                info,
                "totalRevenue",
            )

            revenue_growth = info_value(
                info,
                "revenueGrowth",
            )

            if pd.notna(revenue_growth):

                result["revenue_growth_3y"] = (
                    revenue_growth * 100
                )

        # ====================================================
        # FCF
        # ====================================================

        fcf = info_value(
            info,
            "freeCashflow",
        )

        fcf_series = statement_series(
            cashflow,
            [
                "Free Cash Flow",
                "FreeCashFlow",
            ],
        )

        if pd.isna(fcf):

            if not fcf_series.empty:
                fcf = num(
                    fcf_series.iloc[-1]
                )

        # FCF Margin
        result["fcf_margin"] = safe_divide(
            fcf,
            revenue,
        )

        # FCF Growth
        if not fcf_series.empty:

            result["fcf_growth_3y"] = (
                cagr_from_series(
                    fcf_series,
                    years=3,
                )
            )

        # ====================================================
        # ROE
        # ====================================================

        result["roe"] = info_value(
            info,
            "returnOnEquity",
        )

        # ====================================================
        # ROIC
        # ====================================================

        result["roic"] = calculate_roic(
            info,
            income,
            balance,
        )

        # ====================================================
        # DEBT
        # ====================================================

        total_debt = first_valid(
            info_value(
                info,
                "totalDebt",
            ),
            statement_value(
                balance,
                [
                    "Total Debt",
                    "TotalDebt",
                ],
            ),
        )

        total_cash = first_valid(
            info_value(
                info,
                "totalCash",
            ),
            statement_value(
                balance,
                [
                    "Cash Cash Equivalents And Short Term Investments",
                    "Cash And Cash Equivalents",
                    "Cash",
                ],
            ),
        )

        ebitda = first_valid(
            info_value(
                info,
                "ebitda",
            ),
            info_value(
                info,
                "normalizedEBITDA",
            ),
        )

        # Net Debt / EBITDA
        if (
            pd.notna(total_debt)
            and pd.notna(total_cash)
            and pd.notna(ebitda)
            and ebitda > 0
        ):

            net_debt = (
                total_debt
                - total_cash
            )

            result["net_debt_ebitda"] = (
                net_debt
                / ebitda
            )

        # FCF / Debt
        if (
            pd.notna(fcf)
            and pd.notna(total_debt)
            and total_debt > 0
        ):

            result["fcf_to_debt"] = (
                fcf
                / total_debt
            )

        # ====================================================
        # INTEREST COVERAGE
        # ====================================================

        operating_income = statement_value(
            income,
            [
                "Operating Income",
                "OperatingIncome",
                "EBIT",
            ],
        )

        interest_expense = statement_value(
            income,
            [
                "Interest Expense",
                "Interest Expense Non Operating",
                "InterestExpenseNonOperating",
            ],
        )

        if (
            pd.notna(operating_income)
            and pd.notna(interest_expense)
            and interest_expense != 0
        ):

            result["interest_coverage"] = (
                operating_income
                / abs(interest_expense)
            )

        else:

            # Yahoo kann EBIT/Interest Coverage direkt liefern
            direct_coverage = info_value(
                info,
                "ebitToInterestExpense",
                "interestCoverage",
            )

            result["interest_coverage"] = (
                direct_coverage
            )

        # ====================================================
        # BETA
        # ====================================================

        result["beta"] = info_value(
            info,
            "beta",
        )

        # ====================================================
        # HISTORISCHE KURSE
        # ====================================================

        history = pd.DataFrame()

        try:

            history = ticker.history(
                period="2y",
                interval="1d",
                auto_adjust=False,
            )

        except Exception:
            pass

        if (
            history is not None
            and not history.empty
            and "Close" in history.columns
        ):

            close = (
                pd.to_numeric(
                    history["Close"],
                    errors="coerce",
                )
                .dropna()
            )

            # ------------------------------------------------
            # 52W HIGH
            # ------------------------------------------------

            if len(close) >= 20:

                high_52w = (
                    close.tail(252).max()
                )

                if (
                    pd.notna(current_price)
                    and pd.notna(high_52w)
                    and high_52w > 0
                ):

                    result["dist_52w_high"] = (
                        current_price
                        / high_52w
                        - 1
                    )

            # ------------------------------------------------
            # SMA200
            # ------------------------------------------------

            if len(close) >= 200:

                result["sma_200"] = (
                    close.tail(200).mean()
                )

            # ------------------------------------------------
            # 6 MONATE
            # ------------------------------------------------

            if len(close) >= 126:

                old_price = num(
                    close.iloc[-126]
                )

                new_price = num(
                    close.iloc[-1]
                )

                if (
                    pd.notna(old_price)
                    and pd.notna(new_price)
                    and old_price > 0
                ):

                    result["performance_6m"] = (
                        new_price
                        / old_price
                        - 1
                    )

        # ====================================================
        # DATA QUALITY
        # ====================================================

        valid_count = sum(
            pd.notna(
                result.get(
                    column,
                    np.nan,
                )
            )
            for column in REQUIRED_COLUMNS
        )

        result["data_quality"] = round(
            valid_count
            / len(REQUIRED_COLUMNS)
            * 100
        )

        return result

    except Exception as e:

        result["error"] = (
            f"{type(e).__name__}: {e}"
        )

        # Auch bei komplettem Fehler:
        # alle Spalten vorhanden lassen.
        return result


# ============================================================
# MEHRERE AKTIEN LADEN
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def load_stocks(tickers):

    rows = []

    for ticker in tickers:

        rows.append(
            fetch_stock_data(ticker)
        )

    return pd.DataFrame(rows)


# ============================================================
# SCORE V7
# ============================================================

def calculate_stock_score(df_input):

    df = df_input.copy()

    # --------------------------------------------------------
    # SICHERSTELLEN, DASS ALLE SPALTEN EXISTIEREN
    # --------------------------------------------------------

    for column in REQUIRED_COLUMNS:

        if column not in df.columns:
            df[column] = np.nan

    # ========================================================
    # 1. VALUATION – 25 %
    # ========================================================

    pe_score = ratio_score(
        df["pe_ratio"],
        excellent=12,
        good=20,
        weak=30,
    )

    peg_score = ratio_score(
        df["peg_ratio"],
        excellent=1.0,
        good=1.5,
        weak=2.5,
    )

    pfcf_score = ratio_score(
        df["pfcf_ratio"],
        excellent=15,
        good=25,
        weak=35,
    )

    df["fcf_yield"] = np.where(
        df["pfcf_ratio"] > 0,
        1 / df["pfcf_ratio"],
        np.nan,
    )

    fcf_yield_score = growth_score(
        df["fcf_yield"] * 100,
        excellent=8,
        good=5,
        weak=2,
    )

    # --------------------------------------------------------
    # Lynch Growth Value
    # --------------------------------------------------------

    growth_for_value = np.clip(
        pd.to_numeric(
            df["eps_growth_5y"],
            errors="coerce",
        ),
        0,
        30,
    )

    df["lynch_growth_value"] = (
        df["eps_forward"]
        * growth_for_value
    )

    df["margin_of_safety"] = np.where(
        (
            df["lynch_growth_value"] > 0
        )
        & (
            df["current_price"].notna()
        ),
        (
            df["lynch_growth_value"]
            - df["current_price"]
        )
        / df["lynch_growth_value"],
        np.nan,
    )

    mos_score = np.select(
        [
            df["margin_of_safety"] >= 0.30,
            df["margin_of_safety"] >= 0.10,
            df["margin_of_safety"] >= -0.10,
            df["margin_of_safety"] >= -0.30,
        ],
        [
            100,
            80,
            55,
            30,
        ],
        default=10,
    )

    mos_score = np.asarray(
        mos_score,
        dtype=float,
    )

    mos_score[
        df["margin_of_safety"].isna()
    ] = 50

    # --------------------------------------------------------
    # Analystenpotenzial
    # --------------------------------------------------------

    analyst_score = np.select(
        [
            df["analyst_upside"] >= 0.30,
            df["analyst_upside"] >= 0.15,
            df["analyst_upside"] >= 0,
            df["analyst_upside"] >= -0.15,
        ],
        [
            100,
            80,
            60,
            35,
        ],
        default=15,
    )

    analyst_score = np.asarray(
        analyst_score,
        dtype=float,
    )

    analyst_score[
        df["analyst_upside"].isna()
    ] = 50

    df["score_valuation"] = (
        pe_score * 0.20
        + peg_score * 0.20
        + pfcf_score * 0.15
        + fcf_yield_score * 0.15
        + mos_score * 0.15
        + analyst_score * 0.15
    )

    # ========================================================
    # 2. QUALITY – 30 %
    # ========================================================

    roic_score = growth_score(
        df["roic"] * 100,
        excellent=20,
        good=12,
        weak=6,
    )

    roe_score = growth_score(
        df["roe"] * 100,
        excellent=25,
        good=15,
        weak=8,
    )

    fcf_margin_score = growth_score(
        df["fcf_margin"] * 100,
        excellent=20,
        good=10,
        weak=5,
    )

    revenue_growth_score = growth_score(
        df["revenue_growth_3y"],
        excellent=15,
        good=8,
        weak=3,
    )

    fcf_growth_score = growth_score(
        df["fcf_growth_3y"],
        excellent=15,
        good=8,
        weak=3,
    )

    df["score_quality"] = (
        roic_score * 0.30
        + roe_score * 0.20
        + fcf_margin_score * 0.20
        + revenue_growth_score * 0.15
        + fcf_growth_score * 0.15
    )

    # ========================================================
    # 3. RISK – 15 %
    # ========================================================

    debt_score = np.select(
        [
            df["net_debt_ebitda"] <= 0,
            df["net_debt_ebitda"] <= 1.5,
            df["net_debt_ebitda"] <= 3.0,
            df["net_debt_ebitda"] <= 4.5,
        ],
        [
            100,
            90,
            65,
            35,
        ],
        default=10,
    )

    debt_score = np.asarray(
        debt_score,
        dtype=float,
    )

    debt_score[
        df["net_debt_ebitda"].isna()
    ] = 50

    interest_score = np.select(
        [
            df["interest_coverage"] >= 12,
            df["interest_coverage"] >= 8,
            df["interest_coverage"] >= 4,
            df["interest_coverage"] >= 2,
        ],
        [
            100,
            90,
            70,
            40,
        ],
        default=15,
    )

    interest_score = np.asarray(
        interest_score,
        dtype=float,
    )

    interest_score[
        df["interest_coverage"].isna()
    ] = 50

    fcf_debt_score = growth_score(
        df["fcf_to_debt"] * 100,
        excellent=50,
        good=30,
        weak=10,
    )

    beta_score = np.select(
        [
            df["beta"] <= 0.8,
            df["beta"] <= 1.2,
            df["beta"] <= 1.6,
            df["beta"] <= 2.0,
        ],
        [
            100,
            85,
            60,
            30,
        ],
        default=10,
    )

    beta_score = np.asarray(
        beta_score,
        dtype=float,
    )

    beta_score[
        df["beta"].isna()
    ] = 50

    df["score_risk"] = (
        debt_score * 0.40
        + interest_score * 0.25
        + fcf_debt_score * 0.20
        + beta_score * 0.15
    )

    # ========================================================
    # 4. GROWTH – 15 %
    # ========================================================

    eps_growth_score = growth_score(
        df["eps_growth_5y"],
        excellent=20,
        good=10,
        weak=3,
    )

    revenue_growth_score = growth_score(
        df["revenue_growth_3y"],
        excellent=15,
        good=8,
        weak=3,
    )

    fcf_growth_score = growth_score(
        df["fcf_growth_3y"],
        excellent=15,
        good=8,
        weak=3,
    )

    df["score_growth"] = (
        eps_growth_score * 0.45
        + revenue_growth_score * 0.30
        + fcf_growth_score * 0.25
    )

    # ========================================================
    # 5. DIP & TECHNICALS – 15 %
    # ========================================================

    dip_score = np.select(
        [
            df["dist_52w_high"] <= -0.30,
            df["dist_52w_high"] <= -0.20,
            df["dist_52w_high"] <= -0.10,
            df["dist_52w_high"] <= -0.03,
        ],
        [
            100,
            90,
            70,
            45,
        ],
        default=20,
    )

    dip_score = np.asarray(
        dip_score,
        dtype=float,
    )

    dip_score[
        df["dist_52w_high"].isna()
    ] = 50

    # Abstand zur 200-Tage-Linie
    ma_distance = (
        df["current_price"]
        / df["sma_200"]
        - 1
    )

    sma_score = np.select(
        [
            ma_distance >= 0.15,
            ma_distance >= 0.05,
            ma_distance >= 0,
            ma_distance >= -0.10,
            ma_distance >= -0.20,
        ],
        [
            100,
            90,
            75,
            50,
            30,
        ],
        default=15,
    )

    sma_score = np.asarray(
        sma_score,
        dtype=float,
    )

    sma_score[
        ma_distance.isna()
    ] = 50

    performance_score = np.select(
        [
            df["performance_6m"] <= -0.20,
            df["performance_6m"] <= -0.10,
            df["performance_6m"] <= 0,
            df["performance_6m"] <= 0.20,
        ],
        [
            100,
            80,
            60,
            35,
        ],
        default=20,
    )

    performance_score = np.asarray(
        performance_score,
        dtype=float,
    )

    performance_score[
        df["performance_6m"].isna()
    ] = 50

    df["score_dip_technicals"] = (
        dip_score * 0.40
        + sma_score * 0.35
        + performance_score * 0.25
    )

    # ========================================================
    # GESAMTSCORE
    # ========================================================

    df["total_score_v7"] = (
        df["score_valuation"] * 0.25
        + df["score_quality"] * 0.30
        + df["score_risk"] * 0.15
        + df["score_growth"] * 0.15
        + df["score_dip_technicals"] * 0.15
    ).round(1)

    # ========================================================
    # RATING
    # ========================================================

    df["rating"] = np.select(
        [
            df["total_score_v7"] >= 80,
            df["total_score_v7"] >= 70,
            df["total_score_v7"] >= 55,
        ],
        [
            "🟢 Starker Kauf",
            "🟡 Kaufenswert",
            "🟠 Halten / Beobachten",
        ],
        default="🔴 Verkaufen / Meiden",
    )

    df["stars"] = np.select(
        [
            df["total_score_v7"] >= 85,
            df["total_score_v7"] >= 75,
            df["total_score_v7"] >= 65,
            df["total_score_v7"] >= 50,
        ],
        [
            "⭐⭐⭐⭐⭐",
            "⭐⭐⭐⭐",
            "⭐⭐⭐",
            "⭐⭐",
        ],
        default="⭐",
    )

    return df.sort_values(
        "total_score_v7",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# TICKER EINGABE
# ============================================================

st.subheader("📌 Aktien auswählen")

ticker_input = st.text_input(
    "Ticker eingeben – mehrere mit Komma trennen",
    value=(
        "MSFT, GOOGL, NVDA, MU, SSUN.F, TSM, ASML, "
        "VRT, VST, PEP, NVO, PFE, AMZN, MRVL, NKE, "
        "BYDDY, NOK, AAPL, AVGO, META, TSLA, MC, SAP, "
        "SIE, TTE, SAN, SU, OR, AIR"
    ),
    placeholder="z. B. NVDA, MU, AMD, TSM, ASML",
)

tickers = [
    x.strip().upper()
    for x in ticker_input.split(",")
    if x.strip()
]

tickers = list(dict.fromkeys(tickers))

st.caption(
    f"{len(tickers)} Aktien ausgewählt"
)


# ============================================================
# BUTTON
# ============================================================

col_a, col_b = st.columns([4, 1])

with col_a:

    load_button = st.button(
        "🔄 Live-Daten abrufen",
        type="primary",
        use_container_width=True,
    )

with col_b:

    clear_button = st.button(
        "Cache löschen",
        use_container_width=True,
    )

if clear_button:

    st.cache_data.clear()
    st.rerun()

if load_button:

    st.session_state["data_loaded"] = True


if "data_loaded" not in st.session_state:

    st.session_state["data_loaded"] = False


if not st.session_state["data_loaded"]:

    st.info(
        "Ticker eingeben und anschließend "
        "„Live-Daten abrufen“ klicken."
    )

    st.stop()


# ============================================================
# LIVE DATEN LADEN
# ============================================================

with st.spinner(
    f"Yahoo Finance: {len(tickers)} Aktien werden geladen ..."
):

    df_live = load_stocks(
        tuple(tickers)
    )


# ============================================================
# FEHLER ANZEIGEN
# ============================================================

errors = df_live[
    df_live["error"].fillna("") != ""
].copy()

if not errors.empty:

    st.warning(
        f"{len(errors)} Ticker konnten nicht "
        "vollständig geladen werden."
    )

    st.dataframe(
        errors[
            [
                "symbol",
                "yahoo_symbol",
                "error",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# VERWERTBARE AKTIEN
# ============================================================

df_raw = df_live[
    df_live["current_price"].notna()
].copy()


if df_raw.empty:

    st.error(
        "Keine verwertbaren Live-Kurse gefunden."
    )

    st.stop()


# ============================================================
# SCORE
# ============================================================

df_results = calculate_stock_score(
    df_raw
)


# ============================================================
# TOP AKTIE
# ============================================================

st.markdown("---")

st.subheader("🏆 Höchster Score")

top = df_results.iloc[0]

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Aktie",
    f"{top['symbol']}",
)

c2.metric(
    "Score",
    f"{top['total_score_v7']}/100",
)

c3.metric(
    "Rating",
    top["rating"],
)

c4.metric(
    "Datenqualität",
    f"{top['data_quality']:.0f}%",
)


# ============================================================
# RANKING
# ============================================================

st.markdown("---")

st.subheader("📊 Live Aktien-Ranking")

ranking = df_results[
    [
        "symbol",
        "name",
        "stars",
        "rating",
        "total_score_v7",
        "score_valuation",
        "score_quality",
        "score_risk",
        "score_growth",
        "score_dip_technicals",
        "current_price",
        "currency",
        "analyst_upside",
        "margin_of_safety",
        "data_quality",
    ]
].copy()


st.dataframe(
    ranking,

    column_config={

        "symbol":
            st.column_config.TextColumn(
                "Ticker",
            ),

        "name":
            st.column_config.TextColumn(
                "Unternehmen",
            ),

        "stars":
            st.column_config.TextColumn(
                "Sterne",
            ),

        "rating":
            st.column_config.TextColumn(
                "Rating",
            ),

        "total_score_v7":
            st.column_config.ProgressColumn(
                "Score",
                min_value=0,
                max_value=100,
                format="%.1f",
            ),

        "score_valuation":
            st.column_config.NumberColumn(
                "Valuation 25%",
                format="%.1f",
            ),

        "score_quality":
            st.column_config.NumberColumn(
                "Quality 30%",
                format="%.1f",
            ),

        "score_risk":
            st.column_config.NumberColumn(
                "Risk 15%",
                format="%.1f",
            ),

        "score_growth":
            st.column_config.NumberColumn(
                "Growth 15%",
                format="%.1f",
            ),

        "score_dip_technicals":
            st.column_config.NumberColumn(
                "Dip/Tech 15%",
                format="%.1f",
            ),

        "current_price":
            st.column_config.NumberColumn(
                "Live-Kurs",
                format="%.2f",
            ),

        "currency":
            st.column_config.TextColumn(
                "Währung",
            ),

        "analyst_upside":
            st.column_config.NumberColumn(
                "Analystenpotenzial",
                format="%.1%",
            ),

        "margin_of_safety":
            st.column_config.NumberColumn(
                "Margin of Safety",
                format="%.1%",
            ),

        "data_quality":
            st.column_config.ProgressColumn(
                "Datenqualität",
                min_value=0,
                max_value=100,
                format="%.0f%%",
            ),
    },

    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DETAIL
# ============================================================

st.markdown("---")

st.subheader("🔍 Detailanalyse")

for _, row in df_results.iterrows():

    with st.expander(
        f"{row['stars']} "
        f"{row['symbol']} – "
        f"{row['name']} | "
        f"{row['total_score_v7']}/100"
    ):

        left, right = st.columns(2)

        # ====================================================
        # LINKS: SCORES
        # ====================================================

        with left:

            st.write("### Score-Aufteilung")

            st.progress(
                int(row["score_valuation"]),
                text=(
                    f"Valuation: "
                    f"{row['score_valuation']:.1f}/100"
                ),
            )

            st.progress(
                int(row["score_quality"]),
                text=(
                    f"Quality: "
                    f"{row['score_quality']:.1f}/100"
                ),
            )

            st.progress(
                int(row["score_risk"]),
                text=(
                    f"Risk: "
                    f"{row['score_risk']:.1f}/100"
                ),
            )

            st.progress(
                int(row["score_growth"]),
                text=(
                    f"Growth: "
                    f"{row['score_growth']:.1f}/100"
                ),
            )

            st.progress(
                int(row["score_dip_technicals"]),
                text=(
                    f"Dip/Technicals: "
                    f"{row['score_dip_technicals']:.1f}/100"
                ),
            )

        # ====================================================
        # RECHTS: FUNDAMENTAL
        # ====================================================

        with right:

            st.write("### Fundamentaldaten")

            currency = row["currency"]

            price = row["current_price"]

            if pd.notna(price):

                st.write(
                    f"**Live-Kurs:** "
                    f"{price:.2f} {currency}"
                )

            if pd.notna(row["pe_ratio"]):

                st.write(
                    f"**Forward P/E:** "
                    f"{row['pe_ratio']:.2f}"
                )

            if pd.notna(row["peg_ratio"]):

                st.write(
                    f"**PEG:** "
                    f"{row['peg_ratio']:.2f}"
                )

            if pd.notna(row["pfcf_ratio"]):

                st.write(
                    f"**P/FCF:** "
                    f"{row['pfcf_ratio']:.2f}"
                )

            if pd.notna(row["roic"]):

                st.write(
                    f"**ROIC:** "
                    f"{row['roic']:.1%}"
                )

            if pd.notna(row["roe"]):

                st.write(
                    f"**ROE:** "
                    f"{row['roe']:.1%}"
                )

            if pd.notna(row["fcf_margin"]):

                st.write(
                    f"**FCF-Marge:** "
                    f"{row['fcf_margin']:.1%}"
                )

            if pd.notna(row["revenue_growth_3y"]):

                st.write(
                    f"**Umsatz-CAGR 3J:** "
                    f"{row['revenue_growth_3y']:.1f}%"
                )

            if pd.notna(row["fcf_growth_3y"]):

                st.write(
                    f"**FCF-CAGR 3J:** "
                    f"{row['fcf_growth_3y']:.1f}%"
                )

            if pd.notna(row["eps_growth_5y"]):

                st.write(
                    f"**EPS-Wachstum 5J:** "
                    f"{row['eps_growth_5y']:.1f}%"
                )

            if pd.notna(row["net_debt_ebitda"]):

                st.write(
                    f"**Net Debt / EBITDA:** "
                    f"{row['net_debt_ebitda']:.2f}"
                )

            if pd.notna(row["interest_coverage"]):

                st.write(
                    f"**Interest Coverage:** "
                    f"{row['interest_coverage']:.1f}x"
                )

            if pd.notna(row["fcf_to_debt"]):

                st.write(
                    f"**FCF / Debt:** "
                    f"{row['fcf_to_debt']:.1%}"
                )

            if pd.notna(row["beta"]):

                st.write(
                    f"**Beta:** "
                    f"{row['beta']:.2f}"
                )

            if pd.notna(row["dist_52w_high"]):

                st.write(
                    f"**Abstand 52W-Hoch:** "
                    f"{row['dist_52w_high']:.1%}"
                )

            if pd.notna(row["performance_6m"]):

                st.write(
                    f"**Performance 6M:** "
                    f"{row['performance_6m']:.1%}"
                )

            if pd.notna(row["analyst_upside"]):

                st.write(
                    f"**Analystenpotenzial:** "
                    f"{row['analyst_upside']:.1%}"
                )

            if pd.notna(row["lynch_growth_value"]):

                st.write(
                    f"**Lynch Growth Value:** "
                    f"{row['lynch_growth_value']:.2f}"
                )

            if pd.notna(row["margin_of_safety"]):

                st.write(
                    f"**Margin of Safety:** "
                    f"{row['margin_of_safety']:.1%}"
                )

            st.write(
                f"**Datenqualität:** "
                f"{row['data_quality']:.0f}%"
            )

            st.caption(
                f"Yahoo-Symbol: {row['yahoo_symbol']}"
            )


# ============================================================
# HINWEIS
# ============================================================

st.markdown("---")

st.caption(
    "Datenquelle: Yahoo Finance über yfinance. "
    "Fundamentaldaten und Analystenschätzungen können je "
    "nach Unternehmen/Börsenplatz unterschiedlich vollständig sein."
)
