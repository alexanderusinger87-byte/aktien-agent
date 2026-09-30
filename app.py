import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# STREAMLIT SETUP
# ============================================================

st.set_page_config(
    page_title="Aktien-Screener V7",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Aktien-Screener V7 – Live Multi-Faktor Ranking")


# ============================================================
# TICKER-MAPPING
# ============================================================
# Einige europäische Aktien werden bei Yahoo Finance mit
# Börsenplatz-Suffix geführt.
#
# Der Benutzer kann trotzdem die gewohnten Kürzel eingeben.

TICKER_MAP = {
    "SAP": "SAP.DE",
    "SIE": "SIE.DE",
    "MC": "MC.PA",
    "OR": "OR.PA",
    "AIR": "AIR.PA",
    "SCHN": "SU.PA",
    "SU": "SU.PA",
    "SAN": "SAN.PA",
    "TTE": "TTE.PA",
    "SSUN.F": "SSUN.F",
    "3CP.F": "3CP.F",
    "IRE.MI": "IRE.MI",
    "SMTOF": "SMTOF",
    "BYDDY": "BYDDY",
}


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def clean_number(value):
    """Konvertiert einen Wert sicher in float."""
    try:
        if value is None:
            return np.nan

        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return np.nan
            value = value[0]

        return float(value)

    except (ValueError, TypeError):
        return np.nan


def first_valid(*values):
    """Gibt den ersten gültigen numerischen Wert zurück."""
    for value in values:
        number = clean_number(value)

        if pd.notna(number):
            return number

    return np.nan


def get_info_value(info, *keys):
    """Sucht den ersten vorhandenen Wert aus mehreren Yahoo-Feldern."""
    for key in keys:
        if key in info:
            value = clean_number(info.get(key))

            if pd.notna(value):
                return value

    return np.nan


def safe_divide(a, b):
    """Division ohne Fehler bei 0/NaN."""
    try:
        a = clean_number(a)
        b = clean_number(b)

        if pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan

        return a / b

    except Exception:
        return np.nan


def safe_ratio_score(value, excellent, good, weak, bad=0):
    """
    Score für Kennzahlen, bei denen niedrigere Werte besser sind.
    """

    value = pd.to_numeric(value, errors="coerce")

    score = np.select(
        [
            value <= excellent,
            (value > excellent) & (value <= good),
            (value > good) & (value <= weak),
        ],
        [
            100,
            100 - 30 * (value - excellent) / (good - excellent),
            70 - 40 * (value - good) / (weak - good),
        ],
        default=bad,
    )

    return np.nan_to_num(score, nan=50.0)


def growth_score(value, excellent, good, weak):
    """
    Score für Kennzahlen, bei denen höhere Werte besser sind.
    """

    value = pd.to_numeric(value, errors="coerce")

    score = np.select(
        [
            value >= excellent,
            (value >= good) & (value < excellent),
            (value >= weak) & (value < good),
            value < weak,
        ],
        [
            100,
            70 + 30 * (value - good) / (excellent - good),
            30 + 40 * (value - weak) / (good - weak),
            10,
        ],
        default=50,
    )

    return np.nan_to_num(score, nan=50.0)


# ============================================================
# LIVE DATEN ABRUFEN
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_stock_data(user_ticker):

    yahoo_ticker = TICKER_MAP.get(
        user_ticker.upper(),
        user_ticker.upper(),
    )

    result = {
        "symbol": user_ticker.upper(),
        "yahoo_symbol": yahoo_ticker,
        "name": user_ticker.upper(),
        "data_quality": 0,
        "error": "",
    }

    try:

        ticker = yf.Ticker(yahoo_ticker)

        # ----------------------------------------------------
        # INFO
        # ----------------------------------------------------

        info = ticker.info

        if not info:
            result["error"] = "Keine Yahoo-Finance-Daten gefunden."
            return result

        result["name"] = (
            info.get("longName")
            or info.get("shortName")
            or user_ticker.upper()
        )

        # ----------------------------------------------------
        # KURS
        # ----------------------------------------------------

        current_price = first_valid(
            info.get("currentPrice"),
            info.get("regularMarketPrice"),
            info.get("previousClose"),
        )

        result["current_price"] = current_price

        # ----------------------------------------------------
        # VALUATION
        # ----------------------------------------------------

        result["pe_ratio"] = first_valid(
            info.get("forwardPE"),
            info.get("trailingPE"),
        )

        result["peg_ratio"] = get_info_value(
            info,
            "pegRatio",
        )

        result["pfcf_ratio"] = get_info_value(
            info,
            "priceToFreeCashflow",
            "priceToFreeCashFlows",
        )

        result["eps_forward"] = first_valid(
            info.get("forwardEps"),
            info.get("trailingEps"),
        )

        # ----------------------------------------------------
        # ANALYSTEN-KURSZIEL
        # ----------------------------------------------------

        analyst_upside = np.nan

        try:

            targets = ticker.get_analyst_price_targets()

            target = first_valid(
                targets.get("mean"),
                targets.get("median"),
            )

            if (
                pd.notna(target)
                and pd.notna(current_price)
                and current_price > 0
            ):
                analyst_upside = (
                    target / current_price
                ) - 1

        except Exception:
            pass

        result["analyst_upside"] = analyst_upside

        # ----------------------------------------------------
        # GROWTH
        # ----------------------------------------------------

        result["eps_growth_5y"] = np.nan

        # Zuerst versuchen wir Yahoo's explizite
        # 5-Jahres-Growth-Schätzung.
        try:

            growth_estimates = ticker.get_growth_estimates()

            if (
                growth_estimates is not None
                and not growth_estimates.empty
            ):

                row_candidates = [
                    "+5y",
                    "5y",
                    "+5Y",
                    "5Y",
                ]

                for row_name in row_candidates:

                    if row_name in growth_estimates.index:

                        value = growth_estimates.loc[row_name]

                        if isinstance(value, pd.Series):

                            if "stock" in value.index:
                                value = value["stock"]

                            else:
                                value = value.iloc[0]

                        value = clean_number(value)

                        if pd.notna(value):

                            # Yahoo liefert hier normalerweise
                            # Dezimalwerte, z.B. 0.15 = 15 %
                            if abs(value) < 1:
                                value *= 100

                            result["eps_growth_5y"] = value
                            break

        except Exception:
            pass

        # Fallback auf Yahoo earningsGrowth
        if pd.isna(result["eps_growth_5y"]):

            fallback_growth = get_info_value(
                info,
                "earningsGrowth",
            )

            if pd.notna(fallback_growth):

                if abs(fallback_growth) < 1:
                    fallback_growth *= 100

                result["eps_growth_5y"] = fallback_growth

        # ----------------------------------------------------
        # QUALITY
        # ----------------------------------------------------

        roe = get_info_value(
            info,
            "returnOnEquity",
        )

        result["roe"] = roe if pd.notna(roe) else np.nan

        # ----------------------------------------------------
        # HISTORISCHE FINANZDATEN
        # ----------------------------------------------------

        income = None
        balance = None
        cashflow = None

        try:
            income = ticker.income_stmt

        except Exception:
            pass

        try:
            balance = ticker.balance_sheet

        except Exception:
            pass

        try:
            cashflow = ticker.cashflow

        except Exception:
            pass

        # ----------------------------------------------------
        # REVENUE
        # ----------------------------------------------------

        revenue = np.nan

        if (
            income is not None
            and not income.empty
            and "Total Revenue" in income.index
        ):

            revenue_series = (
                income.loc["Total Revenue"]
                .dropna()
                .sort_index()
            )

            if len(revenue_series) > 0:
                revenue = clean_number(
                    revenue_series.iloc[-1]
                )

            # 3-Jahres-Umsatzwachstum
            if len(revenue_series) >= 4:

                old_revenue = clean_number(
                    revenue_series.iloc[-4]
                )

                new_revenue = clean_number(
                    revenue_series.iloc[-1]
                )

                if (
                    pd.notna(old_revenue)
                    and pd.notna(new_revenue)
                    and old_revenue > 0
                ):

                    result["revenue_growth_3y"] = (
                        (
                            new_revenue
                            / old_revenue
                        ) ** (1 / 3)
                        - 1
                    ) * 100

                else:
                    result["revenue_growth_3y"] = np.nan

            else:

                # Fallback
                result["revenue_growth_3y"] = (
                    get_info_value(
                        info,
                        "revenueGrowth",
                    )
                    * 100
                    if pd.notna(
                        get_info_value(
                            info,
                            "revenueGrowth",
                        )
                    )
                    else np.nan
                )

        else:

            revenue = get_info_value(
                info,
                "totalRevenue",
            )

            revenue_growth = get_info_value(
                info,
                "revenueGrowth",
            )

            result["revenue_growth_3y"] = (
                revenue_growth * 100
                if pd.notna(revenue_growth)
                else np.nan
            )

        # ----------------------------------------------------
        # FCF
        # ----------------------------------------------------

        fcf = get_info_value(
            info,
            "freeCashflow",
        )

        if pd.isna(fcf) and cashflow is not None:

            try:

                if "Free Cash Flow" in cashflow.index:

                    fcf_series = (
                        cashflow.loc[
                            "Free Cash Flow"
                        ]
                        .dropna()
                    )

                    if len(fcf_series) > 0:
                        fcf = clean_number(
                            fcf_series.iloc[0]
                        )

            except Exception:
                pass

        result["fcf_margin"] = safe_divide(
            fcf,
            revenue,
        )

        # ----------------------------------------------------
        # FCF GROWTH
        # ----------------------------------------------------

        result["fcf_growth_3y"] = np.nan

        if (
            cashflow is not None
            and not cashflow.empty
            and "Free Cash Flow" in cashflow.index
        ):

            try:

                fcf_series = (
                    cashflow.loc[
                        "Free Cash Flow"
                    ]
                    .dropna()
                    .sort_index()
                )

                if len(fcf_series) >= 4:

                    old_fcf = clean_number(
                        fcf_series.iloc[-4]
                    )

                    new_fcf = clean_number(
                        fcf_series.iloc[-1]
                    )

                    if (
                        pd.notna(old_fcf)
                        and pd.notna(new_fcf)
                        and old_fcf > 0
                        and new_fcf > 0
                    ):

                        result["fcf_growth_3y"] = (
                            (
                                new_fcf
                                / old_fcf
                            ) ** (1 / 3)
                            - 1
                        ) * 100

            except Exception:
                pass

        # ----------------------------------------------------
        # DEBT / CASH
        # ----------------------------------------------------

        total_debt = get_info_value(
            info,
            "totalDebt",
        )

        cash = get_info_value(
            info,
            "totalCash",
            "cash",
        )

        ebitda = get_info_value(
            info,
            "ebitda",
        )

        if pd.isna(ebitda):

            ebitda = get_info_value(
                info,
                "normalizedEBITDA",
            )

        net_debt = np.nan

        if (
            pd.notna(total_debt)
            and pd.notna(cash)
        ):
            net_debt = total_debt - cash

        if (
            pd.notna(net_debt)
            and pd.notna(ebitda)
            and ebitda > 0
        ):

            result["net_debt_ebitda"] = (
                net_debt / ebitda
            )

        else:

            result["net_debt_ebitda"] = np.nan

        # ----------------------------------------------------
        # INTEREST COVERAGE
        # ----------------------------------------------------

        result["interest_coverage"] = np.nan

        if (
            income is not None
            and not income.empty
        ):

            try:

                ebit = np.nan
                interest_expense = np.nan

                for field in [
                    "EBIT",
                    "Operating Income",
                ]:

                    if field in income.index:

                        ebit = clean_number(
                            income.loc[field]
                            .dropna()
                            .iloc[0]
                        )

                        break

                for field in [
                    "Interest Expense",
                    "Interest Expense Non Operating",
                ]:

                    if field in income.index:

                        interest_expense = abs(
                            clean_number(
                                income.loc[field]
                                .dropna()
                                .iloc[0]
                            )
                        )

                        break

                if (
                    pd.notna(ebit)
                    and pd.notna(interest_expense)
                    and interest_expense > 0
                ):

                    result["interest_coverage"] = (
                        ebit / interest_expense
                    )

            except Exception:
                pass

        # ----------------------------------------------------
        # BETA
        # ----------------------------------------------------

        result["beta"] = get_info_value(
            info,
            "beta",
        )

        # ----------------------------------------------------
        # 52-WEEK-HIGH
        # ----------------------------------------------------

        fifty_two_week_high = get_info_value(
            info,
            "fiftyTwoWeekHigh",
        )

        if (
            pd.notna(current_price)
            and pd.notna(fifty_two_week_high)
            and fifty_two_week_high > 0
        ):

            result["dist_52w_high"] = (
                current_price
                / fifty_two_week_high
                - 1
            )

        else:

            result["dist_52w_high"] = np.nan

        # ----------------------------------------------------
        # HISTORISCHE KURSDATEN
        # ----------------------------------------------------

        history = pd.DataFrame()

        try:

            history = ticker.history(
                period="1y",
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
                history["Close"]
                .dropna()
            )

            # SMA 200
            if len(close) >= 200:

                result["sma_200"] = clean_number(
                    close.tail(200).mean()
                )

            else:

                result["sma_200"] = np.nan

            # 6-Monats-Performance
            if len(close) >= 120:

                price_6m_ago = clean_number(
                    close.iloc[-120]
                )

                price_now = clean_number(
                    close.iloc[-1]
                )

                if (
                    pd.notna(price_6m_ago)
                    and pd.notna(price_now)
                    and price_6m_ago > 0
                ):

                    result["performance_6m"] = (
                        price_now
                        / price_6m_ago
                        - 1
                    )

                else:

                    result["performance_6m"] = np.nan

            else:

                result["performance_6m"] = np.nan

        else:

            result["sma_200"] = np.nan
            result["performance_6m"] = np.nan

        # ----------------------------------------------------
        # DATA QUALITY
        # ----------------------------------------------------

        required = [
            "current_price",
            "pe_ratio",
            "peg_ratio",
            "pfcf_ratio",
            "eps_forward",
            "eps_growth_5y",
            "roe",
            "fcf_margin",
            "beta",
            "net_debt_ebitda",
            "interest_coverage",
            "revenue_growth_3y",
            "dist_52w_high",
            "sma_200",
        ]

        valid_count = sum(
            pd.notna(result.get(col, np.nan))
            for col in required
        )

        result["data_quality"] = round(
            valid_count
            / len(required)
            * 100
        )

        return result

    except Exception as e:

        result["error"] = str(e)
        result["data_quality"] = 0

        return result


# ============================================================
# ALLE TICKER ABFRAGEN
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def load_stocks(tickers):

    rows = []

    for ticker in tickers:

        data = fetch_stock_data(ticker)

        rows.append(data)

    return pd.DataFrame(rows)


# ============================================================
# SCORE-ALGORITHMUS V6
# ============================================================

def calculate_stock_score_v6(df_input):

    df = df_input.copy()

    # ========================================================
    # 1. VALUATION – 25 %
    # ========================================================

    pe_score = safe_ratio_score(
        df["pe_ratio"],
        excellent=12,
        good=20,
        weak=30,
        bad=15,
    )

    peg_score = safe_ratio_score(
        df["peg_ratio"],
        excellent=1.0,
        good=1.5,
        weak=2.5,
        bad=15,
    )

    pfcf_score = safe_ratio_score(
        df["pfcf_ratio"],
        excellent=15,
        good=25,
        weak=35,
        bad=15,
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
    ).round(2)

    df["margin_of_safety"] = np.where(
        df["lynch_growth_value"] > 0,
        (
            df["lynch_growth_value"]
            - df["current_price"]
        )
        / df["lynch_growth_value"],
        -0.50,
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

    roe_score = growth_score(
        df["roe"] * 100,
        excellent=25,
        good=15,
        weak=8,
    )

    roic_score = growth_score(
        df["roic"] * 100,
        excellent=20,
        good=12,
        weak=6,
    )

    fcf_margin_score = growth_score(
        df["fcf_margin"] * 100,
        excellent=20,
        good=10,
        weak=5,
    )

    revenue_growth_quality = growth_score(
        df["revenue_growth_3y"],
        excellent=15,
        good=8,
        weak=3,
    )

    fcf_growth_quality = growth_score(
        df["fcf_growth_3y"],
        excellent=15,
        good=8,
        weak=3,
    )

    df["score_quality"] = (
        roic_score * 0.30
        + roe_score * 0.20
        + fcf_margin_score * 0.20
        + revenue_growth_quality * 0.15
        + fcf_growth_quality * 0.15
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

    interest_coverage_score = np.select(
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

    # FCF / Debt
    # Wird aus FCF und Net Debt berechnet.
    fcf_debt = np.where(
        df["net_debt_ebitda"] > 0,
        df["fcf_margin"]
        / df["net_debt_ebitda"],
        1.0,
    )

    fcf_debt_score = growth_score(
        fcf_debt * 100,
        excellent=50,
        good=30,
        weak=10,
    )

    df["score_risk"] = (
        debt_score * 0.40
        + interest_coverage_score * 0.25
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

    ma200_distance = (
        df["current_price"]
        / df["sma_200"]
        - 1
    )

    sma200_score = np.select(
        [
            ma200_distance >= 0.15,
            ma200_distance >= 0.05,
            ma200_distance >= 0,
            ma200_distance >= -0.10,
            ma200_distance >= -0.20,
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

    performance_6m_score = np.select(
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

    df["score_dip_technicals"] = (
        dip_score * 0.40
        + sma200_score * 0.35
        + performance_6m_score * 0.25
    )

    # ========================================================
    # TOTAL
    # ========================================================

    df["total_score_v6"] = (
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
            df["total_score_v6"] >= 80,
            df["total_score_v6"] >= 70,
            df["total_score_v6"] >= 55,
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
            df["total_score_v6"] >= 85,
            df["total_score_v6"] >= 75,
            df["total_score_v6"] >= 65,
            df["total_score_v6"] >= 50,
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
        by="total_score_v6",
        ascending=False,
    )


# ============================================================
# AKTIEN-EINGABE
# ============================================================

st.subheader("📌 Aktien auswählen")

ticker_input = st.text_input(
    "Ticker eingeben – mehrere mit Komma trennen",
    value=(
        "MSFT, GOOGL, NVDA, MU, SSUN.F, TSM, ASML, "
        "VRT, VST, PEP, NVO, PFE, AMZN, MRVL, NKE, "
        "BYDDY, NOK, AAPL, AVGO, META, TSLA, MC, SAP, "
        "SIE, TTE, SAN, SCHN, SU, OR, AIR"
    ),
    placeholder="z. B. NVDA, MU, AMD, TSM, ASML",
)

tickers = [
    ticker.strip().upper()
    for ticker in ticker_input.split(",")
    if ticker.strip()
]

tickers = list(dict.fromkeys(tickers))

st.caption(
    f"{len(tickers)} Aktien ausgewählt: "
    + ", ".join(tickers)
)


# ============================================================
# DATEN LADEN BUTTON
# ============================================================

load_button = st.button(
    "🔄 Live-Daten abrufen & Ranking berechnen",
    type="primary",
    use_container_width=True,
)


# Automatisch beim ersten Start laden
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False


if load_button:
    st.session_state["data_loaded"] = True


if not st.session_state["data_loaded"]:

    st.info(
        "Ticker eingeben und anschließend "
        "„Live-Daten abrufen & Ranking berechnen“ klicken."
    )

    st.stop()


# ============================================================
# LIVE-DATEN LADEN
# ============================================================

with st.spinner(
    f"Live-Daten für {len(tickers)} Aktien werden abgerufen ..."
):

    df_live = load_stocks(tuple(tickers))


# ============================================================
# FEHLER / DATENQUALITÄT
# ============================================================

error_rows = df_live[
    df_live["error"].fillna("") != ""
]

if not error_rows.empty:

    st.warning(
        "Bei einigen Ticker konnten Daten nicht vollständig "
        "abgerufen werden."
    )

    error_table = error_rows[
        [
            "symbol",
            "yahoo_symbol",
            "error",
        ]
    ].copy()

    st.dataframe(
        error_table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# GÜLTIGE DATEN
# ============================================================

df_raw = df_live[
    df_live["data_quality"] > 20
].copy()


if df_raw.empty:

    st.error(
        "Es konnten keine verwertbaren Live-Daten "
        "abgerufen werden."
    )

    st.stop()


# ============================================================
# SCORE BERECHNEN
# ============================================================

df_results = calculate_stock_score_v6(
    df_raw
)


# ============================================================
# TOP-FAVORIT
# ============================================================

st.markdown("---")

st.subheader("🏆 Top-Favorit")

top = df_results.iloc[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Top-Pick",
    f"{top['name']} ({top['symbol']})",
)

col2.metric(
    "Gesamtwertung",
    f"{top['total_score_v6']} / 100",
)

col3.metric(
    "Sterne-Rating",
    top["stars"],
)

col4.metric(
    "Kaufempfehlung",
    top["rating"],
)


# ============================================================
# RANKING
# ============================================================

st.markdown("---")

st.subheader("📊 Aktien-Ranking & Übersicht")

display_columns = [
    "symbol",
    "name",
    "stars",
    "rating",
    "total_score_v6",
    "score_valuation",
    "score_quality",
    "score_risk",
    "score_growth",
    "score_dip_technicals",
    "current_price",
    "lynch_growth_value",
    "margin_of_safety",
    "analyst_upside",
    "data_quality",
]

display_columns = [
    col
    for col in display_columns
    if col in df_results.columns
]


st.dataframe(
    df_results[display_columns],

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
                "Kaufempfehlung / Ampel",
            ),

        "total_score_v6":
            st.column_config.ProgressColumn(
                "Total Score V6",
                min_value=0,
                max_value=100,
                format="%.1f",
            ),

        "score_valuation":
            st.column_config.NumberColumn(
                "Valuation (25%)",
                format="%.1f",
            ),

        "score_quality":
            st.column_config.NumberColumn(
                "Quality (30%)",
                format="%.1f",
            ),

        "score_risk":
            st.column_config.NumberColumn(
                "Risk (15%)",
                format="%.1f",
            ),

        "score_growth":
            st.column_config.NumberColumn(
                "Growth (15%)",
                format="%.1f",
            ),

        "score_dip_technicals":
            st.column_config.NumberColumn(
                "Dip/Tech (15%)",
                format="%.1f",
            ),

        "current_price":
            st.column_config.NumberColumn(
                "Kurs Live",
                format="%.2f",
            ),

        "lynch_growth_value":
            st.column_config.NumberColumn(
                "Lynch Growth Value",
                format="%.2f",
            ),

        "margin_of_safety":
            st.column_config.NumberColumn(
                "Margin of Safety",
                format="%.1%",
            ),

        "analyst_upside":
            st.column_config.NumberColumn(
                "Analystenpotenzial",
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
# DETAIL-AUFSCHLÜSSELUNG
# ============================================================

st.markdown("---")

st.subheader("🔍 Aufschlüsselung pro Aktie")

for _, row in df_results.iterrows():

    with st.expander(
        f"{row['stars']} **{row['name']} "
        f"({row['symbol']})** – "
        f"{row['rating']} "
        f"(Score: {row['total_score_v6']}/100)"
    ):

        c1, c2 = st.columns(2)

        # ----------------------------------------------------
        # SUB-SCORES
        # ----------------------------------------------------

        with c1:

            st.write(
                "**Sub-Scores Breakdown:**"
            )

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
                    f"Risk & Debt: "
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
                    f"Dip & Technicals: "
                    f"{row['score_dip_technicals']:.1f}/100"
                ),
            )

        # ----------------------------------------------------
        # FUNDAMENTALDATEN
        # ----------------------------------------------------

        with c2:

            st.write(
                "**Bewertung & Fundamentaldaten:**"
            )

            st.write(
                f"- **Aktueller Kurs:** "
                f"{row['current_price']:.2f}"
            )

            if pd.notna(row["lynch_growth_value"]):

                st.write(
                    f"- **Lynch Growth Value:** "
                    f"{row['lynch_growth_value']:.2f}"
                )

            if pd.notna(row["margin_of_safety"]):

                st.write(
                    f"- **Margin of Safety:** "
                    f"{row['margin_of_safety']:.1%}"
                )

            if pd.notna(row["analyst_upside"]):

                st.write(
                    f"- **Analystenpotenzial:** "
                    f"{row['analyst_upside']:.1%}"
                )

            if pd.notna(row["pe_ratio"]):

                st.write(
                    f"- **Forward P/E:** "
                    f"{row['pe_ratio']:.1f}"
                )

            if pd.notna(row["peg_ratio"]):

                st.write(
                    f"- **PEG:** "
                    f"{row['peg_ratio']:.2f}"
                )

            if pd.notna(row["pfcf_ratio"]):

                st.write(
                    f"- **P/FCF:** "
                    f"{row['pfcf_ratio']:.1f}"
                )

            if pd.notna(row["beta"]):

                st.write(
                    f"- **Beta:** "
                    f"{row['beta']:.2f}"
                )

            if pd.notna(row["net_debt_ebitda"]):

                st.write(
                    f"- **Net Debt / EBITDA:** "
                    f"{row['net_debt_ebitda']:.2f}"
                )

            st.write(
                f"- **Datenqualität:** "
                f"{row['data_quality']:.0f}%"
            )
