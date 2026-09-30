import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# STREAMLIT SETUP
# ============================================================

st.set_page_config(
    page_title="Aktien-Screener V6.1",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Aktien-Screener V6.1 – Multi-Faktor Ranking & Buy-Ratings")


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def safe_ratio_score(value, excellent, good, weak, bad=0):
    """
    Kontinuierlicher Score für Kennzahlen,
    bei denen niedrigere Werte besser sind.
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
    Kontinuierlicher Score für Kennzahlen,
    bei denen höhere Werte besser sind.
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
# HAUPTALGORITHMUS V6.1
# ============================================================

def calculate_stock_score_v6(df_input):

    df = df_input.copy()

    # ========================================================
    # 1. VALUATION & MARGIN OF SAFETY – 25 %
    # ========================================================

    # Forward P/E
    pe_score = safe_ratio_score(
        df["pe_ratio"],
        excellent=12,
        good=20,
        weak=30,
        bad=15,
    )

    # PEG
    peg_score = safe_ratio_score(
        df["peg_ratio"],
        excellent=1.0,
        good=1.5,
        weak=2.5,
        bad=15,
    )

    # P/FCF
    pfcf_score = safe_ratio_score(
        df["pfcf_ratio"],
        excellent=15,
        good=25,
        weak=35,
        bad=15,
    )

    # FCF Yield
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
    # EPS Forward × erwartetes EPS-Wachstum
    #
    # Bewusst NICHT als echter Fair Value bezeichnet.
    # Es handelt sich um eine Lynch/PEG-artige Heuristik.
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
        df["eps_forward"] * growth_for_value
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

    # Analysten-Konsenspotenzial
    if "analyst_upside" in df.columns:

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

    else:
        analyst_score = np.full(
            len(df),
            50.0,
        )

    # Valuation Gesamt
    df["score_valuation"] = (
        pe_score * 0.20
        + peg_score * 0.20
        + pfcf_score * 0.15
        + fcf_yield_score * 0.15
        + mos_score * 0.15
        + analyst_score * 0.15
    )

    # ========================================================
    # 2. QUALITY & CAPITAL ALLOCATION – 30 %
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

    if "fcf_growth_3y" in df.columns:

        fcf_growth_quality = growth_score(
            df["fcf_growth_3y"],
            excellent=15,
            good=8,
            weak=3,
        )

    else:

        fcf_growth_quality = revenue_growth_quality

    df["score_quality"] = (
        roic_score * 0.30
        + roe_score * 0.20
        + fcf_margin_score * 0.20
        + revenue_growth_quality * 0.15
        + fcf_growth_quality * 0.15
    )

    # ========================================================
    # 3. RISK & SOLVENCY – 15 %
    # ========================================================

    # Net Debt / EBITDA
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

    # Interest Coverage
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

    # Beta bewusst geringer gewichtet
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
    if "fcf_to_debt" in df.columns:

        fcf_debt_score = growth_score(
            df["fcf_to_debt"] * 100,
            excellent=50,
            good=30,
            weak=10,
        )

    else:

        fcf_debt_score = np.full(
            len(df),
            50.0,
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

    if "fcf_growth_3y" in df.columns:

        fcf_growth_score = growth_score(
            df["fcf_growth_3y"],
            excellent=15,
            good=8,
            weak=3,
        )

    else:

        fcf_growth_score = revenue_growth_score

    df["score_growth"] = (
        eps_growth_score * 0.45
        + revenue_growth_score * 0.30
        + fcf_growth_score * 0.25
    )

    # ========================================================
    # 5. DIP & TECHNICALS – 15 %
    # ========================================================

    # Abstand 52-Wochen-Hoch
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

    # Abstand zur 200-Tage-Linie
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

    # 6-Monats-Performance
    if "performance_6m" in df.columns:

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

    else:

        performance_6m_score = np.full(
            len(df),
            60.0,
        )

    df["score_dip_technicals"] = (
        dip_score * 0.40
        + sma200_score * 0.35
        + performance_6m_score * 0.25
    )

    # ========================================================
    # TOTAL SCORE
    # ========================================================

    df["total_score_v6"] = (
        df["score_valuation"] * 0.25
        + df["score_quality"] * 0.30
        + df["score_risk"] * 0.15
        + df["score_growth"] * 0.15
        + df["score_dip_technicals"] * 0.15
    ).round(1)

    # ========================================================
    # DATENQUALITÄT
    # ========================================================

    required_columns = [
        "pe_ratio",
        "peg_ratio",
        "pfcf_ratio",
        "eps_forward",
        "eps_growth_5y",
        "roe",
        "roic",
        "fcf_margin",
        "beta",
        "net_debt_ebitda",
        "interest_coverage",
        "revenue_growth_3y",
        "dist_52w_high",
        "sma_200",
    ]

    available_columns = [
        col
        for col in required_columns
        if col in df.columns
    ]

    df["data_quality"] = (
        df[available_columns]
        .notna()
        .mean(axis=1)
        * 100
    ).round(0)

    df["data_warning"] = np.where(
        df["data_quality"] < 90,
        "⚠️ Daten unvollständig",
        "✅ OK",
    )

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

    # ========================================================
    # STERNE
    # ========================================================

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
    "Ticker eingeben (mehrere mit Komma trennen)",
    value="MSFT, GOOGL, NVDA, KO",
    placeholder="z. B. MSFT, GOOGL, NVDA, AMD, MU, TSM",
)

# Eingabe bereinigen
tickers = [
    ticker.strip().upper()
    for ticker in ticker_input.split(",")
    if ticker.strip()
]

# Duplikate entfernen
tickers = list(dict.fromkeys(tickers))

if not tickers:

    st.warning(
        "Bitte mindestens einen Ticker eingeben."
    )

    st.stop()

st.caption(
    f"{len(tickers)} Aktien ausgewählt: "
    f"{', '.join(tickers)}"
)


# ============================================================
# DEMO-DATEN
# ============================================================

def get_demo_data():

    return pd.DataFrame([

        {
            "symbol": "MSFT",
            "name": "Microsoft Corp.",
            "current_price": 420.0,
            "pe_ratio": 32.5,
            "peg_ratio": 1.8,
            "pfcf_ratio": 28.0,
            "eps_forward": 13.5,
            "eps_growth_5y": 14.5,
            "roe": 0.38,
            "roic": 0.26,
            "fcf_margin": 0.30,
            "beta": 0.90,
            "net_debt_ebitda": 0.3,
            "interest_coverage": 40.0,
            "revenue_growth_3y": 12.5,
            "dist_52w_high": -0.06,
            "sma_200": 410.0,
            "analyst_upside": 0.12,
            "fcf_growth_3y": 15.0,
            "fcf_to_debt": 1.2,
            "performance_6m": 0.08,
        },

        {
            "symbol": "GOOGL",
            "name": "Alphabet Inc.",
            "current_price": 165.0,
            "pe_ratio": 21.0,
            "peg_ratio": 1.1,
            "pfcf_ratio": 19.5,
            "eps_forward": 7.8,
            "eps_growth_5y": 16.0,
            "roe": 0.29,
            "roic": 0.22,
            "fcf_margin": 0.22,
            "beta": 1.05,
            "net_debt_ebitda": -0.5,
            "interest_coverage": 100.0,
            "revenue_growth_3y": 11.0,
            "dist_52w_high": -0.12,
            "sma_200": 160.0,
            "analyst_upside": 0.18,
            "fcf_growth_3y": 14.0,
            "fcf_to_debt": 1.5,
            "performance_6m": 0.05,
        },

        {
            "symbol": "NVDA",
            "name": "NVIDIA Corp.",
            "current_price": 120.0,
            "pe_ratio": 42.0,
            "peg_ratio": 1.2,
            "pfcf_ratio": 38.0,
            "eps_forward": 3.8,
            "eps_growth_5y": 35.0,
            "roe": 0.52,
            "roic": 0.45,
            "fcf_margin": 0.40,
            "beta": 1.65,
            "net_debt_ebitda": -0.2,
            "interest_coverage": 60.0,
            "revenue_growth_3y": 55.0,
            "dist_52w_high": -0.18,
            "sma_200": 115.0,
            "analyst_upside": 0.22,
            "fcf_growth_3y": 40.0,
            "fcf_to_debt": 2.0,
            "performance_6m": 0.15,
        },

        {
            "symbol": "KO",
            "name": "Coca-Cola Co.",
            "current_price": 68.0,
            "pe_ratio": 24.0,
            "peg_ratio": 2.8,
            "pfcf_ratio": 22.0,
            "eps_forward": 2.85,
            "eps_growth_5y": 6.0,
            "roe": 0.40,
            "roic": 0.16,
            "fcf_margin": 0.21,
            "beta": 0.58,
            "net_debt_ebitda": 1.8,
            "interest_coverage": 12.0,
            "revenue_growth_3y": 5.0,
            "dist_52w_high": -0.02,
            "sma_200": 64.0,
            "analyst_upside": 0.05,
            "fcf_growth_3y": 5.0,
            "fcf_to_debt": 0.35,
            "performance_6m": 0.02,
        },
    ])


# ============================================================
# DATEN LADEN
# ============================================================

df_all = get_demo_data()

# Nur eingegebene Aktien auswählen
df_raw = df_all[
    df_all["symbol"].isin(tickers)
].copy()

# Fehlende Ticker melden
found_tickers = set(df_raw["symbol"])

missing_tickers = [
    ticker
    for ticker in tickers
    if ticker not in found_tickers
]

if missing_tickers:

    st.warning(
        "Für folgende Ticker sind aktuell noch "
        "keine Daten hinterlegt: "
        + ", ".join(missing_tickers)
    )

if df_raw.empty:

    st.error(
        "Für keinen der eingegebenen Ticker "
        "sind aktuell Daten vorhanden."
    )

    st.stop()


# ============================================================
# SCORE BERECHNEN
# ============================================================

df_results = calculate_stock_score_v6(df_raw)


# ============================================================
# TOP-FAVORIT
# ============================================================

st.markdown("---")

st.subheader("🏆 Top-Favorit")

top = df_results.iloc[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Top-Pick",
    f"{top['name']} ({top['symbol']})"
)

col2.metric(
    "Gesamtwertung",
    f"{top['total_score_v6']} / 100"
)

col3.metric(
    "Sterne-Rating",
    top["stars"]
)

col4.metric(
    "Kaufempfehlung",
    top["rating"]
)


# ============================================================
# RANKING
# ============================================================

st.markdown("---")

st.subheader("📊 Aktien-Ranking & Übersicht")

st.dataframe(
    df_results[
        [
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
            "data_warning",
        ]
    ],

    column_config={

        "symbol":
            st.column_config.TextColumn(
                "Ticker"
            ),

        "name":
            st.column_config.TextColumn(
                "Unternehmen"
            ),

        "stars":
            st.column_config.TextColumn(
                "Sterne"
            ),

        "rating":
            st.column_config.TextColumn(
                "Kaufempfehlung / Ampel"
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
                "Kurs",
                format="$%.2f",
            ),

        "lynch_growth_value":
            st.column_config.NumberColumn(
                "Lynch Growth Value",
                format="$%.2f",
            ),

        "margin_of_safety":
            st.column_config.NumberColumn(
                "Margin of Safety",
                format="%.1%",
            ),

        "data_warning":
            st.column_config.TextColumn(
                "Datenqualität"
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
        # LINKE SPALTE
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
        # RECHTE SPALTE
        # ----------------------------------------------------

        with c2:

            st.write(
                "**Bewertung & Fair Value:**"
            )

            st.write(
                f"- **Aktueller Kurs:** "
                f"${row['current_price']:.2f}"
            )

            st.write(
                f"- **Lynch Growth Value:** "
                f"${row['lynch_growth_value']:.2f}"
            )

            st.write(
                f"- **Margin of Safety:** "
                f"{row['margin_of_safety']:.1%}"
            )

            st.write(
                f"- **Beta:** "
                f"{row['beta']}"
            )

            st.write(
                f"- **Net Debt / EBITDA:** "
                f"{row['net_debt_ebitda']}"
            )

            st.write(
                f"- **Datenqualität:** "
                f"{row['data_quality']:.0f}%"
            )
