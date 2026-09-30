import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Aktien-Screener V5", page_icon="📈", layout="wide"
)

st.title("📈 Aktien-Screener V5 (Inkl. Risk-Modul & Live-Price MoS)")


# ---------------------------------------------------------
# V5 Calculation Engine
# ---------------------------------------------------------
def calculate_stock_score_v5(df_input):
    df = df_input.copy()

    # Required columns check to avoid Streamlit crash
    required_cols = [
        "symbol",
        "name",
        "current_price",
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

    for col in required_cols:
        if col not in df.columns:
            st.error(
                f"Fehlende Spalte in den Daten: '{col}'. Berechnung nicht möglich."
            )
            return df

    # 1. VALUATION & MARGIN OF SAFETY (Gewichtung: 25%)
    pe_score = np.select(
        [
            df["pe_ratio"] < 12,
            (df["pe_ratio"] >= 12) & (df["pe_ratio"] <= 20),
            (df["pe_ratio"] > 20) & (df["pe_ratio"] <= 30),
        ],
        [100, 80, 50],
        default=20,
    )

    peg_score = np.select(
        [
            df["peg_ratio"] < 1.0,
            (df["peg_ratio"] >= 1.0) & (df["peg_ratio"] <= 1.5),
            (df["peg_ratio"] > 1.5) & (df["peg_ratio"] <= 2.0),
        ],
        [100, 75, 45],
        default=10,
    )

    pfcf_score = np.select(
        [
            df["pfcf_ratio"] < 15,
            (df["pfcf_ratio"] >= 15) & (df["pfcf_ratio"] <= 25),
        ],
        [100, 65],
        default=30,
    )

    # Margin of Safety (Lynch Fair Value vs Current Price)
    fair_value = df["eps_forward"] * np.maximum(df["eps_growth_5y"], 0)
    df["margin_of_safety"] = np.where(
        fair_value > 0, (fair_value - df["current_price"]) / fair_value, -0.50
    )

    mos_score = np.select(
        [
            df["margin_of_safety"] >= 0.30,
            (df["margin_of_safety"] >= 0.10) & (df["margin_of_safety"] < 0.30),
            (df["margin_of_safety"] >= -0.10) & (df["margin_of_safety"] < 0.10),
        ],
        [100, 75, 50],
        default=15,
    )

    df["score_valuation"] = (
        pe_score * 0.25
        + peg_score * 0.30
        + pfcf_score * 0.20
        + mos_score * 0.25
    )

    # 2. QUALITY & CAPITAL ALLOCATION (Gewichtung: 30%)
    roe_score = np.select(
        [df["roe"] >= 0.20, (df["roe"] >= 0.12) & (df["roe"] < 0.20)],
        [100, 70],
        default=30,
    )

    roic_score = np.select(
        [df["roic"] >= 0.15, (df["roic"] >= 0.10) & (df["roic"] < 0.15)],
        [100, 70],
        default=25,
    )

    fcf_margin_score = np.select(
        [
            df["fcf_margin"] >= 0.15,
            (df["fcf_margin"] >= 0.08) & (df["fcf_margin"] < 0.15),
        ],
        [100, 65],
        default=20,
    )

    df["score_quality"] = (
        roe_score * 0.35 + roic_score * 0.35 + fcf_margin_score * 0.30
    )

    # 3. RISK & SOLVENCY MODULE (Gewichtung: 15%)
    beta_score = np.select(
        [
            df["beta"] <= 0.8,
            (df["beta"] > 0.8) & (df["beta"] <= 1.2),
            (df["beta"] > 1.2) & (df["beta"] <= 1.6),
        ],
        [100, 80, 45],
        default=10,
    )

    debt_score = np.select(
        [
            df["net_debt_ebitda"] <= 1.5,
            (df["net_debt_ebitda"] > 1.5) & (df["net_debt_ebitda"] <= 3.0),
        ],
        [100, 65],
        default=15,
    )

    interest_coverage_score = np.select(
        [
            df["interest_coverage"] >= 8.0,
            (df["interest_coverage"] >= 4.0) & (df["interest_coverage"] < 8.0),
        ],
        [100, 70],
        default=20,
    )

    df["score_risk"] = (
        beta_score * 0.30 + debt_score * 0.40 + interest_coverage_score * 0.30
    )

    # 4. GROWTH (Gewichtung: 15%)
    eps_growth_score = np.select(
        [
            df["eps_growth_5y"] >= 15.0,
            (df["eps_growth_5y"] >= 8.0) & (df["eps_growth_5y"] < 15.0),
        ],
        [100, 70],
        default=25,
    )

    revenue_growth_score = np.select(
        [
            df["revenue_growth_3y"] >= 10.0,
            (df["revenue_growth_3y"] >= 5.0) & (df["revenue_growth_3y"] < 10.0),
        ],
        [100, 65],
        default=20,
    )

    df["score_growth"] = eps_growth_score * 0.60 + revenue_growth_score * 0.40

    # 5. DIP & TECHNICALS (Gewichtung: 15%)
    dip_score = np.select(
        [
            df["dist_52w_high"] <= -0.20,
            (df["dist_52w_high"] > -0.20) & (df["dist_52w_high"] <= -0.10),
            (df["dist_52w_high"] > -0.10) & (df["dist_52w_high"] <= -0.03),
        ],
        [100, 75, 40],
        default=10,
    )

    sma200_score = np.where(df["current_price"] >= df["sma_200"], 100, 30)

    df["score_dip_technicals"] = dip_score * 0.60 + sma200_score * 0.40

    # TOTAL SCORE V5
    df["total_score_v5"] = (
        df["score_valuation"] * 0.25
        + df["score_quality"] * 0.30
        + df["score_risk"] * 0.15
        + df["score_growth"] * 0.15
        + df["score_dip_technicals"] * 0.15
    ).round(2)

    return df.sort_values(by="total_score_v5", ascending=False)


# ---------------------------------------------------------
# Demo-Data Provider (Sicherheitsnetz gegen Startabsturz)
# ---------------------------------------------------------
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
        },
    ])


# ---------------------------------------------------------
# UI Layout & Execution
# ---------------------------------------------------------
st.sidebar.header("Datenquelle wählen")
source = st.sidebar.radio(
    "Quelle:", ["Demo-Daten nutzen", "Eigene CSV hochladen"]
)

df_raw = None

if source == "Demo-Daten nutzen":
    df_raw = get_demo_data()
else:
    uploaded_file = st.sidebar.file_uploader("CSV hochladen", type=["csv"])
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Fehler beim Lesen der CSV-Datei: {e}")

if df_raw is not None:
    # Calculation
    df_results = calculate_stock_score_v5(df_raw)

    # Top Overview Metrics
    top_stock = df_results.iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Top-Pick V5", top_stock["name"])
    col2.metric("Total Score", f"{top_stock['total_score_v5']} / 100")
    col3.metric("Risk-Score", f"{round(top_stock['score_risk'], 1)} / 100")

    st.subheader("Gesamtergebnis V5 Ranking")

    # Display Columns
    display_cols = [
        "symbol",
        "name",
        "total_score_v5",
        "score_valuation",
        "score_quality",
        "score_risk",
        "score_growth",
        "score_dip_technicals",
        "current_price",
        "margin_of_safety",
    ]

    # Formatter for Dataframe
    st.dataframe(
        df_results[display_cols].style.format({
            "total_score_v5": "{:.1f}",
            "score_valuation": "{:.1f}",
            "score_quality": "{:.1f}",
            "score_risk": "{:.1f}",
            "score_growth": "{:.1f}",
            "score_dip_technicals": "{:.1f}",
            "current_price": "${:.2f}",
            "margin_of_safety": "{:.1%}",
        }),
        use_container_width=True,
    )
else:
    st.info("Bitte wähle eine Datenquelle oder lade eine CSV hoch, um zu starten.")
