import numpy as np
import pandas as pd


def calculate_stock_score_v5(df_input):
    """Calculates Stock Score V5 incorporating Risk-Adjustment, Current Live Prices,

    Margin of Safety (Lynch Fair Value), and Fundamental Metrics.
    """
    df = df_input.copy()

    # ---------------------------------------------------------
    # 1. VALUATION & MARGIN OF SAFETY (Gewichtung: 25%)
    # ---------------------------------------------------------
    # P/E Score (KGV)
    pe_score = np.select(
        [
            df["pe_ratio"] < 12,
            (df["pe_ratio"] >= 12) & (df["pe_ratio"] <= 20),
            (df["pe_ratio"] > 20) & (df["pe_ratio"] <= 30),
        ],
        [100, 80, 50],
        default=20,
    )

    # PEG Ratio Score
    peg_score = np.select(
        [
            df["peg_ratio"] < 1.0,
            (df["peg_ratio"] >= 1.0) & (df["peg_ratio"] <= 1.5),
            (df["peg_ratio"] > 1.5) & (df["peg_ratio"] <= 2.0),
        ],
        [100, 75, 45],
        default=10,
    )

    # P/FCF Score
    pfcf_score = np.select(
        [
            df["pfcf_ratio"] < 15,
            (df["pfcf_ratio"] >= 15) & (df["pfcf_ratio"] <= 25),
        ],
        [100, 65],
        default=30,
    )

    # Margin of Safety (Abgleich: Live Price vs. Lynch Fair Value)
    # Fair Value ~ (EPS * Growth Rate)
    fair_value = df["eps_forward"] * np.maximum(df["eps_growth_5y"], 0)
    df["margin_of_safety"] = (fair_value - df["current_price"]) / fair_value

    mos_score = np.select(
        [
            df["margin_of_safety"] >= 0.30,  # 30%+ Unterbewertung
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

    # ---------------------------------------------------------
    # 2. QUALITY & CAPITAL ALLOCATION (Gewichtung: 30%)
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 3. RISK & SOLVENCY MODULE (Gewichtung: 15%)
    # ---------------------------------------------------------
    # Beta / Volatilitäts-Malus
    beta_score = np.select(
        [
            df["beta"] <= 0.8,
            (df["beta"] > 0.8) & (df["beta"] <= 1.2),
            (df["beta"] > 1.2) & (df["beta"] <= 1.6),
        ],
        [100, 80, 45],
        default=10,
    )

    # Net Debt / EBITDA (Verschuldungsgrad)
    debt_score = np.select(
        [
            df["net_debt_ebitda"] <= 1.5,
            (df["net_debt_ebitda"] > 1.5) & (df["net_debt_ebitda"] <= 3.0),
        ],
        [100, 65],
        default=15,
    )

    # Interest Coverage (Zinsdeckungsgrad = EBIT / Zinsaufwand)
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

    # ---------------------------------------------------------
    # 4. GROWTH (Gewichtung: 15%)
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 5. DIP & TECHNICALS (Gewichtung: 15%)
    # ---------------------------------------------------------
    # Abstand zum 52-Wochen-Hoch
    dip_score = np.select(
        [
            df["dist_52w_high"] <= -0.20,  # 20%+ Rücksetzer
            (df["dist_52w_high"] > -0.20) & (df["dist_52w_high"] <= -0.10),
            (df["dist_52w_high"] > -0.10) & (df["dist_52w_high"] <= -0.03),
        ],
        [100, 75, 40],
        default=10,
    )

    # Trend-Filter (z. B. Kurs über 200-Tage-Linie)
    sma200_score = np.where(
        df["current_price"] >= df["sma_200"], 100, 30
    )

    df["score_dip_technicals"] = dip_score * 0.60 + sma200_score * 0.40

    # ---------------------------------------------------------
    # TOTAL SCORE V5 CALCULATION
    # ---------------------------------------------------------
    df["total_score_v5"] = (
        df["score_valuation"] * 0.25
        + df["score_quality"] * 0.30
        + df["score_risk"] * 0.15
        + df["score_growth"] * 0.15
        + df["score_dip_technicals"] * 0.15
    )

    return df.sort_values(by="total_score_v5", ascending=False)
