# ============================================================
# AKTIEN-EINGABE
# ============================================================

st.subheader("📌 Aktien auswählen")

ticker_input = st.text_input(
    "Ticker eingeben (mehrere mit Komma trennen)",
    value="MSFT, GOOGL, NVDA, KO",
    placeholder="z. B. MSFT, GOOGL, NVDA, AMD, MU, TSM",
)

# Ticker bereinigen
tickers = [
    ticker.strip().upper()
    for ticker in ticker_input.split(",")
    if ticker.strip()
]

# Duplikate entfernen, Reihenfolge behalten
tickers = list(dict.fromkeys(tickers))

if not tickers:
    st.warning("Bitte mindestens einen Ticker eingeben.")
    st.stop()

st.caption(
    f"{len(tickers)} Aktien ausgewählt: {', '.join(tickers)}"
)


# ============================================================
# DEMO-DATEN
# ============================================================

def get_demo_data():

    demo = pd.DataFrame([

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

    # Nur eingegebene Ticker behalten
    return demo[demo["symbol"].isin(tickers)].copy()


df_raw = get_demo_data()

# Nicht vorhandene Ticker anzeigen
found_tickers = set(df_raw["symbol"])
missing_tickers = [
    ticker for ticker in tickers
    if ticker not in found_tickers
]

if missing_tickers:
    st.warning(
        "Noch keine Daten für: "
        + ", ".join(missing_tickers)
    )

if df_raw.empty:
    st.error(
        "Für die eingegebenen Ticker sind keine Daten vorhanden."
    )
    st.stop()


# ============================================================
# SCORE BERECHNEN
# ============================================================

df_results = calculate_stock_score_v6(df_raw)
