import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ==============================================================================
# 1. KONFIGURATION & BENCHMARKS
# ==============================================================================
st.set_page_config(
    page_title="Pro-Stock Screener V9 - Optimized",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "TSM",
    "ASML",
    "MU",
    "VRT",
    "VST",
    "KO",
    "PEP",
    "PFE",
    "NVO",
    "NKE",
]

FALLBACK_SECTORS = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Communication Services",
    "AMZN": "Consumer Cyclical",
    "NVDA": "Technology",
    "TSM": "Technology",
    "ASML": "Technology",
    "MU": "Technology",
    "VRT": "Industrials",
    "VST": "Utilities",
    "KO": "Consumer Defensive",
    "PEP": "Consumer Defensive",
    "PFE": "Healthcare",
    "NVO": "Healthcare",
    "NKE": "Consumer Cyclical",
}


# ==============================================================================
# 2. HILFSFUNKTIONEN (DATA CLEANING & EXTRAKTION)
# ==============================================================================
def safe_get(d: Dict[str, Any], key: str, default: Any = np.nan) -> Any:
    """Greift sicher auf Dictionary-Werte zu."""
    if not isinstance(d, dict):
        return default
    val = d.get(key, default)
    return val if val is not None else default


def clean_ratio(val: Any) -> float:
    """Konvertiert Werte sicher in Float und filtert NaNs/Infs."""
    if pd.isna(val) or val is None:
        return np.nan
    try:
        f = float(val)
        return f if np.isfinite(f) else np.nan
    except (ValueError, TypeError):
        return np.nan


def get_row(df: pd.DataFrame, possible_keys: List[str]) -> Any:
    """Durchsucht Finanzberichte robust nach verschiedenen Namensvarianten einer Kennzahl."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return np.nan
    for k in possible_keys:
        if k in df.index:
            val = df.loc[k]
            if isinstance(val, pd.DataFrame):
                val = val.iloc[0]
            if isinstance(val, pd.Series):
                series = val.dropna()
                if not series.empty:
                    try:
                        return float(series.iloc[0])
                    except (ValueError, TypeError):
                        pass
            else:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
    return np.nan


def calculate_period_performance(
    close_series: pd.Series, trading_days: int
) -> float:
    """Berechnet die Performanz über einen Zeitraum in Handelstagen (z.B. 21 Tage ≈ 1 Monat)."""
    if len(close_series) < 2:
        return 0.0
    available_days = min(trading_days, len(close_series) - 1)
    start_val = close_series.iloc[-available_days - 1]
    end_val = close_series.iloc[-1]
    if start_val == 0 or pd.isna(start_val) or pd.isna(end_val):
        return 0.0
    return float((end_val - start_val) / start_val)


# ==============================================================================
# 3. DATENBESCHAFFUNG (PARALLEL & GECACHED)
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_single_ticker(symbol: str) -> Optional[Dict[str, Any]]:
    """Lädt alle Daten für einen einzelnen Ticker via yfinance."""
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1y")
        if hist.empty or len(hist) < 10:
            return None

        info = t.info or {}

        # FIX 1: Gefixtes Iterieren über fast_info anstelle von dict(t.fast_info)
        fast_info_dict = {}
        if hasattr(t, "fast_info"):
            try:
                fast_info_dict = {k: t.fast_info[k] for k in t.fast_info.keys()}
            except Exception:
                fast_info_dict = {}

        try:
            fin = t.financials
        except Exception:
            fin = pd.DataFrame()
        try:
            bs = t.balance_sheet
        except Exception:
            bs = pd.DataFrame()
        try:
            cf = t.cashflow
        except Exception:
            cf = pd.DataFrame()

        return {
            "symbol": symbol,
            "info": info,
            "fast_info": fast_info_dict,
            "hist": hist,
            "financials": fin,
            "balance_sheet": bs,
            "cashflow": cf,
        }
    except Exception:
        return None


def fetch_all_tickers_parallel(
    tickers: List[str], max_workers: int = 8
) -> Dict[str, Dict[str, Any]]:
    """Parallele Datenbeschaffung mittels ThreadPoolExecutor."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        future_to_ticker = {
            executor.submit(fetch_single_ticker, sym): sym for sym in tickers
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            sym = future_to_ticker[future]
            try:
                data = future.result()
                if data:
                    results[sym] = data
            except Exception:
                pass
    return results


# ==============================================================================
# 4. METRICS ENGINE (ANALYSE & BERECHNUNG)
# ==============================================================================
def calculate_advanced_metrics(
    data: Dict[str, Any], symbol: str
) -> Dict[str, Any]:
    info = data["info"]
    fast_info = data["fast_info"]
    hist = data["hist"]
    fin = data["financials"]
    bs = data["balance_sheet"]
    cf = data["cashflow"]

    metrics: Dict[str, Any] = {}

    # Sector & Name
    sector = safe_get(info, "sector")
    if pd.isna(sector) or sector == "Default":
        sector = FALLBACK_SECTORS.get(symbol, "Default")
    metrics["sector"] = sector

    name = safe_get(info, "shortName")
    if pd.isna(name):
        name = safe_get(info, "longName", symbol)
    metrics["name"] = name

    # Preis & Performance
    close = hist["Close"]
    current_price = float(close.iloc[-1])
    metrics["current_price"] = current_price

    max_52w = float(close.max())
    metrics["dist_52w_high"] = (
        (current_price - max_52w) / max_52w if max_52w > 0 else 0.0
    )

    metrics["perf_1m"] = calculate_period_performance(close, trading_days=21)
    metrics["perf_6m"] = calculate_period_performance(close, trading_days=126)

    # Moving Average 200
    if len(close) >= 200:
        metrics["sma_200"] = float(close.rolling(200).mean().iloc[-1])
        metrics["above_sma200"] = current_price > metrics["sma_200"]
    else:
        metrics["sma_200"] = np.nan
        metrics["above_sma200"] = False

    # Market Cap
    market_cap = clean_ratio(
        fast_info.get(
            "market_cap",
            fast_info.get("marketCap", safe_get(info, "marketCap")),
        )
    )
    metrics["market_cap"] = market_cap

    # Bewertungsmultiples
    pe_effective = clean_ratio(safe_get(info, "forwardPE"))
    if pd.isna(pe_effective) or pe_effective <= 0:
        pe_effective = clean_ratio(safe_get(info, "trailingPE"))
    metrics["pe_effective"] = pe_effective

    metrics["ps_ratio"] = clean_ratio(
        safe_get(info, "priceToSalesTrailing12Months")
    )
    metrics["pb_ratio"] = clean_ratio(safe_get(info, "priceToBook"))
    metrics["peg_ratio"] = clean_ratio(safe_get(info, "pegRatio"))

    # Free Cashflow Yield (Erweiterte Suchschlüssel)
    op_cf = get_row(
        cf, ["Operating Cash Flow", "Total Cash From Operating Activities"]
    )
    capex = get_row(
        cf,
        [
            "Capital Expenditure",
            "Capital Expenditures",
            "CapEx",
            "Purchase Of Property And Equipment",
            "Purchase Of Property Plant And Equipment",
            "Net PPE Purchase And Sale",
        ],
    )

    if pd.notna(op_cf) and pd.notna(capex):
        fcf = op_cf - abs(capex)
    else:
        fcf = clean_ratio(safe_get(info, "freeCashflow"))

    metrics["fcf_yield"] = (
        (fcf / market_cap)
        if (pd.notna(fcf) and pd.notna(market_cap) and market_cap > 0)
        else np.nan
    )

    # Qualität & Rentabilität
    metrics["roe"] = clean_ratio(safe_get(info, "returnOnEquity"))
    metrics["gross_margin"] = clean_ratio(safe_get(info, "grossMargins"))

    # ROIC & Verschuldung (Nicht für Finanzdienstleister)
    if sector != "Financial Services":
        ebit = get_row(fin, ["EBIT", "Operating Income"])
        inc_tax = get_row(fin, ["Tax Provision", "Income Tax Expense"])
        pre_tax = get_row(fin, ["Pretax Income", "Income Before Tax"])
        total_assets = get_row(bs, ["Total Assets"])
        curr_liab = get_row(
            bs, ["Current Liabilities", "Total Current Liabilities"]
        )
        cash = get_row(
            bs,
            [
                "Cash And Cash Equivalents",
                "Cash Financial",
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Short Term Investments",
            ],
        )

        if (
            pd.notna(ebit)
            and pd.notna(total_assets)
            and pd.notna(curr_liab)
            and pd.notna(cash)
        ):
            tax_rate = 0.21
            if pd.notna(inc_tax) and pd.notna(pre_tax) and pre_tax > 0:
                tax_rate = max(0.10, min(inc_tax / pre_tax, 0.40))
            nopat = ebit * (1 - tax_rate)
            invested_capital = total_assets - curr_liab - cash
            metrics["roic"] = (
                nopat / invested_capital if invested_capital > 0 else np.nan
            )
        else:
            metrics["roic"] = np.nan

        # FIX 2: Verschuldungsgrad (Net Debt / EBITDA) - Abfangen negativer EBITDAs
        tot_debt = get_row(
            bs,
            [
                "Total Debt",
                "Long Term Debt",
                "Long Term Debt And Capital Lease Obligation",
            ],
        )
        ebitda = get_row(
            fin, ["Normalized EBITDA", "EBITDA", "Operating Income"]
        )

        if pd.isna(tot_debt) and pd.notna(cash):
            tot_debt = 0.0

        if pd.notna(tot_debt) and pd.notna(cash) and pd.notna(ebitda):
            net_debt = tot_debt - cash
            if ebitda > 0:
                metrics["net_debt_ebitda"] = net_debt / ebitda
            else:
                # Extremes Risiko bei Unprofitabilität & Schulden, 0 bei Netto-Guthaben
                metrics["net_debt_ebitda"] = 99.0 if net_debt > 0 else 0.0
        else:
            metrics["net_debt_ebitda"] = np.nan
    else:
        metrics["roic"] = np.nan
        metrics["net_debt_ebitda"] = np.nan

    metrics["current_ratio"] = clean_ratio(safe_get(info, "currentRatio"))

    # Turnaround-Logik
    dist_high = metrics.get("dist_52w_high", 0.0)
    perf_1m = metrics.get("perf_1m", 0.0)
    pe = metrics.get("pe_effective", np.nan)

    is_beaten_down = dist_high <= -0.20
    is_rebounding = pd.notna(perf_1m) and perf_1m >= 0.02
    is_reasonably_priced = pd.notna(pe) and (0 < pe < 25)

    if is_beaten_down:
        if is_rebounding and is_reasonably_priced:
            metrics["turnaround_status"] = "🚀 Aktiver Turnaround"
        elif is_rebounding:
            metrics["turnaround_status"] = "📈 Rebound (KGV hoch/kein KGV)"
        elif pd.notna(perf_1m) and perf_1m < -0.02:
            metrics["turnaround_status"] = "⚠️ Fallendes Messer"
        else:
            metrics["turnaround_status"] = "⏱ Bodenbildung"
    else:
        metrics["turnaround_status"] = "🛡 Trend / Normal"

    # Datenqualitäts-Score (Branchenangepasst)
    if sector != "Financial Services":
        core_fields = [
            metrics["pe_effective"],
            metrics["ps_ratio"],
            metrics["fcf_yield"],
            metrics["market_cap"],
            metrics["gross_margin"],
            metrics["current_ratio"],
            metrics["roic"],
            metrics["net_debt_ebitda"],
        ]
    else:
        core_fields = [
            metrics["pe_effective"],
            metrics["pb_ratio"],
            metrics["market_cap"],
            metrics["roe"],
        ]

    valid_count = sum([1 for f in core_fields if pd.notna(f)])
    metrics["data_quality"] = round((valid_count / len(core_fields)) * 100)

    return metrics


# ==============================================================================
# 5. SCORING ENGINE (SCORING & GEWICHTUNG)
# ==============================================================================
def score_stock_v9(
    metrics: Dict[str, Any]
) -> Tuple[float, Dict[str, float]]:
    """Errechnet Sub-Scores und Gesamtscore (0-100)."""
    scores = {}

    # 1. Valuation Score
    val_points = []
    pe = metrics.get("pe_effective")

    # FIX 3: Abstrafung für negatives KGV (Verlustbringer)
    if pd.notna(pe):
        if pe > 0:
            val_points.append(
                100 if pe < 15 else (70 if pe < 25 else (40 if pe < 40 else 10))
            )
        else:
            val_points.append(0)

    fcf_y = metrics.get("fcf_yield")
    if pd.notna(fcf_y):
        val_points.append(
            100
            if fcf_y > 0.07
            else (70 if fcf_y > 0.04 else (40 if fcf_y > 0.01 else 10))
        )

    peg = metrics.get("peg_ratio")
    if pd.notna(peg) and peg > 0:
        val_points.append(
            100 if peg < 1.0 else (70 if peg < 1.5 else (30 if peg < 2.5 else 0))
        )

    scores["valuation"] = float(np.mean(val_points)) if val_points else np.nan

    # 2. Quality Score
    qual_points = []
    roic = metrics.get("roic")
    roe = metrics.get("roe")
    rentability = roic if pd.notna(roic) else roe
    if pd.notna(rentability):
        qual_points.append(
            100
            if rentability > 0.18
            else (75 if rentability > 0.12 else (40 if rentability > 0.06 else 0))
        )

    gm = metrics.get("gross_margin")
    if pd.notna(gm):
        qual_points.append(
            100 if gm > 0.50 else (70 if gm > 0.30 else (40 if gm > 0.15 else 10))
        )

    scores["quality"] = float(np.mean(qual_points)) if qual_points else np.nan

    # 3. Risk Score
    risk_points = []
    nd_ebitda = metrics.get("net_debt_ebitda")
    if pd.notna(nd_ebitda):
        risk_points.append(
            100
            if nd_ebitda < 1.0
            else (70 if nd_ebitda < 2.5 else (30 if nd_ebitda < 4.0 else 0))
        )

    cr = metrics.get("current_ratio")
    if pd.notna(cr):
        risk_points.append(
            100 if cr > 1.5 else (70 if cr > 1.0 else (30 if cr > 0.8 else 0))
        )

    scores["risk"] = float(np.mean(risk_points)) if risk_points else np.nan

    # 4. Tech / Momentum Score
    tech_points = []
    perf_6m = metrics.get("perf_6m")
    if pd.notna(perf_6m):
        tech_points.append(
            100
            if perf_6m > 0.20
            else (70 if perf_6m > 0.05 else (40 if perf_6m > -0.10 else 10))
        )

    if metrics.get("above_sma200", False):
        tech_points.append(85)
    elif pd.notna(metrics.get("sma_200")):
        tech_points.append(25)

    scores["tech"] = float(np.mean(tech_points)) if tech_points else np.nan

    # Dynamische Neugewichtung
    data_qual = metrics.get("data_quality", 0)
    if data_qual < 40:
        return np.nan, scores

    base_weights = {
        "valuation": 0.30,
        "quality": 0.35,
        "risk": 0.15,
        "tech": 0.20,
    }
    active_weights = {}
    valid_scores = {}

    for cat, weight in base_weights.items():
        if pd.notna(scores.get(cat)):
            valid_scores[cat] = scores[cat]
            active_weights[cat] = weight

    total_weight = sum(active_weights.values())
    if total_weight == 0:
        return np.nan, scores

    final_score = sum(
        valid_scores[cat] * (active_weights[cat] / total_weight)
        for cat in valid_scores
    )
    return float(round(final_score, 1)), scores


# ==============================================================================
# 6. STREAMLIT OBERFLÄCHE (UI)
# ==============================================================================
st.title("📈 Institutional Stock Screener V9 (Optimized)")
st.markdown(
    "Erweiterter Screener zur Analyse von Bewertung, Qualität, Risiko und Momentum."
)

# Sidebar
st.sidebar.header("⚙️ Einstellungen")
input_tickers = st.sidebar.text_area(
    "Ticker-Symbole (kommagetrennt):",
    value=", ".join(DEFAULT_TICKERS),
    height=120,
)
ticker_list = [
    t.strip().upper() for t in input_tickers.split(",") if t.strip()
]

min_score = st.sidebar.slider("Mindest-Score Filter", 0, 100, 50)

if st.sidebar.button("🚀 Cache leeren & Neu laden", type="primary"):
    st.cache_data.clear()

# Daten verarbeiten
with st.spinner("Lade Finanzdaten & erstelle Kennzahlen..."):
    raw_data = fetch_all_tickers_parallel(ticker_list)

processed_stocks = []
for sym in ticker_list:
    if sym in raw_data:
        m = calculate_advanced_metrics(raw_data[sym], sym)
        score, sub_scores = score_stock_v9(m)
        if pd.notna(score):
            processed_stocks.append(
                {
                    "Ticker": sym,
                    "Name": m["name"],
                    "Sektor": m["sector"],
                    "Preis": m["current_price"],
                    "Score": score,
                    "SubScores": sub_scores,
                    "Turnaround": m["turnaround_status"],
                    "P/E": m["pe_effective"],
                    "FCF Yield": m["fcf_yield"],
                    "ROIC": m["roic"],
                    "Datenqualität": m["data_quality"],
                    "Metrics": m,
                }
            )

df_results = pd.DataFrame(processed_stocks)

if df_results.empty:
    st.warning(
        "Keine validen Daten gefunden oder Datenqualität unter 40%. Bitte überprüfe die Ticker-Symbole."
    )
    st.stop()

# FIX 4: Dynamische Sektor-Auswahl basierend auf den tatsächlich geladenen Daten
available_sectors = ["Alle"] + sorted(
    df_results["Sektor"].dropna().unique().tolist()
)
selected_sector = st.sidebar.selectbox("Sektor-Filter", available_sectors)

# Filter anwenden
filtered_df = df_results[df_results["Score"] >= min_score]
if selected_sector != "Alle":
    filtered_df = filtered_df[filtered_df["Sektor"] == selected_sector]

filtered_df = filtered_df.sort_values(by="Score", ascending=False)

# Übersichtstabelle
st.subheader("🏆 Top-Ranking Übersicht")

display_df = filtered_df[
    [
        "Ticker",
        "Name",
        "Sektor",
        "Preis",
        "Score",
        "Turnaround",
        "P/E",
        "FCF Yield",
        "Datenqualität",
    ]
].copy()

# Sichere String-Formatierung gegen NaNs
display_df["Preis"] = display_df["Preis"].map(
    lambda x: f"{x:.2f} $" if pd.notna(x) else "-"
)
display_df["P/E"] = display_df["P/E"].map(
    lambda x: f"{x:.1f}" if pd.notna(x) else "-"
)
display_df["FCF Yield"] = display_df["FCF Yield"].map(
    lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-"
)
display_df["Datenqualität"] = display_df["Datenqualität"].map(
    lambda x: f"{x}%" if pd.notna(x) else "-"
)

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Detailansicht pro Aktie
st.subheader("🔍 Einzelanalysen")

for _, row in filtered_df.iterrows():
    m = row["Metrics"]
    sub = row["SubScores"]

    with st.expander(
        f"**{row['Ticker']}** - {row['Name']} | Score: **{row['Score']}/100** | Status: {row['Turnaround']}"
    ):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write(f"**Sektor:** {m['sector']}")
            st.write(f"**Datenqualität:** {m['data_quality']}%")

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric(
                "KGV (PE)",
                f"{m['pe_effective']:.1f}"
                if pd.notna(m["pe_effective"])
                else "-",
            )
            m_col1.metric(
                "FCF Yield",
                f"{m['fcf_yield']*100:.1f}%"
                if pd.notna(m["fcf_yield"])
                else "-",
            )

            m_col2.metric(
                "ROIC",
                f"{m['roic']*100:.1f}%" if pd.notna(m["roic"]) else "-",
            )
            m_col2.metric(
                "Gross Margin",
                f"{m['gross_margin']*100:.1f}%"
                if pd.notna(m["gross_margin"])
                else "-",
            )

            dist_val = m.get("dist_52w_high")
            dist_str = f"{dist_val * 100:.1f}%" if pd.notna(dist_val) else "-"
            m_col3.metric("Abstand 52W Hoch", dist_str)

            nd_val = m.get("net_debt_ebitda")
            nd_str = f"{nd_val:.1f}x" if pd.notna(nd_val) else "-"
            m_col3.metric("Net Debt / EBITDA", nd_str)

        # Plotly Radar Chart
        with col2:
            categories = ["Valuation", "Quality", "Risk", "Tech"]
            values = [
                sub.get("valuation", 0) if pd.notna(sub.get("valuation")) else 0,
                sub.get("quality", 0) if pd.notna(sub.get("quality")) else 0,
                sub.get("risk", 0) if pd.notna(sub.get("risk")) else 0,
                sub.get("tech", 0) if pd.notna(sub.get("tech")) else 0,
            ]

            categories_closed = categories + [categories[0]]
            values_closed = values + [values[0]]

            fig = go.Figure()
            fig.add_trace(
                go.Scatterpolar(
                    r=values_closed,
                    theta=categories_closed,
                    fill="toself",
                    name=row["Ticker"],
                )
            )

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                margin=dict(l=30, r=30, t=30, b=30),
                height=250,
            )
            st.plotly_chart(
                fig, use_container_width=True, key=f"radar_{row['Ticker']}"
            )
