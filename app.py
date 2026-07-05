st.title("📈 Aktienanalyse Agent – Multi‑Ticker KPI & Score")

tickers_input = st.text_input("Gib mehrere Ticker ein (z.B. MSFT, NVDA, GOOG):")

if tickers_input:
    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    results = []

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info

        pe = info.get("trailingPE")
        fpe = info.get("forwardPE")
        peg = info.get("pegRatio")
        profit_margin = info.get("profitMargins")
        operating_margin = info.get("operatingMargins")
        roe = info.get("returnOnEquity")
        revenue_growth = info.get("revenueGrowth")
        eps_growth = info.get("earningsQuarterlyGrowth")
        debt_to_equity = info.get("debtToEquity")
        cash_to_debt = info.get("totalCash", 0) / info.get("totalDebt", 1)

        if profit_margin: profit_margin *= 100
        if operating_margin: operating_margin *= 100
        if roe: roe *= 100
        if revenue_growth: revenue_growth *= 100
        if eps_growth: eps_growth *= 100

        score = (
            score_roic(roe) * WEIGHTS["roic"] +
            score_peg(peg) * WEIGHTS["peg"] +
            score_operating_margin(operating_margin) * WEIGHTS["operating_margin"] +
            score_forward_pe(fpe) * WEIGHTS["forward_pe"] +
            score_pe(pe) * WEIGHTS["pe"] +
            score_eps_growth(eps_growth) * WEIGHTS["eps_growth"] +
            score_revenue_growth(revenue_growth) * WEIGHTS["revenue_growth"] +
            score_profit_margin(profit_margin) * WEIGHTS["profit_margin"] +
            score_debt_to_equity(debt_to_equity) * WEIGHTS["debt_to_equity"] +
            score_cash_to_debt(cash_to_debt) * WEIGHTS["cash_to_debt"]
        )

        results.append({
            "Ticker": ticker,
            "Score": round(score * 10, 2),
            "P/E": pe,
            "Forward P/E": fpe,
            "PEG": peg,
            "Profit Margin": profit_margin,
            "Operating Margin": operating_margin,
            "ROE": roe,
            "EPS Growth": eps_growth,
            "Revenue Growth": revenue_growth,
            "Debt/Equity": debt_to_equity,
            "Cash/Debt": cash_to_debt
        })

    df = pd.DataFrame(results)
    st.subheader("📊 Ergebnisse aller Aktien")
    st.dataframe(df)
