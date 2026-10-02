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
