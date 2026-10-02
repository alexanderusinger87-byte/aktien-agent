# ============================================================
# TABLE COLORING
# ============================================================

def style_results_table(df):

    score_columns = [
        'Gesamtscore',
        'Valuation',
        'Quality',
        'Risk',
        'Tech'
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
            'Valuation': '{:.0f}',
            'Quality': '{:.0f}',
            'Risk': '{:.0f}',
            'Tech': '{:.0f}',
            'KGV (Eff)': '{:.1f}',
            'Datenvollständigkeit (%)': '{:.0f}%'
        })
    )

    return styler
