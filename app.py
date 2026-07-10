import streamlit as st
import yfinance as yf
import pandas as pd

# -----------------------------
# 1. STANDARDSCORE (Alex-KPI) FUNCTIONS
# -----------------------------
def score_roe_standard(roe):
    if roe is None: return 0
    if roe > 20: return 10
    if roe > 15: return 8
    if roe > 10: return 6
    if roe > 5: return 4
    return 2

def score_peg_standard(peg):
    if peg is None: return 0
    if peg < 1: return 10
    if peg < 1.5: return 8
    if peg < 2: return 6
    return 3

def score_operating_margin_standard(m):
    if m is None: return 0
    if m > 30: return 10
    if m > 20: return 8
    if m > 10: return 6
    return 3

def score_forward_pe_standard(fpe):
    if fpe is None: return 0
    if fpe < 12: return 10
    if fpe < 18: return 8
    if fpe < 25: return 6
    return 3

def score_pe_standard(pe):
    if pe is None: return 0
    if pe < 15: return 10
    if pe < 22: return 8
    if pe < 30: return 6
    return 3

def score_eps_growth_standard(g):
    if g is None: return 0
    if g > 20: return 10
    if g > 10: return 8
    if g > 5: return 6
    return 3

def score_revenue_growth_standard(g):
    if g is None: return 0
    if g > 15: return 10
    if g > 8: return 8
    if g > 3: return 6
    return 3

def score_profit_margin_standard(m):
    if m is None: return 0
    if m > 25: return 10
    if m > 15: return 8
    if m > 8: return 6
    return 3

def score_debt_to_equity_standard(d):
    if d is None: return 0
    if d < 0.3: return 10
    if d < 0.6: return 8
    if d < 1.0: return 6
    return 3

def score_cash_to_debt_standard(c):
    if c is None: return 0
    if c > 1.5: return 10
    if c > 1.0: return 8
    if c > 0.5: return 6
    return 3

def score_fcf_yield_standard(y):
    if y is None: return 0
    if y > 6: return 10
    if y > 4: return 8
    if y > 2: return 6
    return 3

# -----------------------------
# 2. DEFENSIVSCORE FUNCTIONS
# -----------------------------
def score_roe_defensive(roe):
    if roe is None: return 0
    if roe > 18: return 10
    if roe > 12: return 8
    if roe > 8: return 6
    if roe > 4: return 4
    return 1

def score_operating_margin_defensive(m):
    if m is None: return 0
    if m > 25: return 10
    if m > 18: return 8
    if m > 10: return 6
    return 2

def score_forward_pe_defensive(fpe):
    if fpe is None: return 0
    if fpe < 12: return 10
    if fpe < 16: return 8
    if fpe < 22: return 5
    return 1

def score_pe_defensive(pe):
    if pe is None: return 0
    if pe < 14: return 10
    if pe < 20: return 8
    if pe < 26: return 5
    return 1

def score_eps_growth_defensive(g):
    if g is None: return 0
    if g > 12: return 10
    if g > 7: return 8
    if g > 3: return 6
    return 2

def score_revenue_growth_defensive(g):
    if g is None: return 0
    if g > 10: return 10
    if g > 6: return 8
    if g > 2: return 6
    return 2

def score_profit_margin_defensive(m):
    if m is None: return 0
    if m > 20: return 10
    if m > 12: return 8
    if m > 6: return 6
    return 2

def score_debt_to_equity_defensive(d):
    if d is None: return 0
    if d < 0.25: return 10
    if d < 0.50: return 8
    if d < 0.80: return 5
    return 1

def score_cash_to_debt_defensive(c):
    if c is None: return 0
    if c > 2.0: return 10
    if c > 1.2: return 8
    if c > 0.6: return 6
    return 1

def score_fcf_yield_defensive(y):
    if y is None: return 0
    if y > 7: return 10
    if y > 5: return 8
    if y > 3: return 6
    return 1

# -----------------------------
# 3. OFFENSIV/AGGRESSIV FUNCTIONS (Belohnt Hyperwachstum)
# -----------------------------
def score_revenue_growth_offensive(g):
    if g is None: return 0
    if g > 25: return 10  # Extremes Top-Line Wachstum
    if g > 15: return 8
    if g > 10: return 6
    return 2

def score_eps_growth_offensive(g):
    if g is None: return 0
    if g > 30: return 10
    if g > 18: return 8
    if g > 10: return 6
    return 2

def score_peg_offensive(peg):
    if peg is None: return 0
    if peg < 1.0: return 10
    if peg < 1.6: return 8
    if peg < 2.2: return 6
    return 3

def score_roe_offensive(roe):
    if roe is None: return 0
    if roe > 25: return 10
    if roe > 15: return 8
    if roe > 8: return 5
    return 1

def score_operating_margin_offensive(m):
    if m is None: return 0
    if m > 20: return 10
    if m > 12: return 8
    if m > 5: return 5
    return 1

def score_fcf_yield_offensive(y):
    # Auch aggressive Werte brauchen FCF zum Skalieren
    if y is None: return 0
    if y > 5: return 10
    if y > 3: return 8
    if y > 1: return 5
    return 1


# -----------------------------
# GEWICHTUNGS-DICTIONARIES
# -----------------------------
STANDARD_WEIGHTS = {
    "roe": 0.15, "peg": 0.12, "operating_margin": 0.10, "forward_pe": 0.10,
    "pe": 0.08, "eps_growth": 0.10, "revenue_growth": 0.08, "profit_margin": 0.08,
    "debt_to_equity": 0.05, "cash_to_debt": 0.04, "fcf_yield": 0.10
}

DEFENSIVE_WEIGHTS = {
    "operating_margin": 0.15, "roe": 0.13, "forward_pe": 0.13, "debt_to_equity": 0.12,
    "fcf_yield": 0.12, "pe": 0.10, "cash_to_debt": 0.10, "eps_growth": 0.05,
    "revenue_growth": 0.05, "profit_margin": 0.05
}
