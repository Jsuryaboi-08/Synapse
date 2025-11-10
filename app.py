import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import datetime
import time

# ---
# CRITICAL: TA-Lib Installation
# This library is required for pattern recognition. It can be difficult to install.
# 1. On Windows: Try 'pip install ta-lib-binary'
# 2. On Mac/Linux: You may need to install the C-library first:
#    - Mac: 'brew install ta-lib'
#    - Linux: 'sudo apt-get install libta-lib-dev'
#    - THEN: 'pip install TA-Lib'
# If installation fails, you can comment out the 'import talib' line
# and the 'plot_candlestick_patterns' function call to run the rest of the app.
try:
    import talib
    TALIB_LOADED = True
except ImportError:
    TALIB_LOADED = False
    # Don't st.error here, do it in the main app body if needed
    pass

# ---
# PAGE CONFIGURATION & STYLING
# ---

# Set the layout to "wide" for a more professional, dense feel like the reference
st.set_page_config(layout="wide", page_title="Risk-Adjusted Analyst", page_icon="📈")

# Custom CSS to mimic the reference UI (dark, clean, professional)
def load_css():
    st.markdown("""
    <style>
    /* Google Font for clean text */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main app background */
    .stApp {
        background-color: #0b0c0e; /* Even darker background */
        color: #FAFAFA;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #121417; /* Dark sidebar */
        border-right: 1px solid #2D2D2D;
    }
    .st-sidebar .st-emotion-cache-16txtl3 { /* Sidebar title */
        font-weight: 700;
        font-size: 1.75rem;
    }

    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* Metric cards (like your reference) */
    [data-testid="stMetric"] {
        background-color: #121417; /* Dark card background */
        border: 1px solid #2D2D2D;
        border-radius: 12px;
        padding: 1.25rem;
    }
    [data-testid="stMetric"] > label {
        color: #A0A0A0; /* Lighter label text */
        font-weight: 500;
    }
    [data-testid="stMetric"] > div[data-testid="metric-value"] {
        font-size: 2rem;
        font-weight: 600;
    }

    /* Buttons */
    .stButton > button {
        background-color: #00A89C; /* A professional teal accent */
        color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #007A70;
        color: white;
    }
    
    /* Sidebar Radio Navigation (to mimic reference) */
    .stRadio [data-baseweb="radio"] {
        background-color: #121417;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .stRadio [data-baseweb="radio"] label {
        color: #A0A0A0;
        font-weight: 500;
    }
    /* Selected radio item */
    .stRadio [data-baseweb="radio"] input:checked + div {
        background-color: #00A89C !important;
        color: white;
    }
    .stRadio [data-baseweb="radio"] input:checked + div label {
        color: white;
        font-weight: 600;
    }

    /* Reference-style DCF Table */
    .dcf-table {
        font-family: 'Inter', sans-serif;
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    .dcf-table th, .dcf-table td {
        border-top: 1px solid #2D2D2D;
        border-bottom: 1px solid #2D2D2D;
        padding: 12px 14px;
        text-align: right;
        white-space: nowrap;
    }
    .dcf-table th {
        background-color: #121417;
        color: #A0A0A0;
        font-weight: 600;
        text-align: right;
    }
    .dcf-table th:first-child {
        text-align: left;
    }
    .dcf-table td:first-child {
        text-align: left;
        font-weight: 600;
        color: #FAFAFA;
    }
    .dcf-table tr:nth-child(even) {
        background-color: #121417;
    }
    /* Highlight bold rows like reference */
    .dcf-table .highlight-row td {
        font-weight: 700 !important;
        color: #FFFFFF !important;
    }
    .dcf-table .spacer-row td {
        padding: 8px;
        border: none;
    }

    /* Custom Valuation Summary Boxes */
    .summary-box {
        background-color: #121417;
        border: 1px solid #2D2D2D;
        border-radius: 12px;
        padding: 1.5rem;
    }
    .summary-box h4 {
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    .summary-box .price-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    .summary-box .price-label {
        font-size: 1rem;
        color: #A0A0A0;
    }
    .summary-box .price-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FAFAFA;
    }
    .summary-box .upside-bar {
        width: 100%;
        height: 10px;
        background-color: #2D2D2D;
        border-radius: 5px;
        margin-top: 1.25rem;
        margin-bottom: 0.5rem;
    }
    .summary-box .upside-fill {
        height: 10px;
        border-radius: 5px;
    }
    .summary-box .upside-text {
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ---
# DATA PULLING & CACHING
# ---

@st.cache_data
def get_ticker_data(ticker):
    """
    Pulls all necessary yfinance data for a ticker.
    This is our "heavy" initial data pull.
    """
    try:
        tk = yf.Ticker(ticker)
        # Prefetch all data to store in cache
        data = {
            "info": tk.info,
            "history_1y": tk.history(period="1y"),
            "history_5y": tk.history(period="5y"),
            "financials": tk.financials,
            "balance_sheet": tk.balance_sheet,
            "cashflow": tk.cashflow,
            "news": tk.news
        }
        # Check for empty data
        if data["history_1y"].empty:
            return None, f"No historical data found for ticker: {ticker}"
        if not data["info"].get("shortName"):
             return None, f"Ticker not found or info is missing: {ticker}"
        return data, None
    except Exception as e:
        return None, f"An error occurred pulling data: {e}"

# ---
# CALCULATION FUNCTIONS (The "Advanced Analysis")
# ---

@st.cache_data
def calculate_ratios(_data):
    """Calculates all 8 key ratios."""
    ratios = {}
    try:
        # Get data from cached object
        info = _data["info"]
        financials = _data["financials"]
        balance_sheet = _data["balance_sheet"]
        cashflow = _data["cashflow"]

        # Get TTM data (most recent 4 quarters)
        inc_ttm = financials.sum(axis=1)
        bs_latest = balance_sheet.iloc[:, 0]
        cf_ttm = cashflow.sum(axis=1)

        # 1. Gross Margin
        gross_profit = inc_ttm.get("Gross Profit", np.nan)
        revenue = inc_ttm.get("Total Revenue", np.nan)
        ratios["Gross Margin"] = gross_profit / revenue if revenue else np.nan

        # 2. EBIT Margin
        ebit = inc_ttm.get("EBIT", np.nan)
        ratios["EBIT Margin"] = ebit / revenue if revenue else np.nan

        # 3. ROE (Return on Equity)
        net_income = inc_ttm.get("Net Income", np.nan)
        equity = bs_latest.get("Total Stockholder Equity", np.nan)
        ratios["ROE"] = net_income / equity if equity else np.nan

        # 4. ROIC (Return on Invested Capital)
        tax_provision = inc_ttm.get("Tax Provision", 0)
        income_before_tax = inc_ttm.get("Income Before Tax", 1)
        tax_rate = tax_provision / income_before_tax if income_before_tax else 0.21
        nopat = ebit * (1 - tax_rate)
        
        long_term_debt = bs_latest.get("Long Term Debt", 0)
        short_term_debt = bs_latest.get("Short Long Term Debt", 0)
        cash = bs_latest.get("Cash And Cash Equivalents", 0)
        
        net_debt = long_term_debt + short_term_debt - cash
        invested_capital = net_debt + equity
        ratios["ROIC"] = nopat / invested_capital if invested_capital else np.nan

        # 5. Net Debt
        ratios["Net Debt"] = net_debt

        # 6. Gearing (Net Debt / Equity)
        ratios["Gearing"] = net_debt / equity if equity else np.nan

        # 7. Working Capital Cycle (approximation)
        receivables = bs_latest.get("Net Receivables", np.nan)
        inventory = bs_latest.get("Inventory", np.nan)
        payables = bs_latest.get("Accounts Payable", np.nan)
        cogs = inc_ttm.get("Cost Of Revenue", np.nan)
        
        dso = (receivables / revenue) * 365 if revenue else np.nan
        dio = (inventory / cogs) * 365 if cogs else np.nan
        dpo = (payables / cogs) * 365 if cogs else np.nan
        
        ratios["WC Cycle (Days)"] = dso + dio - dpo

        # 8. Altman Z-Score
        wc = bs_latest.get("Total Current Assets", 0) - bs_latest.get("Total Current Liabilities", 0)
        total_assets = bs_latest.get("Total Assets", 0)
        re = bs_latest.get("Retained Earnings", 0)
        mkt_cap = info.get("marketCap", 0)
        total_liab = bs_latest.get("Total Liab", 0)
        
        A = wc / total_assets if total_assets else 0
        B = re / total_assets if total_assets else 0
        C = ebit / total_assets if total_assets else 0
        D = mkt_cap / total_liab if total_liab else 0
        E = revenue / total_assets if total_assets else 0
        
        ratios["Altman Z-Score"] = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E

        return ratios
    
    except Exception as e:
        return {r: np.nan for r in ["Gross Margin", "EBIT Margin", "ROE", "ROIC", "Net Debt", "Gearing", "WC Cycle (Days)", "Altman Z-Score"]}

@st.cache_data
def run_monte_carlo(_data, simulations=10000, time_horizon_days=252):
    """
    Runs the 'heavy' Monte Carlo simulation.
    """
    try:
        hist = _data["history_1y"]['Close']
        log_returns = np.log(1 + hist.pct_change())
        
        u = log_returns.mean()
        var = log_returns.var()
        drift = u - (0.5 * var)
        stdev = log_returns.std()
        T = time_horizon_days
        S0 = hist.iloc[-1]
        
        daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (T, simulations)))
        price_paths = np.zeros_like(daily_returns)
        price_paths[0] = S0
        
        for t in range(1, T):
            price_paths[t] = price_paths[t-1] * daily_returns[t]
            
        ending_prices = price_paths[-1]
        
        prob_gain = (ending_prices > S0).sum() / simulations
        prob_plus_10 = (ending_prices > (S0 * 1.10)).sum() / simulations
        prob_minus_10 = (ending_prices < (S0 * 0.90)).sum() / simulations
        
        percentile_5 = np.percentile(ending_prices, 5)
        var_95 = (S0 - percentile_5) / S0
        cvar_95 = (S0 - ending_prices[ending_prices <= percentile_5].mean()) / S0

        returns_pct = (ending_prices / S0) - 1
        risk_free_rate = 0.04 # Assume 4% risk-free rate
        
        sharpe_ratio = (returns_pct.mean() - risk_free_rate) / returns_pct.std() if returns_pct.std() != 0 else np.nan
        downside_returns = returns_pct[returns_pct < 0]
        downside_std = downside_returns.std()
        sortino_ratio = (returns_pct.mean() - risk_free_rate) / downside_std if downside_std != 0 else np.nan

        return {
            "price_paths": price_paths, "ending_prices": ending_prices, "S0": S0,
            "prob_gain": prob_gain, "prob_plus_10": prob_plus_10, "prob_minus_10": prob_minus_10,
            "var_95": var_95, "cvar_95": cvar_95, "sharpe_ratio": sharpe_ratio, "sortino_ratio": sortino_ratio
        }
    except Exception as e:
        return None

@st.cache_data
def calculate_dcf_assumptions(_data):
    """
    NEW: Automatically calculates DCF assumptions, removing user input.
    This adds "computational weight" and mimics professional platforms.
    """
    try:
        financials = _data["financials"]
        balance_sheet = _data["balance_sheet"].iloc[:, 0]
        info = _data["info"]

        # 1. Revenue Growth (g): 5Y CAGR
        # Get last 5 years of revenue
        revenues = financials.loc["Total Revenue"].dropna()
        if len(revenues) < 2:
            g = 0.05 # Default if not enough data
        else:
            start_rev = revenues.iloc[-1]
            end_rev = revenues.iloc[0]
            if start_rev <= 0: # Handle negative or zero start
                g = 0.05
            else:
                num_years = len(revenues) - 1
                g = (end_rev / start_rev) ** (1 / num_years) - 1
                # Cap growth at a reasonable 20%
                g = min(g, 0.20)

        # 2. WACC (Weighted Average Cost of Capital)
        # This is very complex. We must simplify and state assumptions.
        # Cost of Equity (Re) using CAPM: Re = Rf + Beta * (Rm - Rf)
        risk_free_rate = 0.04 # Assumption: 10-Yr Treasury
        market_risk_premium = 0.06 # Assumption: Standard Market Premium
        beta = info.get("beta", 1.0) # Get Beta from yfinance, default to 1
        cost_of_equity = risk_free_rate + beta * market_risk_premium
        
        # Cost of Debt (Rd)
        interest_expense = financials.loc["Interest Expense"].iloc[0]
        long_term_debt = balance_sheet.get("Long Term Debt", 0)
        short_term_debt = balance_sheet.get("Short Long Term Debt", 0)
        total_debt = long_term_debt + short_term_debt
        cost_of_debt = interest_expense / total_debt if total_debt else 0.05 # 5% default
        
        # Tax Rate
        tax_provision = financials.loc["Tax Provision"].iloc[0]
        income_before_tax = financials.loc["Income Before Tax"].iloc[0]
        tax_rate = tax_provision / income_before_tax if income_before_tax else 0.21
        
        # WACC Calculation
        market_cap = info.get("marketCap", 0)
        equity_value = market_cap
        enterprise_value = market_cap + total_debt - balance_sheet.get("Cash And Cash Equivalents", 0)
        
        e_percent = equity_value / enterprise_value if enterprise_value else 0
        d_percent = total_debt / enterprise_value if enterprise_value else 0
        
        wacc = (e_percent * cost_of_equity) + (d_percent * cost_of_debt * (1 - tax_rate))
        
        # 3. Long-Term Growth (lt_g)
        # Standard assumption: ~long-term inflation/GDP growth
        lt_g = 0.025

        return {
            "g": g,
            "wacc": wacc,
            "lt_g": lt_g,
            "tax_rate": tax_rate
        }
    except Exception as e:
        # Return safe defaults
        return {"g": 0.05, "wacc": 0.08, "lt_g": 0.025, "tax_rate": 0.21}


@st.cache_data
def calculate_dcf_model(_data, assumptions):
    """
    Performs a 5-year DCF calculation based on automated assumptions.
    Rebuilt to match the reference screenshot EXACTLY.
    """
    try:
        # Get TTM data
        financials = _data["financials"]
        cashflow = _data["cashflow"]
        balance_sheet = _data["balance_sheet"]
        info = _data["info"]
        
        # Get TTM values (sum of last 4 quarters)
        ttm_revenue = financials.sum(axis=1).get("Total Revenue", 0)
        ttm_ebitda = financials.sum(axis=1).get("EBITDA", 0)
        ttm_d_a = financials.sum(axis=1).get("Depreciation And Amortization", 0)
        ttm_ebit = ttm_ebitda - ttm_d_a
        ttm_capex = cashflow.sum(axis=1).get("Capital Expenditure", 0)
        
        # NWC calculation (from most recent balance sheet)
        bs_latest = balance_sheet.iloc[:, 0]
        wc_assets = bs_latest.get("Net Receivables", 0) + bs_latest.get("Inventory", 0)
        wc_liab = bs_latest.get("Accounts Payable", 0)
        ttm_nwc = wc_assets - wc_liab
        
        # Get assumptions
        g = assumptions["g"]
        wacc = assumptions["wacc"]
        lt_g = assumptions["lt_g"]
        tax_rate = assumptions["tax_rate"]
        
        # Ratios for projection (as % of revenue)
        ebitda_margin = ttm_ebitda / ttm_revenue if ttm_revenue else 0.15
        d_a_ratio = ttm_d_a / ttm_revenue if ttm_revenue else 0.05
        capex_ratio = ttm_capex / ttm_revenue if ttm_revenue else 0.06 # Note: often negative
        nwc_ratio = ttm_nwc / ttm_revenue if ttm_revenue else 0.1
        
        # Build the 5-year forecast + terminal year
        years = [f"FY{datetime.now().year + i + 1}" for i in range(5)]
        columns = years + ["Terminal"]
        
        df = pd.DataFrame(index=[
            "Revenue", "EBITDA", "EBIT", "NOPAT",
            "(+) Depreciation & Amortization", "(-) CapEx", "(-) Change in NWC",
            "Free Cash Flow to Firm (FCFF)", "Discount Factor", "PV of FCFF"
        ], columns=columns, dtype=float)
        
        last_revenue = ttm_revenue
        last_nwc = ttm_nwc
        
        # Project 5 years
        for i, year in enumerate(years):
            revenue = last_revenue * (1 + g)
            ebitda = revenue * ebitda_margin
            d_a = revenue * d_a_ratio
            ebit = ebitda - d_a
            nopat = ebit * (1 - tax_rate)
            
            capex = revenue * capex_ratio # CapEx is cash out, so it's negative
            nwc = revenue * nwc_ratio
            change_in_nwc = nwc - last_nwc
            
            fcff = nopat + d_a + capex - change_in_nwc # Add CapEx because it's already negative
            
            discount_factor = (1 / (1 + wacc)) ** (i + 1)
            pv_fcff = fcff * discount_factor
            
            # Store values
            df.loc["Revenue", year] = revenue
            df.loc["EBITDA", year] = ebitda
            df.loc["EBIT", year] = ebit
            df.loc["NOPAT", year] = nopat
            df.loc["(+) Depreciation & Amortization", year] = d_a
            df.loc["(-) CapEx", year] = capex
            df.loc["(-) Change in NWC", year] = change_in_nwc
            df.loc["Free Cash Flow to Firm (FCFF)", year] = fcff
            df.loc["Discount Factor", year] = discount_factor
            df.loc["PV of FCFF", year] = pv_fcff
            
            # Update for next loop
            last_revenue = revenue
            last_nwc = nwc

        # Terminal Value Calculation (Perpetuity Growth Model)
        last_fcff = df.loc["Free Cash Flow to Firm (FCFF)", years[-1]]
        terminal_fcff = last_fcff * (1 + lt_g)
        terminal_value = terminal_fcff / (wacc - lt_g)
        
        pv_terminal_value = terminal_value * df.loc["Discount Factor", years[-1]]
        
        df.loc["Terminal Value", "Terminal"] = terminal_value
        df.loc["PV of Terminal Value", "Terminal"] = pv_terminal_value
        
        # Enterprise & Equity Value
        enterprise_value = df.loc["PV of FCFF"].sum() + pv_terminal_value
        
        # Get Net Debt
        net_debt = calculate_ratios(_data)["Net Debt"]
        equity_value = enterprise_value - net_debt
        shares_outstanding = info.get("sharesOutstanding", 0)
        
        if shares_outstanding == 0:
            return None, "Missing shares outstanding data to calculate fair value."
            
        fair_value = equity_value / shares_outstanding
        current_price = info.get("currentPrice", 0)
        if not current_price: # Fallback
            current_price = _data["history_1y"]["Close"].iloc[-1]
            
        upside = (fair_value / current_price) - 1 if current_price else np.nan

        return {
            "fair_value": fair_value,
            "current_price": current_price,
            "upside": upside,
            "dcf_table": df,
            "assumptions": assumptions
        }, None

    except Exception as e:
        return None, f"Error in DCF calculation: {e}"

@st.cache_data
def plot_candlestick_patterns(_data):
    """
    Plots a comprehensive candlestick chart with TA-Lib patterns.
    """
    if not TALIB_LOADED:
        return go.Figure().update_layout(title="TA-Lib not loaded. Chart disabled.", template="plotly_dark")
        
    try:
        df = _data["history_5y"].copy() # Use 5y data for patterns
        
        # 1. Identify patterns
        df['CDLDOJI'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDLHAMMER'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDLENGULFING'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
        
        doji_days = df[df['CDLDOJI'] != 0]
        hammer_days = df[df['CDLHAMMER'] != 0]
        engulfing_days = df[df['CDLENGULFING'] != 0]
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Candlestick'
        ))

        # Add markers
        fig.add_trace(go.Scatter(
            x=doji_days.index, y=doji_days['Low'] * 0.98, mode='markers',
            marker=dict(symbol='circle', color='cyan', size=8), name='Doji'
        ))
        fig.add_trace(go.Scatter(
            x=hammer_days.index, y=hammer_days['Low'] * 0.98, mode='markers',
            marker=dict(symbol='triangle-up', color='yellow', size=8), name='Hammer'
        ))
        fig.add_trace(go.Scatter(
            x=engulfing_days.index, y=engulfing_days['Low'] * 0.98, mode='markers',
            marker=dict(symbol='square', color='magenta', size=8), name='Engulfing'
        ))

        # Style the chart
        fig.update_layout(
            title="Candlestick Chart with TA-Lib Pattern Recognition",
            xaxis_title="Date", yaxis_title="Price", template="plotly_dark",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    except Exception as e:
        return go.Figure().update_layout(title=f"Error plotting chart: {e}", template="plotly_dark")

# ---
# HELPER FUNCTIONS FOR UI
# ---
def format_value(val, type="number"):
    if pd.isna(val): return "N/A"
    if type == "percent": return f"{val:.2%}"
    if type == "decimal": return f"{val:.2f}"
    if type == "days": return f"{val:.1f} Days"
    if type == "currency": return f"${val/1_000_000:,.1f} M"
    if type == "table_currency": return f"{val/1_000_000:,.1f}"
    if type == "table_percent": return f"{val:.1f}%"
    return f"{val:,.0f}"

def create_dcf_html_table(dcf_table):
    """
    Creates the precise HTML for the DCF table from the reference image.
    """
    df = dcf_table.copy()
    
    # Define row groups
    rows = {
        "Revenue": False,
        "EBITDA": False,
        "EBIT": True,
        "NOPAT": False,
        "(+) Depreciation & Amortization": False,
        "(-) CapEx": False,
        "(-) Change in NWC": False,
        "Free Cash Flow to Firm (FCFF)": True,
        "Discount Factor": False,
        "PV of FCFF": False,
        "Terminal Value": False,
        "PV of Terminal Value": False
    }

    html = "<table class='dcf-table'>"
    
    # Header
    html += "<tr><th>(USD in Millions)</th>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr>"

    # Rows
    for idx, is_highlighted in rows.items():
        if idx not in df.index: continue
        
        row_class = "highlight-row" if is_highlighted else ""
        html += f"<tr class='{row_class}'><td>{idx}</td>"
        
        for col in df.columns:
            val = df.loc[idx, col]
            if pd.isna(val):
                html += "<td>-</td>"
                continue
            
            # Format based on row
            if idx == "Discount Factor":
                html += f"<td>{val:.3f}</td>"
            else:
                html += f"<td>{val/1_000_000:,.1f}</td>"
        
        html += "</tr>"

    html += "</table>"
    return html

# ---
# MAIN APPLICATION LOGIC
# ---

# --- Sidebar ---
st.sidebar.title("Risk-Adjusted Analyst")
ticker_input = st.sidebar.text_input("Enter Company Ticker", "AAPL")

if st.sidebar.button("Run Advanced Analysis"):
    # This is the "computing complexity" part.
    with st.spinner(f"Running Advanced Analysis for {ticker_input}... (This may take 10-20 seconds)"):
        # 1. Pull all data
        data, error = get_ticker_data(ticker_input)
        
        if error:
            st.error(error)
            st.stop()
        
        # 2. Store base data in session state
        st.session_state.data = data
        st.session_state.ticker = ticker_input
        
        # 3. Run all "heavy" calculations and store them
        st.session_state.ratios = calculate_ratios(data)
        
        # Simulate extra "computing time" as requested
        time.sleep(1) 
        st.session_state.monte_carlo = run_monte_carlo(data, simulations=10000)
        
        time.sleep(1)
        st.session_state.candlestick_fig = plot_candlestick_patterns(data)
        
        # NEW: Automated DCF
        time.sleep(0.5)
        dcf_assumptions = calculate_dcf_assumptions(data)
        st.session_state.dcf_results, st.session_state.dcf_error = calculate_dcf_model(data, dcf_assumptions)
        
        st.session_state.analysis_run = True
        st.rerun() # Rerun to update the UI with the new state

# --- Main Page Display ---
if "analysis_run" in st.session_state:
    
    data = st.session_state.data
    info = data["info"]
    ticker = st.session_state.ticker
    
    # --- Sidebar Navigation ---
    st.sidebar.markdown("---")
    nav_selection = st.sidebar.radio(
        "Analysis Sections",
        ["Valuation Summary", "DCF Valuation", "Risk Simulation", "Chart & Patterns", "Financials", "Latest News"],
        label_visibility="collapsed"
    )
    
    # ---
    # MAIN CONTENT AREA
    # ---
    
    # Header
    st.title(f"{info.get('shortName', ticker)} ({ticker})")
    st.markdown(f"**{info.get('industry', '')}** | {info.get('sector', '')} | [Website]({info.get('website', '#')})")
    st.markdown("---")

    # --- Valuation Summary Page (NEW) ---
    if nav_selection == "Valuation Summary":
        st.subheader("Valuation Summary")
        
        dcf_results = st.session_state.dcf_results
        dcf_error = st.session_state.dcf_error

        if dcf_error:
            st.error(f"Could not generate valuation: {dcf_error}")
        elif dcf_results:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # --- Valuation Box (from image_16eadb.png) ---
                upside_pct = dcf_results['upside']
                upside_text = f"{upside_pct:.1%}"
                
                if upside_pct > 0:
                    bar_color = "#00A89C" # Teal
                    text_color = "#00A89C"
                else:
                    bar_color = "#D44E52" # Red
                    text_color = "#D44E52"
                    
                st.markdown(f"""
                <div class="summary-box">
                    <h4>Valuation Summary</h4>
                    <div class="price-row">
                        <span class="price-label">Current Price</span>
                        <span class="price-value">${dcf_results['current_price']:,.2f}</span>
                    </div>
                    <div class="price-row">
                        <span class="price-label">Fair Price (DCF)</span>
                        <span class="price-value">${dcf_results['fair_value']:,.2f}</span>
                    </div>
                    <div class="upside-bar">
                        <div class="upside-fill" style="background-color: {bar_color}; width: 100%;"></div>
                    </div>
                    <h3 class="upside-text" style="color: {text_color};">{upside_text}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("") # Spacer
                
                # --- Key Assumptions Box (from image_16eadb.png) ---
                assumptions = dcf_results['assumptions']
                st.markdown(f"""
                <div class="summary-box">
                    <h4>Key Assumptions</h4>
                    <div class="price-row">
                        <span class="price-label">WACC / Discount Rate</span>
                        <span class="price-value">{assumptions['wacc']:.2%}</span>
                    </div>
                    <div class="price-row">
                        <span class="price-label">Revenue Growth (5Y CAGR)</span>
                        <span class="price-value">{assumptions['g']:.2%}</span>
                    </div>
                    <div class="price-row">
                        <span class="price-label">Long-Term Growth Rate</span>
                        <span class="price-value">{assumptions['lt_g']:.2%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # --- Key Ratios ---
                st.subheader("Key Ratios & Health Score")
                ratios = st.session_state.ratios
                
                rcol1, rcol2, rcol3 = st.columns(3)
                
                with rcol1:
                    st.metric("Gross Margin", format_value(ratios.get("Gross Margin"), "percent"))
                    st.metric("Net Debt", format_value(ratios.get("Net Debt"), "currency"))
                with rcol2:
                    st.metric("EBIT Margin", format_value(ratios.get("EBIT Margin"), "percent"))
                    st.metric("ROE", format_value(ratios.get("ROE"), "percent"))
                with rcol3:
                    st.metric("ROIC", format_value(ratios.get("ROIC"), "percent"))
                    z_score = ratios.get("Altman Z-Score")
                    st.metric("Altman Z-Score", format_value(z_score, "decimal"))

                if not pd.isna(z_score):
                    if z_score < 1.81:
                        st.error("Bankruptcy Risk: Distress Zone")
                    elif z_score < 2.99:
                        st.warning("Bankruptcy Risk: Grey Zone")
                    else:
                        st.success("Bankruptcy Risk: Safe Zone")
                        
            st.subheader("Business Summary")
            st.write(info.get("longBusinessSummary", "No summary available."))

    # --- DCF Valuation Page ---
    elif nav_selection == "DCF Valuation":
        st.subheader("Discounted Cash Flow (DCF) Valuation")
        
        dcf_results = st.session_state.dcf_results
        dcf_error = st.session_state.dcf_error
        
        if dcf_error:
            st.error(f"DCF Error: {dcf_error}")
        elif dcf_results:
            st.markdown(f"#### 5-Year DCF Projection (in $ Millions)")
            dcf_table = dcf_results['dcf_table']
            html_table = create_dcf_html_table(dcf_table)
            st.markdown(html_table, unsafe_allow_html=True)
            
            # Show summary numbers at the bottom
            st.markdown("---")
            st.subheader("Valuation Buildup")
            
            pv_fcff_sum = dcf_table.loc["PV of FCFF"].sum()
            pv_tv = dcf_table.loc["PV of Terminal Value", "Terminal"]
            enterprise_value = pv_fcff_sum + pv_tv
            net_debt = calculate_ratios(data)["Net Debt"]
            equity_value = enterprise_value - net_debt
            shares = info.get("sharesOutstanding", 0)
            fair_value = dcf_results["fair_value"]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("PV of 5-Year FCFF", f"${pv_fcff_sum/1_000_000:,.1f} M")
            col2.metric("PV of Terminal Value", f"${pv_tv/1_000_000:,.1f} M")
            col3.metric("Enterprise Value (EV)", f"${enterprise_value/1_000_000:,.1f} M")
            
            col1.metric("(-) Net Debt", f"${net_debt/1_000_000:,.1f} M")
            col2.metric("Equity Value", f"${equity_value/1_000_000:,.1f} M")
            col3.metric("Fair Value per Share", f"${fair_value:,.2f}")


    # --- Risk Simulation Page ---
    elif nav_selection == "Risk Simulation":
        st.subheader("Risk Simulation (Monte Carlo) - 10,000 Paths")
        
        mc = st.session_state.monte_carlo
        
        if mc:
            st.markdown("#### Probability Analysis (1-Year Horizon)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Prob. of Gain", f"{mc['prob_gain']:.1%}", help="Probability the stock will be above its current price in 1 year.")
            col2.metric("Prob. of +10%", f"{mc['prob_plus_10']:.1%}", help="Probability the stock will rise over 10% in 1 year.")
            col3.metric("Prob. of -10%", f"{mc['prob_minus_10']:.1%}", help="Probability the stock will fall over 10% in 1 year.")
            
            st.markdown("#### Risk Metrics (1-Year Horizon)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("VaR (95%)", f"{mc['var_95']:.2%}", help="Value-at-Risk: 95% confidence that the portfolio will not lose more than this percentage in 1 year.")
            col2.metric("CVaR (95%)", f"{mc['cvar_95']:.2%}", help="Conditional VaR: The expected loss if the 5% worst-case scenario occurs.")
            col3.metric("Sharpe Ratio", f"{mc['sharpe_ratio']:.2f}", help="Risk-adjusted return (vs. 4% risk-free rate).")
            col4.metric("Sortino Ratio", f"{mc['sortino_ratio']:.2f}", help="Risk-adjusted return, only penalizing downside volatility.")

            st.markdown("#### Monte Carlo Price Path Simulation")
            paths_to_plot = mc['price_paths'][:, :100] # Plot only 100 paths
            fig = go.Figure()
            for i in range(paths_to_plot.shape[1]):
                fig.add_trace(go.Scatter(y=paths_to_plot[:, i], mode='lines', line=dict(width=0.5, color='rgba(0, 168, 156, 0.3)')))
            fig.update_layout(title="100 Sample Price Paths (from 10,000 simulations)", template="plotly_dark", showlegend=False, xaxis_title="Days", yaxis_title="Stock Price")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Distribution of Ending Prices (1-Year)")
            hist_fig = ff.create_distplot([mc['ending_prices']], ['Ending Prices'], show_hist=True, show_rug=False)
            hist_fig.update_layout(title="Probability Distribution", template="plotly_dark", showlegend=False)
            st.plotly_chart(hist_fig, use_container_width=True)

    # --- Chart & Patterns Page ---
    elif nav_selection == "Chart & Patterns":
        if not TALIB_LOADED:
            st.error("TA-Lib library not found. Candlestick pattern analysis is disabled. Please see code comments for installation instructions.")
        else:
            st.subheader("Advanced Charting & Pattern Recognition")
            st.plotly_chart(st.session_state.candlestick_fig, use_container_width=True)

    # --- Financials Page ---
    elif nav_selection == "Financials":
        st.subheader("Financial Statements")
        
        tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
        
        with tab1:
            st.dataframe(data["financials"])
        with tab2:
            st.dataframe(data["balance_sheet"])
        with tab3:
            st.dataframe(data["cashflow"])

    # --- News Page ---
    elif nav_selection == "Latest News":
        st.subheader("Latest Financial News")
        
        news_list = data.get("news", [])
        if not news_list:
            st.write("No news found for this ticker.")
        
        for article in news_list:
            # --- FIX for TypeError ---
            publish_time = article.get("providerPublishTime")
            time_str = datetime.fromtimestamp(publish_time).strftime("%Y-%m-%d %H:%M") if publish_time else "Timestamp N/A"
            # --- End Fix ---
                
            st.markdown(f"""
            <div style="border: 1px solid #2D2D2D; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; background-color: #121417;">
                <h5 style="margin-bottom: 0.5rem;">{article['title']}</h5>
                <small style="color: #A0A0A0;">{article['publisher']} | {time_str}</small>
                <p style="margin-top: 0.5rem;"><a href="{article['link']}" target="_blank" style="color: #00A89C; text-decoration: none;">Read more</a></p>
            </div>
            """, unsafe_allow_html=True)
            
else:
    # Initial landing page
    st.image("https://placehold.co/1200x300/0b0c0e/2D2D2D?text=Risk-Adjusted+Analyst", use_container_width=True)
    st.info("Please enter a stock ticker and click 'Run Advanced Analysis' to begin.")