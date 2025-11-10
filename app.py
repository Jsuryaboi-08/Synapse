import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import datetime
import time
import json
import requests  # We'll use requests for the FMP API

# ---
# CRITICAL: TA-Lib Installation
# This library is required for pattern recognition. It can be difficult to install.
# 1. On Windows: Try 'pip install ta-lib-binary'
# 2. On Mac/Linux: You may need to install the C-library first:
#    - Mac: 'brew install ta-lib'
#    - Linux: 'sudo apt-get install libta-lib-dev'
#    - THEN: 'pip install TA-Lib'
try:
    import talib
    TALIB_LOADED = True
except ImportError:
    TALIB_LOADED = False
    pass

# ---
# FMP API FUNCTIONS
# We are no longer using yfinance.
# ---

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

def get_fmp_data(ticker, api_key):
    """
    Pulls all necessary data from FMP API.
    This is our new "heavy" data pull.
    """
    endpoints = {
        "profile": f"/profile/{ticker}",
        "peers": f"/stock-peer?symbol={ticker}",
        "key_metrics": f"/key-metrics-ttm/{ticker}",
        "income_statement": f"/income-statement/{ticker}?period=annual&limit=5",
        "balance_sheet": f"/balance-sheet-statement/{ticker}?period=annual&limit=5",
        "cash_flow": f"/cash-flow-statement/{ticker}?period=annual&limit=5",
        "history": f"/historical-price-full/{ticker}?timeseries=3650", # 10 years of data
        "news": f"/stock_news?tickers={ticker}&limit=10"
    }
    
    data = {}
    try:
        for key, endpoint in endpoints.items():
            url = f"{FMP_BASE_URL}{endpoint}&apikey={api_key}"
            response = requests.get(url)
            response.raise_for_status() # Raise an error for bad responses
            data[key] = response.json()
            time.sleep(0.2) # To respect free-tier rate limits

        # Validate critical data
        if not data["profile"]:
            return None, f"Ticker not found: {ticker}"
        if not data["history"] or "historical" not in data["history"]:
            return None, f"No historical data for {ticker}"

        # --- Data Cleaning & Preparation ---
        # Convert statements to DataFrames
        data["income_statement"] = pd.DataFrame(data["income_statement"]).set_index("date")
        data["balance_sheet"] = pd.DataFrame(data["balance_sheet"]).set_index("date")
        data["cash_flow"] = pd.DataFrame(data["cash_flow"]).set_index("date")
        
        # Convert history to DataFrame
        hist_df = pd.DataFrame(data["history"]["historical"]).set_index("date")
        hist_df.index = pd.to_datetime(hist_df.index)
        hist_df = hist_df.sort_index() # Ensure chronological order
        data["history"] = hist_df

        return data, None
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return None, "Invalid FMP API Key. Please get a free key from financialmodelingprep.com"
        if e.response.status_code == 404:
            return None, f"Ticker not found: {ticker}"
        return None, f"FMP API Error: {e}"
    except Exception as e:
        return None, f"An error occurred pulling data: {e}"

@st.cache_data
def get_peers_data(peers_list, api_key):
    """
    NEW: Pulls key data for all peers.
    This is the "computationally complex" part for Companions Analysis.
    """
    peers_data = []
    if not peers_list:
        return pd.DataFrame(), {}
        
    # Limit to first 10 peers to avoid rate-limiting
    for peer in peers_list[:10]:
        try:
            # Pull Profile (for Market Cap) and TTM Key Metrics (for ratios)
            profile_url = f"{FMP_BASE_URL}/profile/{peer}?apikey={api_key}"
            metrics_url = f"{FMP_BASE_URL}/key-metrics-ttm/{peer}?apikey={api_key}"
            
            profile_res = requests.get(profile_url).json()
            metrics_res = requests.get(metrics_url).json()
            
            if profile_res and metrics_res:
                peers_data.append({
                    "Ticker": peer,
                    "Company": profile_res[0].get("companyName", peer),
                    "Market Cap": profile_res[0].get("mktCap", 0),
                    "P/E": metrics_res[0].get("peRatioTTM", np.nan),
                    "EV/EBITDA": metrics_res[0].get("evToEbitdaTTM", np.nan)
                })
            time.sleep(0.2) # Rate limit
        except Exception:
            continue # Skip peer if it fails
            
    df = pd.DataFrame(peers_data)
    
    # Calculate medians
    medians = {
        "P/E": df["P/E"].median(),
        "EV/EBITDA": df["EV/EBITDA"].median()
    }
    
    return df, medians

# ---
# PAGE CONFIGURATION & STYLING
# ---

st.set_page_config(layout="wide", page_title="Risk-Adjusted Analyst", page_icon="📈")

# Custom CSS to mimic the reference UI (dark, clean, professional)
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    
    .stApp { background-color: #0b0c0e; color: #FAFAFA; }

    [data-testid="stSidebar"] {
        background-color: #121417;
        border-right: 1px solid #2D2D2D;
    }
    .st-sidebar .st-emotion-cache-16txtl3 { font-weight: 700; font-size: 1.75rem; }

    .main .block-container {
        padding-top: 2rem; padding-bottom: 2rem;
        padding-left: 3rem; padding-right: 3rem;
    }

    [data-testid="stMetric"] {
        background-color: #121417; border: 1px solid #2D2D2D;
        border-radius: 12px; padding: 1.25rem;
    }
    [data-testid="stMetric"] > label { color: #A0A0A0; font-weight: 500; }
    [data-testid="stMetric"] > div[data-testid="metric-value"] { font-size: 2rem; font-weight: 600; }

    .stButton > button {
        background-color: #00A89C; color: white; border-radius: 8px;
        font-weight: 600; padding: 0.5rem 1rem; width: 100%;
    }
    .stButton > button:hover { background-color: #007A70; color: white; }
    
    .stRadio [data-baseweb="radio"] {
        background-color: #121417; border-radius: 8px;
        padding: 0.75rem 1rem; margin-bottom: 0.5rem;
    }
    .stRadio [data-baseweb="radio"] label { color: #A0A0A0; font-weight: 500; }
    .stRadio [data-baseweb="radio"] input:checked + div { background-color: #00A89C !important; }
    .stRadio [data-baseweb="radio"] input:checked + div label { color: white; font-weight: 600; }

    /* DCF Table */
    .dcf-table {
        font-family: 'Inter', sans-serif; width: 100%; border-collapse: collapse;
        font-size: 0.9rem;
    }
    .dcf-table th, .dcf-table td {
        border-top: 1px solid #2D2D2D; border-bottom: 1px solid #2D2D2D;
        padding: 12px 14px; text-align: right; white-space: nowrap;
    }
    .dcf-table th {
        background-color: #121417; color: #A0A0A0; font-weight: 600;
        text-align: right;
    }
    .dcf-table th:first-child { text-align: left; }
    .dcf-table td:first-child { text-align: left; font-weight: 600; color: #FAFAFA; }
    .dcf-table tr:nth-child(even) { background-color: #121417; }
    .dcf-table .highlight-row td { font-weight: 700 !important; color: #FFFFFF !important; }

    /* Summary Boxes */
    .summary-box {
        background-color: #121417; border: 1px solid #2D2D2D;
        border-radius: 12px; padding: 1.5rem; height: 100%;
    }
    .summary-box h4 { margin-bottom: 1.5rem; font-weight: 600; }
    .summary-box .price-row {
        display: flex; justify-content: space-between;
        align-items: center; margin-bottom: 0.75rem;
    }
    .summary-box .price-label { font-size: 1rem; color: #A0A0A0; }
    .summary-box .price-value { font-size: 1.1rem; font-weight: 600; color: #FAFAFA; }
    .summary-box .upside-bar {
        width: 100%; height: 10px; background-color: #2D2D2D;
        border-radius: 5px; margin-top: 1.25rem; margin-bottom: 0.5rem;
    }
    .summary-box .upside-fill { height: 10px; border-radius: 5px; }
    .summary-box .upside-text { font-size: 1.25rem; font-weight: 700; text-align: center; }

    /* Companions Table */
    .companions-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    .companions-table th, .companions-table td {
        padding: 10px 12px; text-align: right; border-bottom: 1px solid #2D2D2D;
    }
    .companions-table th { color: #A0A0A0; font-weight: 600; text-align: left;}
    .companions-table td { color: #FAFAFA; }
    .companions-table th:first-child, .companions-table td:first-child { text-align: left; }
    .companions-table .median-row td {
        font-weight: 700; color: #FFFFFF; background-color: #1A1C20;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ---
# CALCULATION FUNCTIONS (The "Advanced Analysis")
# ---

@st.cache_data
def calculate_ratios(_data):
    """Calculates key ratios from FMP data."""
    ratios = {}
    try:
        metrics = _data["key_metrics"][0]
        profile = _data["profile"][0]
        
        ratios["Gross Margin"] = metrics.get("grossProfitMarginTTM", np.nan)
        ratios["EBIT Margin"] = metrics.get("operatingMarginTTM", np.nan)
        ratios["ROE"] = metrics.get("roeTTM", np.nan)
        ratios["ROIC"] = metrics.get("roicTTM", np.nan)
        ratios["Net Debt"] = metrics.get("netDebtToEBITDATTM", np.nan) # FMP gives ratio, let's calc absolute
        
        # Get from Balance Sheet TTM
        bs_ttm = _data["balance_sheet"].iloc[:4].sum()
        inc_ttm = _data["income_statement"].iloc[:4].sum()
        
        net_debt = bs_ttm.get("totalDebt", 0) - bs_ttm.get("cashAndCashEquivalents", 0)
        ratios["Net Debt"] = net_debt
        
        equity = bs_ttm.get("totalStockholdersEquity", 0)
        ratios["Gearing"] = net_debt / equity if equity else np.nan
        
        # WC Cycle
        dso = metrics.get("daysSalesOutstandingTTM", np.nan)
        dio = metrics.get("daysOfInventoryOutstandingTTM", np.nan)
        dpo = metrics.get("daysPayablesOutstandingTTM", np.nan)
        ratios["WC Cycle (Days)"] = dso + dio - dpo
        
        # Altman Z-Score
        bs_latest = _data["balance_sheet"].iloc[0]
        inc_latest = _data["income_statement"].iloc[0] # Use latest full year
        
        wc = bs_latest.get("totalCurrentAssets", 0) - bs_latest.get("totalCurrentLiabilities", 0)
        total_assets = bs_latest.get("totalAssets", 0)
        re = bs_latest.get("retainedEarnings", 0)
        ebit = inc_latest.get("ebitda", 0) - inc_latest.get("depreciationAndAmortization", 0)
        mkt_cap = profile.get("mktCap", 0)
        total_liab = bs_latest.get("totalLiabilities", 0)
        revenue = inc_latest.get("revenue", 0)
        
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
        hist = _data["history"]['close']
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
    NEW: Automatically calculates DCF assumptions from FMP data.
    """
    try:
        inc = _data["income_statement"]
        bs = _data["balance_sheet"]
        profile = _data["profile"][0]
        
        # 1. Revenue Growth (g): 3Y Historical CAGR
        revenues = inc["revenue"].dropna().sort_index()
        if len(revenues) < 2:
            g = 0.05
        else:
            start_rev = revenues.iloc[0]
            end_rev = revenues.iloc[-1]
            num_years = len(revenues) - 1
            g = (end_rev / start_rev) ** (1 / num_years) - 1
            g = min(max(g, 0.01), 0.20) # Constrain between 1% and 20%

        # 2. WACC (Weighted Average Cost of Capital)
        risk_free_rate = 0.04
        market_risk_premium = 0.06
        beta = profile.get("beta", 1.0)
        cost_of_equity = risk_free_rate + beta * market_risk_premium
        
        bs_latest = bs.iloc[0]
        inc_latest = inc.iloc[0]
        
        interest_expense = inc_latest.get("interestExpense", 0)
        total_debt = bs_latest.get("totalDebt", 0)
        cost_of_debt = interest_expense / total_debt if total_debt else 0.05
        
        tax_rate = inc_latest.get("incomeTaxExpense", 0) / inc_latest.get("incomeBeforeTax", 1)
        tax_rate = max(min(tax_rate, 0.5), 0.1) # Constrain tax rate
        
        market_cap = profile.get("mktCap", 0)
        e_percent = market_cap / (market_cap + total_debt) if (market_cap + total_debt) else 0
        d_percent = 1 - e_percent
        
        wacc = (e_percent * cost_of_equity) + (d_percent * cost_of_debt * (1 - tax_rate))
        wacc = max(wacc, 0.05) # Floor at 5%
        
        # 3. Long-Term Growth (lt_g)
        lt_g = 0.025 # Standard assumption

        # 4. Ratios for projection
        inc_ttm = _data["income_statement"].iloc[:4].sum()
        cf_ttm = _data["cash_flow"].iloc[:4].sum()
        bs_ttm = _data["balance_sheet"].iloc[:4].sum()

        ttm_revenue = inc_ttm.get("revenue", 1)
        ttm_nwc = bs_ttm.get("netWorkingCapital", 0)
        
        ebitda_margin = inc_ttm.get("ebitda", 0) / ttm_revenue
        d_a_ratio = inc_ttm.get("depreciationAndAmortization", 0) / ttm_revenue
        capex_ratio = cf_ttm.get("capitalExpenditure", 0) / ttm_revenue # Note: Capex is negative
        nwc_ratio = ttm_nwc / ttm_revenue

        return {
            "g": g, "wacc": wacc, "lt_g": lt_g, "tax_rate": tax_rate,
            "ebitda_margin": ebitda_margin, "d_a_ratio": d_a_ratio,
            "capex_ratio": capex_ratio, "nwc_ratio": nwc_ratio
        }
    except Exception as e:
        return {"g": 0.05, "wacc": 0.08, "lt_g": 0.025, "tax_rate": 0.21,
                "ebitda_margin": 0.15, "d_a_ratio": 0.05, "capex_ratio": -0.06, "nwc_ratio": 0.1}

@st.cache_data
def calculate_dcf_model(_data, assumptions):
    """
    Performs a 5-year DCF calculation based on automated assumptions.
    Rebuilt to match the reference screenshot EXACTLY.
    """
    try:
        profile = _data["profile"][0]
        inc_ttm = _data["income_statement"].iloc[:4].sum()
        bs_ttm = _data["balance_sheet"].iloc[:4].sum()
        
        ttm_revenue = inc_ttm.get("revenue", 0)
        ttm_nwc = bs_ttm.get("netWorkingCapital", 0)
        
        # Get assumptions
        g, wacc, lt_g, tax_rate = assumptions["g"], assumptions["wacc"], assumptions["lt_g"], assumptions["tax_rate"]
        ebitda_margin, d_a_ratio, capex_ratio, nwc_ratio = assumptions["ebitda_margin"], assumptions["d_a_ratio"], assumptions["capex_ratio"], assumptions["nwc_ratio"]
        
        years = [f"FY{datetime.now().year + i + 1}" for i in range(5)]
        columns = years + ["Terminal"]
        
        df = pd.DataFrame(index=[
            "Revenue", "% Growth", "EBITDA", "EBITDA Margin", "D&A",
            "EBIT", "EBIT Margin", "(-) Tax", "NOPAT", "(+) D&A", "(-) CapEx",
            "(-) Change in NWC", "Free Cash Flow to Firm (FCFF)",
            "Discount Factor", "PV of FCFF"
        ], columns=columns, dtype=float)
        
        last_revenue = ttm_revenue
        last_nwc = ttm_nwc
        
        for i, year in enumerate(years):
            revenue = last_revenue * (1 + g)
            df.loc["Revenue", year] = revenue
            df.loc["% Growth", year] = g
            
            ebitda = revenue * ebitda_margin
            df.loc["EBITDA", year] = ebitda
            df.loc["EBITDA Margin", year] = ebitda_margin
            
            d_a = revenue * d_a_ratio
            df.loc["D&A", year] = d_a
            df.loc["(+) D&A", year] = d_a
            
            ebit = ebitda - d_a
            df.loc["EBIT", year] = ebit
            df.loc["EBIT Margin", year] = ebit / revenue if revenue else 0
            
            tax = ebit * tax_rate
            df.loc["(-) Tax", year] = tax
            
            nopat = ebit * (1 - tax_rate)
            df.loc["NOPAT", year] = nopat
            
            capex = revenue * capex_ratio # Already negative
            df.loc["(-) CapEx", year] = capex
            
            nwc = revenue * nwc_ratio
            change_in_nwc = nwc - last_nwc
            df.loc["(-) Change in NWC", year] = change_in_nwc
            
            fcff = nopat + d_a + capex - change_in_nwc
            df.loc["Free Cash Flow to Firm (FCFF)", year] = fcff
            
            discount_factor = (1 / (1 + wacc)) ** (i + 1)
            df.loc["Discount Factor", year] = discount_factor
            df.loc["PV of FCFF", year] = fcff * discount_factor
            
            last_revenue = revenue
            last_nwc = nwc

        # Terminal Value
        last_fcff = df.loc["Free Cash Flow to Firm (FCFF)", years[-1]]
        terminal_fcff = last_fcff * (1 + lt_g)
        terminal_value = terminal_fcff / (wacc - lt_g)
        
        pv_terminal_value = terminal_value * df.loc["Discount Factor", years[-1]]
        
        df.loc["Free Cash Flow to Firm (FCFF)", "Terminal"] = terminal_fcff
        df.loc["Terminal Value"] = 0.0
        df.loc["Terminal Value", "Terminal"] = terminal_value
        df.loc["PV of Terminal Value"] = 0.0
        df.loc["PV of Terminal Value", "Terminal"] = pv_terminal_value
        
        # Enterprise & Equity Value
        enterprise_value = df.loc["PV of FCFF"].sum() + pv_terminal_value
        
        net_debt = bs_ttm.get("totalDebt", 0) - bs_ttm.get("cashAndCashEquivalents", 0)
        equity_value = enterprise_value - net_debt
        shares_outstanding = profile.get("sharesOutstanding", 0)
        
        if shares_outstanding == 0:
            return None, "Missing shares outstanding data to calculate fair value."
            
        fair_value = equity_value / shares_outstanding
        current_price = profile.get("price", 0)
        if not current_price:
            current_price = _data["history"]["close"].iloc[-1]
            
        upside = (fair_value / current_price) - 1 if current_price else np.nan

        return {
            "fair_value": fair_value, "current_price": current_price,
            "upside": upside, "dcf_table": df, "assumptions": assumptions
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
        df = _data["history"].tail(365 * 5) # 5 years
        
        # 1. Identify patterns
        df['CDLDOJI'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['CDLHAMMER'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['CDLENGULFING'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        
        doji_days = df[df['CDLDOJI'] != 0]
        hammer_days = df[df['CDLHAMMER'] != 0]
        engulfing_days = df[df['CDLENGULFING'] != 0]
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name='Candlestick'
        ))

        # Add markers
        fig.add_trace(go.Scatter(
            x=doji_days.index, y=doji_days['low'] * 0.98, mode='markers',
            marker=dict(symbol='circle', color='cyan', size=8), name='Doji'
        ))
        fig.add_trace(go.Scatter(
            x=hammer_days.index, y=hammer_days['low'] * 0.98, mode='markers',
            marker=dict(symbol='triangle-up', color='yellow', size=8), name='Hammer'
        ))
        fig.add_trace(go.Scatter(
            x=engulfing_days.index, y=engulfing_days['low'] * 0.98, mode='markers',
            marker=dict(symbol='square', color='magenta', size=8), name='Engulfing'
        ))

        # Style the chart
        fig.update_layout(
            title="Candlestick Chart with TA-Lib Pattern Recognition (5-Year)",
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
    if pd.isna(val) or val is None: return "N/A"
    if type == "percent": return f"{val:.2%}"
    if type == "decimal": return f"{val:.2f}"
    if type == "days": return f"{val:.1f} Days"
    if type == "currency": return f"${val/1_000_000:,.1f} M"
    return f"{val:,.0f}"

def create_dcf_html_table(dcf_table):
    """
    Creates the precise HTML for the DCF table from the reference image.
    """
    df = dcf_table.copy()
    
    # Define row groups
    rows = {
        "Revenue": False, "% Growth": False, "EBITDA": False, "EBITDA Margin": False,
        "D&A": False, "EBIT": True, "EBIT Margin": False, "(-) Tax": False,
        "NOPAT": False, "(+) D&A": False, "(-) CapEx": False, "(-) Change in NWC": False,
        "Free Cash Flow to Firm (FCFF)": True, "Discount Factor": False,
        "PV of FCFF": False, "Terminal Value": False, "PV of Terminal Value": False
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
            if "Margin" in idx or "% Growth" in idx:
                html += f"<td>{val:.2%}</td>"
            elif "Factor" in idx:
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
fmp_api_key = st.sidebar.text_input("Enter FMP API Key", type="password", help="Get a free API key from financialmodelingprep.com")
ticker_input = st.sidebar.text_input("Enter Company Ticker", "AAPL")

if st.sidebar.button("Run Advanced Analysis"):
    if not fmp_api_key:
        st.sidebar.error("Please enter an FMP API Key.")
        st.stop()

    with st.spinner(f"Running Advanced Analysis for {ticker_input}... (This may take 20-30 seconds)"):
        # 1. Pull all data
        data, error = get_fmp_data(ticker_input, fmp_api_key)
        
        if error:
            st.error(error)
            st.stop()
        
        # 2. Store base data in session state
        st.session_state.data = data
        st.session_state.ticker = ticker_input
        
        # 3. Run all "heavy" calculations and store them
        st.session_state.ratios = calculate_ratios(data)
        
        time.sleep(1) 
        st.session_state.monte_carlo = run_monte_carlo(data)
        
        time.sleep(1)
        st.session_state.candlestick_fig = plot_candlestick_patterns(data)
        
        # 4. NEW: Automated DCF
        time.sleep(0.5)
        dcf_assumptions = calculate_dcf_assumptions(data)
        st.session_state.dcf_results, st.session_state.dcf_error = calculate_dcf_model(data, dcf_assumptions)

        # 5. NEW: Companions Analysis
        time.sleep(1)
        peers_list = data.get("peers", [{}])[0].get("peersList", [])
        st.session_state.peers_df, st.session_state.peers_medians = get_peers_data(peers_list, fmp_api_key)
        
        st.session_state.analysis_run = True
        st.rerun() # Rerun to update the UI with the new state

# --- Main Page Display ---
if "analysis_run" in st.session_state:
    
    data = st.session_state.data
    profile = data["profile"][0]
    ticker = st.session_state.ticker
    
    # --- Sidebar Navigation ---
    st.sidebar.markdown("---")
    nav_selection = st.sidebar.radio(
        "Analysis Sections",
        ["Valuation Summary", "Companions Analysis", "DCF Valuation", "Risk Simulation", "Chart & Patterns", "Financials", "Latest News"],
        label_visibility="collapsed"
    )
    
    # ---
    # MAIN CONTENT AREA
    # ---
    
    # Header
    col_img, col_title = st.columns([1, 10])
    with col_img:
        st.image(profile.get("image", ""), width=60)
    with col_title:
        st.title(f"{profile.get('companyName', ticker)} ({ticker})")
        st.markdown(f"**{profile.get('industry', '')}** | {profile.get('sector', '')} | [Website]({profile.get('website', '#')})")
    
    st.markdown("---")

    # --- Valuation Summary Page (NEW) ---
    if nav_selection == "Valuation Summary":
        st.subheader("Valuation Summary")
        
        dcf_results = st.session_state.dcf_results
        dcf_error = st.session_state.dcf_error

        col1, col2 = st.columns([1, 2])
        
        with col1:
            if dcf_error:
                st.error(f"Could not generate DCF valuation: {dcf_error}")
            elif dcf_results:
                upside_pct = dcf_results['upside']
                upside_text = f"{upside_pct:.1%}"
                bar_color = "#00A89C" if upside_pct > 0 else "#D44E52"
                    
                st.markdown(f"""
                <div class="summary-box">
                    <h4>DCF Valuation</h4>
                    <div class="price-row">
                        <span class="price-label">Current Price</span>
                        <span class="price-value">${dcf_results['current_price']:,.2f}</span>
                    </div>
                    <div class="price-row">
                        <span class="price-label">Fair Price (DCF)</span>
                        <span class="price-value">${dcf_results['fair_value']:,.2f}</span>
                    </div>
                    <div class="upside-bar"><div class="upside-fill" style="background-color: {bar_color}; width: 100%;"></div></div>
                    <h3 class="upside-text" style="color: {bar_color};">{upside_text}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("") # Spacer
                
                assumptions = dcf_results['assumptions']
                st.markdown(f"""
                <div class="summary-box">
                    <h4>DCF Assumptions</h4>
                    <div class="price-row">
                        <span class="price-label">WACC / Discount Rate</span>
                        <span class="price-value">{assumptions['wacc']:.2%}</span>
                    </div>
                    <div class="price-row">
                        <span class="price-label">Revenue Growth (Hist. CAGR)</span>
                        <span class="price-value">{assumptions['g']:.2%}</span>
                    </div>
                    <div class="price-row">
                        <span class="price-label">Long-Term Growth Rate</span>
                        <span class="price-value">{assumptions['lt_g']:.2%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
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
                if z_score < 1.81: st.error("Bankruptcy Risk: Distress Zone")
                elif z_score < 2.99: st.warning("Bankruptcy Risk: Grey Zone")
                else: st.success("Bankruptcy Risk: Safe Zone")
                        
        st.subheader("Business Summary")
        st.write(profile.get("description", "No summary available."))

    # --- Companions Analysis Page (NEW) ---
    elif nav_selection == "Companions Analysis":
        st.subheader("Companions Analysis (Trading Multiples)")
        
        peers_df = st.session_state.peers_df
        peers_medians = st.session_state.peers_medians
        
        if peers_df.empty:
            st.warning("No peer data available for this company.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="summary-box">
                    <h4>EV/EBITDA (Median)</h4>
                    <div class="price-row">
                        <span class="price-label">Industry Median</span>
                        <span class="price-value">{peers_medians['EV/EBITDA']:.2f}x</span>
                    </div>
                    <div class="price-row">
                        <span class="price-label">{ticker} TTM</span>
                        <span class="price-value">{data['key_metrics'][0]['evToEbitdaTTM']:.2f}x</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="summary-box">
                    <h4>P/E (Median)</h4>
                    <div class="price-row">
                        <span class="price-label">Industry Median</span>
                        <span class="price-value">{peers_medians['P/E']:.2f}x</span>
                    </div>
                    <div class="price-row">
                        <span class="price-label">{ticker} TTM</span>
                        <span class="price-value">{data['key_metrics'][0]['peRatioTTM']:.2f}x</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### Peer Group Data")
            
            # Build HTML table for peers
            html = "<table class='companions-table'>"
            html += "<tr><th>Company</th><th>Market Cap</th><th>P/E</th><th>EV/EBITDA</th></tr>"
            
            for _, row in peers_df.iterrows():
                html += f"<tr><td><b>{row['Ticker']}</b><br><small>{row['Company'][:30]}...</small></td>"
                html += f"<td>{format_value(row['Market Cap'], 'currency')}</td>"
                html += f"<td>{row['P/E']:.2f}x</td><td>{row['EV/EBITDA']:.2f}x</td></tr>"
            
            # Median Row
            html += "<tr class='median-row'><td><b>Industry Median</b></td>"
            html += f"<td>-</td><td>{peers_medians['P/E']:.2f}x</td><td>{peers_medians['EV/EBITDA']:.2f}x</td></tr>"
            
            html += "</table>"
            st.markdown(html, unsafe_allow_html=True)


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
            
            st.markdown("---")
            st.subheader("Valuation Buildup")
            
            pv_fcff_sum = dcf_table.loc["PV of FCFF"].sum()
            pv_tv = dcf_table.loc["PV of Terminal Value", "Terminal"]
            enterprise_value = pv_fcff_sum + pv_tv
            net_debt = st.session_state.ratios["Net Debt"]
            equity_value = enterprise_value - net_debt
            fair_value = dcf_results["fair_value"]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("PV of 5-Year FCFF", format_value(pv_fcff_sum, 'currency'))
            col2.metric("PV of Terminal Value", format_value(pv_tv, 'currency'))
            col3.metric("Enterprise Value (EV)", format_value(enterprise_value, 'currency'))
            
            col1.metric("(-) Net Debt", format_value(net_debt, 'currency'))
            col2.metric("Equity Value", format_value(equity_value, 'currency'))
            col3.metric("Fair Value per Share", f"${fair_value:,.2f}")

    # --- Risk Simulation Page ---
    elif nav_selection == "Risk Simulation":
        st.subheader("Risk Simulation (Monte Carlo) - 10,000 Paths")
        mc = st.session_state.monte_carlo
        
        if mc:
            st.markdown("#### Probability Analysis (1-Year Horizon)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Prob. of Gain", f"{mc['prob_gain']:.1%}")
            col2.metric("Prob. of +10%", f"{mc['prob_plus_10']:.1%}")
            col3.metric("Prob. of -10%", f"{mc['prob_minus_10']:.1%}")
            
            st.markdown("#### Risk Metrics (1-Year Horizon)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("VaR (95%)", f"{mc['var_95']:.2%}")
            col2.metric("CVaR (95%)", f"{mc['cvar_95']:.2%}")
            col3.metric("Sharpe Ratio", f"{mc['sharpe_ratio']:.2f}")
            col4.metric("Sortino Ratio", f"{mc['sortino_ratio']:.2f}")

            st.markdown("#### Monte Carlo Price Path Simulation")
            paths_to_plot = mc['price_paths'][:, :100]
            fig = go.Figure()
            for i in range(paths_to_plot.shape[1]):
                fig.add_trace(go.Scatter(y=paths_to_plot[:, i], mode='lines', line=dict(width=0.5, color='rgba(0, 168, 156, 0.3)')))
            fig.update_layout(title="100 Sample Price Paths (from 10,000 simulations)", template="plotly_dark", showlegend=False)
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
        st.subheader("Financial Statements (Annual)")
        tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
        with tab1:
            st.dataframe(data["income_statement"])
        with tab2:
            st.dataframe(data["balance_sheet"])
        with tab3:
            st.dataframe(data["cash_flow"])

    # --- News Page ---
    elif nav_selection == "Latest News":
        st.subheader("Latest Financial News")
        news_list = data.get("news", [])
        if not news_list:
            st.write("No news found for this ticker.")
        
        for article in news_list:
            time_str = article.get("publishedDate", "Timestamp N/A")
            
            st.markdown(f"""
            <div style="border: 1px solid #2D2D2D; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; background-color: #121417;">
                <h5 style="margin-bottom: 0.5rem;">{article['title']}</h5>
                <small style="color: #A0A0A0;">{article['site']} | {time_str}</small>
                <p style="margin-top: 0.5rem;"><a href="{article['url']}" target="_blank" style="color: #00A89C; text-decoration: none;">Read more</a></p>
            </div>
            """, unsafe_allow_html=True)
            
else:
    # Initial landing page
    st.image("https://placehold.co/1200x300/0b0c0e/2D2D2D?text=Risk-Adjusted+Analyst", use_container_width=True)
    st.info("Please enter a stock ticker and an FMP API Key to begin.")