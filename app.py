import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats

# Page Configuration
st.set_page_config(
    page_title="Risk-Adjusted Analyst Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Dark Theme Styling ---
# This CSS replaces the previous gradient/shadow-heavy theme
# with a cleaner, solid-color professional design.
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    body {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0F172A; /* Main dark background */
        color: #E2E8F0;
    }
    
    /* Main header */
    .premium-header {
        background-color: #1E293B; /* Slightly lighter card background */
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 700;
    }
    
    /* Custom metric cards */
    .metric-card {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #334155;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #0EA5E9; /* Accent color on hover */
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 10px 20px;
        border: 1px solid #334155;
        color: #94A3B8; /* Muted tab text */
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0EA5E9; /* Accent color */
        color: #FFFFFF;
        border: 1px solid #0EA5E9;
    }
    
    /* Risk badges */
    .risk-badge {
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-top: 10px;
    }
    .risk-badge-safe { background-color: #10B981; }
    .risk-badge-warning { background-color: #F59E0B; }
    .risk-badge-danger { background-color: #EF4444; }
    
    /* Standard Streamlit Metric */
    .stMetric {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
    }
    
    .stMetric label {
        color: #94A3B8 !important; /* Muted label */
        font-size: 14px !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }
    
    /* Expander */
    div[data-testid="stExpander"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
    }
    
    /* Insight box */
    .insight-box {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #0EA5E9; /* Accent color */
        margin: 10px 0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #334155;
    }
    
    .stButton > button[kind="primary"] {
        background-color: #0EA5E9;
        color: #FFFFFF;
        border: 1px solid #0EA5E9;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #0B8DC1;
        border: 1px solid #0B8DC1;
    }
    </style>
""", unsafe_allow_html=True)

# --- Utility Functions ---

@st.cache_data(ttl=3600)
def fetch_company_data(ticker):
    """Fetch comprehensive company data from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        history = stock.history(period="5y")
        quarterly_financials = stock.quarterly_financials
        
        # Check for essential data
        if financials.empty or balance_sheet.empty or cashflow.empty:
            st.error(f"Incomplete financial data for {ticker}. DCF and ratio analysis may fail.")
            
        return {
            'info': info,
            'financials': financials.to_dict() if not financials.empty else {},
            'balance_sheet': balance_sheet.to_dict() if not balance_sheet.empty else {},
            'cashflow': cashflow.to_dict() if not cashflow.empty else {},
            'history': history.to_dict() if not history.empty else {},
            'quarterly_financials': quarterly_financials.to_dict() if not quarterly_financials.empty else {},
            'ticker': ticker
        }
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_risk_free_rate():
    """Fetch 10-Year Treasury yield as risk-free rate"""
    try:
        tnx = yf.Ticker("^TNX")
        history = tnx.history(period="1d")
        if not history.empty:
            return history['Close'].iloc[-1] / 100  # Convert percentage to decimal
        else:
            return 0.04  # Fallback
    except Exception:
        return 0.04 # Fallback

def calculate_advanced_ratios(data):
    """Calculate comprehensive set of financial ratios with trend analysis"""
    try:
        info = data['info']
        financials = pd.DataFrame(data['financials'])
        balance_sheet = pd.DataFrame(data['balance_sheet'])
        cashflow = pd.DataFrame(data['cashflow'])
        
        if financials.empty or balance_sheet.empty or cashflow.empty:
            return {} # Not enough data
            
        latest_financials = financials.iloc[:, 0]
        latest_balance = balance_sheet.iloc[:, 0]
        latest_cf = cashflow.iloc[:, 0]
        
        ratios = {}
        
        # Profitability Ratios
        revenue = latest_financials.get('Total Revenue', 0)
        gross_profit = latest_financials.get('Gross Profit', 0)
        ebit = latest_financials.get('EBIT', 0)
        net_income = latest_financials.get('Net Income', 0)
        
        ratios['Gross Margin'] = (gross_profit / revenue * 100) if revenue > 0 else info.get('grossMargins', 0) * 100
        ratios['EBIT Margin'] = (ebit / revenue * 100) if revenue > 0 else info.get('operatingMargins', 0) * 100
        ratios['Net Margin'] = (net_income / revenue * 100) if revenue > 0 else info.get('profitMargins', 0) * 100
        ratios['ROE'] = info.get('returnOnEquity', 0) * 100
        ratios['ROA'] = info.get('returnOnAssets', 0) * 100
        
        # ROIC Calculation
        total_debt = latest_balance.get('Total Debt', info.get('totalDebt', 0))
        equity = latest_balance.get('Stockholders Equity', info.get('totalStockholderEquity', 0))
        invested_capital = total_debt + equity
        tax_rate = latest_financials.get('Income Tax Expense', 0) / latest_financials.get('Income Before Tax', 1)
        nopat = ebit * (1 - tax_rate)
        ratios['ROIC'] = (nopat / invested_capital * 100) if invested_capital > 0 else 0
        
        # Liquidity Ratios
        current_assets = latest_balance.get('Current Assets', 0)
        current_liabilities = latest_balance.get('Current Liabilities', 1)
        inventory = latest_balance.get('Inventory', 0)
        
        ratios['Current Ratio'] = current_assets / current_liabilities if current_liabilities > 0 else 0
        ratios['Quick Ratio'] = (current_assets - inventory) / current_liabilities if current_liabilities > 0 else 0
        
        # Leverage Ratios
        cash = latest_balance.get('Cash And Cash Equivalents', 0)
        ratios['Net Debt'] = (total_debt - cash) / 1e9
        ratios['Debt/Equity'] = (total_debt / equity * 100) if equity > 0 else 0
        ratios['Debt/Assets'] = (total_debt / latest_balance.get('Total Assets', 1) * 100)
        ratios['Interest Coverage'] = (ebit / abs(latest_financials.get('Interest Expense', 1))) if latest_financials.get('Interest Expense') else 0
        
        # Efficiency Ratios
        cogs = latest_financials.get('Cost Of Revenue', 0)
        receivables = latest_balance.get('Accounts Receivable', 0)
        payables = latest_balance.get('Accounts Payable', 0)
        
        ratios['DIO'] = (inventory / cogs * 365) if cogs > 0 else 0
        ratios['DSO'] = (receivables / revenue * 365) if revenue > 0 else 0
        ratios['DPO'] = (payables / cogs * 365) if cogs > 0 else 0
        ratios['Cash Conversion Cycle'] = ratios['DIO'] + ratios['DSO'] - ratios['DPO']
        ratios['Asset Turnover'] = revenue / latest_balance.get('Total Assets', 1)
        
        # Cash Flow Ratios
        operating_cf = latest_cf.get('Operating Cash Flow', 0)
        capex = abs(latest_cf.get('Capital Expenditure', 0))
        fcf = operating_cf - capex # This is FCFE
        
        ratios['FCF'] = fcf / 1e9
        ratios['FCF Margin'] = (fcf / revenue * 100) if revenue > 0 else 0
        ratios['OCF/Net Income'] = (operating_cf / net_income) if net_income != 0 else 0
        
        # Altman Z-Score (Enhanced)
        total_assets = latest_balance.get('Total Assets', 1)
        retained_earnings = latest_balance.get('Retained Earnings', 0)
        market_cap = info.get('marketCap', 0)
        total_liabilities = latest_balance.get('Total Liabilities Net Minority Interest', 1)
        
        if total_assets > 0:
            x1 = (current_assets - current_liabilities) / total_assets
            x2 = retained_earnings / total_assets
            x3 = ebit / total_assets
            x4 = market_cap / total_liabilities if total_liabilities > 0 else 0
            x5 = revenue / total_assets
            
            ratios['Altman Z-Score'] = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
        else:
            ratios['Altman Z-Score'] = 0
        
        # Piotroski F-Score
        ratios['Piotroski F-Score'] = calculate_piotroski_score(data)
        
        # Valuation Ratios
        ratios['P/E'] = info.get('trailingPE', 0)
        ratios['Forward P/E'] = info.get('forwardPE', 0)
        ratios['PEG'] = info.get('pegRatio', 0)
        ratios['P/B'] = info.get('priceToBook', 0)
        ratios['EV/EBITDA'] = info.get('enterpriseToEbitda', 0)
        
        return ratios
    except Exception as e:
        st.error(f"Error calculating ratios: {str(e)}")
        return {}

def calculate_piotroski_score(data):
    """Calculate Piotroski F-Score (0-9 scale)"""
    try:
        financials = pd.DataFrame(data['financials'])
        balance_sheet = pd.DataFrame(data['balance_sheet'])
        cashflow = pd.DataFrame(data['cashflow'])
        
        if financials.empty or balance_sheet.empty or cashflow.empty or len(financials.columns) < 2:
            return 0
        
        score = 0
        
        # --- Profitability (4 points) ---
        # 1. Net Income
        net_income = financials.loc['Net Income'].iloc[0]
        score += 1 if net_income > 0 else 0
        
        # 2. Operating Cash Flow
        if 'Operating Cash Flow' in cashflow.index:
            ocf = cashflow.loc['Operating Cash Flow'].iloc[0]
            score += 1 if ocf > 0 else 0
            
            # 3. OCF vs Net Income
            score += 1 if ocf > net_income else 0
        
        # 4. ROA
        if 'Total Assets' in balance_sheet.index:
            roa_current = net_income / balance_sheet.loc['Total Assets'].iloc[0]
            roa_previous = financials.loc['Net Income'].iloc[1] / balance_sheet.loc['Total Assets'].iloc[1]
            score += 1 if roa_current > roa_previous else 0
        
        # --- Leverage (3 points) ---
        # 5. Leverage
        if 'Total Debt' in balance_sheet.index:
            debt_ratio_current = balance_sheet.loc['Total Debt'].iloc[0] / balance_sheet.loc['Total Assets'].iloc[0]
            debt_ratio_previous = balance_sheet.loc['Total Debt'].iloc[1] / balance_sheet.loc['Total Assets'].iloc[1]
            score += 1 if debt_ratio_current < debt_ratio_previous else 0
        
        # 6. Current Ratio
        if 'Current Assets' in balance_sheet.index and 'Current Liabilities' in balance_sheet.index:
            current_ratio = balance_sheet.loc['Current Assets'].iloc[0] / balance_sheet.loc['Current Liabilities'].iloc[0]
            previous_current_ratio = balance_sheet.loc['Current Assets'].iloc[1] / balance_sheet.loc['Current Liabilities'].iloc[1]
            score += 1 if current_ratio > previous_current_ratio else 0
        
        # 7. Share Issuance (Using Shareholder Equity as proxy)
        if 'Stockholders Equity' in balance_sheet.index:
            equity_current = balance_sheet.loc['Stockholders Equity'].iloc[0]
            equity_previous = balance_sheet.loc['Stockholders Equity'].iloc[1]
            score += 1 if equity_current >= equity_previous else 0 # Simple check
            
        # --- Operating Efficiency (2 points) ---
        # 8. Gross Margin
        if 'Gross Profit' in financials.index and 'Total Revenue' in financials.index:
            gm_current = financials.loc['Gross Profit'].iloc[0] / financials.loc['Total Revenue'].iloc[0]
            gm_previous = financials.loc['Gross Profit'].iloc[1] / financials.loc['Total Revenue'].iloc[1]
            score += 1 if gm_current > gm_previous else 0
        
        # 9. Asset Turnover
        if 'Total Assets' in balance_sheet.index:
            asset_turnover_current = financials.loc['Total Revenue'].iloc[0] / balance_sheet.loc['Total Assets'].iloc[0]
            asset_turnover_previous = financials.loc['Total Revenue'].iloc[1] / balance_sheet.loc['Total Assets'].iloc[1]
            score += 1 if asset_turnover_current > asset_turnover_previous else 0
        
        return score
    except:
        return 0

def monte_carlo_simulation(current_price, volatility, drift, days, simulations=10000):
    """Enhanced Monte Carlo with drift consideration"""
    dt = 1/252
    
    price_paths = np.zeros((simulations, days))
    price_paths[:, 0] = current_price
    
    for t in range(1, days):
        random_shock = np.random.normal(0, 1, simulations)
        price_paths[:, t] = price_paths[:, t-1] * np.exp(
            (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * random_shock
        )
    
    return price_paths

def calculate_var_cvar(returns, confidence=0.95):
    """Calculate Value at Risk and Conditional Value at Risk"""
    var = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar

def calculate_sharpe_sortino(returns, risk_free_rate=0.02):
    """Calculate Sharpe and Sortino ratios"""
    excess_returns = returns - risk_free_rate/252
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
    sortino = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0
    
    return sharpe, sortino

# --- NEW: Advanced DCF Functions ---

def calculate_wacc(data, rfr, market_premium):
    """Calculates WACC from company data"""
    try:
        info = data['info']
        financials = pd.DataFrame(data['financials'])
        balance_sheet = pd.DataFrame(data['balance_sheet'])
        
        if financials.empty or balance_sheet.empty:
            return None, "Missing financial data for WACC"

        # Cost of Equity (Re) using CAPM
        beta = info.get('beta')
        if beta is None:
            return None, "Beta not available."
        Re = rfr + beta * market_premium

        # Cost of Debt (Rd)
        interest_expense = financials.loc['Interest Expense'].iloc[0] if 'Interest Expense' in financials.index else 0
        total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
        
        # Handle cases where interest expense is positive or zero
        Rd = abs(interest_expense) / total_debt if total_debt > 0 and interest_expense != 0 else rfr + 0.02 # Proxy if no debt/interest

        # Tax Rate (T)
        income_before_tax = financials.loc['Income Before Tax'].iloc[0] if 'Income Before Tax' in financials.index else 0
        income_tax = financials.loc['Income Tax Expense'].iloc[0] if 'Income Tax Expense' in financials.index else 0
        tax_rate = (income_tax / income_before_tax) if income_before_tax > 0 else 0.21 # Fallback to 21%

        # Market Values (E, D, V)
        E = info.get('marketCap')
        if E is None:
            return None, "Market Cap not available."
        D = total_debt # Use book value of debt as proxy for market value
        V = E + D

        # WACC
        wacc = (E/V * Re) + (D/V * Rd * (1 - tax_rate))
        
        wacc_details = {
            "Cost of Equity (Re)": Re,
            "Cost of Debt (Rd)": Rd,
            "Beta": beta,
            "Tax Rate": tax_rate,
            "Market Cap (E)": E,
            "Total Debt (D)": D,
            "E/V": E/V,
            "D/V": D/V,
            "WACC": wacc
        }
        
        return wacc, wacc_details
        
    except Exception as e:
        return None, f"WACC Error: {str(e)}"

def calculate_dcf_valuation_advanced(data, wacc, g_5y, g_10y, g_t):
    """
    Advanced 2-Stage Enterprise Value DCF Model
    Calculates FCFF from components: EBIT, Tax, D&A, NWC, CapEx
    """
    try:
        info = data['info']
        financials = pd.DataFrame(data['financials'])
        balance_sheet = pd.DataFrame(data['balance_sheet'])
        cashflow = pd.DataFrame(data['cashflow'])
        
        if financials.empty or balance_sheet.empty or cashflow.empty or len(balance_sheet.columns) < 2:
            st.error("DCF requires at least 2 years of historical data.")
            return None

        # --- 1. Calculate Base Year (T=0) FCFF ---
        
        # NOPAT = EBIT * (1 - Tax Rate)
        ebit = financials.loc['EBIT'].iloc[0]
        tax_expense = financials.loc['Income Tax Expense'].iloc[0]
        income_before_tax = financials.loc['Income Before Tax'].iloc[0]
        tax_rate = (tax_expense / income_before_tax) if income_before_tax > 0 else 0.21 # Use 21% fallback
        nopat = ebit * (1 - tax_rate)

        # Net Investment = (CapEx - D&A) + (Change in NWC)
        capex = abs(cashflow.loc['Capital Expenditure'].iloc[0])
        d_and_a = cashflow.loc['Depreciation And Amortization'].iloc[0]
        
        # Change in Net Working Capital (NWC)
        # NWC = (Current Assets - Cash) - (Current Liabilities - Short-Term Debt)
        def get_nwc(col):
            ca = balance_sheet.loc['Current Assets'].iloc[col]
            cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[col]
            cl = balance_sheet.loc['Current Liabilities'].iloc[col]
            std = balance_sheet.loc['Short Term Debt'].iloc[col] if 'Short Term Debt' in balance_sheet.index else 0
            return (ca - cash) - (cl - std)

        nwc_0 = get_nwc(0) # Most recent year
        nwc_1 = get_nwc(1) # Prior year
        change_nwc = nwc_0 - nwc_1

        # Base FCFF
        fcff_0 = nopat + d_and_a - capex - change_nwc
        
        if fcff_0 <= 0:
            st.warning("Base year FCFF is negative. DCF model may be unreliable.")
            # We can still proceed, but it's a red flag.
        
        # --- 2. Project FCFF for 10 Years (2-Stage) ---
        projected_fcf = []
        last_fcff = fcff_0
        
        # Stage 1: Years 1-5 (High Growth)
        for year in range(1, 6):
            fcf = last_fcff * (1 + g_5y)
            projected_fcf.append(fcf)
            last_fcff = fcf
            
        # Stage 2: Years 6-10 (Transitional Growth)
        for year in range(1, 6):
            fcf = last_fcff * (1 + g_10y)
            projected_fcf.append(fcf)
            last_fcff = fcf

        # --- 3. Calculate Terminal Value (Gordon Growth) ---
        # TV = FCFF_10 * (1 + g_t) / (WACC - g_t)
        if wacc <= g_t:
            st.error("WACC must be greater than Terminal Growth Rate.")
            return None
        
        terminal_value = projected_fcf[-1] * (1 + g_t) / (wacc - g_t)

        # --- 4. Discount Cash Flows to Present Value ---
        pv_fcf = sum([cf / (1 + wacc)**i for i, cf in enumerate(projected_fcf, 1)])
        pv_terminal = terminal_value / (1 + wacc)**10
        
        enterprise_value = pv_fcf + pv_terminal

        # --- 5. Calculate Equity Value and Intrinsic Price ---
        total_debt = balance_sheet.loc['Total Debt'].iloc[0]
        cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
        minority_interest = balance_sheet.loc['Minority Interest'].iloc[0] if 'Minority Interest' in balance_sheet.index else 0
        
        equity_value = enterprise_value - total_debt + cash - minority_interest
        
        shares_outstanding = info.get('sharesOutstanding', 1)
        intrinsic_value_per_share = equity_value / shares_outstanding
        
        return {
            'enterprise_value': enterprise_value,
            'equity_value': equity_value,
            'intrinsic_price': intrinsic_value_per_share,
            'base_fcff': fcff_0,
            'projected_fcf': projected_fcf,
            'terminal_value': terminal_value,
            'pv_fcf': pv_fcf,
            'pv_terminal': pv_terminal
        }
    except Exception as e:
        st.error(f"Error in DCF calculation: {str(e)}")
        return None

# --- Main Application ---
def main():
    # Header
    st.markdown("""
        <div class'premium-header'>
            <h1 style='margin:0; background: linear-gradient(135deg, #0EA5E9 0%, #0B8DC1 100%); 
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                         background-clip: text; font-size: 48px;'>
                Risk-Adjusted Analyst Pro
            </h1>
            <p style='color: #94A3B8; margin-top: 10px; font-size: 18px;'>
                Advanced Forward-Looking Valuation & Risk Analytics Platform
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Company Selection")
        ticker = st.text_input("Enter Ticker Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL").upper()
        
        st.markdown("---")
        st.markdown("### Analysis Settings")
        
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Quick", "Standard", "Deep"],
            value="Standard"
        )
        
        include_peers = st.checkbox("Compare with Peers", value=False)
        
        if st.button("Run Analysis", type="primary", use_container_width=True):
            with st.spinner(f"Fetching data for {ticker}..."):
                st.session_state['ticker'] = ticker
                st.session_state['data'] = fetch_company_data(ticker)
                st.session_state['analysis_depth'] = analysis_depth
                # Clear old simulation results
                if 'mc_results' in st.session_state:
                    del st.session_state['mc_results']
        
        st.markdown("---")
        st.markdown("### Quick Access")
        st.markdown("**Popular Tickers:**")
        
        popular = {
            "Tech": ["AAPL", "MSFT", "GOOGL", "NVDA"],
            "Finance": ["JPM", "BAC", "GS", "V"],
            "Consumer": ["AMZN", "WMT", "NKE", "COST"]
        }
        
        for sector, tickers in popular.items():
            st.markdown(f"**{sector}**")
            cols = st.columns(4)
            for i, tick in enumerate(tickers):
                if cols[i].button(tick, key=f"quick_{tick}"):
                    st.session_state['ticker'] = tick
                    st.session_state['data'] = fetch_company_data(tick)
                    st.rerun()
    
    # Main Content
    if 'data' in st.session_state and st.session_state['data']:
        data = st.session_state['data']
        info = data['info']
        ticker = st.session_state['ticker']
        history = pd.DataFrame(data['history'])
        
        # Company Overview Header
        col1, col2, col3, col4, col5 = st.columns(5)
        
        current_price = info.get('currentPrice', 0)
        prev_close = info.get('previousClose', current_price)
        price_change = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
        
        with col1:
            st.metric(
                "Current Price", 
                f"${current_price:.2f}",
                f"{price_change:+.2f}%"
            )
        with col2:
            st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B")
        with col3:
            st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}")
        with col4:
            st.metric("52W Range", f"${info.get('fiftyTwoWeekLow', 0):.0f} - ${info.get('fiftyTwoWeekHigh', 0):.0f}")
        with col5:
            st.metric("Avg Volume", f"{info.get('averageVolume', 0)/1e6:.1f}M")
        
        st.markdown(f"""
            <div style='background-color: #1E293B; border: 1px solid #334155; 
                         padding: 15px; border-radius: 10px; margin: 20px 0;'>
                <strong style='color: #0EA5E9; font-size: 18px;'>{info.get('shortName', ticker)}</strong> 
                <span style='color: #94A3B8;'>• {info.get('sector', 'N/A')} • {info.get('industry', 'N/A')}</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Financial Health", 
            "Risk Analytics", 
            "Valuation Models",
            "Technical Analysis"
        ])
        
        # Tab 1: Enhanced Health Scorecard
        with tab1:
            ratios = calculate_advanced_ratios(data)
            
            if not ratios:
                st.warning("Not enough financial data to display ratios.")
            else:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### Profitability & Returns")
                    
                    subcol1, subcol2, subcol3, subcol4 = st.columns(4)
                    with subcol1:
                        st.metric("Gross Margin", f"{ratios.get('Gross Margin', 0):.2f}%")
                        st.metric("ROIC", f"{ratios.get('ROIC', 0):.2f}%")
                    with subcol2:
                        st.metric("EBIT Margin", f"{ratios.get('EBIT Margin', 0):.2f}%")
                        st.metric("ROE", f"{ratios.get('ROE', 0):.2f}%")
                    with subcol3:
                        st.metric("Net Margin", f"{ratios.get('Net Margin', 0):.2f}%")
                        st.metric("ROA", f"{ratios.get('ROA', 0):.2f}%")
                    with subcol4:
                        st.metric("FCF Margin", f"{ratios.get('FCF Margin', 0):.2f}%")
                        st.metric("OCF/NI", f"{ratios.get('OCF/Net Income', 0):.2f}x")
                    
                    st.markdown("---")
                    st.markdown("### Leverage & Liquidity")
                    
                    subcol1, subcol2, subcol3, subcol4 = st.columns(4)
                    with subcol1:
                        st.metric("Current Ratio", f"{ratios.get('Current Ratio', 0):.2f}x")
                        st.metric("Debt/Equity", f"{ratios.get('Debt/Equity', 0):.1f}%")
                    with subcol2:
                        st.metric("Quick Ratio", f"{ratios.get('Quick Ratio', 0):.2f}x")
                        st.metric("Debt/Assets", f"{ratios.get('Debt/Assets', 0):.1f}%")
                    with subcol3:
                        st.metric("Net Debt", f"${ratios.get('Net Debt', 0):.2f}B")
                        st.metric("Interest Coverage", f"{ratios.get('Interest Coverage', 0):.1f}x")
                    with subcol4:
                        st.metric("Cash Conv. Cycle", f"{ratios.get('Cash Conversion Cycle', 0):.0f} days")
                        st.metric("Asset Turnover", f"{ratios.get('Asset Turnover', 0):.2f}x")
                
                with col2:
                    st.markdown("### Quality Scores")
                    
                    # Altman Z-Score
                    z_score = ratios.get('Altman Z-Score', 0)
                    if z_score > 2.99:
                        z_badge = "risk-badge-safe"
                        z_text = "SAFE ZONE"
                    elif z_score > 1.81:
                        z_badge = "risk-badge-warning"
                        z_text = "GREY ZONE"
                    else:
                        z_badge = "risk-badge-danger"
                        z_text = "DISTRESS ZONE"
                    
                    st.markdown(f"""
                        <div class='metric-card' style='text-align: center;'>
                            <h4 style='color: #94A3B8; margin: 0;'>Altman Z-Score</h4>
                            <h1 style='color: #ffffff; margin: 10px 0; font-size: 48px;'>{z_score:.2f}</h1>
                            <span class='risk-badge {z_badge}'>{z_text}</span>
                            <p style='color: #94A3B8; margin-top: 15px; font-size: 12px;'>
                                >2.99: Safe | 1.81-2.99: Grey | <1.81: Distress
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Piotroski F-Score
                    f_score = ratios.get('Piotroski F-Score', 0)
                    if f_score >= 7:
                        f_badge = "risk-badge-safe"
                        f_text = "STRONG"
                    elif f_score >= 4:
                        f_badge = "risk-badge-warning"
                        f_text = "MODERATE"
                    else:
                        f_badge = "risk-badge-danger"
                        f_text = "WEAK"
                    
                    st.markdown(f"""
                        <div class='metric-card' style='text-align: center;'>
                            <h4 style='color: #94A3B8; margin: 0;'>Piotroski F-Score</h4>
                            <h1 style='color: #ffffff; margin: 10px 0; font-size: 48px;'>{f_score}/9</h1>
                            <span class='risk-badge {f_badge}'>{f_text}</span>
                            <p style='color: #94A3B8; margin-top: 15px; font-size: 12px;'>
                                7-9: Strong | 4-6: Moderate | 0-3: Weak
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Insights Section
                st.markdown("---")
                st.markdown("### Key Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.markdown("**Profitability Analysis**")
                    insights = []
                    if ratios.get('Gross Margin', 0) > 40:
                        insights.append("Strong pricing power (Gross Margin > 40%)")
                    if ratios.get('ROIC', 0) > 15:
                        insights.append("Excellent capital efficiency (ROIC > 15%)")
                    if ratios.get('FCF Margin', 0) > 10:
                        insights.append("Strong free cash flow generation")
                    if ratios.get('ROIC', 0) < 10:
                        insights.append("Low return on invested capital")
                    
                    for insight in (insights if insights else ["No specific insights."]):
                        st.markdown(f"- {insight}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.markdown("**Financial Health**")
                    insights = []
                    if ratios.get('Current Ratio', 0) > 1.5:
                        insights.append("Strong liquidity position")
                    if ratios.get('Debt/Equity', 0) < 50:
                        insights.append("Conservative leverage")
                    if ratios.get('Debt/Equity', 0) > 100:
                        insights.append("High leverage - monitor closely")
                    if ratios.get('Interest Coverage', 0) > 5:
                        insights.append("Comfortable debt service capacity")
                    
                    for insight in (insights if insights else ["No specific insights."]):
                        st.markdown(f"- {insight}")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 2: Enhanced Risk Analytics
        with tab2:
            st.markdown("### Monte Carlo Simulation")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("#### Simulation Parameters")
                
                days_forward = st.slider("Forecast Horizon (Days)", 30, 730, 252)
                num_simulations = st.slider("Simulations", 5000, 50000, 10000, step=5000)
                confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, step=0.01)
                
                st.markdown("#### Historical Statistics")
                if not history.empty:
                    returns = history['Close'].pct_change().dropna()
                    hist_vol = returns.std() * np.sqrt(252)
                    hist_drift = returns.mean() * 252
                    
                    st.metric("Historical Volatility", f"{hist_vol*100:.2f}%")
                    st.metric("Historical Drift", f"{hist_drift*100:.2f}%")
                    st.metric("Sharpe Ratio", f"{calculate_sharpe_sortino(returns)[0]:.2f}")
                
                run_sim = st.button("Run Simulation", type="primary", use_container_width=True)
            
            with col1:
                if run_sim or 'mc_results' in st.session_state:
                    if run_sim:
                        with st.spinner("Running Monte Carlo simulation..."):
                            current_price = history['Close'].iloc[-1]
                            returns = history['Close'].pct_change().dropna()
                            volatility = returns.std()
                            drift = returns.mean()
                            
                            price_paths = monte_carlo_simulation(
                                current_price, volatility, drift, days_forward, num_simulations
                            )
                            
                            final_prices = price_paths[:, -1]
                            final_returns = (final_prices - current_price) / current_price
                            
                            var, cvar = calculate_var_cvar(final_returns, confidence_level)
                            sharpe, sortino = calculate_sharpe_sortino(returns)
                            
                            st.session_state['mc_results'] = {
                                'price_paths': price_paths,
                                'current_price': current_price,
                                'volatility': volatility * np.sqrt(252),
                                'drift': drift * 252,
                                'final_returns': final_returns,
                                'var': var,
                                'cvar': cvar,
                                'sharpe': sharpe,
                                'sortino': sortino,
                                'confidence_level': confidence_level
                            }
                    
                    mc = st.session_state.get('mc_results')
                    if mc:
                        price_paths = mc['price_paths']
                        final_prices = price_paths[:, -1]
                        
                        # Percentiles
                        p5 = np.percentile(final_prices, 5)
                        p25 = np.percentile(final_prices, 25)
                        p50 = np.percentile(final_prices, 50)
                        p75 = np.percentile(final_prices, 75)
                        p95 = np.percentile(final_prices, 95)
                        
                        # Enhanced Plot
                        fig = make_subplots(
                            rows=2, cols=1,
                            row_heights=[0.7, 0.3],
                            subplot_titles=("Price Simulation Paths", "Distribution of Final Prices"),
                            vertical_spacing=0.12
                        )
                        
                        # Sample paths
                        sample_paths = price_paths[np.random.choice(price_paths.shape[0], 200, replace=False)]
                        for path in sample_paths:
                            fig.add_trace(
                                go.Scatter(
                                    y=path,
                                    mode='lines',
                                    line=dict(color='rgba(110, 110, 150, 0.1)', width=0.5),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ),
                                row=1, col=1
                            )
                        
                        # Percentile lines
                        days_range = list(range(len(price_paths[0])))
                        fig.add_trace(go.Scatter(x=days_range, y=[p5]*len(days_range), name='P5', 
                                                 line=dict(color='#EF4444', width=2, dash='dash')), row=1, col=1)
                        fig.add_trace(go.Scatter(x=days_range, y=[p50]*len(days_range), name='Median', 
                                                 line=dict(color='#0EA5E9', width=3)), row=1, col=1)
                        fig.add_trace(go.Scatter(x=days_range, y=[p95]*len(days_range), name='P95', 
                                                 line=dict(color='#10B981', width=2, dash='dash')), row=1, col=1)
                        
                        # Distribution histogram
                        fig.add_trace(
                            go.Histogram(
                                x=final_prices,
                                nbinsx=50,
                                name='Distribution',
                                marker=dict(color='#0EA5E9', opacity=0.7),
                                showlegend=False
                            ),
                            row=2, col=1
                        )
                        
                        # Add VaR line
                        var_price = mc['current_price'] * (1 + mc['var'])
                        fig.add_vline(x=var_price, line_dash="dash", line_color="#EF4444", 
                                       annotation_text=f"VaR ({mc['confidence_level']*100:.0f}%)", 
                                       row=2, col=1)
                        
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="#0F172A",
                            paper_bgcolor="#0F172A",
                            height=700,
                            showlegend=True,
                            legend=dict(x=0.01, y=0.99),
                            hovermode='x unified'
                        )
                        
                        fig.update_xaxes(title_text="Trading Days", row=1, col=1)
                        fig.update_xaxes(title_text="Final Price ($)", row=2, col=1)
                        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                        fig.update_yaxes(title_text="Frequency", row=2, col=1)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk Metrics
                        st.markdown("#### Probabilistic Outcomes & Risk Metrics")
                        
                        col_a, col_b, col_c, col_d, col_e = st.columns(5)
                        with col_a:
                            st.metric("P5 (Downside)", f"${p5:.2f}", 
                                      f"{(p5/mc['current_price']-1)*100:.1f}%")
                        with col_b:
                            st.metric("P25", f"${p25:.2f}", 
                                      f"{(p25/mc['current_price']-1)*100:.1f}%")
                        with col_c:
                            st.metric("P50 (Median)", f"${p50:.2f}", 
                                      f"{(p50/mc['current_price']-1)*100:.1f}%")
                        with col_d:
                            st.metric("P75", f"${p75:.2f}", 
                                      f"{(p75/mc['current_price']-1)*100:.1f}%")
                        with col_e:
                            st.metric("P95 (Upside)", f"${p95:.2f}", 
                                      f"{(p95/mc['current_price']-1)*100:.1f}%")
                        
                        st.markdown("---")
                        
                        col_x, col_y, col_z, col_w = st.columns(4)
                        with col_x:
                            st.metric(f"VaR ({mc['confidence_level']*100:.0f}%)", 
                                      f"{mc['var']*100:.2f}%",
                                      help="Maximum expected loss at given confidence level")
                        with col_y:
                            st.metric(f"CVaR ({mc['confidence_level']*100:.0f}%)", 
                                      f"{mc['cvar']*100:.2f}%",
                                      help="Expected loss when VaR is exceeded")
                        with col_z:
                            st.metric("Sharpe Ratio", f"{mc['sharpe']:.2f}",
                                      help="Risk-adjusted return metric")
                        with col_w:
                            st.metric("Sortino Ratio", f"{mc['sortino']:.2f}",
                                      help="Downside risk-adjusted return")
                        
                        # Probability of outcomes
                        st.markdown("#### Probability Analysis")
                        prob_positive = (final_prices > mc['current_price']).sum() / len(final_prices) * 100
                        prob_10_up = (final_prices > mc['current_price'] * 1.1).sum() / len(final_prices) * 100
                        prob_10_down = (final_prices < mc['current_price'] * 0.9).sum() / len(final_prices) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                                <div class='metric-card' style='text-align: center;'>
                                    <h4 style='color: #94A3B8;'>Prob. of Gain</h4>
                                    <h2 style='color: #10B981;'>{prob_positive:.1f}%</h2>
                                </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                                <div class='metric-card' style='text-align: center;'>
                                    <h4 style='color: #94A3B8;'>Prob. of +10%</h4>
                                    <h2 style='color: #10B981;'>{prob_10_up:.1f}%</h2>
                                </div>
                            """, unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"""
                                <div class='metric-card' style='text-align: center;'>
                                    <h4 style='color: #94A3B8;'>Prob. of -10%</h4>
                                    <h2 style='color: #EF4444;'>{prob_10_down:.1f}%</h2>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("Configure parameters and run simulation to see results.")
        
        # Tab 3: Enhanced DCF Valuation
        with tab3:
            st.markdown("### 2-Stage Enterprise Value DCF Model")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("#### Model Parameters")
                
                # WACC Calculation
                st.markdown("##### WACC Inputs")
                rfr = st.slider(
                    "Risk-Free Rate (10y Treasury)",
                    min_value=0.01, max_value=0.07,
                    value=get_risk_free_rate(), step=0.001, format="%.3f%%"
                )
                market_premium = st.slider(
                    "Equity Market Risk Premium",
                    min_value=0.03, max_value=0.08,
                    value=0.05, step=0.005, format="%.1f%%"
                )
                
                wacc, wacc_details = calculate_wacc(data, rfr, market_premium)
                
                if wacc:
                    st.metric("Calculated WACC", f"{wacc*100:.2f}%")
                    with st.expander("Show WACC Details"):
                        st.json({k: (f"{v*100:.2f}%" if "%" in k or "Cost" in k or "Rate" in k else (f"{v:.2f}" if isinstance(v, float) else f"{v:,.0f}")) for k, v in wacc_details.items()})
                else:
                    st.error(f"Could not calculate WACC: {wacc_details}")
                    st.stop()
                
                # Growth Assumptions
                st.markdown("##### Growth Inputs")
                g_5y = st.slider(
                    "FCFF Growth (Years 1-5)",
                    min_value=-0.05, max_value=0.30, value=0.10, step=0.01, format="%.1f%%"
                )
                g_10y = st.slider(
                    "FCFF Growth (Years 6-10)",
                    min_value=-0.05, max_value=0.20, value=0.05, step=0.01, format="%.1f%%"
                )
                g_t = st.slider(
                    "Terminal Growth Rate",
                    min_value=0.01, max_value=0.05, value=0.025, step=0.005, format="%.1f%%"
                )
                
                dcf_result = calculate_dcf_valuation_advanced(data, wacc, g_5y, g_10y, g_t)
            
            with col1:
                if dcf_result:
                    current_price = info.get('currentPrice', 0)
                    intrinsic_price = dcf_result['intrinsic_price']
                    upside = (intrinsic_price / current_price - 1) * 100 if current_price > 0 else 0
                    
                    # Valuation Summary Card
                    if upside > 10:
                        upside_color = "#10B981"
                        upside_text = "UNDERVALUED"
                    elif upside < -10:
                        upside_color = "#EF4444"
                        upside_text = "OVERVALUED"
                    else:
                        upside_color = "#F59E0B"
                        upside_text = "FAIRLY VALUED"
                    
                    st.markdown(f"""
                        <div style='background-color: #1E293B; border: 1px solid #334155;
                                    padding: 30px; border-radius: 15px; margin-bottom: 20px;'>
                            <div style='display: flex; justify-content: space-around; align-items: center;'>
                                <div style='text-align: center;'>
                                    <h4 style='color: #94A3B8; margin: 0;'>Current Price</h4>
                                    <h1 style='color: #ffffff; margin: 10px 0;'>${current_price:.2f}</h1>
                                </div>
                                <div style='text-align: center; font-size: 48px; color: {upside_color};'>
                                    {"→" if abs(upside) < 10 else "↗" if upside > 0 else "↘"}
                                </div>
                                <div style='text-align: center;'>
                                    <h4 style='color: #94A3B8; margin: 0;'>Intrinsic Value</h4>
                                    <h1 style='color: {upside_color}; margin: 10px 0;'>${intrinsic_price:.2f}</h1>
                                </div>
                                <div style='text-align: center;'>
                                    <h4 style='color: #94A3B8; margin: 0;'>Upside/Downside</h4>
                                    <h1 style='color: {upside_color}; margin: 10px 0;'>{upside:+.1f}%</h1>
                                    <span style='background: {upside_color}; color: white; padding: 5px 15px; 
                                                 border-radius: 15px; font-size: 12px; font-weight: 600;'>
                                        {upside_text}
                                    </span>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed Breakdown
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("#### Enterprise Value Build-Up")
                        
                        ev_data = pd.DataFrame({
                            'Component': [
                                'PV of Explicit FCF (Y1-10)',
                                'PV of Terminal Value',
                                'Enterprise Value',
                                'Less: Net Debt',
                                'Equity Value'
                            ],
                            'Value ($B)': [
                                dcf_result['pv_fcf'] / 1e9,
                                dcf_result['pv_terminal'] / 1e9,
                                dcf_result['enterprise_value'] / 1e9,
                                -(dcf_result['enterprise_value'] - dcf_result['equity_value']) / 1e9,
                                dcf_result['equity_value'] / 1e9
                            ]
                        })
                        
                        # Waterfall Chart
                        fig = go.Figure(go.Waterfall(
                            x=ev_data['Component'],
                            y=ev_data['Value ($B)'],
                            measure=['relative', 'relative', 'total', 'relative', 'total'],
                            text=[f"${v:.2f}B" for v in ev_data['Value ($B)']],
                            textposition='outside',
                            connector={"line": {"color": "#94A3B8"}},
                            increasing={"marker": {"color": "#10B981"}},
                            decreasing={"marker": {"color": "#EF4444"}},
                            totals={"marker": {"color": "#0EA5E9"}}
                        ))
                        
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="#0F172A",
                            paper_bgcolor="#0F172A",
                            height=400,
                            showlegend=False,
                            title="Enterprise to Equity Value Waterfall"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_b:
                        st.markdown("#### 10-Year Cash Flow Projection")
                        
                        fcf_df = pd.DataFrame({
                            'Year': range(1, 11),
                            'Projected FCFF ($M)': [fcf/1e6 for fcf in dcf_result['projected_fcf']],
                            'PV of FCFF ($M)': [fcf/(1+wacc)**i/1e6 for i, fcf in enumerate(dcf_result['projected_fcf'], 1)]
                        })
                        
                        # FCF Chart
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=fcf_df['Year'],
                            y=fcf_df['Projected FCFF ($M)'],
                            name='Projected FCFF',
                            marker_color='#0EA5E9'
                        ))
                        fig.add_trace(go.Scatter(
                            x=fcf_df['Year'],
                            y=fcf_df['PV of FCFF ($M)'],
                            name='Present Value',
                            mode='lines+markers',
                            line=dict(color='#10B981', width=3)
                        ))
                        
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="#0F172A",
                            paper_bgcolor="#0F172A",
                            height=300,
                            legend=dict(x=0.01, y=0.99),
                            yaxis_title="Value ($M)",
                            margin=dict(t=20, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        st.metric("Base Year FCFF (T=0)", f"${dcf_result['base_fcff']/1e6:.2f}M")
                        st.metric("Terminal Value (Year 10)", f"${dcf_result['terminal_value']/1e9:.2f}B")
                        
                    
                    # Sensitivity Analysis
                    st.markdown("---")
                    st.markdown("#### Sensitivity Analysis")
                    
                    growth_range = np.linspace(g_5y - 0.02, g_5y + 0.02, 5)
                    wacc_range = np.linspace(wacc - 0.02, wacc + 0.02, 5)
                    
                    sensitivity = np.zeros((5, 5))
                    for i, g in enumerate(growth_range):
                        for j, w in enumerate(wacc_range):
                            # Adjust transitional growth proportionally
                            g_10_sens = g_10y * (g / g_5y) if g_5y != 0 else g_10y
                            if w > g_t:
                                result = calculate_dcf_valuation_advanced(data, w, g, g_10_sens, g_t)
                                if result:
                                    sensitivity[i, j] = result['intrinsic_price']
                    
                    # Heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=sensitivity,
                        x=[f"{w*100:.2f}%" for w in wacc_range],
                        y=[f"{g*100:.1f}%" for g in growth_range],
                        colorscale='RdYlGn',
                        text=sensitivity,
                        texttemplate='$%{text:.2f}',
                        textfont={"size": 10},
                        colorbar=dict(title="Price ($)")
                    ))
                    
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="#0F172A",
                        paper_bgcolor="#0F172A",
                        title="Intrinsic Value Sensitivity (Y1-5 Growth vs WACC)",
                        xaxis_title="WACC",
                        yaxis_title="Y1-5 Growth Rate",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(f"**Reference:** Current market price is **${current_price:.2f}**")
                    
                else:
                    st.error("Unable to calculate DCF - insufficient financial data or negative cash flows")
        
        # Tab 4: Technical Analysis
        with tab4:
            st.markdown("### Technical Analysis & Price Action")
            
            if not history.empty:
                # Price Chart with Moving Averages
                fig = make_subplots(
                    rows=2, cols=1,
                    row_heights=[0.7, 0.3],
                    subplot_titles=("Price & Moving Averages", "Volume"),
                    vertical_spacing=0.1
                )
                
                # Calculate moving averages
                history['SMA_20'] = history['Close'].rolling(20).mean()
                history['SMA_50'] = history['Close'].rolling(50).mean()
                history['SMA_200'] = history['Close'].rolling(200).mean()
                
                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=history.index,
                    open=history['Open'],
                    high=history['High'],
                    low=history['Low'],
                    close=history['Close'],
                    name='Price',
                    increasing_line_color='#10B981',
                    decreasing_line_color='#EF4444'
                ), row=1, col=1)
                
                # Moving averages
                fig.add_trace(go.Scatter(x=history.index, y=history['SMA_20'], 
                                         name='SMA 20', line=dict(color='#0EA5E9', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=history.index, y=history['SMA_50'], 
                                         name='SMA 50', line=dict(color='#F59E0B', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=history.index, y=history['SMA_200'], 
                                         name='SMA 200', line=dict(color='#E11D48', width=2)), row=1, col=1)
                
                # Volume
                colors = ['#10B981' if history['Close'].iloc[i] >= history['Open'].iloc[i] 
                          else '#EF4444' for i in range(len(history))]
                fig.add_trace(go.Bar(x=history.index, y=history['Volume'], 
                                     name='Volume', marker_color=colors, showlegend=False), row=2, col=1)
                
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#0F172A",
                    paper_bgcolor="#0F172A",
                    height=700,
                    xaxis_rangeslider_visible=False,
                    hovermode='x unified'
                )
                
                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical Indicators
                st.markdown("#### Technical Indicators")
                
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = history['Close'].iloc[-1]
                sma_20 = history['SMA_20'].iloc[-1]
                sma_50 = history['SMA_50'].iloc[-1]
                sma_200 = history['SMA_200'].iloc[-1]
                
                with col1:
                    trend = "Bullish" if current_price > sma_20 > sma_50 else "Bearish" if current_price < sma_20 < sma_50 else "Neutral"
                    st.metric("Trend", trend)
                with col2:
                    st.metric("vs SMA 20", f"{(current_price/sma_20-1)*100:+.2f}%")
                with col3:
                    st.metric("vs SMA 50", f"{(current_price/sma_50-1)*100:+.2f}%")
                with col4:
                    st.metric("vs SMA 200", f"{(current_price/sma_200-1)*100:+.2f}%")
                
                # Returns Analysis
                st.markdown("---")
                st.markdown("#### Returns Analysis")
                
                returns_1m = (history['Close'].iloc[-1] / history['Close'].iloc[-21] - 1) * 100 if len(history) > 21 else 0
                returns_3m = (history['Close'].iloc[-1] / history['Close'].iloc[-63] - 1) * 100 if len(history) > 63 else 0
                returns_6m = (history['Close'].iloc[-1] / history['Close'].iloc[-126] - 1) * 100 if len(history) > 126 else 0
                returns_1y = (history['Close'].iloc[-1] / history['Close'].iloc[-252] - 1) * 100 if len(history) > 252 else 0
                returns_ytd = (history['Close'].iloc[-1] / history['Close'].iloc[0] - 1) * 100
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("1 Month", f"{returns_1m:+.2f}%")
                with col2:
                    st.metric("3 Months", f"{returns_3m:+.2f}%")
                with col3:
                    st.metric("6 Months", f"{returns_6m:+.2f}%")
                with col4:
                    st.metric("1 Year", f"{returns_1y:+.2f}%")
                with col5:
                    st.metric("YTD", f"{returns_ytd:+.2f}%")
    
    else:
        # Landing Page
        st.markdown("""
            <div style='text-align: center; padding: 60px 20px;'>
                <h2 style='color: #0EA5E9; font-size: 36px; margin-bottom: 20px;'>
                    Welcome to Risk-Adjusted Analyst Pro
                </h2>
                <p style='color: #94A3B8; font-size: 18px; max-width: 800px; margin: 0 auto;'>
                    Advanced financial analysis platform combining traditional valuation models with 
                    cutting-edge risk analytics and Monte Carlo simulations.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class='metric-card' style='text-align: center;'>
                    <h3 style='color: #0EA5E9;'>Financial Health</h3>
                    <p style='color: #94A3B8;'>
                        Comprehensive ratio analysis including profitability, liquidity, 
                        leverage, and quality scores (Altman Z-Score & Piotroski F-Score)
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='metric-card' style='text-align: center;'>
                    <h3 style='color: #0EA5E9;'>Risk Analytics</h3>
                    <p style='color: #94A3B8;'>
                        Monte Carlo simulations with VaR, CVaR, Sharpe & Sortino ratios 
                        for probabilistic risk assessment
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class='metric-card' style='text-align: center;'>
                    <h3 style='color: #0EA5E9;'>DCF Valuation</h3>
                    <p style='color: #94A3B8;'>
                        2-Stage Enterprise Value DCF with calculated WACC, sensitivity analysis, 
                        and scenario modeling
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown("### Popular Companies to Analyze")
        
        companies = {
            "Tech Giants": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "TSLA"],
            "Financials": ["JPM", "BAC", "GS", "V", "MA", "BRK.B"],
            "Consumer": ["AMZN", "WMT", "HD", "NKE", "COST", "MCD"],
            "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "TMO", "LLY"]
        }
        
        for sector, tickers in companies.items():
            with st.expander(sector, expanded=False):
                cols = st.columns(6)
                for i, tick in enumerate(tickers):
                    if cols[i].button(tick, key=f"landing_{tick}", use_container_width=True):
                        st.session_state['ticker'] = tick
                        st.session_state['data'] = fetch_company_data(tick)
                        st.rerun()

if __name__ == "__main__":
    main()