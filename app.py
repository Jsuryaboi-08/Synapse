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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Dark Theme Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f35 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e2235 0%, #252b42 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #2e3548;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #1e2235 0%, #252b42 100%);
        border-radius: 10px;
        padding: 12px 24px;
        border: 1px solid #2e3548;
        color: #8b92a7;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2e3548 0%, #3a4159 100%);
        color: #ffffff;
        border: 1px solid #4a5269;
    }
    
    .premium-header {
        background: linear-gradient(135deg, #1e2235 0%, #252b42 100%);
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #2e3548;
        margin-bottom: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    
    .risk-badge-safe {
        background: linear-gradient(135deg, #0d7377 0%, #14919b 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .risk-badge-warning {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .risk-badge-danger {
        background: linear-gradient(135deg, #c9184a 0%, #ff0054 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
    }
    
    .stMetric {
        background: transparent;
    }
    
    .stMetric label {
        color: #8b92a7 !important;
        font-size: 14px !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stExpander"] {
        background: linear-gradient(135deg, #1e2235 0%, #252b42 100%);
        border: 1px solid #2e3548;
        border-radius: 10px;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #2e3548 0%, #3a4159 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #14919b;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Utility Functions
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
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_advanced_ratios(data):
    """Calculate comprehensive set of financial ratios with trend analysis"""
    try:
        info = data['info']
        financials = pd.DataFrame(data['financials'])
        balance_sheet = pd.DataFrame(data['balance_sheet'])
        cashflow = pd.DataFrame(data['cashflow'])
        
        latest_financials = financials.iloc[:, 0] if not financials.empty else pd.Series()
        latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
        latest_cf = cashflow.iloc[:, 0] if not cashflow.empty else pd.Series()
        
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
        total_debt = latest_balance.get('Total Debt', 0)
        equity = latest_balance.get('Stockholders Equity', 0)
        invested_capital = total_debt + equity
        nopat = ebit * 0.79  # After-tax operating profit (21% tax)
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
        ratios['Interest Coverage'] = (ebit / latest_financials.get('Interest Expense', 1)) if latest_financials.get('Interest Expense', 0) < 0 else 0
        
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
        fcf = operating_cf - capex
        
        ratios['FCF'] = fcf / 1e9
        ratios['FCF Margin'] = (fcf / revenue * 100) if revenue > 0 else 0
        ratios['OCF/Net Income'] = (operating_cf / net_income) if net_income > 0 else 0
        
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
        
        if financials.empty or balance_sheet.empty:
            return 0
        
        score = 0
        
        # Profitability (4 points)
        net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 0
        score += 1 if net_income > 0 else 0
        
        if not cashflow.empty and 'Operating Cash Flow' in cashflow.index:
            ocf = cashflow.loc['Operating Cash Flow'].iloc[0]
            score += 1 if ocf > 0 else 0
            score += 1 if ocf > net_income else 0
        
        if len(financials.columns) > 1:
            roa_current = net_income / balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 0
            roa_previous = financials.loc['Net Income'].iloc[1] / balance_sheet.loc['Total Assets'].iloc[1] if len(balance_sheet.columns) > 1 else 0
            score += 1 if roa_current > roa_previous else 0
        
        # Leverage (3 points)
        if len(balance_sheet.columns) > 1 and 'Total Debt' in balance_sheet.index:
            debt_current = balance_sheet.loc['Total Debt'].iloc[0]
            debt_previous = balance_sheet.loc['Total Debt'].iloc[1]
            score += 1 if debt_current < debt_previous else 0
        
        if 'Current Assets' in balance_sheet.index and 'Current Liabilities' in balance_sheet.index:
            current_ratio = balance_sheet.loc['Current Assets'].iloc[0] / balance_sheet.loc['Current Liabilities'].iloc[0]
            if len(balance_sheet.columns) > 1:
                previous_current_ratio = balance_sheet.loc['Current Assets'].iloc[1] / balance_sheet.loc['Current Liabilities'].iloc[1]
                score += 1 if current_ratio > previous_current_ratio else 0
        
        # Operating Efficiency (2 points)
        if len(financials.columns) > 1:
            if 'Gross Profit' in financials.index and 'Total Revenue' in financials.index:
                gm_current = financials.loc['Gross Profit'].iloc[0] / financials.loc['Total Revenue'].iloc[0]
                gm_previous = financials.loc['Gross Profit'].iloc[1] / financials.loc['Total Revenue'].iloc[1]
                score += 1 if gm_current > gm_previous else 0
            
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

def calculate_dcf_valuation(data, growth_rate, wacc, terminal_growth=0.025):
    """Enhanced DCF with sensitivity and scenario analysis"""
    try:
        cashflow = pd.DataFrame(data['cashflow'])
        balance_sheet = pd.DataFrame(data['balance_sheet'])
        info = data['info']
        
        latest_cf = cashflow.iloc[:, 0] if not cashflow.empty else pd.Series()
        latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
        
        # FCFF Calculation
        operating_cf = latest_cf.get('Operating Cash Flow', 0)
        capex = latest_cf.get('Capital Expenditure', 0)
        fcff = operating_cf + capex
        
        if fcff <= 0:
            return None
        
        # Project cash flows
        projection_years = 10
        projected_fcf = []
        for year in range(1, projection_years + 1):
            # Declining growth rate model
            year_growth = growth_rate * (1 - year/(projection_years + 5))
            projected_fcf.append(fcff * (1 + year_growth)**year)
        
        # Terminal value with Gordon Growth
        terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
        
        # Discount cash flows
        pv_fcf = sum([cf / (1 + wacc)**i for i, cf in enumerate(projected_fcf, 1)])
        pv_terminal = terminal_value / (1 + wacc)**projection_years
        
        enterprise_value = pv_fcf + pv_terminal
        
        # Equity value calculation
        total_debt = latest_balance.get('Total Debt', 0)
        cash = latest_balance.get('Cash And Cash Equivalents', 0)
        minority_interest = latest_balance.get('Minority Interest', 0)
        
        equity_value = enterprise_value - total_debt + cash - minority_interest
        
        shares_outstanding = info.get('sharesOutstanding', 1)
        intrinsic_value_per_share = equity_value / shares_outstanding
        
        return {
            'enterprise_value': enterprise_value,
            'equity_value': equity_value,
            'intrinsic_price': intrinsic_value_per_share,
            'fcff': fcff,
            'projected_fcf': projected_fcf,
            'terminal_value': terminal_value,
            'pv_fcf': pv_fcf,
            'pv_terminal': pv_terminal
        }
    except Exception as e:
        st.error(f"Error in DCF calculation: {str(e)}")
        return None

# Main Application
def main():
    # Header
    st.markdown("""
        <div class='premium-header'>
            <h1 style='margin:0; background: linear-gradient(135deg, #14919b 0%, #0d7377 100%); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                       background-clip: text; font-size: 48px;'>
                📊 Risk-Adjusted Analyst Pro
            </h1>
            <p style='color: #8b92a7; margin-top: 10px; font-size: 18px;'>
                Advanced Forward-Looking Valuation & Risk Analytics Platform
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Company Selection")
        ticker = st.text_input("Enter Ticker Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL").upper()
        
        st.markdown("---")
        st.markdown("### ⚙️ Analysis Settings")
        
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Quick", "Standard", "Deep"],
            value="Standard"
        )
        
        include_peers = st.checkbox("Compare with Peers", value=False)
        
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Fetching data..."):
                st.session_state['ticker'] = ticker
                st.session_state['data'] = fetch_company_data(ticker)
                st.session_state['analysis_depth'] = analysis_depth
        
        st.markdown("---")
        st.markdown("### 📈 Quick Access")
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
            <div style='background: linear-gradient(135deg, #1e2235 0%, #252b42 100%); 
                        padding: 15px; border-radius: 10px; margin: 20px 0; border: 1px solid #2e3548;'>
                <strong style='color: #14919b; font-size: 18px;'>{info.get('shortName', ticker)}</strong> 
                <span style='color: #8b92a7;'>• {info.get('sector', 'N/A')} • {info.get('industry', 'N/A')}</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Financial Health", 
            "🎲 Risk Analytics", 
            "💰 Valuation Models",
            "📈 Technical Analysis"
        ])
        
        # Tab 1: Enhanced Health Scorecard
        with tab1:
            ratios = calculate_advanced_ratios(data)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 💪 Profitability & Returns")
                
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
                st.markdown("### 🏦 Leverage & Liquidity")
                
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
                st.markdown("### 🎯 Quality Scores")
                
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
                        <h4 style='color: #8b92a7; margin: 0;'>Altman Z-Score</h4>
                        <h1 style='color: #ffffff; margin: 10px 0; font-size: 48px;'>{z_score:.2f}</h1>
                        <span class='{z_badge}'>{z_text}</span>
                        <p style='color: #8b92a7; margin-top: 15px; font-size: 12px;'>
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
                        <h4 style='color: #8b92a7; margin: 0;'>Piotroski F-Score</h4>
                        <h1 style='color: #ffffff; margin: 10px 0; font-size: 48px;'>{f_score}/9</h1>
                        <span class='{f_badge}'>{f_text}</span>
                        <p style='color: #8b92a7; margin-top: 15px; font-size: 12px;'>
                            7-9: Strong | 4-6: Moderate | 0-3: Weak
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Insights Section
            st.markdown("---")
            st.markdown("### 🔍 Key Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown("**💰 Profitability Analysis**")
                insights = []
                if ratios.get('Gross Margin', 0) > 40:
                    insights.append("✅ Strong pricing power with gross margins >40%")
                if ratios.get('ROIC', 0) > 15:
                    insights.append("✅ Excellent capital efficiency (ROIC >15%)")
                if ratios.get('FCF Margin', 0) > 10:
                    insights.append("✅ Strong free cash flow generation")
                if ratios.get('ROIC', 0) < 10:
                    insights.append("⚠️ Low return on invested capital")
                
                for insight in insights:
                    st.markdown(insight)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown("**🏦 Financial Health**")
                insights = []
                if ratios.get('Current Ratio', 0) > 1.5:
                    insights.append("✅ Strong liquidity position")
                if ratios.get('Debt/Equity', 0) < 50:
                    insights.append("✅ Conservative leverage")
                if ratios.get('Debt/Equity', 0) > 100:
                    insights.append("⚠️ High leverage - monitor closely")
                if ratios.get('Interest Coverage', 0) > 5:
                    insights.append("✅ Comfortable debt service capacity")
                
                for insight in insights:
                    st.markdown(insight)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 2: Enhanced Risk Analytics
        with tab2:
            st.markdown("### 🎲 Monte Carlo Simulation")
            
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
                
                run_sim = st.button("🎲 Run Simulation", type="primary", use_container_width=True)
            
            with col1:
                if run_sim or 'mc_results' in st.session_state:
                    if run_sim:
                        with st.spinner("Running Monte Carlo simulation..."):
                            current_price = history['Close'].iloc[-1]
                            returns = history['Close'].pct_change().dropna()
                            volatility = returns.std() * np.sqrt(252)
                            drift = returns.mean() * 252
                            
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
                                'volatility': volatility,
                                'drift': drift,
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
                                    line=dict(color='rgba(100, 150, 200, 0.05)', width=0.5),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ),
                                row=1, col=1
                            )
                        
                        # Percentile lines
                        days_range = list(range(len(price_paths[0])))
                        fig.add_trace(go.Scatter(x=days_range, y=[p5]*len(days_range), name='P5', 
                                                line=dict(color='#ff0054', width=2, dash='dash')), row=1, col=1)
                        fig.add_trace(go.Scatter(x=days_range, y=[p50]*len(days_range), name='Median', 
                                                line=dict(color='#14919b', width=3)), row=1, col=1)
                        fig.add_trace(go.Scatter(x=days_range, y=[p95]*len(days_range), name='P95', 
                                                line=dict(color='#0d7377', width=2, dash='dash')), row=1, col=1)
                        
                        # Distribution histogram
                        fig.add_trace(
                            go.Histogram(
                                x=final_prices,
                                nbinsx=50,
                                name='Distribution',
                                marker=dict(color='#14919b', opacity=0.7),
                                showlegend=False
                            ),
                            row=2, col=1
                        )
                        
                        # Add VaR line
                        var_price = mc['current_price'] * (1 + mc['var'])
                        fig.add_vline(x=var_price, line_dash="dash", line_color="#ff0054", 
                                     annotation_text=f"VaR ({mc['confidence_level']*100:.0f}%)", 
                                     row=2, col=1)
                        
                        fig.update_layout(
                            template="plotly_dark",
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
                        st.markdown("#### 📊 Probabilistic Outcomes & Risk Metrics")
                        
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
                        st.markdown("#### 🎯 Probability Analysis")
                        prob_positive = (final_prices > mc['current_price']).sum() / len(final_prices) * 100
                        prob_10_up = (final_prices > mc['current_price'] * 1.1).sum() / len(final_prices) * 100
                        prob_10_down = (final_prices < mc['current_price'] * 0.9).sum() / len(final_prices) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                                <div class='metric-card' style='text-align: center;'>
                                    <h4 style='color: #8b92a7;'>Prob. of Gain</h4>
                                    <h2 style='color: #14919b;'>{prob_positive:.1f}%</h2>
                                </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                                <div class='metric-card' style='text-align: center;'>
                                    <h4 style='color: #8b92a7;'>Prob. of +10%</h4>
                                    <h2 style='color: #0d7377;'>{prob_10_up:.1f}%</h2>
                                </div>
                            """, unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"""
                                <div class='metric-card' style='text-align: center;'>
                                    <h4 style='color: #8b92a7;'>Prob. of -10%</h4>
                                    <h2 style='color: #ff0054;'>{prob_10_down:.1f}%</h2>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("👈 Configure parameters and run simulation to see results")
        
        # Tab 3: Enhanced DCF Valuation
        with tab3:
            st.markdown("### 💰 Enterprise Value DCF Model")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("#### Model Parameters")
                
                growth_rate = st.slider(
                    "Revenue Growth Rate",
                    min_value=-0.10,
                    max_value=0.40,
                    value=0.08,
                    step=0.01,
                    format="%.1f%%"
                )
                
                wacc = st.slider(
                    "WACC (Discount Rate)",
                    min_value=0.03,
                    max_value=0.25,
                    value=0.10,
                    step=0.01,
                    format="%.1f%%"
                )
                
                terminal_growth = st.slider(
                    "Terminal Growth Rate",
                    min_value=0.01,
                    max_value=0.05,
                    value=0.025,
                    step=0.005,
                    format="%.1f%%"
                )
                
                st.markdown("#### Scenario Analysis")
                scenario = st.radio(
                    "Select Scenario",
                    ["Base Case", "Bull Case", "Bear Case"],
                    horizontal=True
                )
                
                if scenario == "Bull Case":
                    growth_rate *= 1.5
                    wacc *= 0.9
                elif scenario == "Bear Case":
                    growth_rate *= 0.5
                    wacc *= 1.1
                
                dcf_result = calculate_dcf_valuation(data, growth_rate, wacc, terminal_growth)
            
            with col1:
                if dcf_result:
                    current_price = info.get('currentPrice', 0)
                    intrinsic_price = dcf_result['intrinsic_price']
                    upside = (intrinsic_price / current_price - 1) * 100 if current_price > 0 else 0
                    
                    # Valuation Summary Card
                    if upside > 0:
                        upside_color = "#0d7377"
                        upside_text = "UNDERVALUED"
                    else:
                        upside_color = "#ff0054"
                        upside_text = "OVERVALUED"
                    
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #1e2235 0%, #252b42 100%); 
                                    padding: 30px; border-radius: 15px; border: 1px solid #2e3548; 
                                    margin-bottom: 20px;'>
                            <div style='display: flex; justify-content: space-around; align-items: center;'>
                                <div style='text-align: center;'>
                                    <h4 style='color: #8b92a7; margin: 0;'>Current Price</h4>
                                    <h1 style='color: #ffffff; margin: 10px 0;'>${current_price:.2f}</h1>
                                </div>
                                <div style='text-align: center; font-size: 48px; color: {upside_color};'>
                                    {"→" if abs(upside) < 5 else "↗" if upside > 0 else "↘"}
                                </div>
                                <div style='text-align: center;'>
                                    <h4 style='color: #8b92a7; margin: 0;'>Intrinsic Value</h4>
                                    <h1 style='color: {upside_color}; margin: 10px 0;'>${intrinsic_price:.2f}</h1>
                                </div>
                                <div style='text-align: center;'>
                                    <h4 style='color: #8b92a7; margin: 0;'>Upside/Downside</h4>
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
                                'PV of Explicit FCF',
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
                        
                        st.dataframe(
                            ev_data.style.format({'Value ($B)': '${:.2f}B'})
                            .background_gradient(subset=['Value ($B)'], cmap='RdYlGn'),
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Waterfall Chart
                        fig = go.Figure(go.Waterfall(
                            x=ev_data['Component'],
                            y=ev_data['Value ($B)'],
                            measure=['relative', 'relative', 'total', 'relative', 'total'],
                            text=[f"${v:.2f}B" for v in ev_data['Value ($B)']],
                            textposition='outside',
                            connector={"line": {"color": "#8b92a7"}},
                            increasing={"marker": {"color": "#0d7377"}},
                            decreasing={"marker": {"color": "#ff0054"}},
                            totals={"marker": {"color": "#14919b"}}
                        ))
                        
                        fig.update_layout(
                            template="plotly_dark",
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
                        
                        st.dataframe(
                            fcf_df.style.format({
                                'Projected FCFF ($M)': '${:.1f}M',
                                'PV of FCFF ($M)': '${:.1f}M'
                            }).background_gradient(subset=['PV of FCFF ($M)'], cmap='Blues'),
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        st.metric("Terminal Value (Year 10)", f"${dcf_result['terminal_value']/1e9:.2f}B")
                        
                        # FCF Chart
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=fcf_df['Year'],
                            y=fcf_df['Projected FCFF ($M)'],
                            name='Projected FCFF',
                            marker_color='#14919b'
                        ))
                        fig.add_trace(go.Scatter(
                            x=fcf_df['Year'],
                            y=fcf_df['PV of FCFF ($M)'],
                            name='Present Value',
                            mode='lines+markers',
                            line=dict(color='#0d7377', width=3)
                        ))
                        
                        fig.update_layout(
                            template="plotly_dark",
                            height=300,
                            legend=dict(x=0.01, y=0.99),
                            yaxis_title="Value ($M)"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Sensitivity Analysis
                    st.markdown("---")
                    st.markdown("#### 🎛️ Sensitivity Analysis")
                    
                    growth_range = np.linspace(growth_rate - 0.03, growth_rate + 0.03, 7)
                    wacc_range = np.linspace(wacc - 0.03, wacc + 0.03, 7)
                    
                    sensitivity = np.zeros((7, 7))
                    for i, g in enumerate(growth_range):
                        for j, w in enumerate(wacc_range):
                            result = calculate_dcf_valuation(data, g, w, terminal_growth)
                            if result:
                                sensitivity[i, j] = result['intrinsic_price']
                    
                    sensitivity_df = pd.DataFrame(
                        sensitivity,
                        index=[f"{g*100:.1f}%" for g in growth_range],
                        columns=[f"{w*100:.1f}%" for w in wacc_range]
                    )
                    
                    # Heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=sensitivity,
                        x=[f"{w*100:.1f}%" for w in wacc_range],
                        y=[f"{g*100:.1f}%" for g in growth_range],
                        colorscale='RdYlGn',
                        text=sensitivity,
                        texttemplate='$%{text:.2f}',
                        textfont={"size": 10},
                        colorbar=dict(title="Price ($)")
                    ))
                    
                    fig.update_layout(
                        template="plotly_dark",
                        title="Intrinsic Value Sensitivity (Growth Rate vs WACC)",
                        xaxis_title="WACC →",
                        yaxis_title="← Growth Rate",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add current price line for reference
                    st.markdown(f"**Reference:** Current market price is **${current_price:.2f}**")
                    
                else:
                    st.error("❌ Unable to calculate DCF - insufficient financial data or negative cash flows")
        
        # Tab 4: Technical Analysis
        with tab4:
            st.markdown("### 📈 Technical Analysis & Price Action")
            
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
                    increasing_line_color='#0d7377',
                    decreasing_line_color='#ff0054'
                ), row=1, col=1)
                
                # Moving averages
                fig.add_trace(go.Scatter(x=history.index, y=history['SMA_20'], 
                                        name='SMA 20', line=dict(color='#14919b', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=history.index, y=history['SMA_50'], 
                                        name='SMA 50', line=dict(color='#f7931e', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=history.index, y=history['SMA_200'], 
                                        name='SMA 200', line=dict(color='#c9184a', width=2)), row=1, col=1)
                
                # Volume
                colors = ['#0d7377' if history['Close'].iloc[i] >= history['Open'].iloc[i] 
                         else '#ff0054' for i in range(len(history))]
                fig.add_trace(go.Bar(x=history.index, y=history['Volume'], 
                                    name='Volume', marker_color=colors, showlegend=False), row=2, col=1)
                
                fig.update_layout(
                    template="plotly_dark",
                    height=700,
                    xaxis_rangeslider_visible=False,
                    hovermode='x unified'
                )
                
                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical Indicators
                st.markdown("#### 📊 Technical Indicators")
                
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
                st.markdown("#### 📉 Returns Analysis")
                
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
                <h2 style='color: #14919b; font-size: 36px; margin-bottom: 20px;'>
                    Welcome to Risk-Adjusted Analyst Pro
                </h2>
                <p style='color: #8b92a7; font-size: 18px; max-width: 800px; margin: 0 auto;'>
                    Advanced financial analysis platform combining traditional valuation models with 
                    cutting-edge risk analytics and Monte Carlo simulations.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class='metric-card' style='text-align: center;'>
                    <h3 style='color: #14919b;'>📊 Financial Health</h3>
                    <p style='color: #8b92a7;'>
                        Comprehensive ratio analysis including profitability, liquidity, 
                        leverage, and quality scores (Altman Z-Score & Piotroski F-Score)
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='metric-card' style='text-align: center;'>
                    <h3 style='color: #14919b;'>🎲 Risk Analytics</h3>
                    <p style='color: #8b92a7;'>
                        Monte Carlo simulations with VaR, CVaR, Sharpe & Sortino ratios 
                        for probabilistic risk assessment
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class='metric-card' style='text-align: center;'>
                    <h3 style='color: #14919b;'>💰 DCF Valuation</h3>
                    <p style='color: #8b92a7;'>
                        Enterprise Value DCF with real-time sliders, sensitivity analysis, 
                        and scenario modeling
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown("### 🚀 Popular Companies to Analyze")
        
        companies = {
            "🔮 Tech Giants": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "TSLA"],
            "🏦 Financials": ["JPM", "BAC", "GS", "V", "MA", "BRK.B"],
            "🛒 Consumer": ["AMZN", "WMT", "HD", "NKE", "COST", "MCD"],
            "⚕️ Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "TMO", "LLY"]
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