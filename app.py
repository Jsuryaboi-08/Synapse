import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="Risk-Adjusted Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2e3241;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e3241;
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
        
        return {
            'info': info,
            'financials': financials,
            'balance_sheet': balance_sheet,
            'cashflow': cashflow,
            'history': history,
            'stock': stock
        }
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_ratios(data):
    """Calculate 8 key financial ratios"""
    try:
        info = data['info']
        financials = data['financials']
        balance_sheet = data['balance_sheet']
        
        # Get most recent columns
        latest_financials = financials.iloc[:, 0] if not financials.empty else pd.Series()
        latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
        
        ratios = {}
        
        # 1. Gross Margin
        if 'Gross Profit' in latest_financials.index and 'Total Revenue' in latest_financials.index:
            ratios['Gross Margin'] = (latest_financials['Gross Profit'] / latest_financials['Total Revenue'] * 100)
        else:
            ratios['Gross Margin'] = info.get('grossMargins', 0) * 100
        
        # 2. EBIT Margin
        if 'EBIT' in latest_financials.index and 'Total Revenue' in latest_financials.index:
            ratios['EBIT Margin'] = (latest_financials['EBIT'] / latest_financials['Total Revenue'] * 100)
        else:
            ratios['EBIT Margin'] = info.get('operatingMargins', 0) * 100
        
        # 3. ROE
        ratios['ROE'] = info.get('returnOnEquity', 0) * 100
        
        # 4. ROIC (approximation)
        if 'EBIT' in latest_financials.index and 'Total Debt' in latest_balance.index:
            invested_capital = latest_balance.get('Total Debt', 0) + latest_balance.get('Stockholders Equity', 0)
            if invested_capital > 0:
                ratios['ROIC'] = (latest_financials['EBIT'] * 0.79 / invested_capital * 100)  # Assuming 21% tax
            else:
                ratios['ROIC'] = 0
        else:
            ratios['ROIC'] = 0
        
        # 5. Net Debt
        total_debt = latest_balance.get('Total Debt', 0)
        cash = latest_balance.get('Cash And Cash Equivalents', 0)
        ratios['Net Debt'] = (total_debt - cash) / 1e9  # In billions
        
        # 6. Gearing Ratio (Debt/Equity)
        equity = latest_balance.get('Stockholders Equity', 1)
        ratios['Gearing'] = (total_debt / equity * 100) if equity > 0 else 0
        
        # 7. Working Capital Cycle (DIO + DSO - DPO)
        revenue = latest_financials.get('Total Revenue', 0)
        cogs = latest_financials.get('Cost Of Revenue', 0)
        inventory = latest_balance.get('Inventory', 0)
        receivables = latest_balance.get('Accounts Receivable', 0)
        payables = latest_balance.get('Accounts Payable', 0)
        
        dio = (inventory / cogs * 365) if cogs > 0 else 0
        dso = (receivables / revenue * 365) if revenue > 0 else 0
        dpo = (payables / cogs * 365) if cogs > 0 else 0
        ratios['WC Cycle'] = dio + dso - dpo
        
        # 8. Altman Z-Score
        total_assets = latest_balance.get('Total Assets', 1)
        current_assets = latest_balance.get('Current Assets', 0)
        current_liabilities = latest_balance.get('Current Liabilities', 0)
        retained_earnings = latest_balance.get('Retained Earnings', 0)
        ebit = latest_financials.get('EBIT', 0)
        market_cap = info.get('marketCap', 0)
        total_liabilities = latest_balance.get('Total Liabilities Net Minority Interest', 0)
        
        if total_assets > 0:
            x1 = (current_assets - current_liabilities) / total_assets
            x2 = retained_earnings / total_assets
            x3 = ebit / total_assets
            x4 = market_cap / total_liabilities if total_liabilities > 0 else 0
            x5 = revenue / total_assets
            
            ratios['Altman Z-Score'] = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
        else:
            ratios['Altman Z-Score'] = 0
        
        return ratios
    except Exception as e:
        st.error(f"Error calculating ratios: {str(e)}")
        return {}

def monte_carlo_simulation(current_price, volatility, days, simulations=10000):
    """Generate Monte Carlo price paths using Geometric Brownian Motion"""
    dt = 1/252  # Daily time step
    drift = 0.0  # Assume no drift for risk-neutral simulation
    
    price_paths = np.zeros((simulations, days))
    price_paths[:, 0] = current_price
    
    for t in range(1, days):
        random_shock = np.random.normal(0, 1, simulations)
        price_paths[:, t] = price_paths[:, t-1] * np.exp(
            (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * random_shock
        )
    
    return price_paths

def calculate_dcf_valuation(data, growth_rate, wacc):
    """Calculate Enterprise Value using DCF with FCFF"""
    try:
        cashflow = data['cashflow']
        balance_sheet = data['balance_sheet']
        info = data['info']
        
        latest_cf = cashflow.iloc[:, 0] if not cashflow.empty else pd.Series()
        latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
        
        # Free Cash Flow to Firm (FCFF)
        operating_cf = latest_cf.get('Operating Cash Flow', 0)
        capex = latest_cf.get('Capital Expenditure', 0)
        fcff = operating_cf + capex  # capex is negative
        
        if fcff <= 0:
            return None
        
        # Project 5 years of cash flows
        projected_fcf = []
        for year in range(1, 6):
            projected_fcf.append(fcff * (1 + growth_rate)**year)
        
        # Terminal value (perpetuity growth)
        terminal_growth = 0.025  # 2.5% perpetual growth
        terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
        
        # Discount all cash flows
        pv_fcf = sum([cf / (1 + wacc)**i for i, cf in enumerate(projected_fcf, 1)])
        pv_terminal = terminal_value / (1 + wacc)**5
        
        # Enterprise Value
        enterprise_value = pv_fcf + pv_terminal
        
        # Equity Value
        total_debt = latest_balance.get('Total Debt', 0)
        cash = latest_balance.get('Cash And Cash Equivalents', 0)
        equity_value = enterprise_value - total_debt + cash
        
        # Price per share
        shares_outstanding = info.get('sharesOutstanding', 1)
        intrinsic_value_per_share = equity_value / shares_outstanding
        
        return {
            'enterprise_value': enterprise_value,
            'equity_value': equity_value,
            'intrinsic_price': intrinsic_value_per_share,
            'fcff': fcff,
            'projected_fcf': projected_fcf,
            'terminal_value': terminal_value
        }
    except Exception as e:
        st.error(f"Error in DCF calculation: {str(e)}")
        return None

# Main Application
def main():
    st.title("📊 Risk-Adjusted Analyst")
    st.markdown("*Forward-Looking, Risk-Adjusted Valuation Platform*")
    
    # Sidebar
    with st.sidebar:
        st.header("Company Selection")
        ticker = st.text_input("Enter Ticker Symbol", value="AAPL").upper()
        
        if st.button("Analyze", type="primary"):
            st.session_state['ticker'] = ticker
            st.session_state['data'] = fetch_company_data(ticker)
    
    # Main Content
    if 'data' in st.session_state and st.session_state['data']:
        data = st.session_state['data']
        info = data['info']
        ticker = st.session_state['ticker']
        
        # Company Header
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Company", info.get('shortName', ticker))
        with col2:
            st.metric("Current Price", f"${info.get('currentPrice', 0):.2f}")
        with col3:
            st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B")
        with col4:
            st.metric("Sector", info.get('sector', 'N/A'))
        
        st.divider()
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📋 Health Scorecard", "🎲 Future Risk (Monte Carlo)", "💰 Full Valuation (DCF/EV)"])
        
        # Tab 1: Health Scorecard
        with tab1:
            st.header("Financial Health Scorecard")
            st.markdown("*8 Key Historical Ratios*")
            
            ratios = calculate_ratios(data)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Gross Margin", f"{ratios.get('Gross Margin', 0):.2f}%")
                st.metric("Net Debt", f"${ratios.get('Net Debt', 0):.2f}B")
            
            with col2:
                st.metric("EBIT Margin", f"{ratios.get('EBIT Margin', 0):.2f}%")
                st.metric("Gearing Ratio", f"{ratios.get('Gearing', 0):.2f}%")
            
            with col3:
                st.metric("ROE", f"{ratios.get('ROE', 0):.2f}%")
                st.metric("WC Cycle (Days)", f"{ratios.get('WC Cycle', 0):.0f}")
            
            with col4:
                st.metric("ROIC", f"{ratios.get('ROIC', 0):.2f}%")
                z_score = ratios.get('Altman Z-Score', 0)
                z_status = "Safe" if z_score > 2.99 else "Grey" if z_score > 1.81 else "Distress"
                st.metric("Altman Z-Score", f"{z_score:.2f}", delta=z_status)
            
            # Interpretation
            st.divider()
            st.subheader("Quick Interpretation")
            
            interpretations = []
            if ratios.get('Gross Margin', 0) > 40:
                interpretations.append("✅ Strong pricing power with high gross margins")
            if ratios.get('ROE', 0) > 15:
                interpretations.append("✅ Excellent return on equity")
            if ratios.get('Altman Z-Score', 0) > 2.99:
                interpretations.append("✅ Low bankruptcy risk")
            elif ratios.get('Altman Z-Score', 0) < 1.81:
                interpretations.append("⚠️ Elevated bankruptcy risk - caution advised")
            if ratios.get('Gearing', 0) > 100:
                interpretations.append("⚠️ High leverage - debt exceeds equity")
            
            for interp in interpretations:
                st.markdown(interp)
        
        # Tab 2: Monte Carlo Simulation
        with tab2:
            st.header("Stochastic Price Forecasting")
            st.markdown("*Monte Carlo Simulation using Geometric Brownian Motion*")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.subheader("Simulation Parameters")
                days_forward = st.slider("Forecast Horizon (Days)", 30, 365, 252)
                num_simulations = st.slider("Number of Simulations", 1000, 20000, 10000, step=1000)
                
                if st.button("Run Simulation", type="primary"):
                    with st.spinner("Running Monte Carlo simulation..."):
                        history = data['history']
                        current_price = history['Close'].iloc[-1]
                        returns = history['Close'].pct_change().dropna()
                        volatility = returns.std() * np.sqrt(252)  # Annualized
                        
                        price_paths = monte_carlo_simulation(current_price, volatility, days_forward, num_simulations)
                        
                        st.session_state['mc_results'] = {
                            'price_paths': price_paths,
                            'current_price': current_price,
                            'volatility': volatility
                        }
            
            with col1:
                if 'mc_results' in st.session_state:
                    mc = st.session_state['mc_results']
                    price_paths = mc['price_paths']
                    final_prices = price_paths[:, -1]
                    
                    # Calculate percentiles
                    p5 = np.percentile(final_prices, 5)
                    p50 = np.percentile(final_prices, 50)
                    p95 = np.percentile(final_prices, 95)
                    
                    # Plot
                    fig = go.Figure()
                    
                    # Sample paths for visualization
                    sample_paths = price_paths[np.random.choice(price_paths.shape[0], 100, replace=False)]
                    for path in sample_paths:
                        fig.add_trace(go.Scatter(
                            y=path,
                            mode='lines',
                            line=dict(color='rgba(100, 150, 200, 0.1)', width=1),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    # Percentile lines
                    fig.add_trace(go.Scatter(
                        y=[p5]*len(price_paths[0]),
                        mode='lines',
                        name='P5 (5th Percentile)',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        y=[p50]*len(price_paths[0]),
                        mode='lines',
                        name='P50 (Median)',
                        line=dict(color='yellow', width=3)
                    ))
                    fig.add_trace(go.Scatter(
                        y=[p95]*len(price_paths[0]),
                        mode='lines',
                        name='P95 (95th Percentile)',
                        line=dict(color='green', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="Monte Carlo Price Paths",
                        xaxis_title="Trading Days",
                        yaxis_title="Price ($)",
                        template="plotly_dark",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    st.subheader("Probabilistic Outcomes")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("P5 (Downside)", f"${p5:.2f}", f"{(p5/mc['current_price']-1)*100:.1f}%")
                    with col_b:
                        st.metric("P50 (Median)", f"${p50:.2f}", f"{(p50/mc['current_price']-1)*100:.1f}%")
                    with col_c:
                        st.metric("P95 (Upside)", f"${p95:.2f}", f"{(p95/mc['current_price']-1)*100:.1f}%")
                    
                    st.info(f"📊 **Volatility**: {mc['volatility']*100:.2f}% (annualized)")
                else:
                    st.info("👈 Configure parameters and run simulation")
        
        # Tab 3: DCF Valuation
        with tab3:
            st.header("Enterprise Value DCF Model")
            st.markdown("*Free Cash Flow to Firm (FCFF) Approach*")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.subheader("Model Inputs")
                st.markdown("*Adjust sliders to see real-time impact*")
                
                growth_rate = st.slider(
                    "Revenue Growth Rate",
                    min_value=-0.05,
                    max_value=0.30,
                    value=0.05,
                    step=0.01,
                    format="%.1f%%",
                    help="Expected annual growth rate for free cash flows"
                ) 
                
                wacc = st.slider(
                    "WACC (Discount Rate)",
                    min_value=0.03,
                    max_value=0.20,
                    value=0.10,
                    step=0.01,
                    format="%.1f%%",
                    help="Weighted Average Cost of Capital"
                )
                
                dcf_result = calculate_dcf_valuation(data, growth_rate, wacc)
            
            with col1:
                if dcf_result:
                    current_price = info.get('currentPrice', 0)
                    intrinsic_price = dcf_result['intrinsic_price']
                    upside = (intrinsic_price / current_price - 1) * 100 if current_price > 0 else 0
                    
                    st.subheader("Valuation Summary")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col_b:
                        st.metric("Intrinsic Value", f"${intrinsic_price:.2f}")
                    with col_c:
                        st.metric("Upside/Downside", f"{upside:+.1f}%", delta="Undervalued" if upside > 0 else "Overvalued")
                    
                    st.divider()
                    
                    # Detailed breakdown
                    col_d, col_e = st.columns(2)
                    
                    with col_d:
                        st.subheader("Enterprise Value Build-Up")
                        st.metric("Total Enterprise Value", f"${dcf_result['enterprise_value']/1e9:.2f}B")
                        st.metric("Less: Net Debt", f"${(dcf_result['enterprise_value'] - dcf_result['equity_value'])/1e9:.2f}B")
                        st.metric("Equity Value", f"${dcf_result['equity_value']/1e9:.2f}B")
                    
                    with col_e:
                        st.subheader("Cash Flow Projection")
                        fcf_df = pd.DataFrame({
                            'Year': range(1, 6),
                            'Projected FCFF': [f"${fcf/1e6:.1f}M" for fcf in dcf_result['projected_fcf']]
                        })
                        st.dataframe(fcf_df, hide_index=True, use_container_width=True)
                        st.metric("Terminal Value", f"${dcf_result['terminal_value']/1e9:.2f}B")
                    
                    # Sensitivity table
                    st.divider()
                    st.subheader("Sensitivity Analysis")
                    
                    growth_range = np.linspace(growth_rate - 0.02, growth_rate + 0.02, 5)
                    wacc_range = np.linspace(wacc - 0.02, wacc + 0.02, 5)
                    
                    sensitivity = np.zeros((5, 5))
                    for i, g in enumerate(growth_range):
                        for j, w in enumerate(wacc_range):
                            result = calculate_dcf_valuation(data, g, w)
                            if result:
                                sensitivity[i, j] = result['intrinsic_price']
                    
                    sensitivity_df = pd.DataFrame(
                        sensitivity,
                        index=[f"{g*100:.1f}%" for g in growth_range],
                        columns=[f"{w*100:.1f}%" for w in wacc_range]
                    )
                    
                    st.markdown("**Intrinsic Value at Different Growth/WACC Combinations**")
                    st.dataframe(
                        sensitivity_df.style.format("${:.2f}").background_gradient(cmap='RdYlGn'),
                        use_container_width=True
                    )
                else:
                    st.error("Unable to calculate DCF. Insufficient financial data.")
    
    else:
        st.info("👈 Enter a ticker symbol in the sidebar to begin analysis")
        
        # Example section
        st.markdown("---")
        st.subheader("Example Companies to Try")
        examples = ["AAPL", "MSFT", "GOOGL", "TSLA", "JPM", "JNJ", "DIS"]
        cols = st.columns(len(examples))
        for col, ex in zip(cols, examples):
            with col:
                if st.button(ex):
                    st.session_state['ticker'] = ex
                    st.session_state['data'] = fetch_company_data(ex)
                    st.rerun()

if __name__ == "__main__":
    main()