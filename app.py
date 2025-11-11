# Risk-Adjusted Analyst — Streamlit app (cache fix + dark theme)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Risk-Adjusted Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Dark Theme Styling ----------
# Near-black background, muted borders, high-contrast text, dark cards, neon accents
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    :root {
        --bg: #0b0f19;
        --panel: #111318;
        --panel-2: #151922;
        --card: #141821;
        --border: #222733;
        --muted: #9aa3b2;
        --text: #e6e9ef;
        --accent: #60a5fa;      /* blue */
        --accent-2: #34d399;    /* green */
        --accent-3: #f59e0b;    /* amber */
        --danger: #ef4444;
        --success: #10b981;
        --info: #93c5fd;
    }
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background: var(--bg); }
    .block-container { padding-top: 1.2rem; }
    /* Header */
    .main-header {
        background: radial-gradient(1200px 600px at 20% -10%, #1a2440 0%, transparent 60%),
                    linear-gradient(135deg, #0f172a 0%, #0b0f19 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        border: 1px solid var(--border);
        margin-bottom: 2rem;
        box-shadow: 0 20px 50px rgba(0,0,0,0.35);
    }
    .main-title { font-size: 2.1rem; font-weight: 800; color: var(--text); letter-spacing: -0.5px; }
    .subtitle { font-size: 1rem; color: var(--muted); margin-top: .4rem; font-weight: 400; }

    /* Cards / Panels */
    .metric-card, .news-card, .company-header, .valuation-box, .info-card {
        background: var(--card);
        color: var(--text);
        border-radius: 12px;
        border: 1px solid var(--border);
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    }
    .metric-card { padding: 1.2rem; transition: transform .15s ease, box-shadow .15s ease; }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 14px 30px rgba(0,0,0,0.35); }

    .metric-label { font-size: .78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .6px; font-weight: 600; margin-bottom: .5rem; }
    .metric-value { font-size: 1.75rem; font-weight: 700; color: var(--text); line-height: 1.2; }

    /* Sections */
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: var(--text);
        margin: 2.2rem 0 1.1rem 0; padding-bottom: .7rem; border-bottom: 1px dashed var(--border);
    }
    .subsection-header { font-size: 1.05rem; font-weight: 600; color: var(--text); margin: 1.2rem 0 .8rem 0; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap:0; background: var(--panel); border-bottom: 1px solid var(--border); border-radius: 10px 10px 0 0; padding: .5rem; }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border: none; color: var(--muted);
        padding: .7rem 1.2rem; font-weight: 600; font-size: .95rem; border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"]:hover { background: var(--panel-2); color: var(--text); }
    .stTabs [aria-selected="true"] { background: var(--panel-2); color: var(--accent); box-shadow: inset 0 0 0 1px var(--border); }

    .company-header { padding: 1.2rem 1.5rem; margin: 1rem 0; display: flex; align-items: center; justify-content: space-between; }

    .news-card { padding: 1rem; margin-bottom: 1rem; }
    .news-title { font-size: .95rem; font-weight: 600; color: var(--text); margin-bottom: .45rem; }
    .news-meta { font-size: .75rem; color: var(--muted); margin-bottom: .65rem; }
    .news-excerpt { font-size: .88rem; color: #c7cdd9; line-height: 1.55; }

    .info-card { padding: 1rem; border-left: 3px solid var(--accent); }
    .info-card-title { font-size: .85rem; font-weight: 700; color: var(--text); margin-bottom: .4rem; }
    .info-card-text { font-size: .88rem; color: #c7cdd9; line-height: 1.6; }

    /* Valuation box */
    .valuation-box {
        background: linear-gradient(135deg, rgba(37,99,235,0.15) 0%, rgba(2,6,23,0.6) 100%);
        padding: 1.5rem; text-align: center; margin: 1rem 0; border: 1px solid #1f2a44;
    }
    .valuation-label { font-size: .8rem; color: var(--info); font-weight: 700; text-transform: uppercase; letter-spacing: .6px; }
    .valuation-price { font-size: 2.6rem; font-weight: 800; color: #cfe6ff; margin: .4rem 0; }

    /* Badges */
    .badge { display: inline-block; padding: .35rem .8rem; border-radius: 999px; font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .5px; }
    .badge-success { background: rgba(16,185,129,0.15); color: #7bf2c6; border: 1px solid rgba(16,185,129,0.35); }
    .badge-danger { background: rgba(239,68,68,0.15); color: #ffb4b4; border: 1px solid rgba(239,68,68,0.35); }
    .badge-info { background: rgba(96,165,250,0.15); color: #cfe6ff; border: 1px solid rgba(96,165,250,0.35); }

    /* Buttons */
    .stButton button {
        background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
        color: #e6e9ef; border: 1px solid var(--border);
        border-radius: 10px; padding: .6rem 1.2rem; font-weight: 700; letter-spacing: .3px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.35);
    }
    .stButton button:hover { box-shadow: 0 16px 32px rgba(0,0,0,0.45); transform: translateY(-1px); }

    /* Links */
    a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }

    h1, h2, h3, h4, h5, h6, p, span, div { color: var(--text); }
    .css-1d391kg, .st-ek { background: var(--panel) !important; }
    </style>
""", unsafe_allow_html=True)

# ---------- Currency & Country ----------
CURRENCY_MAP = {
    'US': {'symbol': '$', 'name': 'USD'},
    'CA': {'symbol': 'C$', 'name': 'CAD'},
    'GB': {'symbol': '£', 'name': 'GBP'},
    'EU': {'symbol': '€', 'name': 'EUR'},
    'JP': {'symbol': '¥', 'name': 'JPY'},
    'CN': {'symbol': '¥', 'name': 'CNY'},
    'IN': {'symbol': '₹', 'name': 'INR'},
    'AU': {'symbol': 'A$', 'name': 'AUD'},
    'HK': {'symbol': 'HK$', 'name': 'HKD'},
    'DEFAULT': {'symbol': '$', 'name': 'USD'}
}

COUNTRY_DATA = {
    'US': {'risk_free_rate': 0.045, 'market_risk_premium': 0.055, 'gdp_growth': 0.025, 'corp_tax': 0.21},
    'CA': {'risk_free_rate': 0.038, 'market_risk_premium': 0.055, 'gdp_growth': 0.022, 'corp_tax': 0.26},
    'GB': {'risk_free_rate': 0.042, 'market_risk_premium': 0.055, 'gdp_growth': 0.020, 'corp_tax': 0.25},
    'EU': {'risk_free_rate': 0.035, 'market_risk_premium': 0.055, 'gdp_growth': 0.018, 'corp_tax': 0.25},
    'JP': {'risk_free_rate': 0.005, 'market_risk_premium': 0.055, 'gdp_growth': 0.010, 'corp_tax': 0.30},
    'CN': {'risk_free_rate': 0.028, 'market_risk_premium': 0.065, 'gdp_growth': 0.045, 'corp_tax': 0.25},
    'IN': {'risk_free_rate': 0.065, 'market_risk_premium': 0.070, 'gdp_growth': 0.060, 'corp_tax': 0.30},
    'DEFAULT': {'risk_free_rate': 0.045, 'market_risk_premium': 0.055, 'gdp_growth': 0.025, 'corp_tax': 0.21}
}

def detect_currency_and_country(ticker: str):
    t = ticker.upper()
    if t.endswith('.TO') or t.endswith('.V'): return CURRENCY_MAP['CA'], 'CA'
    if t.endswith('.L'): return CURRENCY_MAP['GB'], 'GB'
    if any(t.endswith(x) for x in ['.DE', '.PA', '.AS', '.MI']): return CURRENCY_MAP['EU'], 'EU'
    if t.endswith('.T'): return CURRENCY_MAP['JP'], 'JP'
    if t.endswith('.HK'): return CURRENCY_MAP['HK'], 'HK'
    if any(t.endswith(x) for x in ['.SS', '.SZ']): return CURRENCY_MAP['CN'], 'CN'
    if any(t.endswith(x) for x in ['.NS', '.BO']): return CURRENCY_MAP['IN'], 'IN'
    if t.endswith('.AX'): return CURRENCY_MAP['AU'], 'AU'
    return CURRENCY_MAP['US'], 'US'

# ---------- Data Fetch (CACHE FIX: return only picklable types) ----------
@st.cache_data(ttl=3600)
def fetch_company_data(ticker: str):
    """Return only picklable structures for st.cache_data."""
    try:
        stock = yf.Ticker(ticker)

        # Raw info is a dict (picklable)
        info = stock.info or {}

        # Convert all DataFrames to dicts
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        history = stock.history(period="5y")
        quarterly_financials = stock.quarterly_financials
        quarterly_balance = stock.quarterly_balance_sheet

        return {
            'info': info,
            'financials': financials.to_dict() if hasattr(financials, "empty") and not financials.empty else {},
            'balance_sheet': balance_sheet.to_dict() if hasattr(balance_sheet, "empty") and not balance_sheet.empty else {},
            'cashflow': cashflow.to_dict() if hasattr(cashflow, "empty") and not cashflow.empty else {},
            'history': history.to_dict() if hasattr(history, "empty") and not history.empty else {},
            'quarterly_financials': quarterly_financials.to_dict() if hasattr(quarterly_financials, "empty") and not quarterly_financials.empty else {},
            'quarterly_balance': quarterly_balance.to_dict() if hasattr(quarterly_balance, "empty") and not quarterly_balance.empty else {},
            'ticker': ticker
            # NOTE: do NOT return stock_obj or other non-picklable objects
        }
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

@st.cache_data(ttl=1800)
def fetch_company_news(ticker: str, company_name: str):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news or []
        items = []
        for item in news[:10]:
            ts = item.get('providerPublishTime', None)
            items.append({
                'title': item.get('title', 'No title'),
                'publisher': item.get('publisher', 'Unknown'),
                'link': item.get('link', '#'),
                'timestamp': datetime.fromtimestamp(ts) if ts else datetime.now(),
                'summary': (item.get('summary') or '')[:200] + ('...' if item.get('summary') else '')
            })
        return items
    except Exception:
        return []

# ---------- Modeling Utilities ----------
def calculate_autonomous_wacc(data, country='US'):
    try:
        info = data['info']
        financials = pd.DataFrame(data['financials'])
        balance_sheet = pd.DataFrame(data['balance_sheet'])

        params = COUNTRY_DATA.get(country, COUNTRY_DATA['DEFAULT'])
        risk_free_rate = params['risk_free_rate']
        market_risk_premium = params['market_risk_premium']
        tax_rate = params['corp_tax']

        if not financials.empty:
            latest = financials.iloc[:, 0]
            pretax_income = latest.get('Pretax Income', 0)
            tax_expense = latest.get('Tax Provision', 0)
            if pretax_income and pretax_income > 0:
                eff = abs(tax_expense / pretax_income)
                tax_rate = max(min(eff, 0.35), 0.10)

        beta = info.get('beta', 1.0) or 1.0

        latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series(dtype=float)
        total_debt = latest_balance.get('Total Debt', 0) or 0.0
        market_cap = info.get('marketCap', 0) or 0.0

        cost_of_equity = risk_free_rate + (beta * market_risk_premium)

        if not financials.empty and total_debt > 0:
            interest_expense = abs(financials.iloc[:, 0].get('Interest Expense', 0) or 0.0)
            cost_of_debt = interest_expense / total_debt if total_debt > 0 else 0.04
            cost_of_debt = max(min(cost_of_debt, 0.12), 0.02)
        else:
            cost_of_debt = risk_free_rate + 0.02

        total_capital = market_cap + total_debt
        weight_equity, weight_debt = (market_cap / total_capital, total_debt / total_capital) if total_capital > 0 else (1.0, 0.0)

        wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
        wacc = max(min(wacc, 0.25), 0.04)

        return {
            'wacc': wacc,
            'cost_of_equity': cost_of_equity,
            'cost_of_debt': cost_of_debt,
            'weight_equity': weight_equity,
            'weight_debt': weight_debt,
            'beta': beta,
            'tax_rate': tax_rate,
            'risk_free_rate': risk_free_rate,
            'market_risk_premium': market_risk_premium
        }
    except Exception:
        return {
            'wacc': 0.10, 'cost_of_equity': 0.10, 'cost_of_debt': 0.04,
            'weight_equity': 1.0, 'weight_debt': 0.0, 'beta': 1.0,
            'tax_rate': 0.21, 'risk_free_rate': 0.045, 'market_risk_premium': 0.055
        }

def calculate_historical_metrics(data):
    try:
        financials = pd.DataFrame(data['financials'])
        cashflow = pd.DataFrame(data['cashflow'])

        if financials.empty:
            return {'revenue_cagr_3y': 0.05, 'revenue_cagr_5y': 0.05, 'ebitda_margin_avg': 0.20,
                    'ebitda_margin_trend': 0, 'fcf_margin_avg': 0.10, 'capex_pct_avg': 0.04, 'nwc_pct_avg': 0.10}

        # Revenue CAGR
        if 'Total Revenue' in financials.index:
            rev = financials.loc['Total Revenue'].values
            c3 = (rev[0] / rev[2]) ** (1/2) - 1 if len(rev) >= 3 and rev[2] > 0 else 0.05
            c5 = (rev[0] / rev[4]) ** (1/4) - 1 if len(rev) >= 5 and rev[4] > 0 else c3
        else:
            c3 = c5 = 0.05

        # EBITDA Margin
        if 'EBITDA' in financials.index and 'Total Revenue' in financials.index:
            margins = (financials.loc['EBITDA'] / financials.loc['Total Revenue']).values
            margins = margins[np.isfinite(margins)]
            m_avg = np.mean(margins[margins > 0]) if margins.size else 0.20
            m_trend = (margins[0] - margins[-1]) if len(margins) > 1 else 0
        else:
            m_avg, m_trend = 0.20, 0

        # OCF, CapEx, FCF margins
        if not cashflow.empty and 'Operating Cash Flow' in cashflow.index:
            ocf = cashflow.loc['Operating Cash Flow'].values
            capex = abs(cashflow.loc['Capital Expenditure'].values) if 'Capital Expenditure' in cashflow.index else ocf * 0.04
            fcf = ocf - capex
            if 'Total Revenue' in financials.index:
                rev_cf = financials.loc['Total Revenue'].values[:len(fcf)]
            else:
                rev_cf = np.maximum(fcf / 0.10, 1.0)
            fcf_margins = fcf / rev_cf
            fcf_avg = np.mean(fcf_margins[fcf_margins > 0]) if np.any(fcf_margins > 0) else 0.10
            capex_pct = np.mean((capex / rev_cf)[(capex / rev_cf) > 0]) if np.any((capex / rev_cf) > 0) else 0.04
        else:
            fcf_avg, capex_pct = 0.10, 0.04

        # Caps
        c3 = max(min(c3, 0.50), -0.20)
        c5 = max(min(c5, 0.50), -0.20)
        m_avg = max(min(m_avg, 0.60), 0.05)
        fcf_avg = max(min(fcf_avg, 0.40), 0.00)
        capex_pct = max(min(capex_pct, 0.20), 0.01)

        return {'revenue_cagr_3y': c3, 'revenue_cagr_5y': c5, 'ebitda_margin_avg': m_avg,
                'ebitda_margin_trend': m_trend, 'fcf_margin_avg': fcf_avg, 'capex_pct_avg': capex_pct, 'nwc_pct_avg': 0.10}
    except Exception:
        return {'revenue_cagr_3y': 0.05, 'revenue_cagr_5y': 0.05, 'ebitda_margin_avg': 0.20,
                'ebitda_margin_trend': 0, 'fcf_margin_avg': 0.10, 'capex_pct_avg': 0.04, 'nwc_pct_avg': 0.10}

def project_complete_financials(data, forecast_years, wacc_params, country='US'):
    try:
        hist = calculate_historical_metrics(data)
        recent_growth = hist['revenue_cagr_3y']
        long_term_growth = COUNTRY_DATA.get(country, COUNTRY_DATA['DEFAULT'])['gdp_growth']

        # Fade recent growth to GDP growth
        growth_rates = []
        for year in range(1, forecast_years + 1):
            fade = 1 - (year / (forecast_years + 5))
            yr_growth = long_term_growth + (recent_growth - long_term_growth) * fade
            growth_rates.append(max(min(yr_growth, 0.30), -0.05))

        base_ebitda = hist['ebitda_margin_avg']
        target_ebitda = max(min(base_ebitda + (hist['ebitda_margin_trend'] * 0.3), 0.50), 0.10)
        capex_pct = hist['capex_pct_avg']
        nwc_pct = hist['nwc_pct_avg']

        financials = pd.DataFrame(data['financials'])
        balance_sheet = pd.DataFrame(data['balance_sheet'])
        if financials.empty: return None

        latest_fin = financials.iloc[:, 0]
        latest_bal = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series(dtype=float)

        base_revenue = latest_fin.get('Total Revenue', 0) or 0.0
        base_da = latest_fin.get('Reconciled Depreciation', base_revenue * 0.03) or (base_revenue * 0.03)
        if base_revenue <= 0: return None

        projections = []
        prev_nwc = (latest_bal.get('Current Assets', 0) or 0) - (latest_bal.get('Current Liabilities', 0) or 0) - (latest_bal.get('Cash And Cash Equivalents', 0) or 0)

        for year in range(1, forecast_years + 1):
            cumulative_growth = np.prod([1 + g for g in growth_rates[:year]])
            revenue = base_revenue * cumulative_growth

            margin_progress = (year / forecast_years)
            ebitda_margin = base_ebitda + (target_ebitda - base_ebitda) * margin_progress
            ebitda = revenue * ebitda_margin

            da_pct = (base_da / base_revenue) if base_revenue > 0 else 0.03
            da = revenue * da_pct

            ebit = ebitda - da
            tax_rate = wacc_params['tax_rate']
            nopat = ebit * (1 - tax_rate)

            capex = revenue * capex_pct
            nwc = revenue * nwc_pct
            change_nwc = nwc - prev_nwc

            ufcf = nopat + da - capex - change_nwc
            discount_factor = 1 / ((1 + wacc_params['wacc']) ** year)
            pv_ufcf = ufcf * discount_factor

            projections.append({
                'Year': year, 'Revenue': revenue, 'Revenue_Growth': growth_rates[year-1],
                'EBITDA': ebitda, 'EBITDA_Margin': ebitda_margin,
                'D&A': da, 'EBIT': ebit, 'EBIT_Margin': (ebit / revenue) if revenue > 0 else 0.0,
                'Tax': ebit * tax_rate, 'NOPAT': nopat, 'D&A_Add_Back': da,
                'CapEx': capex, 'CapEx_Pct': capex_pct, 'NWC': nwc, 'Change_NWC': change_nwc,
                'UFCF': ufcf, 'Discount_Factor': discount_factor, 'PV_UFCF': pv_ufcf
            })
            prev_nwc = nwc

        return pd.DataFrame(projections)
    except Exception as e:
        st.error(f"Projection error: {e}")
        return None

def calculate_autonomous_dcf(data, country='US', currency_symbol='$'):
    try:
        wacc_params = calculate_autonomous_wacc(data, country)
        terminal_growth = COUNTRY_DATA.get(country, COUNTRY_DATA['DEFAULT'])['gdp_growth']
        projections = project_complete_financials(data, 10, wacc_params, country)
        if projections is None or projections.empty: return None

        final_ufcf = projections['UFCF'].iloc[-1]
        wacc = wacc_params['wacc']

        terminal_value = (final_ufcf * (1 + terminal_growth)) / (wacc - terminal_growth) if (final_ufcf > 0 and wacc > terminal_growth) else 0.0
        pv_terminal = terminal_value / ((1 + wacc) ** 10)
        total_pv_ufcf = projections['PV_UFCF'].sum()
        enterprise_value = total_pv_ufcf + pv_terminal

        balance_sheet = pd.DataFrame(data['balance_sheet'])
        latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series(dtype=float)

        cash = latest_balance.get('Cash And Cash Equivalents', 0) or 0.0
        debt = latest_balance.get('Total Debt', 0) or 0.0
        minority_interest = latest_balance.get('Minority Interest', 0) or 0.0
        preferred_stock = latest_balance.get('Preferred Stock', 0) or 0.0
        investments = latest_balance.get('Investments And Advances', 0) or 0.0

        equity_value = enterprise_value + cash + investments - debt - minority_interest - preferred_stock
        shares = data['info'].get('sharesOutstanding', 1) or 1
        intrinsic_price = (equity_value / shares) if shares > 0 else 0.0

        return {
            'projections': projections, 'wacc_params': wacc_params, 'terminal_growth': terminal_growth,
            'terminal_value': terminal_value, 'pv_terminal': pv_terminal, 'total_pv_ufcf': total_pv_ufcf,
            'enterprise_value': enterprise_value, 'equity_value': equity_value, 'intrinsic_price': intrinsic_price,
            'adjustments': {'cash': cash, 'debt': debt, 'minority_interest': minority_interest,
                            'preferred_stock': preferred_stock, 'investments': investments},
            'shares': shares
        }
    except Exception as e:
        st.error(f"DCF calculation error: {e}")
        return None

# ---------- UI ----------
def main():
    # Header
    st.markdown("""
        <div class='main-header'>
            <div class='main-title'>Risk-Adjusted Analyst</div>
            <div class='subtitle'>Professional Investment Research & Valuation Platform</div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🔍 Company Search")
        ticker = st.text_input("Enter Ticker Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL").upper()
        if st.button("Analyze Company", type="primary", use_container_width=True):
            with st.spinner("Fetching comprehensive data..."):
                st.session_state['ticker'] = ticker
                st.session_state['data'] = fetch_company_data(ticker)
                currency, country = detect_currency_and_country(ticker)
                st.session_state['currency'] = currency
                st.session_state['country'] = country

        st.markdown("---")
        st.markdown("### 📊 Popular Companies")
        popular = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA"],
            "Finance": ["JPM", "BAC", "V", "MA", "BRK.B", "GS"],
            "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "LLY", "MRK"],
            "Consumer": ["AMZN", "WMT", "HD", "NKE", "COST", "DIS"]
        }
        for sector, tickers in popular.items():
            with st.expander(sector):
                cols = st.columns(3)
                for i, tick in enumerate(tickers):
                    if cols[i % 3].button(tick, key=f"pop_{tick}", use_container_width=True):
                        st.session_state['ticker'] = tick
                        st.session_state['data'] = fetch_company_data(tick)
                        currency, country = detect_currency_and_country(tick)
                        st.session_state['currency'] = currency
                        st.session_state['country'] = country
                        st.rerun()

    # Main content
    if 'data' in st.session_state and st.session_state['data']:
        data = st.session_state['data']
        info = data['info']
        ticker = st.session_state['ticker']
        history = pd.DataFrame(data['history'])
        currency = st.session_state.get('currency', CURRENCY_MAP['US'])
        country = st.session_state.get('country', 'US')
        curr_symbol = currency['symbol']

        current_price = info.get('currentPrice', 0) or 0.0
        prev_close = info.get('previousClose', current_price) or current_price
        price_change = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

        st.markdown(f"""
            <div class='company-header'>
                <div>
                    <div class='company-name' style='font-size:1.4rem;font-weight:700;color:#e6e9ef;'>{info.get('shortName', ticker)}</div>
                    <div class='company-meta' style='color:#9aa3b2;'>{ticker} | {info.get('sector', 'N/A')} | {info.get('industry', 'N/A')}</div>
                </div>
                <div style='text-align: right;'>
                    <div style='font-size: 1.8rem; font-weight: 800; color: #e6e9ef;'>{curr_symbol}{current_price:.2f}</div>
                    <div style='font-size: 1rem; font-weight: 700; color: {"#34d399" if price_change >= 0 else "#ef4444"};'>{price_change:+.2f}%</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            mc = (info.get('marketCap', 0) or 0) / 1e9
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Market Cap</div><div class='metric-value'>{curr_symbol}{mc:.2f}B</div></div>", unsafe_allow_html=True)
        with col2:
            pe = info.get('trailingPE', 0) or 0.0
            st.markdown(f"<div class='metric-card'><div class='metric-label'>P/E Ratio</div><div class='metric-value'>{pe:.2f}</div></div>", unsafe_allow_html=True)
        with col3:
            ev_ebitda = info.get('enterpriseToEbitda', 0) or 0.0
            st.markdown(f"<div class='metric-card'><div class='metric-label'>EV/EBITDA</div><div class='metric-value'>{ev_ebitda:.2f}</div></div>", unsafe_allow_html=True)
        with col4:
            divy = (info.get('dividendYield', 0) or 0.0) * 100
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Div Yield</div><div class='metric-value'>{divy:.2f}%</div></div>", unsafe_allow_html=True)
        with col5:
            beta = info.get('beta', 0) or 0.0
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Beta</div><div class='metric-value'>{beta:.2f}</div></div>", unsafe_allow_html=True)

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview & Valuation", "📊 Detailed DCF Model", "📰 News & Analysis", "💹 Technical Analysis"])

        # Tab 1
        with tab1:
            dcf_result = calculate_autonomous_dcf(data, country, curr_symbol)
            if dcf_result:
                intrinsic_price = dcf_result['intrinsic_price']
                upside = (intrinsic_price / current_price - 1) * 100 if current_price > 0 else 0

                col_val1, col_val2 = st.columns([1, 2])
                with col_val1:
                    upside_color = "#34d399" if upside > 0 else "#ef4444"
                    status = "UNDERVALUED" if upside > 0 else "OVERVALUED"
                    badge_class = "badge-success" if upside > 0 else "badge-danger"
                    st.markdown(f"""
                        <div class='valuation-box'>
                            <div class='valuation-label'>Intrinsic Value</div>
                            <div class='valuation-price'>{curr_symbol}{intrinsic_price:.2f}</div>
                            <div style='font-size: 1.05rem; font-weight: 800; color: {upside_color}; margin: .5rem 0;'>{upside:+.1f}% vs Current</div>
                            <span class='badge {badge_class}'>{status}</span>
                        </div>
                    """, unsafe_allow_html=True)

                    w = dcf_result['wacc_params']
                    st.markdown("<div class='info-card'><div class='info-card-title'>WACC Components</div><div class='info-card-text'>", unsafe_allow_html=True)
                    st.markdown(f"**WACC:** {w['wacc']*100:.2f}%")
                    st.markdown(f"**Cost of Equity:** {w['cost_of_equity']*100:.2f}%")
                    st.markdown(f"**Cost of Debt:** {w['cost_of_debt']*100:.2f}%")
                    st.markdown(f"**Tax Rate:** {w['tax_rate']*100:.1f}%")
                    st.markdown(f"**Beta:** {w['beta']:.2f}")
                    st.markdown("</div></div>", unsafe_allow_html=True)

                with col_val2:
                    st.markdown("<div class='subsection-header'>Enterprise Value to Equity Bridge</div>", unsafe_allow_html=True)
                    bridge_data = pd.DataFrame({
                        'Component': [
                            'PV of Forecast UFCF (10Y)', 'PV of Terminal Value', 'Enterprise Value',
                            'Plus: Cash & Equivalents', 'Plus: Investments', 'Less: Total Debt',
                            'Less: Minority Interest', 'Less: Preferred Stock', 'Equity Value',
                            'Shares Outstanding (M)', 'Intrinsic Value per Share'
                        ],
                        'Value': [
                            dcf_result['total_pv_ufcf'], dcf_result['pv_terminal'], dcf_result['enterprise_value'],
                            dcf_result['adjustments']['cash'], dcf_result['adjustments']['investments'],
                            -dcf_result['adjustments']['debt'], -dcf_result['adjustments']['minority_interest'],
                            -dcf_result['adjustments']['preferred_stock'], dcf_result['equity_value'],
                            dcf_result['shares'] / 1e6, intrinsic_price
                        ]
                    })
                    bridge_data['Formatted'] = bridge_data['Value'].apply(
                        lambda x: f"{curr_symbol}{x/1e9:.2f}B" if abs(x) > 1e9 else (f"{curr_symbol}{x/1e6:.1f}M" if abs(x) > 1e6 else f"{curr_symbol}{x:.2f}")
                    )
                    st.dataframe(bridge_data[['Component', 'Formatted']].rename(columns={'Formatted': 'Value'}),
                                 hide_index=True, use_container_width=True, height=420)

                # Projections summary
                st.markdown("<div class='section-header'>10-Year Financial Projections</div>", unsafe_allow_html=True)
                proj = dcf_result['projections'].copy()
                summary_proj = proj[['Year', 'Revenue', 'EBITDA', 'EBITDA_Margin', 'NOPAT', 'CapEx', 'UFCF', 'PV_UFCF']].copy()
                for col in ['Revenue', 'EBITDA', 'NOPAT', 'CapEx', 'UFCF', 'PV_UFCF']:
                    summary_proj[col] = summary_proj[col] / 1e9
                summary_proj['EBITDA_Margin'] *= 100
                summary_proj.columns = ['Year', f'Revenue ({curr_symbol}B)', f'EBITDA ({curr_symbol}B)', 'EBITDA %',
                                        f'NOPAT ({curr_symbol}B)', f'CapEx ({curr_symbol}B)', f'UFCF ({curr_symbol}B)', f'PV UFCF ({curr_symbol}B)']
                st.dataframe(summary_proj.style.format({
                    f'Revenue ({curr_symbol}B)': '{:.2f}', f'EBITDA ({curr_symbol}B)': '{:.2f}',
                    'EBITDA %': '{:.1f}%', f'NOPAT ({curr_symbol}B)': '{:.2f}',
                    f'CapEx ({curr_symbol}B)': '{:.2f}', f'UFCF ({curr_symbol}B)': '{:.2f}', f'PV UFCF ({curr_symbol}B)': '{:.2f}'
                }), hide_index=True, use_container_width=True)

                # Charts (dark)
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    fig1 = go.Figure()
                    fig1.add_trace(go.Bar(x=proj['Year'], y=proj['Revenue'] / 1e9, name='Revenue'))
                    fig1.add_trace(go.Bar(x=proj['Year'], y=proj['EBITDA'] / 1e9, name='EBITDA'))
                    fig1.update_layout(title='Revenue & EBITDA Projection', xaxis_title='Year',
                                       yaxis_title=f'Value ({curr_symbol}B)', barmode='group', height=350, template='plotly_dark')
                    st.plotly_chart(fig1, use_container_width=True)
                with col_chart2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=proj['Year'], y=proj['UFCF'] / 1e9, name='UFCF', mode='lines+markers'))
                    fig2.add_trace(go.Scatter(x=proj['Year'], y=proj['PV_UFCF'] / 1e9, name='PV of UFCF', mode='lines+markers'))
                    fig2.update_layout(title='Unlevered Free Cash Flow', xaxis_title='Year',
                                       yaxis_title=f'Cash Flow ({curr_symbol}B)', height=350, template='plotly_dark')
                    st.plotly_chart(fig2, use_container_width=True)

                # Insights
                st.markdown("<div class='section-header'>Valuation Insights</div>", unsafe_allow_html=True)
                col_ins1, col_ins2, col_ins3 = st.columns(3)
                tv_pct = (dcf_result['pv_terminal'] / dcf_result['enterprise_value'] * 100) if dcf_result['enterprise_value'] else 0
                with col_ins1:
                    st.markdown(f"""
                        <div class='info-card'><div class='info-card-title'>Terminal Value</div>
                        <div class='info-card-text'>Terminal value represents <strong>{tv_pct:.1f}%</strong> of EV.
                        Perpetuity growth: <strong>{dcf_result['terminal_growth']*100:.2f}%</strong>.</div></div>
                    """, unsafe_allow_html=True)
                with col_ins2:
                    avg_growth = proj['Revenue_Growth'].mean() * 100
                    st.markdown(f"<div class='info-card'><div class='info-card-title'>Growth Assumptions</div><div class='info-card-text'>Average revenue growth <strong>{avg_growth:.1f}%</strong> over 10 years, fading to GDP.</div></div>", unsafe_allow_html=True)
                with col_ins3:
                    avg_margin = proj['EBITDA_Margin'].mean() * 100
                    st.markdown(f"<div class='info-card'><div class='info-card-title'>Margin Profile</div><div class='info-card-text'>Average EBITDA margin <strong>{avg_margin:.1f}%</strong> projected.</div></div>", unsafe_allow_html=True)
            else:
                st.error("Unable to calculate autonomous DCF valuation. Insufficient financial data.")

        # Tab 2
        with tab2:
            dcf_result = calculate_autonomous_dcf(data, country, curr_symbol)
            if dcf_result:
                st.markdown("<div class='section-header'>Complete Financial Model</div>", unsafe_allow_html=True)
                full = dcf_result['projections'].copy()

                # to millions
                for col in ['Revenue','EBITDA','D&A','EBIT','Tax','NOPAT','D&A_Add_Back','CapEx','NWC','Change_NWC','UFCF','PV_UFCF']:
                    if col in full.columns: full[col] = full[col] / 1e6
                for col in ['Revenue_Growth','EBITDA_Margin','EBIT_Margin','CapEx_Pct']:
                    if col in full.columns: full[col] = full[col] * 100

                st.markdown("### Income Statement Projection")
                inc_cols = ['Year','Revenue','Revenue_Growth','EBITDA','EBITDA_Margin','D&A','EBIT','EBIT_Margin','Tax','NOPAT']
                inc = full[inc_cols].copy()
                inc.columns = ['Year', f'Revenue ({curr_symbol}M)', 'Growth %', f'EBITDA ({curr_symbol}M)', 'EBITDA %',
                               f'D&A ({curr_symbol}M)', f'EBIT ({curr_symbol}M)', 'EBIT %', f'Tax ({curr_symbol}M)', f'NOPAT ({curr_symbol}M)']
                st.dataframe(inc.style.format({
                    f'Revenue ({curr_symbol}M)': '{:.1f}', 'Growth %': '{:.1f}%', f'EBITDA ({curr_symbol}M)': '{:.1f}',
                    'EBITDA %': '{:.1f}%', f'D&A ({curr_symbol}M)': '{:.1f}', f'EBIT ({curr_symbol}M)': '{:.1f}',
                    'EBIT %': '{:.1f}%', f'Tax ({curr_symbol}M)': '{:.1f}', f'NOPAT ({curr_symbol}M)': '{:.1f}'
                }), hide_index=True, use_container_width=True)

                st.markdown("### Free Cash Flow Calculation")
                fcf_cols = ['Year','NOPAT','D&A_Add_Back','CapEx','CapEx_Pct','Change_NWC','UFCF']
                fcf = full[fcf_cols].copy()
                fcf.columns = ['Year', f'NOPAT ({curr_symbol}M)', f'+ D&A ({curr_symbol}M)', f'- CapEx ({curr_symbol}M)',
                               'CapEx %', f'- ∆NWC ({curr_symbol}M)', f'= UFCF ({curr_symbol}M)']
                st.dataframe(fcf.style.format({
                    f'NOPAT ({curr_symbol}M)': '{:.1f}', f'+ D&A ({curr_symbol}M)': '{:.1f}',
                    f'- CapEx ({curr_symbol}M)': '{:.1f}', 'CapEx %': '{:.1f}%',
                    f'- ∆NWC ({curr_symbol}M)': '{:.1f}', f'= UFCF ({curr_symbol}M)': '{:.1f}'
                }), hide_index=True, use_container_width=True)

                st.markdown("### Discounted Cash Flow Analysis")
                dcf_cols = ['Year','UFCF','Discount_Factor','PV_UFCF']
                dcf_display = full[dcf_cols].copy()
                dcf_display.columns = ['Year', f'UFCF ({curr_symbol}M)', 'Discount Factor', f'PV UFCF ({curr_symbol}M)']
                st.dataframe(dcf_display.style.format({
                    f'UFCF ({curr_symbol}M)': '{:.1f}', 'Discount Factor': '{:.4f}', f'PV UFCF ({curr_symbol}M)': '{:.1f}'
                }), hide_index=True, use_container_width=True)

                st.markdown("<div class='section-header'>Model Assumptions</div>", unsafe_allow_html=True)
                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    st.markdown("<div class='info-card'><div class='info-card-title'>Revenue Assumptions</div></div>", unsafe_allow_html=True)
                    st.markdown(f"**Year 1 Growth:** {full['Revenue_Growth'].iloc[0]:.2f}%")
                    st.markdown(f"**Year 10 Growth:** {full['Revenue_Growth'].iloc[-1]:.2f}%")
                    st.markdown(f"**Average Growth:** {full['Revenue_Growth'].mean():.2f}%")
                with col_a2:
                    st.markdown("<div class='info-card'><div class='info-card-title'>Margin Assumptions</div></div>", unsafe_allow_html=True)
                    st.markdown(f"**Year 1 EBITDA Margin:** {full['EBITDA_Margin'].iloc[0]:.2f}%")
                    st.markdown(f"**Year 10 EBITDA Margin:** {full['EBITDA_Margin'].iloc[-1]:.2f}%")
                    st.markdown(f"**Average CapEx %:** {full['CapEx_Pct'].mean():.2f}%")
                with col_a3:
                    w = dcf_result['wacc_params']
                    st.markdown("<div class='info-card'><div class='info-card-title'>Discount Rate</div></div>", unsafe_allow_html=True)
                    st.markdown(f"**WACC:** {w['wacc']*100:.2f}%")
                    st.markdown(f"**Terminal Growth:** {dcf_result['terminal_growth']*100:.2f}%")
                    st.markdown(f"**Forecast Period:** 10 years")

        # Tab 3
        with tab3:
            st.markdown("<div class='section-header'>Latest News & Company Updates</div>", unsafe_allow_html=True)
            company_name = info.get('shortName', ticker)
            news_items = fetch_company_news(ticker, company_name)
            if news_items:
                for item in news_items:
                    st.markdown(f"""
                        <div class='news-card'>
                            <div class='news-title'>{item['title']}</div>
                            <div class='news-meta'>{item['publisher']} • {item['timestamp'].strftime('%b %d, %Y %I:%M %p')}</div>
                            <div class='news-excerpt'>{item['summary']}</div>
                            <a href="{item['link']}" target="_blank">Read more →</a>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent news available for this company.")

            st.markdown("<div class='section-header'>Company Profile</div>", unsafe_allow_html=True)
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("<div class='info-card'><div class='info-card-title'>Business Description</div><div class='info-card-text'>", unsafe_allow_html=True)
                st.write(info.get('longBusinessSummary', 'No description available.'))
                st.markdown("</div></div>", unsafe_allow_html=True)
            with col_p2:
                st.markdown("<div class='info-card'><div class='info-card-title'>Company Information</div><div class='info-card-text'>", unsafe_allow_html=True)
                st.markdown(f"**Country:** {info.get('country', 'N/A')}")
                site = info.get('website', 'N/A')
                st.markdown(f"**Website:** [{site}]({site})" if site and site != 'N/A' else "**Website:** N/A")
                employees = info.get('fullTimeEmployees', None)
                st.markdown(f"**Employees:** {employees:,}" if isinstance(employees, int) else "**Employees:** N/A")
                st.markdown(f"**Founded:** {info.get('founded', 'N/A')}")
                st.markdown("</div></div>", unsafe_allow_html=True)

        # Tab 4
        with tab4:
            st.markdown("<div class='section-header'>Price Chart & Technical Indicators</div>", unsafe_allow_html=True)
            if not history.empty:
                hist_df = pd.DataFrame(history)
                try: hist_df.index = pd.to_datetime(hist_df.index)
                except Exception: pass

                if 'Close' in hist_df.columns:
                    hist_df['SMA_50'] = hist_df['Close'].rolling(50).mean()
                    hist_df['SMA_200'] = hist_df['Close'].rolling(200).mean()

                    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                                        subplot_titles=('Price & Moving Averages', 'Volume'),
                                        vertical_spacing=0.1)

                    if all(k in hist_df.columns for k in ['Open','High','Low','Close']):
                        fig.add_trace(go.Candlestick(
                            x=hist_df.index, open=hist_df['Open'], high=hist_df['High'],
                            low=hist_df['Low'], close=hist_df['Close'], name='Price'
                        ), row=1, col=1)

                    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_50'], name='50-Day MA'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_200'], name='200-Day MA'), row=1, col=1)

                    if 'Volume' in hist_df.columns:
                        fig.add_trace(go.Bar(x=hist_df.index, y=hist_df['Volume'], name='Volume', showlegend=False), row=2, col=1)

                    fig.update_layout(height=600, template='plotly_dark', xaxis_rangeslider_visible=False, hovermode='x unified')
                    fig.update_xaxes(title_text="Date", row=2, col=1)
                    fig.update_yaxes(title_text=f"Price ({curr_symbol})", row=1, col=1)
                    fig.update_yaxes(title_text="Volume", row=2, col=1)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("<div class='subsection-header'>Technical Indicators</div>", unsafe_allow_html=True)
                    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                    current = float(hist_df['Close'].iloc[-1])
                    sma50 = float(hist_df['SMA_50'].iloc[-1]) if not np.isnan(hist_df['SMA_50'].iloc[-1]) else current
                    sma200 = float(hist_df['SMA_200'].iloc[-1]) if not np.isnan(hist_df['SMA_200'].iloc[-1]) else current

                    with col_t1:
                        trend = "Bullish" if current > sma50 > sma200 else "Bearish" if current < sma50 < sma200 else "Neutral"
                        badge_class = "badge-success" if trend == "Bullish" else "badge-danger" if trend == "Bearish" else "badge-info"
                        st.markdown(f"<div class='metric-card' style='text-align:center;'><div class='metric-label'>Trend</div><span class='badge {badge_class}'>{trend}</span></div>", unsafe_allow_html=True)
                    with col_t2:
                        vs_50 = (current / sma50 - 1) * 100 if sma50 else 0.0
                        st.markdown(f"<div class='metric-card'><div class='metric-label'>vs 50-Day MA</div><div class='metric-value' style='font-size:1.4rem;color:{'#34d399' if vs_50>0 else '#ef4444'};'>{vs_50:+.2f}%</div></div>", unsafe_allow_html=True)
                    with col_t3:
                        vs_200 = (current / sma200 - 1) * 100 if sma200 else 0.0
                        st.markdown(f"<div class='metric-card'><div class='metric-label'>vs 200-Day MA</div><div class='metric-value' style='font-size:1.4rem;color:{'#34d399' if vs_200>0 else '#ef4444'};'>{vs_200:+.2f}%</div></div>", unsafe_allow_html=True)
                    with col_t4:
                        returns = hist_df['Close'].pct_change().dropna()
                        volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else 0.0
                        st.markdown(f"<div class='metric-card'><div class='metric-label'>Volatility (Ann.)</div><div class='metric-value' style='font-size:1.4rem;'>{volatility:.1f}%</div></div>", unsafe_allow_html=True)
            else:
                st.info("No price history available.")
    else:
        # Landing page (dark)
        st.markdown("""
            <div style='text-align:center; padding: 4rem 2rem;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>📊</div>
                <h2 style='color: #e6e9ef; font-size: 2.3rem; margin-bottom: 1rem; font-weight: 800;'>
                    Welcome to Risk-Adjusted Analyst
                </h2>
                <p style='color: #9aa3b2; font-size: 1.15rem; max-width: 720px; margin: 0 auto 3rem auto; line-height: 1.7;'>
                    Professional investment research platform with autonomous DCF valuation,
                    comprehensive financial modeling, and real-time market analysis.
                </p>
            </div>
        """, unsafe_allow_html=True)

        col_feat1, col_feat2, col_feat3 = st.columns(3)
        with col_feat1:
            st.markdown("""
                <div class='metric-card' style='text-align:center; padding: 2rem;'>
                    <div style='font-size: 2.2rem; margin-bottom: 1rem;'>🤖</div>
                    <div style='font-size: 1.1rem; font-weight: 700; color: #e6e9ef; margin-bottom: .6rem;'>
                        Autonomous DCF
                    </div>
                    <div style='font-size: .95rem; color: #c7cdd9; line-height: 1.6;'>
                        Automated 10-year projections, intelligent growth assumptions, WACC
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col_feat2:
            st.markdown("""
                <div class='metric-card' style='text-align:center; padding: 2rem;'>
                    <div style='font-size: 2.2rem; margin-bottom: 1rem;'>📈</div>
                    <div style='font-size: 1.1rem; font-weight: 700; color: #e6e9ef; margin-bottom: .6rem;'>
                        Detailed Modeling
                    </div>
                    <div style='font-size: .95rem; color: #c7cdd9; line-height: 1.6;'>
                        Income, cash flow, and balance sheet projections with drivers
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col_feat3:
            st.markdown("""
                <div class='metric-card' style='text-align:center; padding: 2rem;'>
                    <div style='font-size: 2.2rem; margin-bottom: 1rem;'>📰</div>
                    <div style='font-size: 1.1rem; font-weight: 700; color: #e6e9ef; margin-bottom: .6rem;'>
                        Real-Time News
                    </div>
                    <div style='font-size: .95rem; color: #c7cdd9; line-height: 1.6;'>
                        Latest headlines and company profile
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            <div class='info-card' style='text-align:center; padding: 1.4rem; margin-top: 1.5rem;'>
                <div style='font-size: 1.2rem; font-weight: 800; color: #e6e9ef; margin-bottom: .6rem;'>
                    Get Started
                </div>
                <div style='font-size: .98rem; color: #c7cdd9;'>
                    Enter a ticker symbol in the sidebar or select from popular companies
                </div>
            </div>
        """, unsafe_allow_html=True)

        popular_showcase = [
            ("AAPL", "Apple Inc.", "Technology"),
            ("MSFT", "Microsoft Corp.", "Technology"),
            ("GOOGL", "Alphabet Inc.", "Technology"),
            ("AMZN", "Amazon.com", "Consumer"),
            ("TSLA", "Tesla Inc.", "Automotive"),
            ("JPM", "JPMorgan Chase", "Finance"),
            ("V", "Visa Inc.", "Finance"),
            ("JNJ", "Johnson & Johnson", "Healthcare")
        ]
        cols = st.columns(4)
        for i, (tick, name, sector) in enumerate(popular_showcase):
            with cols[i % 4]:
                if st.button(f"**{tick}**\n{name}\n*{sector}*", key=f"land_{tick}", use_container_width=True):
                    st.session_state['ticker'] = tick
                    st.session_state['data'] = fetch_company_data(tick)
                    currency, country = detect_currency_and_country(tick)
                    st.session_state['currency'] = currency
                    st.session_state['country'] = country
                    st.rerun()

if __name__ == "__main__":
    main()
