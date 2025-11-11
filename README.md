# Finance Analysis Toolkit

This is a single-file Python project that combines several finance analysis techniques in one place. It allows users to enter a stock ticker and automatically run DCF valuation, Monte Carlo simulation, technical analysis, and fetch recent financial news. The project is still in an early stage, and I am working toward making it cleaner, more structured, and open-source friendly.

I am new to open-source development, so any guidance, improvements, or pull requests are appreciated.

---

## Features

### DCF Valuation
- Estimates intrinsic value
- Includes projections and terminal value
- Uses financial data fetched from APIs

### Monte Carlo Simulation
- Generates simulated price paths
- Helps analyze uncertainty and risk

### Technical Analysis
- Includes commonly used indicators
- Generates charts and trend signals

### Financial News
- Pulls recent and relevant articles for the selected ticker

---

## How It Works

The entire project currently runs from a single Python file.  
You provide a ticker, and the program:

1. Fetches financial and market data  
2. Runs valuation calculations  
3. Generates simulations  
4. Computes technical indicators  
5. Displays everything using a Streamlit interface  

---

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
