# Stock Analysis App with Bollinger Bands & DPO

A Streamlit-based stock analysis application that provides technical analysis using Bollinger Bands and Detrended Price Oscillator (DPO) indicators.

## Features

- **Stock Symbol Input**: Enter any valid stock ticker symbol
- **Bollinger Bands Analysis**: 
  - 20-day moving average
  - ±1 standard deviation bands
  - ±3 standard deviation bands
- **Detrended Price Oscillator (DPO)**: 20-day period analysis
- **Interactive Charts**: Synchronized charts with hover information
- **Real-time Values**: Display current band values and positions

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and go to `http://localhost:8501`

3. Enter a stock symbol (e.g., AAPL, TSLA, MSFT) in the sidebar

4. Click "Search" to fetch data and view analysis

## Technical Indicators Explained

### Bollinger Bands
- **Upper/Lower Bands**: Price channels based on standard deviation from moving average
- **1σ Bands**: Contain ~68% of price movements
- **3σ Bands**: Contain ~99.7% of price movements
- **Position %**: Shows where current price sits within the bands

### Detrended Price Oscillator (DPO)
- Removes trend to show cyclical price patterns
- Positive values: Price above its trend
- Negative values: Price below its trend
- Zero line: Trend line reference

## Charts
- **Main Chart**: Stock price, moving average, and Bollinger Bands
- **DPO Chart**: Detrended price oscillator below main chart
- **Interactive**: Hover for detailed values at any point

## Data Source
Stock data is fetched from Yahoo Finance using the `yfinance` library.