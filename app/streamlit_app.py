import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="Neural Network Stock Predictor",
    layout="wide",
)

DATA_DIR = "data/outputs"

AVAILABLE_SYMBOLS = sorted([
    f.replace("_lstm_results.csv", "")
    for f in os.listdir(DATA_DIR)
    if f.endswith("_lstm_results.csv")
])

SECTOR_TAGS = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AMZN": "Consumer Discretionary",
    "META": "Communication Services",
    "GOOGL": "Communication Services",
    "GOOG": "Communication Services",
    "TSLA": "Automotive",
    "ADBE": "Technology",
    "AMD": "Technology",
    "CRM": "Technology",
    "NFLX": "Communication Services",
    "AVGO": "Technology",
    "COST": "Consumer Staples",
    "WMT": "Consumer Staples",
    "MCD": "Consumer Discretionary",
    "DIS": "Communication Services",
    "HD": "Consumer Discretionary",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "JPM": "Financials",
    "GS": "Financials",
    "SCHW": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "UNH": "Healthcare",
    "ABBV": "Healthcare",
    "TMO": "Healthcare",
    "PFE": "Healthcare",
    "JNJ": "Healthcare",
    "XOM": "Energy",
    "CVX": "Energy",
    "CAT": "Industrials",
    "BA": "Industrials",
    "VZ": "Communication Services",
    "T": "Communication Services",
    "TM": "Automotive",
    "MC.PA": "Consumer Luxury",
    "^GSPC": "Index",
    "^NDX": "Index",
    "^DJI": "Index",
    "SPY": "ETF",
    "QQQ": "ETF",
    "DIA": "ETF",
}

def load_symbol(symbol):
    path = os.path.join(DATA_DIR, f"{symbol}_lstm_results.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

st.title("📈 Neural Network Stock Predictor (Cached Results Only)")

symbol = st.selectbox("Select a symbol", AVAILABLE_SYMBOLS)

df = load_symbol(symbol)

if df is None or df.empty:
    st.error(f"No saved prediction data found for {symbol}.")
    st.stop()

st.success(f"Loaded {len(df)} rows for {symbol}.")
st.caption(f"Sector: **{SECTOR_TAGS.get(symbol, 'Unknown')}**")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 True Close Price")
    st.line_chart(df["Close"])

with col2:
    st.subheader("🔮 LSTM Predicted Price")
    st.line_chart(df["Predicted"])

df["Error"] = df["Predicted"] - df["Close"]

st.subheader("📊 Model Error Distribution")
st.line_chart(df["Error"])

st.subheader("📃 Latest Predictions")
st.dataframe(df.tail(20))

