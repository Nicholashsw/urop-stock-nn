import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# List of symbols we generated outputs for
SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA",
    "ADBE", "AMD", "CRM", "NFLX", "AVGO", "COST", "WMT", "MCD", "DIS",
    "HD", "KO", "PEP", "JPM", "GS", "SCHW", "V", "MA", "UNH", "ABBV",
    "TMO", "PFE", "JNJ", "XOM", "CVX", "CAT", "BA", "VZ", "T", "TM",
    "MC.PA", "^GSPC", "^NDX", "^DJI", "SPY", "QQQ", "DIA",
]


DATA_DIR = Path("data/outputs")


def load_symbol_data(symbol: str) -> pd.DataFrame:
    """
    Load precomputed outputs for a symbol from data/outputs.
    Returns an empty DataFrame if the file does not exist or is invalid.
    """
    csv_path = DATA_DIR / f"{symbol}_lstm_results.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except Exception as e:
        st.error(f"Could not read CSV for {symbol}: {e}")
        return pd.DataFrame()

    # Basic sanity check
    required_cols = [
        "date",
        "close",
        "return",
        "naive_pred",
        "lstm_pred",
        "direction_true",
        "direction_lstm",
        "signal",
        "strategy_return",
        "bh_return",
        "cum_strategy",
        "cum_bh",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"{symbol}_lstm_results.csv is missing columns: {missing}")
        return pd.DataFrame()

    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return math.nan
    mse = np.mean((y_true[mask] - y_pred[mask]) ** 2)
    return float(np.sqrt(mse))


def compute_directional_accuracy(direction_true: np.ndarray,
                                 direction_pred: np.ndarray) -> float:
    mask = np.isfinite(direction_true) & np.isfinite(direction_pred)
    if mask.sum() == 0:
        return math.nan
    hits = (direction_true[mask] == direction_pred[mask]).astype(float)
    return float(hits.mean())


def compute_sharpe(returns: np.ndarray, trading_days: int = 252) -> float:
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return math.nan
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma == 0:
        return math.nan
    return float(mu / sigma * math.sqrt(trading_days))


def main():
    st.set_page_config(
        page_title="Neural Network Stock Prediction",
        layout="wide",
    )

    st.title("App for Predicting Stock Price Fluctuations with Neural Networks")

    st.markdown(
        """
This Streamlit app visualises **precomputed outputs** from a simple
neural network style pipeline.  
For each symbol we load:

- Daily prices
- Naive baseline predictions (random walk)
- A lightweight LSTM like proxy signal
- A simple long only strategy driven by the signal
        """
    )

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        symbol = st.selectbox("Select symbol", SYMBOLS, index=0)

        st.caption(
            "All results are loaded from CSV files in `data/outputs/`. "
            "No live data download and no torch is used at runtime."
        )

    df = load_symbol_data(symbol)

    if df.empty:
        st.warning(
            f"No precomputed data available for {symbol}. "
            f"Run `python scripts/generate_all_outputs.py` locally "
            f"and commit the CSVs to the repo."
        )
        return

    # Latest snapshot
    latest = df.iloc[-1]
    latest_price = latest["close"]
    latest_date = latest["date"]

    # Metrics
    rmse_naive = compute_rmse(df["return"].values, df["naive_pred"].values)
    rmse_lstm = compute_rmse(df["return"].values, df["lstm_pred"].values)

    da_lstm = compute_directional_accuracy(
        df["direction_true"].values,
        df["direction_lstm"].values,
    )

    sharpe_strategy = compute_sharpe(df["strategy_return"].values)
    sharpe_bh = compute_sharpe(df["bh_return"].values)

    cum_strategy = float(df["cum_strategy"].iloc[-1])
    cum_bh = float(df["cum_bh"].iloc[-1])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Latest close",
            f"{latest_price:,.2f}",
            help=f"As of {latest_date.date()}",
        )
    with col2:
        st.metric(
            "Cumulative return (strategy)",
            f"{(cum_strategy - 1.0) * 100:,.1f} %",
        )
    with col3:
        st.metric(
            "Cumulative return (buy and hold)",
            f"{(cum_bh - 1.0) * 100:,.1f} %",
        )

    st.markdown("### Model vs Baseline (Error and Direction)")

    col4, col5 = st.columns(2)
    with col4:
        st.write("#### RMSE on daily returns")
        st.write(f"Naive baseline: **{rmse_naive:.6f}**")
        st.write(f"LSTM like proxy: **{rmse_lstm:.6f}**")
    with col5:
        st.write("#### Directional accuracy")
        st.write(f"LSTM like proxy: **{da_lstm * 100:.2f} %**")

    st.markdown("### Price and Strategy Behaviour")

    tab1, tab2 = st.tabs(["Price and signals", "Equity curves"])

    with tab1:
        chart_df = df[["date", "close"]].copy()
        chart_df = chart_df.rename(columns={"date": "Date", "close": "Close"})
        st.line_chart(chart_df, x="Date", y="Close", height=350)

        st.caption(
            "Price series used to generate returns and model signals. "
            "Naive and LSTM like predictions are applied on returns, not prices."
        )

    with tab2:
        eq_df = df[["date", "cum_strategy", "cum_bh"]].copy()
        eq_df = eq_df.rename(
            columns={
                "date": "Date",
                "cum_strategy": "Strategy",
                "cum_bh": "Buy and hold",
            }
        )
        st.line_chart(eq_df, x="Date", y=["Strategy", "Buy and hold"], height=350)

        st.write("#### Sharpe ratios (daily returns, annualised)")
        st.write(f"Strategy Sharpe: **{sharpe_strategy:.3f}**")
        st.write(f"Buy and hold Sharpe: **{sharpe_bh:.3f}**")

    st.markdown(
        """
### Interpretation

These results are **illustrative**:

- The naive baseline is the one day random walk.
- The LSTM like proxy is a rolling average of returns.
- The long only strategy goes long when the proxy is positive.

This mirrors the structure of your UROP report:

- Compare naive vs neural style model
- Evaluate both statistical error and trading behaviour
- Show that small edges are hard and markets are noisy
        """
    )


if __name__ == "__main__":
    main()
