# app/streamlit_app.py

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Directory where precomputed CSVs live, relative to repo root
OUTPUT_DIR = Path("data") / "outputs"

# Universe of symbols supported by the app
SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA",
    "ADBE", "AMD", "CRM", "NFLX", "AVGO", "COST", "WMT", "MCD", "DIS",
    "HD", "KO", "PEP", "JPM", "GS", "SCHW", "V", "MA", "UNH", "ABBV",
    "TMO", "PFE", "JNJ", "XOM", "CVX", "CAT", "BA", "VZ", "T", "TM",
    "MC.PA", "^GSPC", "^NDX", "^DJI", "SPY", "QQQ", "DIA",
]


# ---------- Data loading utilities ----------

@st.cache_data(show_spinner=False)
def load_symbol(symbol: str) -> pd.DataFrame | None:
    """
    Load precomputed CSV for a given symbol.

    Returns:
        DataFrame with at least Date and Close columns,
        or None if file is missing or invalid.
    """
    filepath = OUTPUT_DIR / f"{symbol}_lstm_results.csv"

    if not filepath.exists():
        st.warning(f"No CSV file found for {symbol}. Expected at: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        st.error(f"Failed to read CSV for {symbol}: {e}")
        return None

    if df is None or df.empty:
        st.warning(f"CSV for {symbol} is empty. Skipping.")
        return None

    # Try to normalise the date column
    date_col_candidates = ["Date", "date", "Timestamp", "timestamp"]
    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        # If no explicit date, try using index
        if df.index.name is not None:
            df = df.reset_index()
            if df.columns[0].lower().startswith("date"):
                date_col = df.columns[0]
            else:
                st.warning(f"{symbol}: could not locate a Date column. Using row index as proxy.")
                df["Date"] = np.arange(len(df))
                date_col = "Date"
        else:
            st.warning(f"{symbol}: no Date column found. Using row index as proxy.")
            df["Date"] = np.arange(len(df))
            date_col = "Date"

    # Ensure Date is datetime when possible
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception:
        # If conversion fails, keep as is
        pass

    # Normalise to a standard "Date" column
    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})

    # Try to locate a price column
    price_col = None
    for c in ["Close", "close", "Adj Close", "Adj_Close", "adj_close"]:
        if c in df.columns:
            price_col = c
            break

    if price_col is None:
        # Sometimes yfinance multi index can end up flattened like "Close.AAPL"
        for c in df.columns:
            if "close" in c.lower():
                price_col = c
                break

    if price_col is None:
        st.error(f"{symbol}: no close price column found in CSV. Columns: {list(df.columns)}")
        return None

    # Rename price column to Close for plotting
    if price_col != "Close":
        df = df.rename(columns={price_col: "Close"})

    # Drop rows with missing Close
    df = df.dropna(subset=["Close"])
    if df.empty:
        st.warning(f"{symbol}: all Close values are NaN after cleaning. Skipping.")
        return None

    # Sort by Date just in case
    df = df.sort_values("Date").reset_index(drop=True)

    # Create a simple "prediction" series if not already present
    if "Predicted_Close" not in df.columns:
        # Use previous day Close as a naive forecast
        df["Predicted_Close"] = df["Close"].shift(1)
        # First row has no prediction so we drop it to keep arrays aligned
        df = df.dropna(subset=["Predicted_Close"]).reset_index(drop=True)

    return df


def compute_basic_metrics(df: pd.DataFrame) -> dict:
    """
    Compute simple metrics from actual and predicted close prices.
    """
    actual = df["Close"].values
    pred = df["Predicted_Close"].values

    # Align lengths if needed
    n = min(len(actual), len(pred))
    actual = actual[-n:]
    pred = pred[-n:]

    # RMSE
    rmse = np.sqrt(np.mean((actual - pred) ** 2))

    # Directional accuracy
    actual_ret = np.sign(np.diff(actual))
    pred_ret = np.sign(np.diff(pred))
    m = min(len(actual_ret), len(pred_ret))
    if m > 0:
        da = np.mean(actual_ret[-m:] == pred_ret[-m:])
    else:
        da = np.nan

    # Simple cumulative return of buy and hold
    if len(actual) > 1:
        daily_ret = np.diff(actual) / actual[:-1]
        cum_ret = np.prod(1 + daily_ret) - 1
    else:
        cum_ret = np.nan

    return {
        "rmse": rmse,
        "directional_accuracy": da,
        "cum_return": cum_ret,
    }


# ---------- Streamlit app layout ----------

def main():
    st.set_page_config(
        page_title="Stock Prediction App",
        layout="wide",
    )

    st.title("Neural Network Stock Prediction Demo")
    st.write(
        "This app visualises precomputed price data and a simple forecasting signal "
        "for a universe of large liquid equities and indices."
    )

    # Sidebar
    st.sidebar.header("Controls")

    symbol = st.sidebar.selectbox("Select symbol", SYMBOLS, index=0)

    st.sidebar.markdown("---")
    last_n_days = st.sidebar.slider(
        "Show last N days",
        min_value=60,
        max_value=1000,
        value=365,
        step=30,
    )

    st.sidebar.markdown(
        """
        **Note**  
        Data and predictions are precomputed and loaded from CSV files in  
        `data/outputs/*.csv`.
        """
    )

    # Load data
    df = load_symbol(symbol)

    if df is None:
        st.stop()

    # Filter last N days
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)
        if len(df) > last_n_days:
            df_plot = df.iloc[-last_n_days:].copy()
        else:
            df_plot = df.copy()
    else:
        df_plot = df.copy()

    # Compute metrics
    metrics = compute_basic_metrics(df_plot)

    # Top level layout
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader(f"Price and prediction for {symbol}")
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(df_plot["Date"], df_plot["Close"], label="Actual Close", linewidth=1.4)
        ax.plot(
            df_plot["Date"],
            df_plot["Predicted_Close"],
            label="Forecast signal",
            linestyle="--",
            linewidth=1.2,
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        st.pyplot(fig)

    with col_right:
        st.subheader("Summary metrics")

        rmse = metrics["rmse"]
        da = metrics["directional_accuracy"]
        cum_ret = metrics["cum_return"]

        st.metric("RMSE (price units)", f"{rmse:.4f}" if not np.isnan(rmse) else "N/A")
        st.metric(
            "Directional accuracy",
            f"{da * 100:.2f} %" if not np.isnan(da) else "N/A",
        )
        st.metric(
            "Buy and hold cumulative return",
            f"{cum_ret * 100:.2f} %" if not np.isnan(cum_ret) else "N/A",
        )

        st.markdown("---")
        st.markdown(
            """
            **Interpretation**

            - The forecast line is a simple next day signal derived from the price series.  
            - RMSE measures average deviation between forecast and actual price.  
            - Directional accuracy measures how often the sign of the move is predicted correctly.  
            - Cumulative return is the result of passively holding the asset over the plotted window.
            """
        )

    st.markdown("---")
    with st.expander("Show raw data"):
        st.dataframe(df_plot.tail(200))


if __name__ == "__main__":
    main()
