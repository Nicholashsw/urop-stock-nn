import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
# Paths and config
# ============================================================

# Repo root: .../urop-stock-nn
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "outputs"

st.set_page_config(
    page_title="Neural Network Stock Prediction Demo",
    page_icon="📈",
    layout="wide",
)

# ============================================================
# Symbol universe
# ============================================================

SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "ADBE",
    "AMD", "CRM", "NFLX", "AVGO", "COST", "WMT", "MCD", "DIS", "HD", "KO",
    "PEP", "JPM", "GS", "SCHW", "V", "MA", "UNH", "ABBV", "TMO", "PFE", "JNJ",
    "XOM", "CVX", "CAT", "BA", "VZ", "T", "TM", "MC.PA",
    "^GSPC", "^NDX", "^DJI",
    "SPY", "QQQ", "DIA",
]


# ============================================================
# Helpers
# ============================================================

@st.cache_data(show_spinner=False)
def load_symbol(symbol: str) -> pd.DataFrame:
    """
    Load precomputed CSV output for the given symbol.

    Expected filename: data/outputs/{symbol}_lstm_results.csv

    The function is defensive:
    - returns empty DataFrame if file missing or unreadable
    - parses Date column to datetime if present
    """
    file_path = DATA_DIR / f"{symbol}_lstm_results.csv"

    if not file_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)

        # Handle possible index column from pandas to_csv
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # Standardise date column name if possible
        date_cols = [c for c in df.columns if c.lower() in ["date", "timestamp"]]
        if date_cols:
            date_col = date_cols[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
        else:
            # Create a fake date index if none exists, just to allow plotting
            df = df.copy()
            df["index"] = np.arange(len(df))
            date_col = "index"

        df = df.reset_index(drop=True)
        return df, date_col
    except Exception as e:
        st.error(f"Failed to load {file_path.name}: {e}")
        return pd.DataFrame(), None


def pick_price_column(df: pd.DataFrame) -> str | None:
    """
    Try to choose a reasonable price column for plotting.
    """
    candidates = ["Close", "Adj Close", "close", "adj_close", "Price", "price"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback to first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols[0] if numeric_cols else None


def pick_prediction_columns(df: pd.DataFrame) -> list[str]:
    """
    Try to find prediction columns if available.
    """
    candidates = [
        "pred_lstm", "lstm_pred", "y_pred_lstm",
        "pred_naive", "naive_pred",
        "pred_ma", "ma_pred",
    ]
    found = [c for c in candidates if c in df.columns]
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for c in found:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def compute_simple_metrics(df: pd.DataFrame, price_col: str, pred_col: str | None):
    """
    Compute simple performance metrics:
    - last price
    - last return
    - RMSE if prediction column present
    - directional accuracy if prediction column present and target exists
    """
    out = {}

    prices = df[price_col].astype(float)
    if len(prices) >= 2:
        last_price = prices.iloc[-1]
        prev_price = prices.iloc[-2]
        last_ret = (last_price / prev_price) - 1.0
        out["last_price"] = last_price
        out["last_return"] = last_ret
    else:
        out["last_price"] = None
        out["last_return"] = None

    if pred_col is None:
        return out

    y_cols = [c for c in df.columns if c.lower() in ["target", "y_true", "true", "return"]]
    if not y_cols:
        return out

    y_col = y_cols[0]
    y_true = df[y_col].astype(float)
    y_pred = df[pred_col].astype(float)

    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return out

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = math.sqrt(mse)
    out["rmse"] = rmse

    # Directional accuracy
    sign_true = np.sign(y_true.values)
    sign_pred = np.sign(y_pred.values)
    correct = (sign_true == sign_pred).sum()
    da = correct / len(y_true)
    out["directional_accuracy"] = da

    return out


def make_price_chart(df: pd.DataFrame, date_col: str, price_col: str, preds: list[str]):
    """
    Build Altair chart: price series and optional prediction overlays.
    """
    base = alt.Chart(df).encode(
        x=alt.X(date_col, title="Date"),
    )

    price_line = base.mark_line().encode(
        y=alt.Y(price_col, title="Price"),
        color=alt.value("#1f77b4"),
    )

    layers = [price_line]

    color_map = {
        "pred_lstm": "#d62728",
        "lstm_pred": "#d62728",
        "y_pred_lstm": "#d62728",
        "pred_ma": "#2ca02c",
        "ma_pred": "#2ca02c",
        "pred_naive": "#ff7f0e",
        "naive_pred": "#ff7f0e",
    }

    for col in preds:
        if col not in df.columns:
            continue
        line = base.mark_line(strokeDash=[4, 3]).encode(
            y=alt.Y(col, title=""),
            color=alt.value(color_map.get(col, "#888888")),
        )
        layers.append(line)

    chart = alt.layer(*layers).resolve_scale(y="shared").interactive()
    return chart


# ============================================================
# Layout
# ============================================================

st.title("📈 Neural Network Stock Prediction Demo")

st.markdown(
    """
This app visualises **precomputed outputs** from your UROP stock prediction project.

It reads CSV files from `data/outputs/` and shows:

- Price history for the selected symbol  
- Optional neural network or baseline predictions, if present in the CSV  
- Simple metrics such as last price, last return, and RMSE if targets are available  

All heavy lifting (training, downloading, backtesting) was done **offline**.
"""
)

# Sidebar controls
st.sidebar.header("Controls")

symbol = st.sidebar.selectbox("Select symbol", SYMBOLS, index=0)

lookback_years = st.sidebar.slider(
    "Lookback (years)",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
)

st.sidebar.info(
    "Data and predictions are loaded from precomputed CSV files. "
    "No live API calls are made inside this app."
)

# ============================================================
# Main content
# ============================================================

df, date_col = load_symbol(symbol)

if df is None or df.empty or date_col is None:
    st.error(
        f"No data available for {symbol}. "
        f"Check that `data/outputs/{symbol}_lstm_results.csv` exists in the repo."
    )
    st.stop()

price_col = pick_price_column(df)
pred_cols = pick_prediction_columns(df)

if price_col is None:
    st.error("Could not identify a price column to plot.")
    st.write("Columns found:", list(df.columns))
    st.stop()

# Filter by lookback
if df[date_col].dtype == "datetime64[ns]":
    max_date = df[date_col].max()
    min_date = max_date - pd.DateOffset(years=lookback_years)
    df_view = df[df[date_col] >= min_date].copy()
else:
    # If date is fake index, just take last N rows
    n = 252 * lookback_years
    df_view = df.tail(n).copy()

st.subheader(f"{symbol} price and model outputs")

chart = make_price_chart(df_view, date_col, price_col, pred_cols)
st.altair_chart(chart, use_container_width=True)

# Metrics
pred_col_for_metrics = pred_cols[0] if pred_cols else None
metrics = compute_simple_metrics(df_view, price_col, pred_col_for_metrics)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Last price",
        f"{metrics.get('last_price'):.2f}" if metrics.get("last_price") is not None else "n/a",
    )

with col2:
    if metrics.get("last_return") is not None:
        st.metric(
            "Last daily return",
            f"{metrics['last_return'] * 100:.2f} %",
        )
    else:
        st.metric("Last daily return", "n/a")

with col3:
    if "rmse" in metrics:
        st.metric("RMSE (test window)", f"{metrics['rmse']:.4f}")
    else:
        st.metric("RMSE (test window)", "n/a")

with col4:
    if "directional_accuracy" in metrics:
        st.metric(
            "Directional accuracy",
            f"{metrics['directional_accuracy'] * 100:.2f} %",
        )
    else:
        st.metric("Directional accuracy", "n/a")

st.markdown("### Raw data snapshot")
st.dataframe(df_view.tail(20))

st.caption(
    "This is a static research demo built from your UROP neural network project. "
    "All predictions are precomputed offline and loaded from CSV."
)
