import sys
from pathlib import Path

# Make src importable no matter where Streamlit runs from
ROOT = Path(__file__).resolve().parent.parent  # .../urop-stock-nn
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
import yfinance as yf

import config as cfg  # src/config.py

EXPERIMENT_SYMBOLS = cfg.EXPERIMENT_SYMBOLS
DEFAULT_SYMBOL = cfg.DEFAULT_SYMBOL
LOOKBACK = cfg.LOOKBACK
TEST_SIZE = cfg.TEST_SIZE
MODELS_DIR = cfg.MODELS_DIR
DATA_PROCESSED_DIR = cfg.DATA_PROCESSED_DIR
DEVICE = cfg.DEVICE

# ---------- Model definition ----------

class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        out = self.fc(last)
        return out


# ---------- Utilities ----------

def load_price_history(
    symbol: str,
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily data for a single symbol."""
    if period is not None:
        df = yf.download(symbol, period=period, interval="1d", progress=False, threads=False)
    else:
        df = yf.download(symbol, start=start, end=end, interval="1d", progress=False, threads=False)

    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol} with period={period}, start={start}, end={end}")

    # If MultiIndex appears (should not for single ticker, but just in case), flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

    df = df.sort_index()
    df.index.name = "Date"
    return df


def choose_price_column(df: pd.DataFrame, desired: str) -> str:
    cols = list(df.columns)

    # exact match
    if desired in cols:
        return desired

    # flattened like "Adj Close_AAPL"
    prefix_matches = [c for c in cols if c.startswith(desired)]
    if prefix_matches:
        return prefix_matches[0]

    # fallbacks
    for base in ["Adj Close", "Close", "Open", "High", "Low"]:
        if base in cols:
            return base
        cand = [c for c in cols if c.startswith(base)]
        if cand:
            return cand[0]

    # last resort
    return cols[0]


def load_processed_arrays(symbol: str):
    X_train = np.load(DATA_PROCESSED_DIR / f"{symbol}_X_train.npy")
    y_train = np.load(DATA_PROCESSED_DIR / f"{symbol}_y_train.npy")
    X_test = np.load(DATA_PROCESSED_DIR / f"{symbol}_X_test.npy")
    y_test = np.load(DATA_PROCESSED_DIR / f"{symbol}_y_test.npy")
    scaler_params = np.load(DATA_PROCESSED_DIR / f"{symbol}_scaler.npz")
    return X_train, y_train, X_test, y_test, scaler_params


# IMPORTANT: match sklearn MinMaxScaler
# scaled = (raw - min_) / scale_
# raw    = scaled * scale_ + min_

def scale_values(raw: np.ndarray, scaler_params: np.lib.npyio.NpzFile) -> np.ndarray:
    """Scale raw prices into [0,1] using saved min_ and scale_."""
    min_ = scaler_params["min_"]
    scale_ = scaler_params["scale_"]
    raw_2d = raw.reshape(-1, 1)
    return (raw_2d - min_) / scale_


def inverse_scale(scaled: np.ndarray, scaler_params: np.lib.npyio.NpzFile) -> np.ndarray:
    """Inverse-transform scaled values back to original price space."""
    min_ = scaler_params["min_"]
    scale_ = scaler_params["scale_"]
    scaled_2d = scaled.reshape(-1, 1)
    return (scaled_2d * scale_) + min_


def get_dates_for_test(df: pd.DataFrame) -> pd.DatetimeIndex:
    n = len(df)
    split_idx = int(n * (1.0 - TEST_SIZE))
    return df.index[split_idx:]


def load_trained_model(symbol: str) -> LSTMRegressor | None:
    model_path = MODELS_DIR / f"lstm_{symbol}.pt"
    if not model_path.exists():
        return None

    model = LSTMRegressor(input_size=1)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def compute_baselines(targets_price: np.ndarray) -> dict[str, np.ndarray]:
    naive = np.concatenate([[targets_price[0]], targets_price[:-1]])
    ma5 = pd.Series(targets_price).rolling(window=5, min_periods=1).mean().values
    ma20 = pd.Series(targets_price).rolling(window=20, min_periods=1).mean().values

    return {
        "Naive lag1": naive,
        "MA 5 day": ma5,
        "MA 20 day": ma20,
    }


def compute_metrics(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    return mae, rmse


def forecast_next_days(
    model: LSTMRegressor,
    df: pd.DataFrame,
    scaler_params: np.lib.npyio.NpzFile,
    price_col: str,
    n_steps: int = 5,
) -> pd.DataFrame:
    closes = df[price_col].values
    if len(closes) < LOOKBACK:
        return pd.DataFrame()

    last_window_raw = closes[-LOOKBACK:]
    last_window_scaled = scale_values(last_window_raw, scaler_params)
    window = last_window_scaled.copy()

    preds_scaled = []
    for _ in range(n_steps):
        x = torch.tensor(window.reshape(1, LOOKBACK, 1), dtype=torch.float32).to(DEVICE)
        with torch.inference_mode():
            y_scaled = model(x).cpu().numpy().reshape(-1)
        preds_scaled.append(y_scaled[0])
        window = np.concatenate([window[1:], y_scaled.reshape(1, 1)], axis=0)

    preds_scaled_arr = np.array(preds_scaled).reshape(-1, 1)
    preds_price = inverse_scale(preds_scaled_arr, scaler_params).reshape(-1)

    last_date = df.index[-1]
    future_dates = pd.date_range(last_date, periods=n_steps + 1, freq="B")[1:]

    return pd.DataFrame({"LSTM forecast": preds_price}, index=future_dates)


def get_ticker_info(symbol: str) -> dict:
    try:
        t = yf.Ticker(symbol)
        info = t.info
    except Exception:
        info = {}
    wanted = {
        "shortName": info.get("shortName"),
        "longName": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "marketCap": info.get("marketCap"),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "beta": info.get("beta"),
    }
    return wanted


def select_period_label(label: str) -> str | None:
    mapping = {
        "1M": "1mo",
        "3M": "3mo",
        "6M": "6mo",
        "1Y": "1y",
        "5Y": "5y",
        "Max": "max",
    }
    return mapping.get(label, "1y")


# ---------- Streamlit app ----------

def main():
    # Page title in browser tab
    st.set_page_config(page_title="Equity Model Lab", layout="wide")

    # Main title in the app
    st.title("Equity Model Lab")

    # Short description under the title
    st.caption(
        "Interactive research console for stock price prediction with neural networks "
        "and time series benchmark models."
    )


    st.sidebar.header("Mode and universe")

    mode = st.sidebar.radio("Mode", ["Predefined universe", "Custom symbol"])

    if mode == "Predefined universe":
        symbol = st.sidebar.selectbox(
            "Choose asset",
            options=EXPERIMENT_SYMBOLS,
            index=EXPERIMENT_SYMBOLS.index(DEFAULT_SYMBOL),
        )
    else:
        symbol = st.sidebar.text_input("Enter Yahoo Finance symbol", value="AAPL").strip().upper()

    window_label = st.sidebar.selectbox(
        "Time window",
        options=["1M", "3M", "6M", "1Y", "5Y", "Max"],
        index=3,
    )

    forecast_horizon = st.sidebar.slider(
        "Forecast horizon (days ahead)",
        min_value=1,
        max_value=10,
        value=5,
    )

    price_field = st.sidebar.selectbox(
        "Price field",
        options=["Adj Close", "Close", "Open", "High", "Low"],
        index=0,
    )

    period = select_period_label(window_label)

    tabs = st.tabs(["Overview", "Models", "Fundamentals"])

    # Data loading with proper error handling
    try:
        with st.spinner(f"Loading price history for {symbol}"):
            df_full = load_price_history(symbol, period=period)
    except Exception as e:
        for tab in tabs:
            with tab:
                st.error(f"Could not load data for {symbol}: {e}")
        return

    price_col = choose_price_column(df_full, price_field)
    df = df_full.copy()

    # -------- Overview tab --------
    with tabs[0]:
        st.subheader(f"{symbol} price overview")

        price_series = df[price_col].rename("Price")
        st.line_chart(price_series)

        st.write("Recent daily data")
        cols_to_show = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        if cols_to_show:
            st.dataframe(
                df[cols_to_show].tail(15),
                use_container_width=True,
            )
        else:
            st.info("No OHLCV columns available to display.")

    # -------- Models tab --------
    with tabs[1]:
        st.subheader(f"{symbol} model comparison on test set")

        model = load_trained_model(symbol) if mode == "Predefined universe" else None

        if model is None:
            st.info(
                "LSTM model is available only for symbols in the predefined universe "
                "that you have trained. For other symbols, only price overview is shown."
            )
        else:
            try:
                _, _, X_test, y_test, scaler_params = load_processed_arrays(symbol)
            except FileNotFoundError:
                st.error(
                    f"Processed arrays for {symbol} not found. "
                    f"Run the preprocessing and training scripts first."
                )
            else:
                X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
                with torch.inference_mode():
                    preds_t = model(X_test_t)
                preds_scaled = preds_t.cpu().numpy().reshape(-1, 1)
                targets_scaled = y_test.reshape(-1, 1)

                preds_price = inverse_scale(preds_scaled, scaler_params).reshape(-1)
                targets_price = inverse_scale(targets_scaled, scaler_params).reshape(-1)

                dates_test_full = get_dates_for_test(df_full)
                L = min(len(dates_test_full), len(preds_price), len(targets_price))
                dates_test = dates_test_full[-L:]
                preds_price = preds_price[-L:]
                targets_price = targets_price[-L:]

                baselines = compute_baselines(targets_price)

                data_dict = {"Actual": targets_price, "LSTM": preds_price}
                data_dict.update(baselines)
                results_df = pd.DataFrame(data_dict, index=dates_test)

                st.line_chart(results_df)

                rows = []
                mae_lstm, rmse_lstm = compute_metrics(preds_price, targets_price)
                rows.append({"Model": "LSTM", "MAE": mae_lstm, "RMSE": rmse_lstm})

                for name, pred in baselines.items():
                    mae, rmse = compute_metrics(pred, targets_price)
                    rows.append({"Model": name, "MAE": mae, "RMSE": rmse})

                metrics_df = pd.DataFrame(rows).set_index("Model")
                st.write("Error metrics on test set")
                st.dataframe(metrics_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}"}))

                st.write("Short horizon LSTM forecast")

                forecast_df = forecast_next_days(
                    model=model,
                    df=df_full,
                    scaler_params=scaler_params,
                    price_col=price_col,
                    n_steps=forecast_horizon,
                )

                if forecast_df.empty:
                    st.info("Not enough history to compute forecast.")
                else:
                    hist = df_full[[price_col]].tail(60).rename(columns={price_col: "History"})
                    combined = pd.concat([hist, forecast_df], axis=0)
                    st.line_chart(combined)
                    st.dataframe(forecast_df)

    # -------- Fundamentals tab --------
    with tabs[2]:
        st.subheader(f"{symbol} fundamentals snapshot")
        info = get_ticker_info(symbol)
        if not info.get("shortName") and not info.get("longName"):
            st.info("Fundamental info not available for this symbol.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Name**", info.get("longName") or info.get("shortName"))
                st.write("**Sector**", info.get("sector"))
                st.write("**Industry**", info.get("industry"))
            with col_b:
                st.write("**Market cap**", f"{info.get('marketCap'):,}" if info.get("marketCap") else "N/A")
                st.write("**Trailing PE**", info.get("trailingPE"))
                st.write("**Forward PE**", info.get("forwardPE"))
                st.write("**Beta**", info.get("beta"))


if __name__ == "__main__":
    main()