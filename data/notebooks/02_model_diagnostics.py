"""
02_model_diagnostics.py

Model diagnostics for LSTM vs simple baselines.

Usage (from project root):
    source .venv/bin/activate
    python -m data.notebooks.02_model_diagnostics

This script will, for each symbol where a trained LSTM exists:
  * Load processed train/test arrays and scaler
  * Load the saved LSTM model
  * Compute predictions on the test set
  * Compare against naive lag, 5 day and 20 day moving averages
  * Compute MAE and RMSE for all models
  * Plot Actual vs LSTM vs baselines on the test period
  * Save figures and a metrics table into data/notebooks/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import (
    EXPERIMENT_SYMBOLS,
    MODELS_DIR,
    DATA_PROCESSED_DIR,
    TEST_SIZE,
    DEVICE,
)
from src.data.load import load_price_history


OUT_DIR = Path("data/notebooks")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# LSTM definition must match training
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


def load_processed_arrays(symbol: str):
    X_train = np.load(DATA_PROCESSED_DIR / f"{symbol}_X_train.npy")
    y_train = np.load(DATA_PROCESSED_DIR / f"{symbol}_y_train.npy")
    X_test = np.load(DATA_PROCESSED_DIR / f"{symbol}_X_test.npy")
    y_test = np.load(DATA_PROCESSED_DIR / f"{symbol}_y_test.npy")
    scaler_params = np.load(DATA_PROCESSED_DIR / f"{symbol}_scaler.npz")
    return X_train, y_train, X_test, y_test, scaler_params


def scale_inverse(scaled: np.ndarray, scaler_params: np.lib.npyio.NpzFile) -> np.ndarray:
    """Inverse of sklearn MinMaxScaler with feature_range (0, 1)."""
    min_ = scaler_params["min_"]
    scale_ = scaler_params["scale_"]
    scaled_2d = scaled.reshape(-1, 1)
    return (scaled_2d * scale_) + min_


def get_dates_for_test(df: pd.DataFrame) -> pd.DatetimeIndex:
    n = len(df)
    split_idx = int(n * (1.0 - TEST_SIZE))
    return df.index[split_idx:]


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


def main():
    metrics_rows: list[dict] = []

    for symbol in EXPERIMENT_SYMBOLS:
        model_path = MODELS_DIR / f"lstm_{symbol}.pt"
        if not model_path.exists():
            print(f"[Diagnostics] Skipping {symbol}: no model at {model_path.name}")
            continue

        print(f"[Diagnostics] Processing {symbol}...")

        # Load full price history only to get test dates
        df_full = load_price_history(symbol)
        price_col = "Adj Close" if "Adj Close" in df_full.columns else "Close"

        # Load processed arrays and scaler
        _, _, X_test, y_test, scaler_params = load_processed_arrays(symbol)

        # Load model
        model = LSTMRegressor(input_size=1)
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        # Predict on test set
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        with torch.inference_mode():
            preds_t = model(X_test_t)

        preds_scaled = preds_t.cpu().numpy().reshape(-1, 1)
        targets_scaled = y_test.reshape(-1, 1)

        preds_price = scale_inverse(preds_scaled, scaler_params).reshape(-1)
        targets_price = scale_inverse(targets_scaled, scaler_params).reshape(-1)

        # Align with actual test dates
        dates_test_full = get_dates_for_test(df_full)
        L = min(len(dates_test_full), len(preds_price), len(targets_price))
        dates = dates_test_full[-L:]
        preds_price = preds_price[-L:]
        targets_price = targets_price[-L:]

        # Baselines
        baselines = compute_baselines(targets_price)

        # One big DataFrame of models
        data = {"Actual": targets_price, "LSTM": preds_price}
        data.update(baselines)
        results_df = pd.DataFrame(data, index=dates)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(results_df.index, results_df["Actual"], label="Actual", linewidth=1.3)
        ax.plot(results_df.index, results_df["LSTM"], label="LSTM", linewidth=1.0)
        ax.plot(results_df.index, results_df["Naive lag1"], label="Naive lag1", alpha=0.7)
        ax.plot(results_df.index, results_df["MA 5 day"], label="MA 5 day", alpha=0.7)
        ax.plot(results_df.index, results_df["MA 20 day"], label="MA 20 day", alpha=0.7)
        ax.set_title(f"{symbol} test set: actual vs models")
        ax.set_ylabel(price_col)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"fig_{symbol}_test_models.png", dpi=150)
        plt.close(fig)

        # Metrics
        mae_lstm, rmse_lstm = compute_metrics(results_df["LSTM"], results_df["Actual"])
        metrics_rows.append(
            {"Symbol": symbol, "Model": "LSTM", "MAE": mae_lstm, "RMSE": rmse_lstm}
        )

        for name in baselines.keys():
            mae, rmse = compute_metrics(results_df[name], results_df["Actual"])
            metrics_rows.append(
                {"Symbol": symbol, "Model": name, "MAE": mae, "RMSE": rmse}
            )

    if not metrics_rows:
        print("[Diagnostics] No metrics produced. Make sure you have trained some models.")
        return

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / "metrics_all_symbols.csv", index=False)
    print("\n[Diagnostics] Metrics summary:")
    print(metrics_df.pivot(index="Symbol", columns="Model", values="RMSE"))
    print(f"\n[Diagnostics] Saved metrics to {OUT_DIR / 'metrics_all_symbols.csv'}")


if __name__ == "__main__":
    main()
