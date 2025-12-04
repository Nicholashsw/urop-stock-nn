"""
02_model_diagnostics

Purpose
    For each symbol with a trained LSTM model
        Evaluate on test set
        Compare against naive and moving average baselines
        Save consolidated metrics table

Output
    metrics_all_symbols.csv in this folder
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import (
    EXPERIMENT_SYMBOLS,
    START_DATE,
    END_DATE,
    LOOKBACK,
    TEST_SIZE,
    MODELS_DIR,
    DATA_PROCESSED_DIR,
    DEVICE,
)
from src.data.load import load_price_history


class LSTMRegressor(nn.Module):
    """Same simple LSTM regressor used in training and in the app."""
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
    """Load saved numpy arrays and scaler params for one symbol."""
    X_train = np.load(DATA_PROCESSED_DIR / f"{symbol}_X_train.npy")
    y_train = np.load(DATA_PROCESSED_DIR / f"{symbol}_y_train.npy")
    X_test = np.load(DATA_PROCESSED_DIR / f"{symbol}_X_test.npy")
    y_test = np.load(DATA_PROCESSED_DIR / f"{symbol}_y_test.npy")
    scaler_params = np.load(DATA_PROCESSED_DIR / f"{symbol}_scaler.npz")
    return X_train, y_train, X_test, y_test, scaler_params


def scale_values(raw: np.ndarray, scaler_params: np.lib.npyio.NpzFile) -> np.ndarray:
    """Scale raw prices into zero to one using saved min and scale."""
    min_ = scaler_params["min_"]
    scale_ = scaler_params["scale_"]
    raw_2d = raw.reshape(-1, 1)
    return (raw_2d - min_) / scale_


def inverse_scale(scaled: np.ndarray, scaler_params: np.lib.npyio.NpzFile) -> np.ndarray:
    """Inverse transform scaled values back to original price space."""
    min_ = scaler_params["min_"]
    scale_ = scaler_params["scale_"]
    scaled_2d = scaled.reshape(-1, 1)
    return (scaled_2d * scale_) + min_


def get_dates_for_test(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Split index according to TEST_SIZE to align test predictions with dates."""
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


def compute_baselines(targets_price: np.ndarray) -> Dict[str, np.ndarray]:
    """Naive one day lag and simple moving averages."""
    naive = np.concatenate([[targets_price[0]], targets_price[:-1]])
    ma5 = pd.Series(targets_price).rolling(window=5, min_periods=1).mean().values
    ma20 = pd.Series(targets_price).rolling(window=20, min_periods=1).mean().values

    return {
        "Naive lag1": naive,
        "MA 5 day": ma5,
        "MA 20 day": ma20,
    }


def compute_metrics(pred: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    return mae, rmse


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    rows = []

    for symbol in EXPERIMENT_SYMBOLS:
        print("=" * 60)
        print(f"[Diagnostics] Processing {symbol}...")
        model = load_trained_model(symbol)
        if model is None:
            print(f"[Diagnostics] Skipping {symbol}: no model at {MODELS_DIR / f'lstm_{symbol}.pt'}")
            continue

        try:
            _, _, X_test, y_test, scaler_params = load_processed_arrays(symbol)
        except FileNotFoundError:
            print(f"[Diagnostics] Skipping {symbol}: processed arrays not found")
            continue

        # Load full history to get date index for plots and alignment
        df_full = load_price_history(symbol, start=START_DATE, end=END_DATE)
        dates_test_full = get_dates_for_test(df_full)

        # Predict on test set
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        with torch.inference_mode():
            preds_t = model(X_test_t)

        preds_scaled = preds_t.cpu().numpy().reshape(-1, 1)
        targets_scaled = y_test.reshape(-1, 1)

        preds_price = inverse_scale(preds_scaled, scaler_params).reshape(-1)
        targets_price = inverse_scale(targets_scaled, scaler_params).reshape(-1)

        # Align lengths
        L = min(len(dates_test_full), len(preds_price), len(targets_price))
        dates_test = dates_test_full[-L:]
        preds_price = preds_price[-L:]
        targets_price = targets_price[-L:]

        baselines = compute_baselines(targets_price)

        # Metrics for LSTM
        mae_lstm, rmse_lstm = compute_metrics(preds_price, targets_price)
        rows.append({
            "Symbol": symbol,
            "Model": "LSTM",
            "MAE": mae_lstm,
            "RMSE": rmse_lstm,
        })

        # Metrics for baselines
        for name, pred in baselines.items():
            mae, rmse = compute_metrics(pred, targets_price)
            rows.append({
                "Symbol": symbol,
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
            })

        print(f"[Diagnostics] {symbol} LSTM MAE {mae_lstm:.6f}  RMSE {rmse_lstm:.6f}")

    if not rows:
        print("[Diagnostics] No symbols had trained models, nothing to save")
        return

    metrics_df = pd.DataFrame(rows)
    metrics_path = out_dir / "metrics_all_symbols.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print()
    print("[Diagnostics] Metrics summary:")
    print(metrics_df.pivot(index="Symbol", columns="Model", values="MAE"))
    print(f"\n[Diagnostics] Saved metrics to {metrics_path}")
    print("[Diagnostics] Done")


if __name__ == "__main__":
    main()
