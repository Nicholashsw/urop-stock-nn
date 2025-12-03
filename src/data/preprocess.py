from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..config import (
    DATA_PROCESSED_DIR,
    EXPERIMENT_SYMBOLS,
    LOOKBACK,
    TEST_SIZE,
)
from .load import load_price_history


def create_sequences(data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn a 1D series into overlapping sequences of length `lookback`
    with a 1-step-ahead prediction target.
    """
    xs, ys = [], []
    for i in range(len(data) - lookback):
        x = data[i : i + lookback]
        y = data[i + lookback]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def preprocess_symbol(symbol: str) -> None:
    print(f"[preprocess] Processing {symbol}...")
    df = load_price_history(symbol)

    # Use Adjusted Close if available, else Close
    if "Adj Close" in df.columns:
        close = df["Adj Close"].values.reshape(-1, 1)
    else:
        close = df["Close"].values.reshape(-1, 1)

    # Train-test split on raw series (time-wise)
    n = len(close)
    split_idx = int(n * (1.0 - TEST_SIZE))
    train_raw = close[:split_idx]
    test_raw = close[split_idx:]

    # Scale using train-only fit
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    # Recombine for sequence creation
    series_scaled = np.concatenate([train_scaled, test_scaled], axis=0)

    X_all, y_all = create_sequences(series_scaled, LOOKBACK)
    # Align split index for sequences
    split_seq_idx = split_idx - LOOKBACK
    X_train, X_test = X_all[:split_seq_idx], X_all[split_seq_idx:]
    y_train, y_test = y_all[:split_seq_idx], y_all[split_seq_idx:]

    # Ensure processed dir exists
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save arrays and scaler
    np.save(DATA_PROCESSED_DIR / f"{symbol}_X_train.npy", X_train)
    np.save(DATA_PROCESSED_DIR / f"{symbol}_y_train.npy", y_train)
    np.save(DATA_PROCESSED_DIR / f"{symbol}_X_test.npy", X_test)
    np.save(DATA_PROCESSED_DIR / f"{symbol}_y_test.npy", y_test)

    # Save scaler params as a small npz (min_, scale_)
    np.savez(
        DATA_PROCESSED_DIR / f"{symbol}_scaler.npz",
        min_=scaler.min_,
        scale_=scaler.scale_,
        data_min_=scaler.data_min_,
        data_max_=scaler.data_max_,
        data_range_=scaler.data_range_,
    )

    print(
        f"[preprocess] {symbol}: X_train {X_train.shape}, X_test {X_test.shape}, "
        f"y_train {y_train.shape}, y_test {y_test.shape}"
    )


def preprocess_all_experiments() -> None:
    for symbol in EXPERIMENT_SYMBOLS:
        preprocess_symbol(symbol)


if __name__ == "__main__":
    preprocess_all_experiments()
