from __future__ import annotations

from src.training.train_lstm import train_symbol
from src.config import EXPERIMENT_SYMBOLS


def train_all_symbols() -> None:
    for symbol in EXPERIMENT_SYMBOLS:
        print("=" * 60)
        print(f"Training LSTM for {symbol}")
        train_symbol(symbol)
        print("=" * 60)


if __name__ == "__main__":
    train_all_symbols()
