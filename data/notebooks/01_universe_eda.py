"""
01_universe_eda.py

Universe level EDA:

- Download price history for all EXPERIMENT_SYMBOLS
- Build a clean Close price matrix
- Plot:
    1. Normalised price paths for all symbols
    2. Correlation heatmap of daily returns

Outputs:
- data/notebooks/close_prices_matrix.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import EXPERIMENT_SYMBOLS
from src.data.load import load_price_history


PLOTS_DIR = Path("data/notebooks/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_MATRIX_CSV = Path("data/notebooks/close_prices_matrix.csv")


def pick_price_column(df: pd.DataFrame) -> str:
    """
    Choose the best price column for EDA.
    Preference: Adj Close then Close then first numeric column.
    """
    for col in ["Adj Close", "Close", "Open", "High", "Low"]:
        if col in df.columns:
            return col

    # fallback: first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns available for price series")
    return numeric_cols[0]


def main():
    print("[EDA] Loading universe...")

    series_list: list[pd.Series] = []

    for symbol in EXPERIMENT_SYMBOLS:
        print(f"[EDA] Loading {symbol}...")
        df = load_price_history(symbol)
        start, end = df.index[0].date(), df.index[-1].date()
        print(f"  {symbol}: {start} to {end}, {len(df)} rows")

        price_col = pick_price_column(df)
        s = df[price_col].rename(symbol)
        series_list.append(s)

    # Align by date and build wide matrix
    close_df = pd.concat(series_list, axis=1)
    close_df = close_df.sort_index()

    # Save matrix for later use
    OUT_MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    close_df.to_csv(OUT_MATRIX_CSV)
    print(f"[EDA] Saved close price matrix to {OUT_MATRIX_CSV}")

    # Compute daily log returns
    log_ret = np.log(close_df / close_df.shift(1)).dropna(how="all")

    # 1. Normalised price paths
    norm_prices = close_df / close_df.iloc[0]

    plt.figure(figsize=(12, 6))
    for col in norm_prices.columns:
        plt.plot(norm_prices.index, norm_prices[col], linewidth=1.0, alpha=0.9)
    plt.title("Normalised price paths (start = 1.0)")
    plt.xlabel("Date")
    plt.ylabel("Normalised price")
    plt.tight_layout()
    path_norm = PLOTS_DIR / "01_normalised_price_paths.png"
    plt.savefig(path_norm, dpi=150)
    plt.close()
    print(f"[EDA] Saved {path_norm}")

    # 2. Correlation heatmap of daily log returns
    corr = log_ret.corr()

    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation of daily log returns")
    plt.tight_layout()
    path_corr = PLOTS_DIR / "02_corr_heatmap_log_returns.png"
    plt.savefig(path_corr, dpi=150)
    plt.close()
    print(f"[EDA] Saved {path_corr}")

    print("[EDA] Done.")


if __name__ == "__main__":
    main()
