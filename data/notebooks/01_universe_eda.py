"""
01_universe_eda

Purpose
    Universe level exploration
    Build close price and return matrices
    Save basic summary stats and correlation heatmap

Output files (in this folder)
    close_prices_matrix.csv
    daily_returns_matrix.csv
    returns_summary_stats.csv
    corr_heatmap.png
    universe_normalized_prices.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import EXPERIMENT_SYMBOLS, START_DATE, END_DATE
from src.data.load import load_price_history


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    all_data: dict[str, pd.DataFrame] = {}

    print("[EDA] Loading universe...")
    # 1. Load history for every symbol in the universe
    for symbol in EXPERIMENT_SYMBOLS:
        print(f"[EDA] Loading {symbol}...")
        df = load_price_history(symbol, start=START_DATE, end=END_DATE)
        all_data[symbol] = df
        print(f"  {symbol}: {df.index.min().date()} to {df.index.max().date()}, {len(df)} rows")

    # 2. Build close price matrix using Adj Close if available else Close
    close_series_list: list[pd.Series] = []

    for symbol, df in all_data.items():
        if "Adj Close" in df.columns:
            series = df["Adj Close"]
        elif "Close" in df.columns:
            series = df["Close"]
        else:
            # fallback: first numeric column
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) == 0:
                print(f"[EDA] Warning: no numeric columns for {symbol}, skipping")
                continue
            series = df[numeric_cols[0]]

        # ensure we have a Series, not a DataFrame
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        series = series.rename(symbol)
        close_series_list.append(series)

    # concat along columns, align by date index
    close_df = pd.concat(close_series_list, axis=1).sort_index()
    close_df = close_df.dropna(how="all")

    print("[EDA] Built close price matrix with shape", close_df.shape)

    # 3. Save close price matrix
    close_path = out_dir / "close_prices_matrix.csv"
    close_df.to_csv(close_path)
    print(f"[EDA] Saved close prices to {close_path}")

    # 4. Build and save daily returns matrix
    returns_df = close_df.pct_change().dropna(how="all")
    returns_path = out_dir / "daily_returns_matrix.csv"
    returns_df.to_csv(returns_path)
    print(f"[EDA] Saved daily returns to {returns_path}")

    # 5. Summary stats for returns mean and standard deviation
    summary = returns_df.agg(["mean", "std"]).T
    summary_path = out_dir / "returns_summary_stats.csv"
    summary.to_csv(summary_path)
    print(f"[EDA] Saved returns summary stats to {summary_path}")

    # 6. Correlation heatmap across all assets in the universe
    corr = returns_df.corr()

    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr, interpolation="nearest")
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation of daily returns")
    plt.tight_layout()
    corr_fig_path = out_dir / "corr_heatmap.png"
    plt.savefig(corr_fig_path, dpi=150)
    plt.close()
    print(f"[EDA] Saved correlation heatmap to {corr_fig_path}")

    # 7. Normalized price plot for the entire universe
    plt.figure(figsize=(10, 5))
    (close_df / close_df.iloc[0]).plot(ax=plt.gca(), legend=False)
    plt.title("Universe normalized prices (start = 1.0)")
    plt.tight_layout()
    uni_fig_path = out_dir / "universe_normalized_prices.png"
    plt.savefig(uni_fig_path, dpi=150)
    plt.close()
    print(f"[EDA] Saved normalized price chart to {uni_fig_path}")

    print("[EDA] Done")


if __name__ == "__main__":
    main()
