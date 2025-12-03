"""
03_results_summary.py

High level summary for the report.

Usage (from project root):
    source .venv/bin/activate
    python -m data.notebooks.03_results_summary

This script will:
  * Load metrics_all_symbols.csv produced by 02_model_diagnostics.py
  * For each symbol, identify which model has the lowest RMSE
  * Produce a simple ranking table of LSTM vs baselines
  * Save a clean CSV and print it in a report friendly format
"""

from pathlib import Path

import pandas as pd

OUT_DIR = Path("data/notebooks")


def main():
    metrics_path = OUT_DIR / "metrics_all_symbols.csv"
    if not metrics_path.exists():
        print(
            "[Summary] metrics_all_symbols.csv not found. "
            "Run 02_model_diagnostics.py first."
        )
        return

    metrics_df = pd.read_csv(metrics_path)

    # For each symbol, pick the model with lowest RMSE
    idx = metrics_df.groupby("Symbol")["RMSE"].idxmin()
    best_df = metrics_df.loc[idx].reset_index(drop=True)
    best_df = best_df.sort_values("Symbol")

    # Save best model per symbol
    best_path = OUT_DIR / "best_model_per_symbol.csv"
    best_df.to_csv(best_path, index=False)

    print("[Summary] Best model per symbol based on RMSE:\n")
    print(best_df)

    # Pivot for a wide table (symbols x models) useful for the report
    wide = metrics_df.pivot(index="Symbol", columns="Model", values="RMSE")
    wide_path = OUT_DIR / "rmse_pivot_table.csv"
    wide.to_csv(wide_path)

    print(f"\n[Summary] Saved:")
    print(f"  - {best_path}")
    print(f"  - {wide_path}")


if __name__ == "__main__":
    main()
