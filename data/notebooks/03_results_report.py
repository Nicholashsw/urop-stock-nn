"""
03_results_report

Purpose
    Summarize results across the universe
    Combine
        metrics_all_symbols.csv
        daily_returns_matrix.csv
    Produce
        per symbol performance ranking
        basic risk and return measures
        csv and a couple of simple figures

Output
    results_summary_table.csv
    model_rankings_by_symbol.csv
    mae_barplot_LSTM_vs_best_baseline.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    here = Path(__file__).resolve().parent

    metrics_path = here / "metrics_all_symbols.csv"
    returns_path = here / "daily_returns_matrix.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}, run 02_model_diagnostics first")
    if not returns_path.exists():
        raise FileNotFoundError(f"Missing {returns_path}, run 01_universe_eda first")

    metrics = pd.read_csv(metrics_path)
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)

    # 1. Risk and return per symbol
    returns_summary = returns.agg(["mean", "std"]).T
    returns_summary.rename(columns={"mean": "mean_daily_return", "std": "daily_volatility"}, inplace=True)

    # 2. Pivot metrics by symbol and model
    pivot_mae = metrics.pivot(index="Symbol", columns="Model", values="MAE")
    pivot_rmse = metrics.pivot(index="Symbol", columns="Model", values="RMSE")

    # 3. For each symbol, find which model has lowest MAE
    best_model = pivot_mae.idxmin(axis=1)
    best_mae = pivot_mae.min(axis=1)

    ranking_df = pd.DataFrame({
        "best_model_by_MAE": best_model,
        "best_MAE": best_mae,
    })

    # 4. Merge everything into one summary table
    results_summary = returns_summary.join(pivot_mae, how="left", rsuffix="_MAE")
    results_summary = results_summary.join(pivot_rmse, how="left", rsuffix="_RMSE")
    results_summary = results_summary.join(ranking_df, how="left")

    summary_path = here / "results_summary_table.csv"
    results_summary.to_csv(summary_path)
    print(f"[Report] Saved combined results summary to {summary_path}")

    ranking_path = here / "model_rankings_by_symbol.csv"
    ranking_df.to_csv(ranking_path)
    print(f"[Report] Saved per symbol best model ranking to {ranking_path}")

    # 5. LSTM vs best baseline bar plot for quick visual comparison
    if "LSTM" in pivot_mae.columns:
        lstm_mae = pivot_mae["LSTM"]

        # For each symbol, best baseline among non LSTM models
        baseline_cols = [c for c in pivot_mae.columns if c != "LSTM"]
        if baseline_cols:
            best_baseline_mae = pivot_mae[baseline_cols].min(axis=1)

            comp_df = pd.DataFrame({
                "LSTM": lstm_mae,
                "Best baseline": best_baseline_mae,
            }).dropna()

            # Sort by LSTM error for cleaner plot
            comp_df_sorted = comp_df.sort_values("LSTM")

            plt.figure(figsize=(10, 6))
            x = np.arange(len(comp_df_sorted))
            width = 0.4

            plt.bar(x - width / 2, comp_df_sorted["LSTM"], width, label="LSTM")
            plt.bar(x + width / 2, comp_df_sorted["Best baseline"], width, label="Best baseline")

            plt.xticks(x, comp_df_sorted.index, rotation=90)
            plt.ylabel("MAE")
            plt.title("LSTM vs best baseline by symbol (lower is better)")
            plt.legend()
            plt.tight_layout()

            fig_path = here / "mae_barplot_LSTM_vs_best_baseline.png"
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"[Report] Saved MAE comparison bar plot to {fig_path}")
        else:
            print("[Report] No baseline models found in metrics, skipping bar plot")
    else:
        print("[Report] LSTM column not found in metrics, skipping comparison plot")

    print("[Report] Done")


if __name__ == "__main__":
    main()
