from __future__ import annotations

from ..config import EXPERIMENT_SYMBOLS
from .load import load_price_history


def snapshot_experiment_universe() -> None:
    """
    Ensure that all symbols used in the UROP experiments have
    local CSV snapshots in data/raw/.

    Run this once (or occasionally) so that your report analysis
    is based on a stable dataset.
    """
    for symbol in EXPERIMENT_SYMBOLS:
        print(f"Snapshotting {symbol}...")
        df = load_price_history(symbol)
        print(f"{symbol}: {len(df)} rows")


if __name__ == "__main__":
    snapshot_experiment_universe()
