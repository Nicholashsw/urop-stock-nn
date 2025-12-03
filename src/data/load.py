from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf

from ..config import START_DATE, END_DATE


def load_price_history(
    symbol: str,
    start: Optional[str] = START_DATE,
    end: Optional[str] = END_DATE,
) -> pd.DataFrame:
    """
    Simple loader:
    Always fetch from yfinance.

    This avoids any CSV format issues and keeps the pipeline simple.
    """
    print(f"[load_price_history] Downloading {symbol} from Yahoo Finance...")
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise RuntimeError(f"Could not download data for {symbol}.")

    df = df.sort_index()
    df.index.name = "Date"
    return df
