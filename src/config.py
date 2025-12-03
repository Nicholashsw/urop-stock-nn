from __future__ import annotations

from pathlib import Path
import torch

# ==========
# Paths
# ==========

# Root data folders
DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

# Make sure directories exist when imported in scripts
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==========
# Data Universe (Organized + Expanded)
# ==========

DEFAULT_SYMBOL = "AAPL"

EXPERIMENT_SYMBOLS = [

    # ==========
    # US Large-Cap Tech
    # ==========
    "AAPL",     # Apple
    "MSFT",     # Microsoft
    "NVDA",     # Nvidia
    "AMZN",     # Amazon
    "META",     # Meta Platforms
    "GOOGL",    # Alphabet Class A
    "GOOG",     # Alphabet Class C
    "TSLA",     # Tesla
    "ADBE",     # Adobe
    "AMD",      # Advanced Micro Devices
    "CRM",      # Salesforce
    "NFLX",     # Netflix
    "AVGO",     # Broadcom

    # ==========
    # Consumer & Retail
    # ==========
    "COST",     # Costco
    "WMT",      # Walmart
    "MCD",      # McDonald's
    "DIS",      # Disney
    "HD",       # Home Depot
    "KO",       # Coca-Cola
    "PEP",      # PepsiCo

    # ==========
    # Finance & Payments
    # ==========
    "JPM",      # JPMorgan Chase
    "GS",       # Goldman Sachs
    "SCHW",     # Charles Schwab
    "V",        # Visa
    "MA",       # Mastercard  (optional but recommended)
    # "BAC",    # Bank of America (uncomment if needed)
    # "MS",     # Morgan Stanley (optional)

    # ==========
    # Healthcare & Pharma
    # ==========
    "UNH",      # UnitedHealth Group
    "ABBV",     # AbbVie
    "TMO",      # Thermo Fisher Scientific
    "PFE",      # Pfizer (optional)
    "JNJ",      # Johnson & Johnson (optional)

    # ==========
    # Energy & Industrials
    # ==========
    "XOM",      # ExxonMobil
    "CVX",      # Chevron
    "CAT",      # Caterpillar
    "BA",       # Boeing

    # ==========
    # Telecom & Utilities
    # ==========
    "VZ",       # Verizon
    "T",        # AT&T (optional)
    # "NEE",    # NextEra Energy (clean energy)

    # ==========
    # Global / International Majors
    # ==========
    "TM",       # Toyota Motor Corp
    "MC.PA",    # LVMH (Euronext Paris)
    # "NSRGY",  # Nestlé ADR (optional)
    # "BABA",   # Alibaba (optional)

    # ==========
    # ETFs & Indices
    # ==========
    "^GSPC",    # S&P 500 Index
    "^NDX",     # Nasdaq 100
    "^DJI",     # Dow Jones Industrial Average
    "SPY",      # SPDR S&P 500 ETF
    "QQQ",      # Invesco QQQ ETF
    "DIA",      # Dow Jones ETF
]

START_DATE = "2010-01-01"
END_DATE: str | None = None




# For backward compatibility with some scripts
SYMBOL = DEFAULT_SYMBOL

# ==========
# Model / training config
# ==========

# Window and split
LOOKBACK = 30         # number of days in each input sequence
TEST_SIZE = 0.2       # last 20 percent of samples for test

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Random seed for reproducibility
RANDOM_SEED = 42
