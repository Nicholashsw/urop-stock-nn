# ==== normal imports below ====
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import (
    EXPERIMENT_SYMBOLS,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    LOOKBACK,
    TEST_SIZE,
    DEVICE,
)
from src.data.load import load_price_history



class LSTMRegressor(nn.Module):
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


def inverse_scale(scaled, scaler_params):
    min_ = scaler_params["min_"]
    scale_ = scaler_params["scale_"]
    return (scaled - min_) / scale_


def get_dates_for_test(df):
    n = len(df)
    split_idx = int(n * (1.0 - TEST_SIZE))
    return df.index[split_idx:]


def compute_metrics(pred, actual):
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    return mae, rmse


# %% choose symbol for analysis
symbol = "AAPL"

# %% load data and model
X_train = np.load(DATA_PROCESSED_DIR / f"{symbol}_X_train.npy")
y_train = np.load(DATA_PROCESSED_DIR / f"{symbol}_y_train.npy")
X_test = np.load(DATA_PROCESSED_DIR / f"{symbol}_X_test.npy")
y_test = np.load(DATA_PROCESSED_DIR / f"{symbol}_y_test.npy")
scaler_params = np.load(DATA_PROCESSED_DIR / f"{symbol}_scaler.npz")

df = load_price_history(symbol)
price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

model = LSTMRegressor(input_size=1)
state_dict = torch.load(MODELS_DIR / f"lstm_{symbol}.pt", map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# %% plot train vs test split
plt.figure(figsize=(10, 4))
plt.plot(df[price_col], label="Full series")
n = len(df)
split_idx = int(n * (1.0 - TEST_SIZE))
plt.axvline(df.index[split_idx], color="red")
plt.legend()
plt.title(f"{symbol} price with train test split")
plt.tight_layout()
plt.savefig(f"fig_{symbol}_train_test_split.png", dpi=300)
plt.show()

# %% LSTM predictions on test set
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
with torch.inference_mode():
    preds_t = model(X_test_t)
preds_scaled = preds_t.cpu().numpy().reshape(-1, 1)
targets_scaled = y_test.reshape(-1, 1)

preds_price = inverse_scale(preds_scaled, scaler_params).reshape(-1)
targets_price = inverse_scale(targets_scaled, scaler_params).reshape(-1)

dates_test_full = get_dates_for_test(df)
L = min(len(dates_test_full), len(preds_price), len(targets_price))
dates_test = dates_test_full[-L:]
preds_price = preds_price[-L:]
targets_price = targets_price[-L:]

mae, rmse = compute_metrics(preds_price, targets_price)
print("MAE", mae, "RMSE", rmse)

# %% plot actual vs predicted on test set
plt.figure(figsize=(10, 4))
plt.plot(dates_test, targets_price, label="Actual")
plt.plot(dates_test, preds_price, label="LSTM")
plt.legend()
plt.title(f"{symbol} test set actual vs LSTM")
plt.tight_layout()
plt.savefig(f"fig_{symbol}_test_actual_vs_lstm.png", dpi=300)
plt.show()
