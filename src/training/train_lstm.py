from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    SYMBOL,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    BATCH_SIZE,
    EPOCHS,
    LR,
    DEVICE,
)


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


def load_processed(symbol: str):
    X_train = np.load(DATA_PROCESSED_DIR / f"{symbol}_X_train.npy")
    y_train = np.load(DATA_PROCESSED_DIR / f"{symbol}_y_train.npy")
    X_test = np.load(DATA_PROCESSED_DIR / f"{symbol}_X_test.npy")
    y_test = np.load(DATA_PROCESSED_DIR / f"{symbol}_y_test.npy")
    return X_train, y_train, X_test, y_test


def train_symbol(symbol: str) -> None:
    X_train, y_train, X_test, y_test = load_processed(symbol)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMRegressor(input_size=1).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_test_loss = float("inf")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"lstm_{symbol}.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(Xb).view(-1, 1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        test_losses = []
        with torch.inference_mode():
            for Xb, yb in test_loader:
                preds = model(Xb).view(-1, 1)
                loss = criterion(preds, yb)
                test_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        test_loss = float(np.mean(test_losses))

        print(f"Epoch {epoch:03d}  train_loss {train_loss:.6f}  test_loss {test_loss:.6f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), model_path)
            print(f"  Saved new best model for {symbol} to {model_path}")

    print(f"Training complete for {symbol}. Best test loss {best_test_loss}")


if __name__ == "__main__":
    train_symbol(SYMBOL)
