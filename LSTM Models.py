"""
Bitcoin LSTM Model
------------------
End-to-end workflow for univariate & multivariate LSTM on Bitcoin time series,
using yfinance to fetch BTC-USD data directly.

This script is meant as a clean, working baseline you can extend.
"""

# =========================
# 0. IMPORTS
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# 1. DATA PREPROCESSING
# =========================

def load_bitcoin_data(ticker: str = "BTC-USD", start_date: str = "2018-01-01") -> pd.DataFrame:
    """
    Load Bitcoin OHLCV data using yfinance and return a cleaned DataFrame.

    Args:
        ticker: e.g. "BTC-USD"
        start_date: earliest date to download (YYYY-MM-DD)

    Returns:
        DataFrame indexed by date with columns:
        Open, High, Low, Close, Adj Close, Volume
    """
    df = yf.download(ticker, interval="1h", start=start_date, auto_adjust=True)

    # Ensure datetime index is sorted
    df = df.sort_index()

    # Some versions of yfinance return MultiIndex columns (e.g. ("Close","BTC-USD"))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Keep only standard OHLCV columns if present
    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep_cols]

    # Drop rows with missing Close prices
    df = df.dropna(subset=["Close"])

    return df


# =========================
# 2. FEATURE ENGINEERING
# =========================

def add_features_univariate(df: pd.DataFrame) -> pd.DataFrame:
    """
    For a univariate LSTM, we ultimately only feed the Close price,
    but we can still engineer features for analysis or future use.
    """
    df = df.copy()
    # df["return_1d"] = df["Close"].pct_change()
    df["log_return_1"] = np.log(df["Close"] / df["Close"].shift(1))
    df["ma_7"] = df["Close"].rolling(window=7).mean()
    df["ma_30"] = df["Close"].rolling(window=30).mean()
    df = df.dropna()
    return df


def add_features_multivariate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example multivariate feature set. Modify as you like.
    """
    df = df.copy()
    # df["return_1d"] = df["Close"].pct_change()
    df["log_return_1"] = np.log(df["Close"] / df["Close"].shift(1))
    df["ma_7"] = df["Close"].rolling(window=7).mean()
    df["ma_30"] = df["Close"].rolling(window=30).mean()
    df["high_low_spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["volatility_7d"] = df["log_return_1"].rolling(7).std()
    df["volatility_30d"] = df["log_return_1"].rolling(30).std()
    df["momentum_3"] = df["Close"].pct_change(3)
    df["momentum_7"] = df["Close"].pct_change(7)

    df = df.dropna()
    return df


# =========================
# 3. CREATE SLIDING WINDOWS
# =========================

def create_sliding_windows(
    data: np.ndarray,
    target: np.ndarray,
    window_size: int,
):
    """
    Create sequences (X, y) using a sliding window over the time series.

    Args:
        data:   2D array of shape (num_timesteps, num_features)
        target: 1D array of shape (num_timesteps,) – usually Close price shifted
        window_size: length of each input sequence

    Returns:
        X: 3D array (samples, window_size, num_features)
        y: 1D array (samples,)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)


# =========================
# 4. TRAIN/VAL/TEST SPLIT (strict chronological)
# =========================

def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    Chronological split for time series.

    Args:
        X, y: sequences
        train_ratio, val_ratio: fractions of total data

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    n_samples = X.shape[0]
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


# =========================
# 7. DESIGN A UNIVARIATE LSTM
# =========================

def build_univariate_lstm(window_size: int) -> Sequential:
    """
    Build a simple univariate LSTM model.
    Input shape: (timesteps, 1)
    """
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(window_size, 1)))
    model.add(Dropout(0.3))

    model.add(LSTM(64))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# =========================
# 8. DESIGN A MULTIVARIATE LSTM
# =========================

def build_multivariate_lstm(window_size: int, num_features: int) -> Sequential:
    """
    Build a multivariate LSTM model.
    Input shape: (timesteps, num_features)
    """
    model = Sequential()
    model.add(
        LSTM(
            64,
            input_shape=(window_size, num_features),
            return_sequences=True,
        )
    )
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # predict next Close price

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# =========================
# 9. TRAIN MODELS (fit)
# =========================

def train_lstm_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 50,
    batch_size: int = 32,
):
    """
    Train an LSTM model with early stopping.
    """
    es = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1,
    )
    return history


# =========================
# 10. EVALUATE MODELS
# =========================

def directional_accuracy(y_true, y_pred) -> float:
    """
    Percentage of times the predicted direction (up vs down) matches the true direction.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))

    min_len = min(len(true_dir), len(pred_dir))
    if min_len == 0:
        return 0.0

    true_dir = true_dir[:min_len]
    pred_dir = pred_dir[:min_len]
    correct = (true_dir == pred_dir).sum()
    return correct / len(true_dir)


def evaluate_forecast(y_true, y_pred, label: str = ""):
    """
    Print basic regression metrics and directional accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    da = directional_accuracy(y_true, y_pred)

    print(f"=== {label} ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"Directional Accuracy: {da:.4f}")


# =========================
# 11. MAIN WORKFLOW – UNIVARIATE
# =========================

def run_univariate_pipeline(ticker: str, window_size: int = 30):
    """
    Full univariate LSTM workflow:
        load data -> engineer features -> scale target -> windows -> split -> train -> evaluate
    """
    # 1) Load & engineer
    df = load_bitcoin_data(ticker)
    df_feat = add_features_univariate(df)

    # 2) Use only Close price for univariate LSTM
    close_vals = df_feat["Close"].values.reshape(-1, 1)

    # 3) Scale Close price to [0, 1]
    target_scaler = MinMaxScaler()
    close_scaled = target_scaler.fit_transform(close_vals).squeeze()

    # 4) Create sliding windows on scaled series
    X, y = create_sliding_windows(
        data=close_scaled.reshape(-1, 1),  # shape (timesteps, 1)
        target=close_scaled,
        window_size=window_size,
    )

    # 5) Chronological train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # 6) Build and train model
    model = build_univariate_lstm(window_size)
    print(model.summary())
    _ = train_lstm_model(model, X_train, y_train, X_val, y_val)

    # 7) Predict on test set (still in scaled space)
    y_pred_test_scaled = model.predict(X_test).flatten()

    # 8) Inverse-scale predictions and test targets back to price space
    y_pred_test = target_scaler.inverse_transform(
        y_pred_test_scaled.reshape(-1, 1)
    ).flatten()
    y_test_unscaled = target_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    # 9) Evaluate
    evaluate_forecast(y_test_unscaled, y_pred_test, label="Univariate LSTM")

    # 10) Plot
    plt.figure()
    plt.plot(y_test_unscaled, label="True")
    plt.plot(y_pred_test, label="Pred")
    plt.title("Univariate LSTM - Test Set")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# 12. MAIN WORKFLOW – MULTIVARIATE
# =========================

def run_multivariate_pipeline(ticker: str, window_size: int = 30):
    """
    Full multivariate LSTM workflow.
    """
    # 1) Load & engineer
    df = load_bitcoin_data(ticker)
    df_feat = add_features_multivariate(df)

    # 2) Feature matrix and target
    feature_cols = [
        "Close",
        "Open",
        "High",
        "Low",
        "Volume",
        "return_1d",
        "ma_7",
        "ma_30",
        "high_low_spread",
    ]
    feature_cols = [c for c in feature_cols if c in df_feat.columns]

    data = df_feat[feature_cols].values
    target = df_feat["Close"].values.reshape(-1, 1)

    # 3) Scale features and target separately
    feature_scaler = MinMaxScaler()
    data_scaled = feature_scaler.fit_transform(data)

    target_scaler = MinMaxScaler()
    target_scaled = target_scaler.fit_transform(target).squeeze()

    # 4) Create sliding windows on scaled data
    X, y = create_sliding_windows(
        data=data_scaled,
        target=target_scaled,
        window_size=window_size,
    )

    # 5) Chronological train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # 6) Build and train model
    num_features = X_train.shape[2]
    model = build_multivariate_lstm(window_size, num_features)
    print(model.summary())
    _ = train_lstm_model(model, X_train, y_train, X_val, y_val)

    # 7) Predict on test set (scaled)
    y_pred_test_scaled = model.predict(X_test).flatten()

    # 8) Inverse-scale predictions and targets
    y_pred_test = target_scaler.inverse_transform(
        y_pred_test_scaled.reshape(-1, 1)
    ).flatten()
    y_test_unscaled = target_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    # 9) Evaluate
    evaluate_forecast(y_test_unscaled, y_pred_test, label="Multivariate LSTM")

    # 10) Plot
    plt.figure()
    plt.plot(y_test_unscaled, label="True")
    plt.plot(y_pred_test, label="Pred")
    plt.title("Multivariate LSTM - Test Set")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    # You can change this ticker if you want to experiment with other assets.
    TICKER = "BTC-USD"

    # Run one or both pipelines
    # run_univariate_pipeline(TICKER, window_size=30)
    run_multivariate_pipeline(TICKER, window_size=30)
