"""Features known at the close of each origin day, and h-day-ahead targets."""

import numpy as np
import pandas as pd


EWMA_LAMBDA = 0.94  # RiskMetrics daily decay


def log_returns(prices: pd.DataFrame) -> pd.Series:
    return np.log(prices["Close"]).diff().rename("r1")


def ewma_volatility(returns: pd.Series, decay: float = EWMA_LAMBDA) -> pd.Series:
    """RiskMetrics volatility at each close, using returns up to and including that day."""
    squared = returns.fillna(0.0) ** 2
    variance = squared.ewm(alpha=1 - decay, adjust=False).mean()
    return np.sqrt(variance).rename("ewma_vol")


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Every column at date t uses only prices and volume observed up to the close of t."""
    close = np.log(prices["Close"])
    r1 = log_returns(prices)
    volume = np.log(prices["Volume"].clip(lower=1))
    features = pd.DataFrame(index=prices.index)
    features["r1"] = r1
    for lag in [5, 21, 63, 252]:
        features[f"mom{lag}"] = close - close.shift(lag)
    for window in [5, 21, 63]:
        features[f"vol{window}"] = r1.rolling(window).std()
    features["ewma_vol"] = ewma_volatility(r1)
    features["dist_52w_high"] = close - close.rolling(252).max()
    features["range_5d"] = (np.log(prices["High"]) - np.log(prices["Low"])).rolling(5).mean()
    features["volume_z"] = (volume - volume.rolling(63).mean()) / volume.rolling(63).std()
    features["weekday"] = prices.index.dayofweek.astype(float)
    return features


def target(prices: pd.DataFrame, horizon: int) -> pd.Series:
    """Log return from the close of t to the close of t + horizon trading days."""
    close = np.log(prices["Close"])
    return (close.shift(-horizon) - close).rename(f"y{horizon}")
