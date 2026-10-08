"""Synthetic GARCH(1,1) price paths: volatility clusters, direction is unpredictable."""

import numpy as np
import pandas as pd


def garch_prices(n: int = 1200, seed: int = 7, start: str = "2012-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2e-6, 0.08, 0.9
    variance = omega / (1 - alpha - beta)
    returns = np.empty(n)
    for t in range(n):
        returns[t] = 0.0004 + np.sqrt(variance) * rng.standard_normal()
        variance = omega + alpha * (returns[t] - 0.0004) ** 2 + beta * variance
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": rng.integers(5e7, 1e8, n).astype(float)},
        index=pd.DatetimeIndex(dates, name="Date"),
    )
