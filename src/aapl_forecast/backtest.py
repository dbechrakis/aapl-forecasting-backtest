"""Rolling-origin backtest of h-day log-return forecasts.

At every origin t the forecasters see only data up to the close of t. Learned models are
refitted every ``refit_every`` trading days on rows whose h-day target had matured by t,
so a training label never extends past the origin it is used from.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aapl_forecast.features import build_features, target


HORIZONS = (1, 5, 21)
FEATURES = [
    "r1", "mom5", "mom21", "mom63", "mom252", "vol5", "vol21", "vol63",
    "ewma_vol", "dist_52w_high", "range_5d", "volume_z", "weekday",
]
SES_ALPHAS = np.round(np.arange(0.05, 1.0, 0.05), 2)
MA_WINDOWS = [5, 10, 21, 42, 63]


def learned_models() -> dict[str, Callable[[], object]]:
    return {
        "Ridge (features)": lambda: make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))
        ),
        "Gradient boosting (features)": lambda: HistGradientBoostingRegressor(
            learning_rate=0.03,
            max_iter=200,
            max_depth=3,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def ses_level(log_close: pd.Series, alpha: float) -> pd.Series:
    return log_close.ewm(alpha=alpha, adjust=False).mean()


def tune_on_burn_in(log_close: pd.Series, y: pd.Series, burn_in_end: int, horizon: int) -> tuple[float, int]:
    """Pick the SES alpha and MA window on rows whose targets matured before the first origin."""
    matured = slice(252, burn_in_end - horizon + 1)
    actual = y.iloc[matured]

    def mae(forecast: pd.Series) -> float:
        return float((actual - forecast.iloc[matured]).abs().mean())

    alpha = min(SES_ALPHAS, key=lambda a: mae(ses_level(log_close, a) - log_close))
    window = min(MA_WINDOWS, key=lambda w: mae(log_close.rolling(w).mean() - log_close))
    return float(alpha), int(window)


def baseline_forecasts(prices: pd.DataFrame, horizon: int, burn_in_end: int) -> tuple[dict, dict]:
    log_close = np.log(prices["Close"])
    y = target(prices, horizon)
    r1 = log_close.diff()
    alpha, window = tune_on_burn_in(log_close, y, burn_in_end, horizon)
    forecasts = {
        "Random walk": pd.Series(0.0, index=prices.index),
        "Drift (252d mean)": r1.rolling(252).mean() * horizon,
        "SES level": ses_level(log_close, alpha) - log_close,
        "Moving-average level": log_close.rolling(window).mean() - log_close,
    }
    return forecasts, {"ses_alpha": alpha, "ma_window": window}


def learned_forecasts(
    features: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    origins: np.ndarray,
    refit_every: int,
    factories: dict[str, Callable[[], object]],
) -> dict[str, pd.Series]:
    usable = features[FEATURES].notna().all(axis=1).to_numpy()
    X = features[FEATURES].to_numpy()
    target_values = y.to_numpy()
    out = {name: np.full(len(features), np.nan) for name in factories}
    for block_start in range(0, len(origins), refit_every):
        block = origins[block_start:block_start + refit_every]
        fit_at = block[0]
        # Row j's label is the close at j + horizon, observed by the origin only if j + horizon <= fit_at.
        train = np.flatnonzero(usable[: fit_at - horizon + 1])
        for name, factory in factories.items():
            model = factory().fit(X[train], target_values[train])
            rows = block[usable[block]]
            out[name][rows] = model.predict(X[rows])
    return {name: pd.Series(values, index=features.index) for name, values in out.items()}


def run_backtest(
    prices: pd.DataFrame,
    start: str = "2016-01-01",
    horizons: tuple[int, ...] = HORIZONS,
    refit_every: int = 21,
    factories: dict[str, Callable[[], object]] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return long-format forecasts for every origin on/after ``start`` and tuning metadata."""
    factories = learned_models() if factories is None else factories
    features = build_features(prices)
    first_origin = int(prices.index.searchsorted(pd.Timestamp(start)))
    if first_origin < 504:
        raise ValueError("Need at least two years of history before the first origin")
    origins = np.arange(first_origin, len(prices))
    frames, tuning = [], {}
    for horizon in horizons:
        y = target(prices, horizon)
        forecasts, tuning[f"h{horizon}"] = baseline_forecasts(prices, horizon, first_origin)
        forecasts.update(learned_forecasts(features, y, horizon, origins, refit_every, factories))
        for model, series in forecasts.items():
            frames.append(
                pd.DataFrame(
                    {
                        "origin": prices.index[origins],
                        "horizon": horizon,
                        "model": model,
                        "forecast": series.iloc[origins].to_numpy(),
                        "actual": y.iloc[origins].to_numpy(),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True), tuning
