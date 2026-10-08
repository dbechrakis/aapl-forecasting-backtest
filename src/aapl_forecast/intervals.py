"""Prediction intervals for h-day returns and their out-of-sample calibration.

The point forecast is the random walk (zero log return), so these intervals isolate one
question: can the *size* of the next move be forecast, even when its direction cannot?
"""

import numpy as np
import pandas as pd
from scipy import stats

from aapl_forecast.features import ewma_volatility, log_returns, target
from aapl_forecast.garch import garch_scale


LEVELS = (0.80, 0.95)
METHODS = ["static", "ewma", "conformal", "garch", "garch_conformal"]
CONFORMAL_WINDOW = 1000


def standardized_history(prices: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """h-day targets, EWMA scale at each origin, and targets divided by that scale."""
    vol = ewma_volatility(log_returns(prices)).to_numpy()
    y = target(prices, horizon).to_numpy()
    scale = vol * np.sqrt(horizon)
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized = y / scale
    return y, scale, standardized


def conformal_bounds(standardized: np.ndarray, scale: np.ndarray, i: int, horizon: int, level: float) -> tuple[float, float]:
    """Empirical quantiles of matured standardized outcomes, rescaled by today's EWMA scale."""
    end = i - horizon + 1  # targets observed by the close of origin i
    recent = standardized[max(1, end - CONFORMAL_WINDOW): end]
    recent = recent[np.isfinite(recent)]
    lo, hi = np.quantile(recent, [(1 - level) / 2, (1 + level) / 2])
    return float(lo * scale[i]), float(hi * scale[i])


def interval_forecasts(
    prices: pd.DataFrame,
    start: str,
    horizons: tuple[int, ...],
    levels: tuple[float, ...] = LEVELS,
) -> pd.DataFrame:
    """Static-normal, EWMA (normal, conformal) and GARCH (normal, conformal) intervals."""
    vol = ewma_volatility(log_returns(prices)).to_numpy()
    first = int(prices.index.searchsorted(pd.Timestamp(start)))
    origins = np.arange(first, len(prices))
    garch_scales, _ = garch_scale(prices, horizons, first)
    frames = []
    for horizon in horizons:
        y, scale, standardized = standardized_history(prices, horizon)
        g_scale = garch_scales[horizon]
        with np.errstate(divide="ignore", invalid="ignore"):
            g_standardized = y / g_scale
        for level in levels:
            z = stats.norm.ppf(0.5 + level / 2)
            rows = []
            for i in origins:
                history = y[1: i - horizon + 1]  # targets observed by the close of i
                static = z * history[~np.isnan(history)].std(ddof=1)
                lo_c, hi_c = conformal_bounds(standardized, scale, i, horizon, level)
                lo_g, hi_g = conformal_bounds(g_standardized, g_scale, i, horizon, level)
                rows.append(
                    (prices.index[i], -static, static, -z * scale[i], z * scale[i], lo_c, hi_c,
                     -z * g_scale[i], z * g_scale[i], lo_g, hi_g, y[i], vol[i])
                )
            block = pd.DataFrame(
                rows,
                columns=["origin", "static_lo", "static_hi", "ewma_lo", "ewma_hi",
                         "conformal_lo", "conformal_hi", "garch_lo", "garch_hi",
                         "garch_conformal_lo", "garch_conformal_hi", "actual", "ewma_vol"],
            )
            block["horizon"] = horizon
            block["level"] = level
            frames.append(block)
    return pd.concat(frames, ignore_index=True)


def interval_score(lower, upper, actual, level: float) -> np.ndarray:
    """Winkler/interval score: width plus a penalty for misses; lower is better."""
    alpha = 1 - level
    return (
        (upper - lower)
        + (2 / alpha) * (lower - actual) * (actual < lower)
        + (2 / alpha) * (actual - upper) * (actual > upper)
    )


def kupiec_pof(misses: int, n: int, level: float) -> float:
    """p-value of Kupiec's proportion-of-failures test for the nominal miss rate."""
    expected = 1 - level
    observed = misses / n
    if observed in (0.0, 1.0):
        log_lik_obs = 0.0
    else:
        log_lik_obs = misses * np.log(observed) + (n - misses) * np.log(1 - observed)
    log_lik_null = misses * np.log(expected) + (n - misses) * np.log(1 - expected)
    return float(stats.chi2.sf(-2 * (log_lik_null - log_lik_obs), df=1))


def calibration_table(intervals: pd.DataFrame) -> pd.DataFrame:
    """Coverage, width and interval score overall and in the high-volatility half."""
    scored = intervals.dropna(subset=["actual"]).copy()
    rows = []
    for (horizon, level), group in scored.groupby(["horizon", "level"]):
        high_vol = group["ewma_vol"] >= group["ewma_vol"].expanding().median()
        for method in METHODS:
            lo, hi = group[f"{method}_lo"], group[f"{method}_hi"]
            inside = (group["actual"] >= lo) & (group["actual"] <= hi)
            rows.append(
                {
                    "horizon": horizon,
                    "level": level,
                    "method": method,
                    "origins": len(group),
                    "coverage": inside.mean(),
                    "coverage_high_vol": inside[high_vol].mean(),
                    "coverage_low_vol": inside[~high_vol].mean(),
                    "mean_width": (hi - lo).mean(),
                    "interval_score": interval_score(lo, hi, group["actual"], level).mean(),
                    # Overlapping h>1 outcomes are dependent, so the test is valid only at h=1.
                    "kupiec_p": kupiec_pof(int((~inside).sum()), len(group), level)
                    if horizon == 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)
