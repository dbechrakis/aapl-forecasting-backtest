"""GARCH(1,1) volatility forecasts whose h-day horizon reverts towards the long-run level.

Scaling today's EWMA volatility by sqrt(h) assumes volatility stays where it is for the
whole horizon. GARCH instead forecasts each future day's variance decaying geometrically,
at rate alpha + beta, towards the unconditional variance, and sums them over the horizon.
Parameters are re-estimated every ``refit_every`` trading days on returns up to the origin.
"""

import warnings

from arch import arch_model
import numpy as np
import pandas as pd

from aapl_forecast.features import log_returns


SCALE = 100.0  # arch optimises better on percentage returns


def fit_garch(returns: np.ndarray) -> dict:
    """Maximum-likelihood GARCH(1,1) with a constant mean on returns observed so far."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = arch_model(returns * SCALE, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(
            disp="off", options={"maxiter": 500}
        )
    params = result.params
    return {
        "mu": float(params["mu"]) / SCALE,
        "omega": float(params["omega"]) / SCALE**2,
        "alpha": float(params["alpha[1]"]),
        "beta": float(params["beta[1]"]),
    }


def next_day_variance(returns: np.ndarray, params: dict, end: int) -> np.ndarray:
    """One-step-ahead variance for days 1..end, filtered with returns before each day."""
    omega, alpha, beta, mu = params["omega"], params["alpha"], params["beta"], params["mu"]
    residuals = returns[:end] - mu
    variance = np.empty(end + 1)
    variance[0] = np.var(residuals[: min(end, 252)]) if end > 1 else omega
    for t in range(end):
        variance[t + 1] = omega + alpha * residuals[t] ** 2 + beta * variance[t]
    return variance


def horizon_variance(next_variance: float, params: dict, horizon: int) -> float:
    """Sum of the 1..h-day-ahead variance forecasts, reverting at rate alpha + beta."""
    persistence = params["alpha"] + params["beta"]
    long_run = params["omega"] / max(1e-9, 1 - persistence)
    weights = persistence ** np.arange(horizon)
    return float(np.sum(long_run + weights * (next_variance - long_run)))


def garch_scale(
    prices: pd.DataFrame,
    horizons: tuple[int, ...],
    first_origin: int,
    refit_every: int = 21,
    min_history: int = 252,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    """h-day volatility forecast at every origin from ``min_history`` on.

    Origins before ``first_origin`` are also filled (needed as matured history for conformal
    intervals) with the parameters fitted at the first origin, so they never use later data
    than the origins that consume them.
    """
    returns = log_returns(prices).to_numpy()
    returns[0] = 0.0
    n = len(prices)
    scales = {h: np.full(n, np.nan) for h in horizons}
    fits = []
    refits = list(range(first_origin, n, refit_every))
    start = min_history
    for k, fit_at in enumerate(refits):
        params = fit_garch(returns[1: fit_at + 1])  # returns observed by the close of fit_at
        fits.append({"fit_at": str(prices.index[fit_at].date()), **params})
        block_end = refits[k + 1] if k + 1 < len(refits) else n
        block_start = start if k == 0 else fit_at
        # variance[t + 1] uses returns up to t, so it is the forecast made at the close of t.
        variance = next_day_variance(returns, params, block_end)
        for i in range(block_start, block_end):
            for horizon in horizons:
                scales[horizon][i] = np.sqrt(horizon_variance(variance[i + 1], params, horizon))
    return scales, fits
