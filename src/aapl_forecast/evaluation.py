"""Point-forecast accuracy and significance against the random walk."""

import numpy as np
import pandas as pd
from scipy import stats


BENCHMARK = "Random walk"


def diebold_mariano(loss_model: np.ndarray, loss_benchmark: np.ndarray, horizon: int) -> tuple[float, float]:
    """Harvey-Leybourne-Newbold corrected DM test; negative statistic favours the model.

    Overlapping h-step errors are autocorrelated up to lag h-1, so the long-run variance
    uses autocovariances to that lag (Bartlett weights keep it positive).
    """
    d = np.asarray(loss_model, dtype=float) - np.asarray(loss_benchmark, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 30:
        return np.nan, np.nan
    centred = d - d.mean()
    variance = centred @ centred / n
    for lag in range(1, horizon):
        weight = 1 - lag / horizon
        variance += 2 * weight * (centred[lag:] @ centred[:-lag]) / n
    if variance <= 0:
        return np.nan, np.nan
    statistic = d.mean() / np.sqrt(variance / n)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic *= correction
    p_value = 2 * stats.t.sf(abs(statistic), df=n - 1)
    return float(statistic), float(p_value)


def accuracy_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """MAE/RMSE, ratios to the random walk, direction hit rate and DM tests per horizon."""
    scored = predictions.dropna(subset=["forecast", "actual"])
    rows = []
    for horizon, group in scored.groupby("horizon"):
        # Compare every model on the origins where all models have a forecast.
        wide = group.pivot(index="origin", columns="model", values="forecast").dropna()
        actual = group.drop_duplicates("origin").set_index("origin")["actual"].loc[wide.index]
        bench_error = actual - wide[BENCHMARK]
        bench_mae = bench_error.abs().mean()
        bench_rmse = np.sqrt((bench_error**2).mean())
        up_share = float((actual > 0).mean())
        for model in wide.columns:
            error = actual - wide[model]
            nonzero = wide[model] != 0
            hit = (
                float((np.sign(wide[model][nonzero]) == np.sign(actual[nonzero])).mean())
                if nonzero.any()
                else np.nan
            )
            dm_abs = diebold_mariano(error.abs(), bench_error.abs(), horizon)
            dm_sq = diebold_mariano(error**2, bench_error**2, horizon)
            rows.append(
                {
                    "horizon": horizon,
                    "model": model,
                    "origins": len(actual),
                    "mae": error.abs().mean(),
                    "rmse": np.sqrt((error**2).mean()),
                    "mae_vs_rw": error.abs().mean() / bench_mae,
                    "rmse_vs_rw": np.sqrt((error**2).mean()) / bench_rmse,
                    "direction_hit_rate": hit,
                    "always_up_hit_rate": up_share,
                    "dm_abs_stat": dm_abs[0],
                    "dm_abs_p": dm_abs[1],
                    "dm_sq_stat": dm_sq[0],
                    "dm_sq_p": dm_sq[1],
                }
            )
    return pd.DataFrame(rows)
