"""Run the AAPL backtest on a fixed panel of tickers, as fixed in VALIDATION.md before the run.

Every ticker gets the same data validation, origins, horizons, models, intervals and tests as AAPL. A ticker that fails to
download or to validate is dropped and named. Prices are not committed. The manifest records each snapshot's fingerprint.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from aapl_forecast.backtest import HORIZONS, run_backtest
from aapl_forecast.data import download_prices, load_prices
from aapl_forecast.evaluation import accuracy_table
from aapl_forecast.intervals import calibration_table, interval_forecasts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "panel"
CACHE = ROOT / "data" / "panel"
START = "2016-01-01"
TICKERS = ("AAPL", "GOOGL", "MSFT", "AMZN", "XOM", "BRK-B", "META", "GE", "JNJ", "WFC",
           "JPM", "PG", "WMT", "T", "VZ", "PFE", "KO", "DIS", "CVX", "HD")
ALPHA = 0.05
BENCHMARK_METHOD = "static"


def benjamini_hochberg(p_values, alpha: float = ALPHA) -> np.ndarray:
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    ranked = p[order]
    threshold = alpha * (np.arange(1, len(p) + 1) / len(p))
    passed = np.nonzero(ranked <= threshold)[0]
    keep = np.zeros(len(p), bool)
    if len(passed):
        keep[order[: passed.max() + 1]] = True
    return keep


def analyse_ticker(prices: pd.DataFrame, start: str = START, factories=None, refit_every: int = 21) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions, _ = run_backtest(prices, start=start, horizons=HORIZONS, refit_every=refit_every, factories=factories)
    accuracy = accuracy_table(predictions)
    calibration = calibration_table(interval_forecasts(prices, start, HORIZONS))
    return accuracy, calibration


def fetch_and_analyse(symbol: str) -> dict:
    saved = CACHE / f"result_{symbol}.pkl"
    if saved.exists():
        return pd.read_pickle(saved)
    try:
        snapshot, source = download_prices(symbol, CACHE / f"{symbol}.csv")
        prices = load_prices(snapshot.path, snapshot.sha256)
        accuracy, calibration = analyse_ticker(prices)
    except Exception as error:  # noqa: BLE001
        return {"symbol": symbol, "error": str(error)}
    result = {"symbol": symbol, "source": source, "sha256": snapshot.sha256, "rows": snapshot.rows, "first_date": snapshot.first_date,
              "last_date": snapshot.last_date, "accuracy": accuracy.assign(ticker=symbol), "calibration": calibration.assign(ticker=symbol)}
    pd.to_pickle(result, saved)
    print(f"{symbol} done", flush=True)
    return result


def point_summary(accuracy: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    rows = []
    for (horizon, model), group in accuracy.groupby(["horizon", "model"]):
        if model == "Random walk":
            continue
        p = group["dm_sq_p"].to_numpy()
        better = group["rmse_vs_rw"].to_numpy() < 1
        survive = benjamini_hochberg(p, alpha)
        rows.append({"horizon": horizon, "model": model, "tickers": len(group), "median_rmse_vs_rw": float(group["rmse_vs_rw"].median()),
                     "better_p_below_005": int(((p < alpha) & better).sum()), "worse_p_below_005": int(((p < alpha) & ~better).sum()),
                     "better_after_bh": int((survive & better).sum()), "worse_after_bh": int((survive & ~better).sum())})
    return pd.DataFrame(rows)


def interval_summary(calibration: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    rows = []
    reference = calibration[calibration["method"] == BENCHMARK_METHOD].set_index(["ticker", "horizon", "level"])["interval_score"]
    for (horizon, level, method), group in calibration.groupby(["horizon", "level", "method"]):
        relative = group.set_index("ticker")["interval_score"] / reference.xs((horizon, level), level=("horizon", "level"))
        row = {"horizon": horizon, "level": level, "method": method, "tickers": len(group), "mean_coverage": float(group["coverage"].mean()),
               "median_score_vs_static": float(relative.median())}
        if horizon == 1:
            row["kupiec_pass"] = int((group["kupiec_p"] >= alpha).sum())
        rows.append(row)
    return pd.DataFrame(rows)


MODEL_NAMES = {"Drift (252d mean)": "Drift", "Gradient boosting (features)": "Gradient boosting", "Moving-average level": "Moving-average level",
               "Ridge (features)": "Ridge", "SES level": "SES level"}
METHOD_NAMES = {"static": "Static normal", "ewma": "EWMA normal", "conformal": "EWMA conformal", "garch": "GARCH normal", "garch_conformal": "GARCH conformal"}
START_MARK, END_MARK = "<!-- panel:start -->", "<!-- panel:end -->"


def readme_block(point: pd.DataFrame, intervals: pd.DataFrame) -> str:
    rows = ["| Horizon | Model | Median RMSE vs random walk | Better, p < 0.05 | Worse, p < 0.05 | Better after BH | Worse after BH |", "|---:|---|---:|---:|---:|---:|---:|"]
    for r in point.sort_values(["horizon", "model"]).itertuples():
        rows.append(f"| {r.horizon}d | {MODEL_NAMES[r.model]} | {r.median_rmse_vs_rw:.3f} | {r.better_p_below_005} | {r.worse_p_below_005} | {r.better_after_bh} | {r.worse_after_bh} |")
    cover = ["| Level | Method | Tickers passing Kupiec at 1 day | Mean coverage at 1 day | Median interval score vs static at 1, 5 and 21 days |", "|---|---|---:|---:|---|"]
    for level in (0.8, 0.95):
        for method in METHOD_NAMES:
            g = intervals[(intervals["level"] == level) & (intervals["method"] == method)].set_index("horizon")
            cover.append(f"| {int(level * 100)}% | {METHOD_NAMES[method]} | {int(g.loc[1, 'kupiec_pass'])} of {int(g.loc[1, 'tickers'])} | {g.loc[1, 'mean_coverage']:.3f} "
                         f"| {g.loc[1, 'median_score_vs_static']:.3f}, {g.loc[5, 'median_score_vs_static']:.3f}, {g.loc[21, 'median_score_vs_static']:.3f} |")
    return "\n".join(["**Point forecasts, ratio of RMSE to the random walk, Diebold-Mariano on squared error, 20 tickers.**", "", *rows, "", "**Intervals, 20 tickers.**", "", *cover])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch_and_analyse, TICKERS))
    dropped = {r["symbol"]: r["error"] for r in results if "error" in r}
    kept = [r for r in results if "error" not in r]
    accuracy = pd.concat([r["accuracy"] for r in kept], ignore_index=True)
    calibration = pd.concat([r["calibration"] for r in kept], ignore_index=True)
    accuracy.to_csv(OUT / "accuracy_by_ticker.csv", index=False, float_format="%.6f")
    calibration.to_csv(OUT / "calibration_by_ticker.csv", index=False, float_format="%.6f")
    point = point_summary(accuracy)
    intervals = interval_summary(calibration)
    point.to_csv(OUT / "point_summary.csv", index=False, float_format="%.6f")
    intervals.to_csv(OUT / "interval_summary.csv", index=False, float_format="%.6f")
    manifest = {"run_utc": datetime.now(UTC).isoformat(timespec="seconds"), "start": START, "tickers_requested": list(TICKERS),
                "dropped": dropped, "snapshots": {r["symbol"]: {k: r[k] for k in ("source", "sha256", "rows", "first_date", "last_date")} for r in kept}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with pd.option_context("display.width", 220):
        print(point.round(3).to_string(index=False))
        print(intervals.round(3).to_string(index=False))
    print("dropped:", dropped)


if __name__ == "__main__":
    main()
