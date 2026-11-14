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
