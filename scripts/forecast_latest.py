"""Issue today's interval forecasts and score every earlier forecast that has matured.

Runs on a schedule (see .github/workflows/live-forecast.yml). The log stores bounds in
log-return space and scores them with close-to-close ratios from a fresh snapshot, so
later dividend re-adjustment of the price history cannot move past scores.

    python scripts/forecast_latest.py                      # download, forecast, score
    python scripts/forecast_latest.py --prices data/raw/aapl_daily.csv
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from aapl_forecast.backtest import HORIZONS
from aapl_forecast.data import download_prices, load_prices
from aapl_forecast.intervals import LEVELS, conformal_bounds, standardized_history


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "live" / "forecast_log.csv"
SUMMARY = ROOT / "outputs" / "live" / "track_record.csv"
COLUMNS = [
    "issued_at_utc", "origin", "origin_close", "horizon", "level",
    "lower_return", "upper_return", "lower_price", "upper_price",
    "target_date", "realized_return", "inside",
]


def issue(prices: pd.DataFrame, issued_at: str) -> pd.DataFrame:
    i = len(prices) - 1
    close = float(prices["Close"].iloc[i])
    rows = []
    for horizon in HORIZONS:
        _, scale, standardized = standardized_history(prices, horizon)
        for level in LEVELS:
            lo, hi = conformal_bounds(standardized, scale, i, horizon, level)
            rows.append(
                {
                    "issued_at_utc": issued_at,
                    "origin": prices.index[i].date().isoformat(),
                    "origin_close": round(close, 4),
                    "horizon": horizon,
                    "level": level,
                    "lower_return": lo,
                    "upper_return": hi,
                    "lower_price": round(close * np.exp(lo), 2),
                    "upper_price": round(close * np.exp(hi), 2),
                }
            )
    return pd.DataFrame(rows)


def score(log: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    log = log.copy()
    log["target_date"] = log["target_date"].astype(object)
    positions = {date.date().isoformat(): k for k, date in enumerate(prices.index)}
    close = prices["Close"].to_numpy()
    for idx, row in log[log["inside"].isna()].iterrows():
        k = positions.get(row["origin"])
        if k is None or k + int(row["horizon"]) >= len(prices):
            continue
        end = k + int(row["horizon"])
        realized = float(np.log(close[end] / close[k]))
        log.loc[idx, "target_date"] = prices.index[end].date().isoformat()
        log.loc[idx, "realized_return"] = realized
        log.loc[idx, "inside"] = float(row["lower_return"] <= realized <= row["upper_return"])
    return log


def track_record(log: pd.DataFrame) -> pd.DataFrame:
    scored = log.dropna(subset=["inside"])
    if scored.empty:
        return pd.DataFrame(columns=["horizon", "level", "scored", "coverage"])
    return (
        scored.groupby(["horizon", "level"])["inside"]
        .agg(scored="size", coverage="mean")
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", type=Path, help="Use a local snapshot instead of downloading")
    args = parser.parse_args()
    if args.prices:
        prices = load_prices(args.prices)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot, source = download_prices("AAPL", Path(tmp) / "aapl.csv")
            prices = load_prices(snapshot.path)
        print(f"Downloaded {snapshot.rows} rows from {source}; last close {snapshot.last_date}")

    log = pd.read_csv(LOG, dtype={"origin": str}) if LOG.exists() else pd.DataFrame(columns=COLUMNS)
    origin = prices.index[-1].date().isoformat()
    if origin in set(log["origin"]):
        print(f"Forecasts for {origin} already issued; scoring only")
    else:
        issued = issue(prices, datetime.now(timezone.utc).isoformat(timespec="seconds"))
        log = pd.concat([log, issued], ignore_index=True) if len(log) else issued.reindex(columns=COLUMNS)
    log = score(log.reindex(columns=COLUMNS), prices)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG, index=False, float_format="%.6f")
    summary = track_record(log)
    summary.to_csv(SUMMARY, index=False, float_format="%.4f")
    latest = log[log["origin"] == origin][["horizon", "level", "lower_price", "upper_price"]]
    print(f"AAPL close {prices['Close'].iloc[-1]:.2f} on {origin}")
    print(latest.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
