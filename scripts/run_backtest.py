"""Run the rolling-origin backtest on the committed snapshot and write the evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aapl_forecast.backtest import HORIZONS, run_backtest  # noqa: E402
from aapl_forecast.data import load_prices  # noqa: E402
from aapl_forecast.evaluation import BENCHMARK, accuracy_table  # noqa: E402
from aapl_forecast.intervals import calibration_table, interval_forecasts  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "backtest"
START = "2016-01-01"


def plot_cumulative_advantage(predictions: pd.DataFrame, path: Path) -> None:
    """Cumulative squared-error advantage over the random walk; above zero means better."""
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(15, 4), sharex=True)
    scored = predictions.dropna(subset=["forecast", "actual"])
    for ax, horizon in zip(axes, HORIZONS):
        wide = scored[scored.horizon == horizon].pivot(index="origin", columns="model", values="forecast").dropna()
        actual = scored[scored.horizon == horizon].drop_duplicates("origin").set_index("origin")["actual"].loc[wide.index]
        bench = (actual - wide[BENCHMARK]) ** 2
        for model in wide.columns.drop(BENCHMARK):
            ax.plot(wide.index, (bench - (actual - wide[model]) ** 2).cumsum(), label=model, linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{horizon}-day horizon")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Cumulative SSE saved vs random walk")
    axes[-1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rolling_coverage(intervals: pd.DataFrame, path: Path) -> None:
    group = intervals[(intervals.horizon == 1) & (intervals.level == 0.95)].dropna(subset=["actual"])
    group = group.set_index("origin")
    fig, ax = plt.subplots(figsize=(12, 4))
    for method, label in [("static", "Static normal"), ("ewma", "EWMA normal"), ("conformal", "EWMA conformal"),
                          ("garch_conformal", "GARCH conformal")]:
        inside = (group["actual"] >= group[f"{method}_lo"]) & (group["actual"] <= group[f"{method}_hi"])
        ax.plot(inside.rolling(252).mean(), label=label, linewidth=1.2)
    ax.axhline(0.95, color="black", linestyle="--", linewidth=0.8, label="Nominal 95%")
    ax.set_ylabel("Trailing 252-day coverage")
    ax.set_title("1-day 95% interval coverage")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    prices = load_prices(ROOT / manifest["path"], manifest["sha256"])
    predictions, tuning = run_backtest(prices, start=START)
    accuracy = accuracy_table(predictions)
    intervals = interval_forecasts(prices, START, HORIZONS)
    calibration = calibration_table(intervals)

    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT / "predictions.csv.gz", index=False, float_format="%.8f", compression="gzip")
    accuracy.to_csv(OUT / "accuracy.csv", index=False, float_format="%.6f")
    calibration.to_csv(OUT / "interval_calibration.csv", index=False, float_format="%.6f")
    plot_cumulative_advantage(predictions, OUT / "cumulative_advantage.png")
    plot_rolling_coverage(intervals, OUT / "rolling_coverage.png")
    scored = predictions.dropna(subset=["actual"])
    (OUT / "run_manifest.json").write_text(
        json.dumps(
            {
                "data_sha256": manifest["sha256"],
                "data_last_date": manifest["last_date"],
                "first_origin": str(scored["origin"].min().date()),
                "last_scored_origin": str(scored["origin"].max().date()),
                "horizons": list(HORIZONS),
                "refit_every_trading_days": 21,
                "burn_in_tuning": tuning,
            },
            indent=2,
        )
        + "\n"
    )
    with pd.option_context("display.width", 200):
        print(accuracy[["horizon", "model", "origins", "mae_vs_rw", "rmse_vs_rw",
                        "direction_hit_rate", "always_up_hit_rate", "dm_sq_p"]].round(4).to_string(index=False))
        print(calibration[["horizon", "level", "method", "coverage", "coverage_high_vol",
                           "mean_width", "interval_score", "kupiec_p"]].round(4).to_string(index=False))
    print(f"Tuning on burn-in: {tuning}; mean absolute 1-day return {np.abs(scored[scored.horizon == 1].actual).mean():.4f}")


if __name__ == "__main__":
    main()
