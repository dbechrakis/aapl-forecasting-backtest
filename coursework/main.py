# AAPL Stock Exploratory Analysis
# Forecasting framework: January training window, February-December evaluation.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "results"
for folder in ["2023", "2024", "comparison"]:
    (RESULTS / folder).mkdir(parents=True, exist_ok=True)


def mad(actual, forecast):
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.mean(np.abs(actual - forecast)))


def mape(actual, forecast):
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


def metrics(actual, forecast):
    return {"MAD": mad(actual, forecast), "MAPE": mape(actual, forecast)}


def naive_forecast(train, horizon):
    return np.repeat(train[-1], horizon)


def moving_average_forecast(train, horizon, window):
    return np.repeat(np.mean(train[-window:]), horizon)


def ses_forecast(train, horizon, alpha):
    level = float(train[0])
    for value in train[1:]:
        level = alpha * value + (1 - alpha) * level
    return np.repeat(level, horizon)


def choose_ma_window(train, candidates=range(2, 10)):
    best_window, best_error = None, np.inf
    candidates = [w for w in candidates if w < len(train)]
    if not candidates:
        raise ValueError("Not enough training observations for moving-average tuning")
    common_start = max(candidates)
    for window in candidates:
        if window >= len(train):
            continue
        one_step = np.array([np.mean(train[i-window:i]) for i in range(common_start, len(train))])
        error = mad(train[common_start:], one_step)
        if error < best_error:
            best_window, best_error = window, error
    return best_window


def choose_ses_alpha(train, candidates=np.linspace(0.1, 0.9, 9)):
    best_alpha, best_error = None, np.inf
    for alpha in candidates:
        level = float(train[0])
        errors = []
        for value in train[1:]:
            errors.append(abs(value - level))  # forecast before observing this value
            level = alpha * value + (1 - alpha) * level
        error = float(np.mean(errors))
        if error < best_error:
            best_alpha, best_error = float(alpha), error
    return best_alpha


def trend_season_forecast(train, horizon, cycle=21):
    """Extrapolate a linear trend plus fixed cycle effects learned only from train."""
    x = np.arange(len(train), dtype=float)
    slope, intercept, *_ = stats.linregress(x, train)
    trend_train = intercept + slope * x
    residuals = train - trend_train

    seasonal = np.zeros(cycle)
    for position in range(cycle):
        values = residuals[position::cycle]
        seasonal[position] = np.mean(values) if len(values) else 0.0
    seasonal -= seasonal.mean()

    x_future = np.arange(len(train), len(train) + horizon, dtype=float)
    trend_future = intercept + slope * x_future
    seasonal_future = seasonal[x_future.astype(int) % cycle]
    return trend_future, trend_future + seasonal_future


def load_year(year):
    path = PROCESSED / f"AAPL_{year}_cleaned.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed dataset: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    price_col = "Price" if "Price" in df.columns else "Close Price"
    if price_col not in df.columns:
        raise ValueError(f"Expected Price/Close Price column in {path}")
    return df.sort_values("Date").reset_index(drop=True), price_col


def analyze_year(year):
    df, price_col = load_year(year)
    jan = df["Date"].dt.month.eq(1)
    train = df.loc[jan, price_col].to_numpy(dtype=float)
    test = df.loc[~jan, price_col].to_numpy(dtype=float)
    train_dates = df.loc[jan, "Date"]
    test_dates = df.loc[~jan, "Date"]
    horizon = len(test)

    ma_window = choose_ma_window(train)
    alpha = choose_ses_alpha(train)

    forecasts = {
        "Naive": naive_forecast(train, horizon),
        f"MA (n={ma_window})": moving_average_forecast(train, horizon, ma_window),
        f"SES (alpha={alpha:.1f})": ses_forecast(train, horizon, alpha),
    }
    trend, seasonal = trend_season_forecast(train, horizon, cycle=21)
    forecasts["Linear Trend"] = trend
    forecasts["Trend + Seasonal (21-day cycle)"] = seasonal

    rows = []
    for method, forecast in forecasts.items():
        result = metrics(test, forecast)
        rows.append({"Method": method, **result})

    output = RESULTS / str(year)
    pd.DataFrame(rows).sort_values("MAD").to_csv(output / f"metrics_{year}.csv", index=False)

    plot_df = pd.DataFrame({"Date": test_dates, "Actual": test})
    for method, forecast in forecasts.items():
        plot_df[method] = forecast
    plot_df.to_csv(output / f"forecasts_{year}.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(train_dates, train, label="January training")
    plt.plot(test_dates, test, label="Actual Feb-Dec")
    for method, forecast in forecasts.items():
        plt.plot(test_dates, forecast, linewidth=1.5, label=method)
    plt.title(f"AAPL {year}: January-trained forecasts vs February-December actuals")
    plt.xlabel("Date")
    plt.ylabel("Closing price")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / f"forecast_comparison_{year}.png", dpi=200)
    plt.close()

    return pd.DataFrame(rows)


def main():
    results = {2023: analyze_year(2023), 2024: analyze_year(2024)}

    comparison = results[2023].merge(
        results[2024], on="Method", suffixes=("_2023", "_2024")
    )
    comparison["MAD_Difference_2024_minus_2023"] = (
        comparison["MAD_2024"] - comparison["MAD_2023"]
    )
    comparison["MAPE_Difference_2024_minus_2023"] = (
        comparison["MAPE_2024"] - comparison["MAPE_2023"]
    )
    comparison.sort_values("MAD_2024").to_csv(
        RESULTS / "comparison" / "year_over_year_comparison.csv", index=False
    )

    print("Analysis complete. All trend/seasonality components are fitted on January only.")


if __name__ == "__main__":
    main()
