# AAPL Forecasting — Rolling-Origin Backtest and Live Interval Forecasts

[![Evidence checks](https://github.com/dbechrakis/aapl-stock-exploratory-analysis/actions/workflows/evidence.yml/badge.svg)](https://github.com/dbechrakis/aapl-stock-exploratory-analysis/actions/workflows/evidence.yml)

A forecasting study built to answer two separate questions honestly:

1. **Can any model beat the random walk at forecasting AAPL's 1-, 5- and 21-day returns?** If it does, is the gain statistically significant?
2. **Can the *size* of the next move be forecast**, so the intervals hold their stated coverage in both calm and volatile markets?

The two questions are kept apart because they usually have different answers. Direction is close to unpredictable for a liquid large-cap stock. Volatility clusters, so uncertainty can be forecast.

**Stack:** Python · pandas · NumPy · SciPy · scikit-learn · Matplotlib · GitHub Actions

## Evidence status

> **AAPL results are not committed yet.** This environment could not reach a price provider. The first run of the [`Refresh backtest`](.github/workflows/refresh-backtest.yml) workflow downloads a fingerprinted snapshot, runs the backtest and commits `outputs/backtest/`. Results tables will be added here from that run, not before.

Until then, the evidence is the tested method. The section [Sanity check on synthetic data](#sanity-check-on-synthetic-data) shows the pipeline behaving as expected where the true answer is known.

## Design

```mermaid
flowchart LR
    A["Stooq / Yahoo<br/>(adjusted daily bars)"] --> B["Validated, SHA-256<br/>fingerprinted snapshot"]
    B --> C["Features known<br/>at the close of t"]
    C --> D["Rolling-origin backtest<br/>2016 → today, every trading day"]
    D --> E["Accuracy vs random walk<br/>+ Diebold–Mariano tests"]
    B --> F["Interval forecasts<br/>static · EWMA · conformal"]
    F --> G["Coverage, interval score,<br/>Kupiec test, by volatility regime"]
    B --> H["Scheduled live forecast<br/>+ public track record"]
```

### Point forecasts of h-day log returns (h = 1, 5, 21)

| Model | What it forecasts | Fitted on |
|---|---|---|
| Random walk | zero return (benchmark) | — |
| Drift | trailing 252-day mean return × h | expanding history |
| SES level / moving-average level | mean reversion towards a smoothed price, from the original coursework | α and window tuned once, on the pre-2016 burn-in |
| Ridge | 13 features: momentum, realised and EWMA volatility, 52-week-high distance, range, volume z-score, weekday | refitted every 21 trading days |
| Gradient boosting | the same 13 features, shallow trees | refitted every 21 trading days |

### Leakage rules (each one is tested)

- Every feature at origin *t* uses prices and volume up to the close of *t* only.
- At a refit on day *t*, a training row *j* is used only if its label had matured, meaning *j + h ≤ t*. Overlapping multi-day labels therefore never reach past the origin.
- Smoother hyperparameters are tuned once, on targets that matured before the first origin.
- Conformal intervals at *t* use only standardized outcomes that had matured by *t*.
- **Test:** perturbing every price after day *t* must leave features, point forecasts, tuning and interval bounds at or before *t* bit-identical. Deliberately reintroducing either leak makes the test fail; this was checked by mutation.

### Evaluation

- **Accuracy:** MAE and RMSE as ratios to the random walk (below 1 beats it).
- **Significance:** Diebold–Mariano with the Harvey–Leybourne–Newbold small-sample correction. A long-run variance up to lag *h − 1* accounts for overlapping multi-day errors.
- **Direction:** hit rate, compared with the trivial "always up" rate over the same period. A directional claim must beat that rate, not 50%.
- **Intervals at 80% and 95%:**
  - Three methods are compared, all centred on the random walk: static normal, EWMA normal (RiskMetrics λ = 0.94), and EWMA-scaled split-conformal (empirical quantiles of the last 1,000 matured standardized outcomes).
  - Each is scored on coverage overall and in the high- and low-volatility halves, on mean width and on the Winkler interval score.
  - Kupiec's proportion-of-failures test is reported at h = 1, the only horizon where outcomes are independent.

## Live forecasting in production

[`live-forecast.yml`](.github/workflows/live-forecast.yml) runs after every US close:

1. Downloads a fresh snapshot.
2. Issues 1-, 5- and 21-day conformal intervals at 80% and 95%.
3. Scores every earlier forecast whose horizon has passed.
4. Commits the result to [`outputs/live/forecast_log.csv`](outputs/live/) and [`track_record.csv`](outputs/live/).

The bounds are stored in log-return space and scored with close-to-close ratios. A later dividend re-adjustment of the price history therefore cannot change a past score; a test covers this. The log is append-only and timestamped. Over time it becomes an out-of-sample record that no backtest can be tuned to.

## Sanity check on synthetic data

On 4,200 days of simulated GARCH(1,1) prices, where direction is unpredictable by construction and volatility clusters, the pipeline reports what it should:

- **No model significantly beats the random walk.** Diebold–Mariano p > 0.05 for every learned model at every horizon. The moving-average reversion forecast is significantly *worse*.
- **The conformal interval is calibrated at h = 1.** It covers 94.9% against a nominal 95% (Kupiec p = 0.77).
- **The static interval fails when volatility is high.** It covers 92.4% in high-volatility periods, and its trailing one-year coverage falls to about 83% after a volatility spike.

These numbers validate the method, not AAPL. They are not reported as results.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
make check        # lint + evidence + 20 tests (~5 s)
make data         # download and fingerprint the snapshot (needs network access to Stooq or Yahoo)
make backtest     # ~2 min; writes outputs/backtest/
make forecast     # issue and score live forecasts
```

## Repository structure

```text
src/aapl_forecast/
├── data.py          # Download (Stooq → Yahoo fallback), validation, fingerprinting
├── features.py      # Features known at the close; h-day targets
├── backtest.py      # Rolling-origin backtest with matured-label refits
├── evaluation.py    # Accuracy vs random walk, Diebold–Mariano
└── intervals.py     # Static / EWMA / conformal intervals, coverage, Kupiec
scripts/             # download_data, run_backtest, forecast_latest
tests/               # look-ahead, statistics, validation and live-scoring tests
coursework/          # Original January-window coursework experiment (archived)
.github/workflows/   # CI, manual backtest refresh, scheduled live forecast
```

## Original coursework (archived)

The project began as an MSc coursework experiment. It trained on January only and forecast February–December price levels for 2023 and 2024, using naive, moving-average, SES and trend-seasonal methods. That code now sits in [`coursework/`](coursework/), with its corrected tuning rules still under test.

Its historical score table stays withdrawn: the original source CSV was never committed, so it cannot be reproduced. The rolling-origin study replaces that design, which had one training window per year and a single long horizon. The report and presentation are archived coursework artifacts. The original report credits Nikitas Valtadoros, David Frederick and Dimitrios Bechrakis.

## Limitations

- Prices come from a free provider's adjusted series. Adjustment conventions differ between providers, so the snapshot fingerprint matters more than the source name.
- One asset only. A positive finding on AAPL would still need testing on other tickers before it could be called general.
- No transaction costs or trading rules. This is forecast evaluation, not a strategy backtest.
- Learned models are refitted every 21 trading days, not daily, to keep compute small.

## Author

**Dimitris Bechrakis** — MSc Data Science, The American College of Greece.

## Licensing

See [licensing scope](LICENSING.md) for the MIT-licensed code and the separately governed coursework materials.
