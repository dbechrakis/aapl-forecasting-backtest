# Validation record

Review date: 2026-09-05 (UTC).

Forecasting logic tested on controlled sequences; historical metrics remain unverified without the original source CSV.

Historical records are not labelled as freshly reproduced results.

## Rolling-origin forecasting framework — 2026-10-08

- Added `src/aapl_forecast/`: a rolling-origin backtest of 1-, 5- and 21-day log returns with six models, Diebold–Mariano tests against the random walk, and static, EWMA and conformal prediction intervals.
- A look-ahead test perturbs every price after a cutoff. It requires features, point forecasts, tuning and interval bounds up to the cutoff to stay identical. Two deliberate leaks were injected (training on unmatured labels, conformal quantiles over unmatured outcomes), and each made the test fail.
- 20 tests pass on Python 3.12 with the pinned requirements, and Ruff passes.
- The full pipeline ran on a 4,200-day synthetic GARCH series: backtest about 90 s, then evidence check and live forecast/scoring. Results are described in the README as a method sanity check only.
- **No AAPL results were produced.** The development environment's network policy blocked every price provider tried (Stooq, Yahoo, Alpha Vantage, Nasdaq). AAPL evidence will come from the `Refresh backtest` workflow and will be recorded here with its snapshot fingerprint.
