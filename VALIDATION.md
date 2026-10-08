# Validation record

Review date: 2026-09-05 (UTC).

Forecasting logic tested on controlled sequences; historical metrics remain unverified without the original source CSV.

Historical records are not labelled as freshly reproduced results.

## Rolling-origin forecasting framework — 2026-10-08

- Added `src/aapl_forecast/`: a rolling-origin backtest of 1-, 5- and 21-day log returns with six models, Diebold–Mariano tests against the random walk, and static, EWMA and conformal prediction intervals.
- A look-ahead test perturbs every price after a cutoff. It requires features, point forecasts, tuning and interval bounds up to the cutoff to stay identical. Two deliberate leaks were injected (training on unmatured labels, conformal quantiles over unmatured outcomes), and each made the test fail.
- 20 tests pass on Python 3.12 with the pinned requirements, and Ruff passes.
- The full pipeline ran on a 4,200-day synthetic GARCH series: backtest about 90 s, then evidence check and live forecast/scoring. Results are described in the README as a method sanity check only.
- No AAPL results were produced in the development environment; AAPL evidence comes from the `Refresh backtest` workflow (below).

## First AAPL backtest — 2026-10-08

- The first `Refresh backtest` run failed. AAPL's genuine -52% day on 2000-09-29 tripped the split check before the pre-2010 history was trimmed. Validation now applies to the analysed window only (with a test), and Yahoo is tried first because Stooq serves a browser challenge to scripts.
- The second run downloaded 4,217 Yahoo adjusted daily rows (2010-01-04 to 2026-10-08), SHA-256 `44d92e48ad954becd81a8ca010aef1b0e79bbbfaff08b061d6bdee5ffae3ee1c`. It backtested 2,706 origins (2016-01-04 to 2026-10-07), passed the evidence check, and committed `outputs/backtest/`.
- Headline results, reported in the README:
  - No model is significantly better than the random walk at any horizon.
  - Gradient boosting is significantly worse at 5 and 21 days.
  - Only the conformal 1-day intervals pass Kupiec's coverage test, at both 80% and 95%.

## Unsettled-session fix — 2026-10-08

- The first manual `Live forecast` run (16:51 UTC, during US trading hours) used an intraday price as the 2026-10-08 close. The first backtest snapshot (downloaded 16:37 UTC) also ended with that partial bar, which entered one scored 1-day outcome.
- Downloads now drop any session that has not settled. A session counts as settled after 17:00 New York time; a test covers this. The intraday live forecast rows were removed rather than kept as a track record, and the backtest was refreshed on settled bars only. The refreshed snapshot ends 2026-10-07 (4,216 rows, SHA-256 `c9c5e42cd31b448857a8933586d1a2631cc9b57755d4cfdc09070ddd60cd6fb0`). It gives 2,705 1-day origins, and every conclusion in the README is unchanged.

## GARCH volatility intervals — 2026-10-08

- Added GARCH(1,1) h-day volatility, refitted every 21 trading days on returns up to the origin, with the variance forecast reverting geometrically to its long-run level. GARCH-normal and GARCH-conformal intervals were added to the comparison.
- The look-ahead test covers the GARCH bounds. A deliberate leak (fitting on all returns) made it fail.
- Rerun on the committed snapshot (SHA-256 `c9c5e42c…`):
  - Point forecasts are byte-identical to the previous run.
  - GARCH-conformal has the best interval score at 80% for all three horizons, and GARCH-normal the best at 95%.
  - At 21 days and 80%, GARCH-conformal covers 79.5% in high-volatility and 78.2% in low-volatility periods, against 87.4% and 69.4% for EWMA-conformal.
- The live forecast now issues GARCH-conformal intervals and records the method in each log row.
- 24 tests pass and Ruff passes.
