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

## Does it hold beyond AAPL? Design fixed before the panel was run — 2026-11-14

Written before any ticker other than AAPL was downloaded or scored. The README says a finding on AAPL alone would need testing on other tickers. This fixes the test.

- **Tickers.** Twenty large US stocks, chosen by one rule that does not look at returns: the largest companies by market capitalisation among S&P 500 members at 31 December 2015, the date before the first origin. The ranking was recalled from memory and was not looked up, so the list below is what is fixed and the ranking may be off at the margin. Yahoo symbols: AAPL, GOOGL, MSFT, AMZN, XOM, BRK-B, META, GE, JNJ, WFC, JPM, PG, WMT, T, VZ, PFE, KO, DIS, CVX, HD. All were still listed in 2026. That is survivorship, and it is stated as a limit.
- **Everything else as AAPL.** The same download and validation, the same window from 2010, origins from 2016-01-01, horizons 1, 5 and 21 days, refits every 21 trading days, the same models, the same intervals and the same tests. No setting is retuned per ticker.
- **A ticker that fails** to download or to validate is dropped and named. It is not replaced.
- **Point forecasts.** Per ticker, model and horizon, the ratio of RMSE to the random walk and the Diebold-Mariano p-value on squared error. Per model and horizon, across tickers: the number with p below 0.05, split into better and worse than the random walk, and the number that survive Benjamini-Hochberg at 5% across the tickers in that family.
- **Intervals.** Per ticker, method, horizon and level: coverage and interval score. At horizon 1: the number of tickers whose Kupiec p-value is at least 0.05, for each method and level. Across tickers: the median interval score of each method relative to static normal.
- **How it is read.** The AAPL findings are that no model beats the random walk, that gradient boosting is worse at 5 and 21 days, and that only the conformal 1-day intervals pass Kupiec at both levels. The README says for each of these whether the panel supports it, with counts. Nothing is retuned after the run.
- **Limits.** Twenty related stocks share market moves, so their tests are not independent. Survivors only. Yahoo adjusted prices for every ticker.
