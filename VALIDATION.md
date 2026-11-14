# Validation record

**Scope of the latest review.** Rerun in the review of 2026-11-14: the 20-ticker panel (`outputs/panel/`), the unit tests, the evidence check and lint. Recorded history and not rerun in that review: the AAPL backtest in `outputs/backtest/` and the live forecast log, which its workflows repeat. Command: `make check`, then `PYTHONPATH=src python scripts/run_panel.py`.

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

## Wider lint rules — 2026-11-14

- Ruff now selects E, F, I, UP, B and SIM in `pyproject.toml`, with line length (E501), zip without strict (B905) and a call in a default argument (B008) ignored. The Makefile and the evidence workflow use that configuration. Before this the check selected syntax errors and unused names only.
- 37 findings were fixed automatically, import order and `datetime.UTC`. No behaviour changed. The 34 tests and the evidence check pass. The backtest was not rerun.

## A rule for which interval method goes live — fixed 2026-11-14, applied to nothing yet

The live forecast uses GARCH-conformal at both levels, although plain GARCH has a slightly better interval score at 95% in the backtest. This fixes how that choice is made so it is not made after seeing a result.

- **When.** Once a year, on the first trading day of November, on the backtest refreshed through the previous month. Not at any other time.
- **Rule.** A method replaces the live one only if, on that refreshed backtest, it has a lower interval score than the live method at both 80% and 95% for all three horizons, and its one-day coverage passes Kupiec's test at both levels. Otherwise the live method stays.
- **What is never done.** The method is not changed inside the year, after a bad week, or because of the live track record, whose 5- and 21-day intervals overlap and are not independent.
- **Status.** The rule has not been applied. The live method is unchanged.

### Result of the 20-ticker panel (same day)

`scripts/run_panel.py` ran once as designed. All 20 tickers downloaded and validated, none was dropped (`outputs/panel/manifest.json`, with each snapshot's fingerprint). The first launch oversubscribed the four cores (worker threads multiplied) and made no progress in an hour, so it was stopped and restarted with one thread per worker. That changed speed only. Nothing in the design, the settings or the data was changed, and the second launch is the only result.

- **No model beats the random walk on any ticker after Benjamini-Hochberg.** At most one ticker of 20 is better at p below 0.05 for any model.
- **Gradient boosting is significantly worse** on 15 of 20 tickers at 5 days (14 after correction), 7 at 21 days (2 after) and 10 at 1 day (3 after).
- **Ridge's 21-day gain on AAPL does not repeat.** The median RMSE ratio is above 1 at every horizon.
- **Only the conformal 1-day intervals pass Kupiec on every ticker.** EWMA conformal 20 of 20 at both levels, GARCH conformal 20 of 20 at 80% and 19 of 20 at 95%. GARCH normal 1 and 10, EWMA normal 1 and 4, static 3 and 4.
- **The interval-score ranking is only partly the AAPL one.** GARCH conformal is lowest at 80% for 1 and 5 days and plain GARCH at 21 days, and plain GARCH is lowest at 95% for every horizon.

The README states these with the tables, generated from `outputs/panel/` and checked by a test. The AAPL findings are supported. The one difference is in the interval-score ranking, reported as it came out. Nothing was retuned.
