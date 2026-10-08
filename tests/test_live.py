import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic import garch_prices

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from forecast_latest import COLUMNS, issue, score, track_record  # noqa: E402


class LiveForecastTests(unittest.TestCase):
    def test_issue_then_score_after_horizon(self):
        prices = garch_prices(n=1300)
        history = prices.iloc[:1250]
        log = issue(history, "2026-01-01T00:00:00+00:00").reindex(columns=COLUMNS)
        self.assertEqual(len(log), 6)
        self.assertTrue((log["lower_price"] < log["origin_close"]).all())
        self.assertTrue((log["upper_price"] > log["origin_close"]).all())
        # Ten more sessions mature the 1- and 5-day forecasts, not the 21-day one.
        scored = score(log, prices.iloc[:1261])
        matured = scored.dropna(subset=["inside"])
        self.assertEqual(set(matured["horizon"]), {1, 5})
        k = 1249
        expected = np.log(prices["Close"].iloc[k + 5] / prices["Close"].iloc[k])
        five = matured[matured.horizon == 5].iloc[0]
        self.assertAlmostEqual(five["realized_return"], expected)

    def test_scores_survive_dividend_rescaling(self):
        prices = garch_prices(n=1300)
        log = issue(prices.iloc[:1250], "t").reindex(columns=COLUMNS)
        rescaled = prices.copy()
        rescaled[["Open", "High", "Low", "Close"]] *= 0.97
        a = score(log, prices)
        b = score(log, rescaled)
        pd.testing.assert_series_equal(a["inside"], b["inside"])
        self.assertEqual(len(track_record(a)), 6)


if __name__ == "__main__":
    unittest.main()
