import unittest

import numpy as np
import pandas as pd

from aapl_forecast.data import validate_prices
from aapl_forecast.evaluation import accuracy_table, diebold_mariano
from aapl_forecast.intervals import calibration_table, interval_forecasts, interval_score, kupiec_pof
from synthetic import garch_prices


class DieboldMarianoTests(unittest.TestCase):
    def test_clearly_better_model_is_significant(self):
        rng = np.random.default_rng(0)
        benchmark = rng.normal(0, 1, 500) ** 2 + 1
        model = benchmark - 0.5 + rng.normal(0, 0.1, 500)
        statistic, p_value = diebold_mariano(model, benchmark, horizon=1)
        self.assertLess(statistic, 0)
        self.assertLess(p_value, 1e-6)

    def test_equal_losses_are_not_significant(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0, 1, 2000) ** 2
        b = rng.permutation(a)
        _, p_value = diebold_mariano(a, b, horizon=5)
        self.assertGreater(p_value, 0.01)

    def test_short_samples_return_nan(self):
        self.assertTrue(np.isnan(diebold_mariano(np.ones(10), np.zeros(10), 1)[0]))


class IntervalTests(unittest.TestCase):
    def test_interval_score_penalises_misses(self):
        inside = interval_score(np.array([-1.0]), np.array([1.0]), np.array([0.0]), 0.9)
        outside = interval_score(np.array([-1.0]), np.array([1.0]), np.array([2.0]), 0.9)
        self.assertAlmostEqual(inside[0], 2.0)
        self.assertAlmostEqual(outside[0], 2.0 + 20.0)

    def test_kupiec_accepts_nominal_and_rejects_wrong_coverage(self):
        self.assertGreater(kupiec_pof(50, 1000, 0.95), 0.9)
        self.assertLess(kupiec_pof(120, 1000, 0.95), 1e-6)

    def test_conformal_intervals_are_calibrated_on_garch(self):
        prices = garch_prices(n=2500, seed=3)
        table = calibration_table(interval_forecasts(prices, "2015-01-01", (1,), levels=(0.9,)))
        conformal = table.set_index("method").loc["conformal"]
        self.assertAlmostEqual(conformal["coverage"], 0.9, delta=0.03)
        # A constant-width interval under-covers when volatility is high.
        static = table.set_index("method").loc["static"]
        self.assertLess(static["coverage_high_vol"], conformal["coverage_high_vol"])


class AccuracyTableTests(unittest.TestCase):
    def test_random_walk_ratios_are_one(self):
        rng = np.random.default_rng(2)
        origins = pd.bdate_range("2020-01-01", periods=300)
        actual = rng.normal(0, 0.01, 300)
        frame = pd.concat(
            [
                pd.DataFrame({"origin": origins, "horizon": 1, "model": "Random walk", "forecast": 0.0, "actual": actual}),
                pd.DataFrame({"origin": origins, "horizon": 1, "model": "Oracle-ish",
                              "forecast": actual * 0.5, "actual": actual}),
            ]
        )
        table = accuracy_table(frame).set_index("model")
        self.assertAlmostEqual(table.loc["Random walk", "mae_vs_rw"], 1.0)
        self.assertAlmostEqual(table.loc["Oracle-ish", "mae_vs_rw"], 0.5)
        self.assertEqual(table.loc["Oracle-ish", "direction_hit_rate"], 1.0)
        self.assertTrue(np.isnan(table.loc["Random walk", "direction_hit_rate"]))


class ValidationTests(unittest.TestCase):
    def frame(self):
        return garch_prices(n=50).reset_index()

    def test_accepts_clean_series(self):
        self.assertEqual(len(validate_prices(self.frame())), 50)

    def test_rejects_duplicates_and_unadjusted_splits(self):
        duplicated = pd.concat([self.frame(), self.frame().tail(1)])
        with self.assertRaises(ValueError):
            validate_prices(duplicated)
        split = self.frame()
        for column in ["Open", "High", "Low", "Close"]:
            split.loc[25:, column] /= 4
        with self.assertRaises(ValueError):
            validate_prices(split)


if __name__ == "__main__":
    unittest.main()
