import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from aapl_forecast.backtest import run_backtest
from aapl_forecast.features import build_features, target
from aapl_forecast.intervals import interval_forecasts
from synthetic import garch_prices


START = "2014-06-02"
FAST_MODELS = {"Ridge (features)": lambda: Ridge(alpha=10.0)}


def perturb_after(prices: pd.DataFrame, position: int) -> pd.DataFrame:
    changed = prices.copy()
    factor = np.exp(np.linspace(0.05, 0.4, len(prices) - position - 1))
    for column in ["Open", "High", "Low", "Close"]:
        changed.iloc[position + 1:, changed.columns.get_loc(column)] *= factor
    changed.iloc[position + 1:, changed.columns.get_loc("Volume")] *= 3
    return changed


class LookAheadTests(unittest.TestCase):
    """Changing prices after the close of t must not change anything forecast at t."""

    @classmethod
    def setUpClass(cls):
        cls.prices = garch_prices()
        cls.cut = int(cls.prices.index.searchsorted(pd.Timestamp(START))) + 150
        cls.changed = perturb_after(cls.prices, cls.cut)

    def test_features_use_only_the_past(self):
        before = build_features(self.prices).iloc[: self.cut + 1]
        after = build_features(self.changed).iloc[: self.cut + 1]
        pd.testing.assert_frame_equal(before, after)

    def test_point_forecasts_ignore_the_future(self):
        kwargs = dict(start=START, horizons=(1, 5, 21), refit_every=21, factories=FAST_MODELS)
        original, tuning = run_backtest(self.prices, **kwargs)
        changed, tuning_changed = run_backtest(self.changed, **kwargs)
        self.assertEqual(tuning, tuning_changed)
        cutoff = self.prices.index[self.cut]
        left = original[original.origin <= cutoff].reset_index(drop=True)
        right = changed[changed.origin <= cutoff].reset_index(drop=True)
        pd.testing.assert_series_equal(left["forecast"], right["forecast"])
        # The labels of late origins do change, which proves the perturbation reached the targets.
        self.assertFalse(np.allclose(left["actual"].fillna(0), right["actual"].fillna(0)))

    def test_intervals_ignore_the_future(self):
        original = interval_forecasts(self.prices, START, (1, 5))
        changed = interval_forecasts(self.changed, START, (1, 5))
        cutoff = self.prices.index[self.cut]
        bounds = ["static_lo", "static_hi", "ewma_lo", "ewma_hi", "conformal_lo", "conformal_hi",
                  "garch_lo", "garch_hi", "garch_conformal_lo", "garch_conformal_hi"]
        pd.testing.assert_frame_equal(
            original.loc[original.origin <= cutoff, bounds].reset_index(drop=True),
            changed.loc[changed.origin <= cutoff, bounds].reset_index(drop=True),
        )


class BacktestShapeTests(unittest.TestCase):
    def test_every_model_forecasts_every_origin(self):
        prices = garch_prices()
        predictions, _ = run_backtest(prices, start=START, horizons=(5,), factories=FAST_MODELS)
        counts = predictions.groupby("model")["forecast"].count()
        expected = len(prices) - prices.index.searchsorted(pd.Timestamp(START))
        self.assertEqual(set(counts), {expected})
        self.assertEqual(predictions["actual"].isna().sum(), 5 * predictions["model"].nunique())

    def test_target_is_forward_log_return(self):
        prices = garch_prices(n=30)
        y = target(prices, 3)
        expected = np.log(prices["Close"].iloc[3] / prices["Close"].iloc[0])
        self.assertAlmostEqual(y.iloc[0], expected)
        self.assertTrue(y.iloc[-3:].isna().all())

    def test_requires_history_before_first_origin(self):
        with self.assertRaises(ValueError):
            run_backtest(garch_prices(n=600), start="2012-06-01", factories=FAST_MODELS)


if __name__ == "__main__":
    unittest.main()
