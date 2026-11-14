import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from synthetic import garch_prices
from test_backtest import START, perturb_after

from aapl_forecast.backtest import run_backtest
from aapl_forecast.intervals import interval_forecasts

SPEC = importlib.util.spec_from_file_location("run_panel", Path(__file__).resolve().parents[1] / "scripts" / "run_panel.py")
run_panel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_panel)


class BenjaminiHochbergTests(unittest.TestCase):
    def test_matches_a_hand_worked_example(self):
        p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
        keep = run_panel.benjamini_hochberg(p, 0.05)
        self.assertEqual(keep.tolist(), [True, True, False, False, False, False, False, False, False, False])

    def test_keeps_everything_up_to_the_largest_passing_rank(self):
        keep = run_panel.benjamini_hochberg([0.04, 0.001, 0.049], 0.05)
        self.assertTrue(keep.all())

    def test_keeps_nothing_when_no_p_value_passes(self):
        self.assertFalse(run_panel.benjamini_hochberg([0.5, 0.6, 0.7], 0.05).any())


class SummaryTests(unittest.TestCase):
    def test_point_summary_counts_better_and_worse_before_and_after_correction(self):
        accuracy = pd.DataFrame({
            "horizon": 1, "model": ["Ridge"] * 4 + ["Random walk"] * 4, "ticker": list("ABCD") * 2,
            "rmse_vs_rw": [0.9, 0.95, 1.1, 1.2] + [1.0] * 4, "dm_sq_p": [0.001, 0.4, 0.002, 0.8] + [1.0] * 4})
        row = run_panel.point_summary(accuracy).iloc[0]
        self.assertEqual((row["better_p_below_005"], row["worse_p_below_005"]), (1, 1))
        self.assertEqual((row["better_after_bh"], row["worse_after_bh"]), (1, 1))
        self.assertAlmostEqual(row["median_rmse_vs_rw"], 1.025)

    def test_interval_summary_counts_kupiec_passes_at_horizon_one_only(self):
        rows = []
        for ticker, p in (("A", 0.3), ("B", 0.01)):
            for method, score in (("static", 1.0), ("garch_conformal", 0.9)):
                rows.append({"ticker": ticker, "horizon": 1, "level": 0.95, "method": method, "coverage": 0.94, "interval_score": score, "kupiec_p": p})
                rows.append({"ticker": ticker, "horizon": 5, "level": 0.95, "method": method, "coverage": 0.9, "interval_score": score, "kupiec_p": np.nan})
        summary = run_panel.interval_summary(pd.DataFrame(rows))
        one = summary[(summary.horizon == 1) & (summary.method == "garch_conformal")].iloc[0]
        five = summary[(summary.horizon == 5) & (summary.method == "garch_conformal")].iloc[0]
        self.assertEqual(one["kupiec_pass"], 1)
        self.assertAlmostEqual(one["median_score_vs_static"], 0.9)
        self.assertNotIn("kupiec_pass", five.dropna().index)


class SecondTickerLookAheadTests(unittest.TestCase):
    def test_point_forecasts_and_intervals_of_another_series_ignore_the_future(self):
        prices = garch_prices(seed=23)
        cut = int(prices.index.searchsorted(pd.Timestamp(START))) + 150
        changed = perturb_after(prices, cut)
        cutoff = prices.index[cut]
        kwargs = dict(start=START, horizons=(1, 5), refit_every=21, factories={"Ridge (features)": lambda: Ridge(alpha=10.0)})
        original, _ = run_backtest(prices, **kwargs)
        altered, _ = run_backtest(changed, **kwargs)
        left = original[original.origin <= cutoff].reset_index(drop=True)
        right = altered[altered.origin <= cutoff].reset_index(drop=True)
        pd.testing.assert_series_equal(left["forecast"], right["forecast"])
        bounds_a = interval_forecasts(prices, START, (1,))
        bounds_b = interval_forecasts(changed, START, (1,))
        columns = [c for c in bounds_a.columns if c.endswith("_lo") or c.endswith("_hi")]
        a = bounds_a[bounds_a.origin <= cutoff].reset_index(drop=True)
        b = bounds_b[bounds_b.origin <= cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(a[columns], b[columns])


if __name__ == "__main__":
    unittest.main()


class ReadmeMatchesResultsTests(unittest.TestCase):
    def test_the_readme_block_is_generated_from_the_committed_summaries(self):
        root = Path(__file__).resolve().parents[1]
        point = pd.read_csv(root / "outputs" / "panel" / "point_summary.csv")
        intervals = pd.read_csv(root / "outputs" / "panel" / "interval_summary.csv")
        readme = (root / "README.md").read_text(encoding="utf-8")
        block = readme[readme.index(run_panel.START_MARK) + len(run_panel.START_MARK): readme.index(run_panel.END_MARK)].strip("\n")
        self.assertEqual(block, run_panel.readme_block(point, intervals))

    def test_the_prose_claims_match_the_counts(self):
        root = Path(__file__).resolve().parents[1]
        point = pd.read_csv(root / "outputs" / "panel" / "point_summary.csv")
        self.assertEqual(int(point["better_after_bh"].sum()), 0)
        gb = point[point["model"].str.startswith("Gradient")].set_index("horizon")
        self.assertEqual((int(gb.loc[5, "worse_p_below_005"]), int(gb.loc[5, "worse_after_bh"])), (15, 14))
        self.assertEqual((int(gb.loc[21, "worse_p_below_005"]), int(gb.loc[21, "worse_after_bh"])), (7, 2))
        self.assertEqual((int(gb.loc[1, "worse_p_below_005"]), int(gb.loc[1, "worse_after_bh"])), (10, 3))
        self.assertLessEqual(int(point["better_p_below_005"].max()), 1)
        self.assertTrue((point[point["model"].str.startswith("Ridge")]["median_rmse_vs_rw"] > 1).all())
        intervals = pd.read_csv(root / "outputs" / "panel" / "interval_summary.csv")
        one = intervals[intervals["horizon"] == 1].set_index(["level", "method"])["kupiec_pass"]
        self.assertEqual([int(one[(0.8, "conformal")]), int(one[(0.95, "conformal")]), int(one[(0.8, "garch_conformal")]), int(one[(0.95, "garch_conformal")])], [20, 20, 20, 19])
        self.assertEqual([int(one[(0.8, "garch")]), int(one[(0.95, "garch")]), int(one[(0.8, "ewma")]), int(one[(0.95, "ewma")]), int(one[(0.8, "static")]), int(one[(0.95, "static")])], [1, 10, 1, 4, 3, 4])

    def test_the_interval_score_statements_match_the_summary(self):
        root = Path(__file__).resolve().parents[1]
        intervals = pd.read_csv(root / "outputs" / "panel" / "interval_summary.csv")
        score = intervals.set_index(["level", "horizon", "method"])["median_score_vs_static"]
        for horizon in (1, 5):
            self.assertEqual(score.loc[0.8, horizon].idxmin(), "garch_conformal")
        self.assertEqual(score.loc[0.8, 21].idxmin(), "garch")
        for horizon in (1, 5, 21):
            self.assertEqual(score.loc[0.95, horizon].idxmin(), "garch")
        self.assertGreater(score.loc[0.8, 21, "conformal"], 1.0)
        self.assertGreater(score.loc[0.95, 21, "conformal"], 1.0)
