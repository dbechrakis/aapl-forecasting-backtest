import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_track_record as rtr  # noqa: E402


class TrackRecordTests(unittest.TestCase):
    def test_wilson_matches_a_known_case_and_handles_no_data(self):
        lo, hi = rtr.wilson(5, 100)
        self.assertAlmostEqual(lo, 0.0215, places=3)
        self.assertAlmostEqual(hi, 0.1118, places=3)
        self.assertTrue(all(math.isnan(v) for v in rtr.wilson(0, 0)))

    def test_the_block_reports_live_and_backtest_coverage_side_by_side(self):
        summary = pd.DataFrame({"horizon": [1], "level": [0.8], "scored": [10], "coverage": [0.9]})
        log = pd.DataFrame({"origin": ["2026-10-08", "2026-10-09"], "inside": [1.0, None], "target_date": ["2026-10-09", None]})
        backtest = pd.DataFrame({"method": ["garch_conformal"], "horizon": [1], "level": [0.8], "coverage": [0.803]})
        block = rtr.render(summary, log, backtest)
        self.assertIn("| 1d | 80% | 10 | 90.0% (60% to 98%) | 80.3% |", block)
        self.assertIn("2 origins, 1 intervals scored", block)

    def test_update_needs_the_markers_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            bare = Path(tmp) / "README.md"
            bare.write_text("no markers\n")
            with self.assertRaises(ValueError):
                rtr.update(bare)
            marked = Path(tmp) / "M.md"
            marked.write_text(f"a\n{rtr.START}\nold\n{rtr.END}\nb\n")
            self.assertTrue(rtr.update(marked))
            self.assertFalse(rtr.update(marked))
            self.assertTrue(marked.read_text().startswith("a\n") and marked.read_text().endswith("\nb\n"))


if __name__ == "__main__":
    unittest.main()
