import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_track_record as rtr  # noqa: E402


def test_wilson_matches_a_known_case_and_handles_no_data():
    assert rtr.wilson(5, 100) == pytest.approx((0.0215, 0.1118), abs=1e-3)
    assert all(v != v for v in rtr.wilson(0, 0))


def test_the_block_reports_live_and_backtest_coverage_side_by_side():
    summary = pd.DataFrame({"horizon": [1], "level": [0.8], "scored": [10], "coverage": [0.9]})
    log = pd.DataFrame({"origin": ["2026-10-08", "2026-10-09"], "inside": [1.0, None], "target_date": ["2026-10-09", None]})
    backtest = pd.DataFrame({"method": ["garch_conformal"], "horizon": [1], "level": [0.8], "coverage": [0.803]})
    block = rtr.render(summary, log, backtest)
    assert "| 1d | 80% | 10 | 90.0% (60% to 98%) | 80.3% |" in block and "2 origins, 1 intervals scored" in block


def test_update_needs_the_markers_and_is_idempotent(tmp_path):
    bare = tmp_path / "README.md"
    bare.write_text("no markers\n")
    with pytest.raises(ValueError):
        rtr.update(bare)
    marked = tmp_path / "M.md"
    marked.write_text(f"a\n{rtr.START}\nold\n{rtr.END}\nb\n")
    assert rtr.update(marked) is True and rtr.update(marked) is False
    assert marked.read_text().startswith("a\n") and marked.read_text().endswith("\nb\n")
