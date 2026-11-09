"""Write the live track record into the README between two markers, so the page always shows the current score.

    python scripts/render_track_record.py          # run by the live-forecast workflow after each forecast

Coverage is shown with a 95% Wilson interval, because a few dozen scored forecasts cannot pin a rate down, and with
the backtest's coverage for the same horizon and level as the reference it should be compared with.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SUMMARY = ROOT / "outputs" / "live" / "track_record.csv"
LOG = ROOT / "outputs" / "live" / "forecast_log.csv"
BACKTEST = ROOT / "outputs" / "backtest" / "interval_calibration.csv"
START, END = "<!-- live-track-record:start -->", "<!-- live-track-record:end -->"


def wilson(hits: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def render(summary: pd.DataFrame, log: pd.DataFrame, backtest: pd.DataFrame) -> str:
    ref = backtest[backtest["method"] == "garch_conformal"].set_index(["horizon", "level"])["coverage"]
    scored = log.dropna(subset=["inside"])
    lines = [
        f"Live GARCH-conformal forecasts issued since {log['origin'].min()}: {log['origin'].nunique()} origins, "
        f"{len(scored)} intervals scored so far (latest outcome {scored['target_date'].max() if len(scored) else 'none'}). "
        "Updated automatically after each US close.",
        "",
        "| Horizon | Level | Scored | Live coverage (95% interval) | Backtest coverage |",
        "|---:|---:|---:|---|---:|",
    ]
    for row in summary.itertuples():
        lo, hi = wilson(row.coverage * row.scored, int(row.scored))
        back = ref.get((row.horizon, round(row.level, 2)), float("nan"))
        lines.append(f"| {row.horizon}d | {row.level:.0%} | {row.scored} | {row.coverage:.1%} ({lo:.0%} to {hi:.0%}) | {back:.1%} |")
    lines += ["", "Overlapping multi-day intervals are not independent, so the 5- and 21-day intervals are optimistic. "
              "A live coverage far outside its interval is a reason to look, not proof of a miscalibrated method."]
    return "\n".join(lines)


def update(readme: Path = README) -> bool:
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError("README is missing the live-track-record markers")
    block = render(pd.read_csv(SUMMARY), pd.read_csv(LOG), pd.read_csv(BACKTEST))
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{START}\n{block}\n{END}{tail}"
    if new != text:
        readme.write_text(new, encoding="utf-8", newline="\n")
    return new != text


if __name__ == "__main__":
    print("README updated" if update() else "README already current")
