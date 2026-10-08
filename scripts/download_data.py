"""Download a fingerprinted AAPL daily price snapshot and record it in the manifest."""

from datetime import datetime, timezone
import json
from pathlib import Path

from aapl_forecast.data import download_prices


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "raw" / "aapl_daily.csv"
MANIFEST = ROOT / "data" / "manifest.json"


def main() -> None:
    snapshot, source = download_prices("AAPL", TARGET)
    manifest = {
        "symbol": "AAPL",
        "source": source,
        "adjustment": "split- and dividend-adjusted closes as provided by the source",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "path": str(TARGET.relative_to(ROOT)),
        "sha256": snapshot.sha256,
        "rows": snapshot.rows,
        "first_date": snapshot.first_date,
        "last_date": snapshot.last_date,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
