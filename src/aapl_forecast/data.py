"""Price download, validation and fingerprinting."""

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}.us&i=d"
YAHOO_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?period1=0&period2=9999999999&interval=1d&events=split%2Cdiv"
)
COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


@dataclass(frozen=True)
class Snapshot:
    path: Path
    sha256: str
    rows: int
    first_date: str
    last_date: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, date-indexed daily OHLCV frame or raise on a broken series."""
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    prices = frame[COLUMNS].copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="raise")
    prices = prices.set_index("Date").sort_index()
    if prices.index.has_duplicates:
        raise ValueError("Duplicate trading dates")
    if prices[["Open", "High", "Low", "Close"]].isna().any().any():
        raise ValueError("Missing prices")
    if (prices[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")
    if (prices["High"] < prices[["Open", "Close", "Low"]].max(axis=1) - 1e-9).any():
        raise ValueError("High is below another price on the same day")
    if (prices["Low"] > prices[["Open", "Close", "High"]].min(axis=1) + 1e-9).any():
        raise ValueError("Low is above another price on the same day")
    daily = np.log(prices["Close"]).diff().abs()
    if (daily > np.log(2)).any():
        # A >100% one-day move in a large-cap series almost always means an unadjusted split.
        raise ValueError(f"Implausible one-day move on {daily.idxmax().date()}; check split adjustment")
    return prices.astype({"Volume": "float64"})


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (aapl-forecast research)"})
    return urlopen(request, timeout=60).read()


def from_stooq(symbol: str) -> pd.DataFrame:
    """Stooq daily bars, adjusted by Stooq for splits and dividends."""
    return pd.read_csv(io.BytesIO(_fetch(STOOQ_URL.format(symbol=symbol.lower()))))


def from_yahoo(symbol: str) -> pd.DataFrame:
    """Yahoo daily bars with OHLC rescaled by the split- and dividend-adjusted close."""
    result = json.loads(_fetch(YAHOO_URL.format(symbol=symbol.upper())))["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(result["timestamp"], unit="s").normalize(),
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": quote["close"],
            "Volume": quote["volume"],
        }
    ).dropna()
    adjusted = pd.Series(result["indicators"]["adjclose"][0]["adjclose"]).loc[frame.index]
    factor = adjusted.to_numpy() / frame["Close"].to_numpy()
    for column in ["Open", "High", "Low", "Close"]:
        frame[column] = frame[column] * factor
    return frame


SOURCES = {"stooq": from_stooq, "yahoo": from_yahoo}


def download_prices(symbol: str, target: Path, start: str = "2010-01-01") -> tuple[Snapshot, str]:
    """Try each source in turn, validate, and save a fingerprinted CSV snapshot."""
    errors = []
    for name, fetch in SOURCES.items():
        try:
            prices = validate_prices(fetch(symbol)).loc[start:]
        except Exception as error:  # noqa: BLE001 - report every source failure together
            errors.append(f"{name}: {error}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        body = prices.reset_index().to_csv(index=False, float_format="%.6f").encode()
        target.write_bytes(body)
        snapshot = Snapshot(
            target,
            sha256_bytes(body),
            len(prices),
            str(prices.index[0].date()),
            str(prices.index[-1].date()),
        )
        return snapshot, name
    raise RuntimeError("All price sources failed: " + "; ".join(errors))


def load_prices(path: Path, expected_sha256: str | None = None) -> pd.DataFrame:
    payload = path.read_bytes()
    if expected_sha256 and sha256_bytes(payload) != expected_sha256:
        raise ValueError("Price snapshot fingerprint does not match the manifest")
    return validate_prices(pd.read_csv(io.BytesIO(payload)))
