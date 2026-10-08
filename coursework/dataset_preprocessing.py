from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data/raw/AAPL_daily_2023_2024_raw.csv"
OUTPUT_DIR = ROOT / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_FILE}. "
            "Place the AAPL daily dataset in data/raw/ before running preprocessing."
        )

    df = pd.read_csv(RAW_FILE)
    df = df.dropna(how="all").copy()

    # Named fields prevent silently treating Open or Adjusted Close as Close.
    df.columns = [str(c).strip() for c in df.columns]
    aliases = {"Close Price": "Close", "Price": "Close", "Close/Last": "Close"}
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df and "Close" not in df})
    required = {"Date", "Close", "High", "Low", "Open", "Volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected named columns {sorted(required)}; got {list(df.columns)}")
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names in source data")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    if df["Date"].duplicated().any():
        raise ValueError("Duplicate trading dates in source data")
    forecast_df = df[["Date", "Close"]].rename(columns={"Close": "Price"})

    for year in [2023, 2024]:
        year_df = forecast_df[forecast_df["Date"].dt.year == year].copy()
        if year_df.empty:
            raise ValueError(f"No observations found for {year}.")
        year_df.to_csv(OUTPUT_DIR / f"AAPL_{year}_cleaned.csv", index=False)

    forecast_df.to_csv(OUTPUT_DIR / "AAPL_cleaned.csv", index=False)

    summary = pd.DataFrame([
        {
            "dataset": "AAPL_cleaned",
            "rows": len(forecast_df),
            "start": forecast_df["Date"].min().date(),
            "end": forecast_df["Date"].max().date(),
        },
        *[
            {
                "dataset": f"AAPL_{year}_cleaned",
                "rows": len(forecast_df[forecast_df["Date"].dt.year == year]),
                "start": forecast_df[forecast_df["Date"].dt.year == year]["Date"].min().date(),
                "end": forecast_df[forecast_df["Date"].dt.year == year]["Date"].max().date(),
            }
            for year in [2023, 2024]
        ],
    ])
    summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
