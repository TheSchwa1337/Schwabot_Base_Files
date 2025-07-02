from pathlib import Path

import pandas as pd



HIST_DIR = Path("data/historical")
PREPROC_DIR = Path("data/preprocessed")


def concat_csv_to_parquet(asset: str, quote: str = "usdc"):
    folder = HIST_DIR / f"{asset.lower()}_{quote.lower()}"
    if not folder.exists():
        raise FileNotFoundError(
            f"No folder found for {asset.upper()}_{quote.upper()} history."
        )

    all_csvs = sorted(folder.glob("*.csv"))
    if not all_csvs:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    frames = [pd.read_csv(f) for f in all_csvs]
    combined = pd.concat(frames, ignore_index=True)

    # Optional: sort and clean
    combined = combined.drop_duplicates().sort_values("timestamp")
    combined.reset_index(drop=True, inplace=True)

    out_path = PREPROC_DIR / f"{asset.lower()}_{quote.lower()}.parquet"
    combined.to_parquet(out_path)
    print(f"[✓] Saved {asset.upper()}_{quote.upper()} history → {out_path}")
    return out_path


def load_historical_data(asset: str, quote: str = "usdc") -> pd.DataFrame:
    parquet_file = PREPROC_DIR / f"{asset.lower()}_{quote.lower()}.parquet"
    if not parquet_file.exists():
        print(f"[!] Parquet not found, creating for {asset.upper()}...")
        concat_csv_to_parquet(asset, quote)
    return pd.read_parquet(parquet_file)
