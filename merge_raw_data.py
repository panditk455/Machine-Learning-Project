import pandas as pd
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

def parse_category_and_date(filename: str):
    m = re.match(r"^(.*)_(\d{8})\.csv$", filename)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def main():
    csv_files = list(DATA_DIR.rglob("*.csv"))
    print(f"Searching: {DATA_DIR}")
    print(f"Found {len(csv_files)} CSV files")

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {DATA_DIR}")

    dfs = []
    skipped = 0

    for f in csv_files:
        try:
            df = pd.read_csv(f)

            category, datestr = parse_category_and_date(f.name)
            df["category"] = category if category else "unknown"
            df["date"] = pd.to_datetime(datestr, format="%Y%m%d", errors="coerce")
            df["source_path"] = str(f.relative_to(ROOT))

            dfs.append(df)
        except Exception as e:
            skipped += 1
            print(f"Skipping {f}: {e}")

    print(f"Loaded {len(dfs)} files, skipped {skipped}")
    if not dfs:
        raise ValueError("No CSVs successfully loaded.")

    data = pd.concat(dfs, ignore_index=True)
    print("Merged shape:", data.shape)

    out_path = ROOT / "merged_all.csv"
    data.to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()