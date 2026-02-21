import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MERGED_PATH = ROOT / "merged_all.csv"
OUT_PANEL_PATH = ROOT / "panel_data.csv"

def main():
    if not MERGED_PATH.exists():
        raise FileNotFoundError(f"Missing {MERGED_PATH}. Run 01_merge_raw.py first.")

    data = pd.read_csv(MERGED_PATH)

    # ---- Basic cleaning / typing ----
    # Use the parsed 'date' from merge if present; otherwise fallback to date_scraped.
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
    elif "date_scraped" in data.columns:
        data["date"] = pd.to_datetime(data["date_scraped"], format="%Y%m%d", errors="coerce")
    else:
        raise ValueError("No 'date' or 'date_scraped' column found.")

    data["price"] = pd.to_numeric(data.get("currentMin"), errors="coerce")
    data["original_price"] = pd.to_numeric(data.get("originalMin"), errors="coerce")

    data["averageRating"] = pd.to_numeric(data.get("averageRating"), errors="coerce")
    data["reviewCount"] = pd.to_numeric(data.get("reviewCount"), errors="coerce")

    # Keep only rows with a SKU and a price
    data = data.dropna(subset=["skuId", "price", "date"])

    # ---- Discount features ----
    # Avoid divide-by-zero by masking invalid original_price
    valid_orig = (data["original_price"].notna()) & (data["original_price"] > 0)
    data["discount"] = 0.0
    data.loc[valid_orig, "discount"] = (
        (data.loc[valid_orig, "original_price"] - data.loc[valid_orig, "price"])
        / data.loc[valid_orig, "original_price"]
    )

    # Clip extreme values just in case scraping issues produce weird numbers
    data["discount"] = data["discount"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data["discount"] = data["discount"].clip(lower=-1.0, upper=1.0)

    data["on_sale"] = (data["discount"] > 0).astype(int)

    # ---- Panel sorting ----
    data = data.sort_values(["skuId", "date"])

    # ---- Lag / lead features ----
    data["price_lag1"] = data.groupby("skuId")["price"].shift(1)
    data["discount_lag1"] = data.groupby("skuId")["discount"].shift(1)

    data["price_lead1"] = data.groupby("skuId")["price"].shift(-1)
    data["discount_lead1"] = data.groupby("skuId")["discount"].shift(-1)

    # ---- Targets (tomorrow) ----
    data["price_change_tomorrow"] = (data["price_lead1"] != data["price"]).astype(int)
    data["discount_increase_tomorrow"] = (data["discount_lead1"] > data["discount"]).astype(int)
    data["price_delta"] = data["price_lead1"] - data["price"]

    # Drop last observation per SKU (no tomorrow)
    panel = data.dropna(subset=["price_lead1"]).copy()

    # ---- Extra features ----
    panel["log_reviews"] = np.log1p(panel["reviewCount"].fillna(0))
    panel["day_of_week"] = panel["date"].dt.dayofweek

    # If all your dates are in the same holiday week, this feature won’t help much,
    # but we can keep it for completeness:
    panel["holiday_week"] = 1

    print("Panel shape:", panel.shape)
    print("Price change rate:", panel["price_change_tomorrow"].mean())
    print("Discount increase rate:", panel["discount_increase_tomorrow"].mean())

    # panel = panel.drop_duplicates(subset=["skuId", "date"])

    # Keep one row per (skuId, date) to avoid duplicate transitions/leakiness
    panel = panel.sort_values(["skuId", "date"]).drop_duplicates(subset=["skuId", "date"], keep="last")

    dup = panel.duplicated(subset=["skuId", "date"]).mean()
    print("Duplicate (skuId,date) rate:", dup)

    panel.to_csv(OUT_PANEL_PATH, index=False)
    print("Saved:", OUT_PANEL_PATH)

if __name__ == "__main__":
    main()