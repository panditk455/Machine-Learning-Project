import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
PANEL_PATH = ROOT / "panel_data.csv"

def main():
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing {PANEL_PATH}. Run 02_build_panel.py first.")

    df = pd.read_csv(PANEL_PATH)

    features = [
        "price",
        "discount",
        "averageRating",
        "log_reviews",
        "category",
        "day_of_week",
    ]

    target = "price_change_tomorrow"

    X = pd.get_dummies(df[features], drop_first=True)
    y = df[target].astype(int)

    print("Positive rate overall:", y.mean())
    print("Unique SKUs:", df["skuId"].nunique())
    print("Date range:", pd.to_datetime(df["date"]).min(), "to", pd.to_datetime(df["date"]).max())

    print("Positive rate overall:", y.mean())
    print("Unique SKUs:", df["skuId"].nunique())
    print("Date range:", pd.to_datetime(df["date"]).min(), "to", pd.to_datetime(df["date"]).max())

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42, stratify=y
    # )
    # ---- SKU-based group split ----
    # groups = df["skuId"].astype(str)

    # gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # train_idx, test_idx = next(gss.split(X, y, groups=groups))

    # X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    # y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # print("Train SKUs:", df.iloc[train_idx]["skuId"].nunique())
    # print("Test SKUs:", df.iloc[test_idx]["skuId"].nunique())
    # ---- Time-based split: train on early dates, test on later dates ----
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    cutoff = df["date"].quantile(0.75)  # last 25% of dates as test (or pick a fixed date)
    train_mask = df["date"] < cutoff
    test_mask = df["date"] >= cutoff

    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    print("Train date range:", df.loc[train_mask, "date"].min(), "to", df.loc[train_mask, "date"].max())
    print("Test date range:", df.loc[test_mask, "date"].min(), "to", df.loc[test_mask, "date"].max())
    print("Train SKUs:", df.loc[train_mask, "skuId"].nunique())
    print("Test SKUs:", df.loc[test_mask, "skuId"].nunique())

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print("AUC:", auc)

    # Optional: threshold report at 0.5
    yhat = (preds >= 0.5).astype(int)
    print(classification_report(y_test, yhat))

if __name__ == "__main__":
    main()