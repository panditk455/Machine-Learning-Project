import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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