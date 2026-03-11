import pandas as pd
import numpy as np
from pathlib import Path
import json

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


def train_logreg(csv_path, features, label):

    print("\n" + "="*70)
    print(f"Training Logistic Regression for {label}")
    print("="*70)

    df = pd.read_csv(csv_path)

    y = df["ConclusionClass"]
    X = df[features]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())
                ]),
                num_cols
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                cat_cols
            ),
        ]
    )

    model = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced"
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)

    acc = accuracy_score(y_test, pred)

    roc = roc_auc_score(
        pd.get_dummies(y_test),
        prob,
        average="macro",
        multi_class="ovr"
    )

    print("Accuracy:", acc)
    print("ROC AUC:", roc)

    print("\nClassification Report")
    print(classification_report(y_test, pred))

    cm = confusion_matrix(y_test, pred)

    # cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    results = {
        "target": "ConclusionClass",
        "test_accuracy": float(acc),
        "roc_auc_macro": float(roc),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "features": features
    }

    out = Path("logreg_results_" + label + ".json")
    out.write_text(json.dumps(results, indent=2))

    print("\nSaved results to", out)


if __name__ == "__main__":

    qq_features = [
        "Difficulty",
        "Intervening bullets (min)",
        "Intervening bullets (max)",
        "Q2 Quality (prescreen)",
        "Caliber Q1",
        "Subsession  Q2-K3",
        "Phase",
        "Q1 Quality (Voted)",
        "Caliber Q2-K3",
        "Q1 Quality (prescreen)",
        "Type",
        "Q2 Quality (Voted)",
        "Assigns (total)",
        "Assigns (Baseline+Repeat)",
        "Assigns (Baseline)"
    ]

    kq_features = [
        "Comparability",
        "Intervening bullets (max)",
        "Difficulty",
        "Subsession  Q2-K3",
        "Q1 Quality (prescreen)",
        "Q1 Quality (Voted)",
        "Caliber Q2-K3",
        "Assigns (total)",
        "Intervening bullets (min)",
        "Phase",
        "Caliber Q1",
        "Assigns (Baseline+Repeat)",
        "Assigns (Baseline)",
        "Type"
    ]

    train_logreg("QQ_model_ready.csv", qq_features, "QQ")
    train_logreg("KQ_model_ready.csv", kq_features, "KQ")