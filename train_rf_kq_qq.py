import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance


# def train_rf(csv_path: str, label: str):
#     print("\n" + "="*80)
#     print(f"Training Random Forest for: {label}")
#     print("="*80)

#     df = pd.read_csv(csv_path)

#     # Drop duplicate columns if they exist (like Cset repeated)
#     df = df.loc[:, ~df.columns.duplicated()]

#     # Target
#     y = df["ConclusionClass"].copy()

#     # Drop leakage/IDs/raw targets
#     drop_cols = ["ConclusionClass", "ConclusionRaw", "AnonID"]  # AnonID shouldn't exist, but safe
#     X = df.drop(columns=[c for c in drop_cols if c in df.columns])

#     # Optional: if you want baseline only
#     # Xy = pd.concat([X, y], axis=1)
#     # Xy = Xy[Xy["Phase"].str.contains("Baseline", na=False)]
#     # y = Xy["ConclusionClass"]
#     # X = Xy.drop(columns=["ConclusionClass"])

#     # Train/test split (random row split since AnonID is removed)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y,
#         test_size=0.2,
#         random_state=42,
#         stratify=y
#     )

#     # Identify categorical vs numeric columns
#     cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
#     num_cols = [c for c in X.columns if c not in cat_cols]

#     preprocess = ColumnTransformer(
#         transformers=[
#             ("num", Pipeline(steps=[
#                 ("imputer", SimpleImputer(strategy="median")),
#             ]), num_cols),
#             ("cat", Pipeline(steps=[
#                 ("imputer", SimpleImputer(strategy="most_frequent")),
#                 ("onehot", OneHotEncoder(handle_unknown="ignore")),
#             ]), cat_cols),
#         ],
#         remainder="drop"
#     )

#     rf = RandomForestClassifier(
#         n_estimators=600,
#         random_state=42,
#         class_weight="balanced",
#         n_jobs=-1
#     )

#     model = Pipeline(steps=[
#         ("preprocess", preprocess),
#         ("rf", rf),
#     ])

#     model.fit(X_train, y_train)

#     pred = model.predict(X_test)
#     print("Balanced accuracy:", balanced_accuracy_score(y_test, pred))
#     print("\nClassification report:\n", classification_report(y_test, pred))

#     labels_sorted = sorted(y.unique())
#     cm = confusion_matrix(y_test, pred, labels=labels_sorted)
#     print("Labels order:", labels_sorted)
#     print("Confusion matrix:\n", cm)

#     # Permutation importance on original (pre-one-hot) columns
#     perm = permutation_importance(
#         model,
#         X_test,
#         y_test,
#         n_repeats=10,
#         random_state=42,
#         scoring="balanced_accuracy"
#     )

#     imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
#     out_imp = f"rf_perm_importance_{label}.csv"
#     imp.to_csv(out_imp)

#     print(f"\nTop 20 permutation importances ({label}):")
#     print(imp.head(20))
#     print(f"\nSaved: {out_imp}")

#     return model

def train_rf(csv_path: str, label: str):
    print("\n" + "="*80)
    print(f"Training Random Forest for: {label}")
    print("="*80)

    df = pd.read_csv(csv_path)

    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Drop columns that are entirely empty
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        print("Dropping all-NaN columns:", all_nan_cols)
        df = df.drop(columns=all_nan_cols)

    # Target
    y = df["ConclusionClass"].copy()

    # Drop raw target + identifiers
    drop_cols = ["ConclusionClass", "ConclusionRaw", "AnonID"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Drop any Cset identifier columns (Cset, Cset.1, etc.)
    X = X.drop(columns=[c for c in X.columns if str(c).lower().startswith("cset")], errors="ignore")

    # Train/test split (row-wise)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop"
    )

    rf = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("rf", rf),
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("Balanced accuracy:", balanced_accuracy_score(y_test, pred))
    print("\nClassification report:\n", classification_report(y_test, pred))

    labels_sorted = sorted(y.unique())
    cm = confusion_matrix(y_test, pred, labels=labels_sorted)
    print("Labels order:", labels_sorted)
    print("Confusion matrix:\n", cm)

    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=42,
        scoring="balanced_accuracy"
    )

    imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
    out_imp = f"rf_perm_importance_{label}_noCset.csv"
    imp.to_csv(out_imp)

    print(f"\nTop 20 permutation importances ({label}):")
    print(imp.head(20))
    print(f"\nSaved: {out_imp}")

    return model


if __name__ == "__main__":
    train_rf("KQ_model_ready.csv", "KQ")
    train_rf("QQ_model_ready.csv", "QQ")