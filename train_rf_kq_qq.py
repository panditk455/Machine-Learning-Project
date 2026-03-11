import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
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
    return preprocess


def clean_df(csv_path: str):
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

    return X, y


def tune_n_estimators_oob(X_train, y_train, preprocess, label: str, out_dir: Path,
                          candidates=(50, 100, 200, 300, 400, 600, 800, 1000),
                          random_state=42):
    """
    Sweeps n_estimators and uses OOB score (accuracy) as a tuning signal.
    OOB error = 1 - oob_score.
    """
    rows = []
    best_n = None
    best_oob = -np.inf

    for n in candidates:
        rf = RandomForestClassifier(
            n_estimators=n,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
            bootstrap=True,
            oob_score=True,
            max_features="sqrt",
        )

        model = Pipeline(steps=[
            ("preprocess", preprocess),
            ("rf", rf),
        ])

        model.fit(X_train, y_train)
        oob = model.named_steps["rf"].oob_score_
        rows.append({
            "label": label,
            "n_estimators": n,
            "oob_score_accuracy": oob,
            "oob_error": 1.0 - oob
        })

        if oob > best_oob:
            best_oob = oob
            best_n = n

        print(f"[{label}] n_estimators={n:4d} | OOB score={oob:.4f} | OOB error={1.0-oob:.4f}")

    tune_df = pd.DataFrame(rows).sort_values("oob_error")
    tune_path = out_dir / f"rf_oob_tuning_{label}.csv"
    tune_df.to_csv(tune_path, index=False)
    print(f"Saved OOB tuning curve to: {tune_path}")
    print(f"Best n_estimators by OOB: {best_n} (OOB score={best_oob:.4f})")

    return best_n, tune_df


def train_rf(csv_path: str, label: str, do_oob_tuning=True):
    print("\n" + "="*80)
    print(f"Training Random Forest for: {label}")
    print("="*80)

    out_dir = Path("rf_outputs")
    out_dir.mkdir(exist_ok=True)

    X, y = clean_df(csv_path)

    # Train/test split (row-wise)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = build_preprocessor(X_train)

    # ---- OOB tuning for #trees ----
    if do_oob_tuning:
        candidates = (50, 100, 200, 300, 400, 600, 800, 1000)
        best_n, _ = tune_n_estimators_oob(
            X_train, y_train, preprocess, label=label, out_dir=out_dir, candidates=candidates
        )
    else:
        best_n = 600

    # ---- Final model ----
    rf = RandomForestClassifier(
        n_estimators=best_n,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        bootstrap=True,
        oob_score=True,
        max_features="sqrt",
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("rf", rf),
    ])

    model.fit(X_train, y_train)

    # Evaluate
    pred = model.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, pred)
    report_txt = classification_report(y_test, pred)

    labels_sorted = sorted(y.unique())
    cm = confusion_matrix(y_test, pred, labels=labels_sorted)

    print("Chosen n_estimators:", best_n)
    print("OOB score (accuracy):", model.named_steps["rf"].oob_score_)
    print("Balanced accuracy:", bal_acc)
    print("\nClassification report:\n", report_txt)
    print("Labels order:", labels_sorted)
    print("Confusion matrix:\n", cm)

    # Permutation importance
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=42,
        scoring="balanced_accuracy"
    )
    imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

    # ---- Save outputs ----
    results_txt = out_dir / f"rf_results_{label}.txt"
    with open(results_txt, "w") as f:
        f.write(f"Dataset: {label}\n")
        f.write(f"Chosen n_estimators: {best_n}\n")
        f.write(f"OOB score (accuracy): {model.named_steps['rf'].oob_score_:.6f}\n")
        f.write(f"OOB error: {1.0 - model.named_steps['rf'].oob_score_:.6f}\n")
        f.write(f"Balanced accuracy (test): {bal_acc:.6f}\n\n")
        f.write("Classification report:\n")
        f.write(report_txt)
        f.write("\n\nLabels order:\n")
        f.write(str(labels_sorted))
        f.write("\n\nConfusion matrix:\n")
        f.write(np.array2string(cm))

    # Save report / confusion / importances
    # pd.DataFrame(classification_report(y_test, pred, output_dict=True)).T.to_csv(out_dir / f"rf_report_{label}.csv")
    # pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(out_dir / f"rf_confusion_{label}.csv")
    # imp.to_csv(out_dir / f"rf_perm_importance_{label}.csv")

    print(f"\nSaved results to: {results_txt}")
    print(f"Saved report to: {out_dir / f'rf_report_{label}.csv'}")
    print(f"Saved confusion to: {out_dir / f'rf_confusion_{label}.csv'}")
    print(f"Saved importances to: {out_dir / f'rf_perm_importance_{label}.csv'}")

    return model


if __name__ == "__main__":
    train_rf("KQ_model_ready.csv", "KQ", do_oob_tuning=True)
    train_rf("QQ_model_ready.csv", "QQ", do_oob_tuning=True)