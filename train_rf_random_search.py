import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
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

    return ColumnTransformer(
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


# def clean_df(csv_path: str):
#     df = pd.read_csv(csv_path)

#     # Drop duplicate columns
#     df = df.loc[:, ~df.columns.duplicated()]

#     # Drop columns that are entirely empty
#     all_nan_cols = [c for c in df.columns if df[c].isna().all()]
#     if all_nan_cols:
#         print("Dropping all-NaN columns:", all_nan_cols)
#         df = df.drop(columns=all_nan_cols)

#     y = df["ConclusionClass"].copy()

#     drop_cols = ["ConclusionClass", "ConclusionRaw", "AnonID"]
#     X = df.drop(columns=[c for c in drop_cols if c in df.columns])

#     # Drop Cset identifier columns (Cset, Cset.1, etc.)
#     X = X.drop(columns=[c for c in X.columns if str(c).lower().startswith("cset")], errors="ignore")

#     return X, y

def clean_df(csv_path: str, extra_drop_cols=None):
    extra_drop_cols = extra_drop_cols or []

    df = pd.read_csv(csv_path)

    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Drop columns that are entirely empty
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        print("Dropping all-NaN columns:", all_nan_cols)
        df = df.drop(columns=all_nan_cols)

    y = df["ConclusionClass"].copy()

    # Always drop label + raw + examiner id
    drop_cols = ["ConclusionClass", "ConclusionRaw", "AnonID"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Drop Cset identifier columns (Cset, Cset.1, etc.)
    X = X.drop(columns=[c for c in X.columns if str(c).lower().startswith("cset")], errors="ignore")

    # Drop additional columns for variants (e.g., objective-only)
    X = X.drop(columns=[c for c in extra_drop_cols if c in X.columns], errors="ignore")

    return X, y

# def run_random_search(csv_path: str, label: str, n_iter: int = 30, cv_splits: int = 5, random_state: int = 42):
def run_random_search(csv_path: str, label: str, extra_drop_cols=None, n_iter: int = 30, cv_splits: int = 5, random_state: int = 42):
    print("\n" + "="*80)
    print(f"RandomizedSearchCV for: {label}")
    print("="*80)

    out_dir = Path("rf_outputs")
    out_dir.mkdir(exist_ok=True)

    # X, y = clean_df(csv_path)
    X, y = clean_df(csv_path, extra_drop_cols=extra_drop_cols)

    # Hold-out test set (never touched during CV)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )

    preprocess = build_preprocessor(X_train)

    # Base RF (values here are just defaults; search will override)
    rf = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
        bootstrap=True,
        oob_score=True,   # works during fit; not used for scoring in CV, but useful to log afterward
        criterion="gini"
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("rf", rf),
    ])

    # ---- Small, sensible search space ----
    # Note: use "rf__" prefix because it's inside the Pipeline
    param_dist = {
        "rf__n_estimators": [200, 400, 600, 800, 1000],
        "rf__max_features": ["sqrt", "log2", 0.5],
        "rf__max_depth": [None, 10, 20, 30, 40],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="balanced_accuracy",
        n_jobs=-1,
        cv=cv,
        verbose=2,
        random_state=random_state,
        return_train_score=True
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_cv = search.best_score_

    print("\nBest CV balanced accuracy:", best_cv)
    print("Best params:", best_params)

    # ---- Evaluate on untouched test set ----
    pred = best_model.predict(X_test)
    bal_acc_test = balanced_accuracy_score(y_test, pred)
    report_txt = classification_report(y_test, pred)

    labels_sorted = sorted(y.unique())
    cm = confusion_matrix(y_test, pred, labels=labels_sorted)

    # Try to get OOB score from best RF (may exist; if not, set NA)
    try:
        oob = best_model.named_steps["rf"].oob_score_
    except Exception:
        oob = np.nan

    print("\nTest balanced accuracy:", bal_acc_test)
    print(report_txt)
    print("Confusion matrix:\n", cm)
    print("OOB score (accuracy) of best model:", oob)

    # ---- Permutation importance (test set) ----
    perm = permutation_importance(
        best_model, X_test, y_test,
        n_repeats=10,
        random_state=random_state,
        scoring="balanced_accuracy"
    )
    imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

    # ---- Save everything ----
    # CV results
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(out_dir / f"rf_randomsearch_cvresults_{label}.csv", index=False)

    # Best params JSON
    (out_dir / f"rf_randomsearch_bestparams_{label}.json").write_text(
        pd.Series(best_params).to_json(indent=2)
    )

    # Text summary
    results_txt = out_dir / f"rf_randomsearch_results_{label}.txt"
    with open(results_txt, "w") as f:
        f.write(f"Dataset: {label}\n")
        f.write(f"Best CV balanced accuracy: {best_cv:.6f}\n")
        f.write(f"Best params: {best_params}\n")
        f.write(f"OOB score (accuracy): {oob}\n")
        f.write(f"Test balanced accuracy: {bal_acc_test:.6f}\n\n")
        f.write("Classification report:\n")
        f.write(report_txt)
        f.write("\n\nLabels order:\n")
        f.write(str(labels_sorted))
        f.write("\n\nConfusion matrix:\n")
        f.write(np.array2string(cm))

    # Report CSV
    pd.DataFrame(classification_report(y_test, pred, output_dict=True)).T.to_csv(
        out_dir / f"rf_randomsearch_report_{label}.csv"
    )

    # Confusion CSV
    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(
        out_dir / f"rf_randomsearch_confusion_{label}.csv"
    )

    # Importances CSV
    imp.to_csv(out_dir / f"rf_randomsearch_perm_importance_{label}.csv")

    # Save the fitted best model
    joblib.dump(best_model, out_dir / f"rf_randomsearch_bestmodel_{label}.joblib")

    print("\nSaved:")
    print(out_dir / f"rf_randomsearch_cvresults_{label}.csv")
    print(out_dir / f"rf_randomsearch_bestparams_{label}.json")
    print(out_dir / f"rf_randomsearch_results_{label}.txt")
    print(out_dir / f"rf_randomsearch_report_{label}.csv")
    print(out_dir / f"rf_randomsearch_confusion_{label}.csv")
    print(out_dir / f"rf_randomsearch_perm_importance_{label}.csv")
    print(out_dir / f"rf_randomsearch_bestmodel_{label}.joblib")

    return best_model



if __name__ == "__main__":
    run_random_search("KQ_model_ready.csv", "KQ", n_iter=30, cv_splits=5)
    run_random_search("QQ_model_ready.csv", "QQ", n_iter=30, cv_splits=5)