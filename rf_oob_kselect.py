import numpy as np
import pandas as pd
from pathlib import Path
import json

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    balanced_accuracy_score, classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance

CONFIGS = {
    "FULL": {
        "drop_cols": []  # keep everything (except the ID columns you already drop)
    },
    "OBJECTIVE": {
        # remove subjective / examiner-judgment fields
        "drop_cols": ["Difficulty", "Comparability", "Phase"]
        # If you decide Phase is OK to keep, delete it from this list.
    }
}


# -------------------------
# Data prep
# -------------------------
# def clean_df(csv_path: str):
#     df = pd.read_csv(csv_path)
#     df = df.loc[:, ~df.columns.duplicated()]

#     # drop all-NaN cols
#     all_nan_cols = [c for c in df.columns if df[c].isna().all()]
#     if all_nan_cols:
#         df = df.drop(columns=all_nan_cols)

#     y = df["ConclusionClass"].copy()

#     drop_cols = ["ConclusionClass", "ConclusionRaw", "AnonID"]
#     X = df.drop(columns=[c for c in drop_cols if c in df.columns])

#     # drop any cset identifiers
#     X = X.drop(columns=[c for c in X.columns if str(c).lower().startswith("cset")], errors="ignore")

#     return X, y
def clean_df(csv_path: str, drop_cols=None):
    drop_cols = drop_cols or []

    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.duplicated()]

    # drop all-NaN cols
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        df = df.drop(columns=all_nan_cols)

    y = df["ConclusionClass"].copy()

    base_drop = ["ConclusionClass", "ConclusionRaw", "AnonID"]
    X = df.drop(columns=[c for c in base_drop if c in df.columns])

    # drop any cset identifiers
    X = X.drop(columns=[c for c in X.columns if str(c).lower().startswith("cset")], errors="ignore")

    # drop config-specific columns
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

    return X, y

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


# -------------------------
# Step A: OOB-tune n_estimators
# -------------------------
def compute_optimized_tree_count_oob(
    X_train, y_train, preprocess, out_dir: Path, label: str,
    min_trees=50, max_trees=1000, step=50
):
    out_dir.mkdir(exist_ok=True)
    trees_range = list(range(min_trees, max_trees + 1, step))
    rows = []

    best_n = None
    best_oob = -np.inf

    for n_tree in trees_range:
        rf = RandomForestClassifier(
            n_estimators=n_tree,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
            bootstrap=True,
            oob_score=True,
            max_features="sqrt",
        )
        model = Pipeline([
            ("preprocess", preprocess),
            ("rf", rf),
        ])
        model.fit(X_train, y_train)
        oob = model.named_steps["rf"].oob_score_
        oob_error = 1.0 - oob

        rows.append({"n_estimators": n_tree, "oob_score": oob, "oob_error": oob_error})

        if oob > best_oob:
            best_oob = oob
            best_n = n_tree

        print(f"[{label}] n={n_tree:4d} | OOB score={oob:.4f} | OOB error={oob_error:.4f}")

    tune_df = pd.DataFrame(rows).sort_values("oob_error")
    tune_df.to_csv(out_dir / f"oob_curve_{label}.csv", index=False)

    print(f"\n[{label}] Optimal n_estimators by OOB error: {best_n} (OOB score={best_oob:.4f})")
    return best_n, tune_df


# -------------------------
# Step B: pick best k features (CV, no leakage)
# We use permutation importance on validation fold, then test top-k subsets.
# -------------------------
def top_features_from_fold(rf_pipe, X_val, y_val, feature_names, scoring="balanced_accuracy", repeats=5):
    perm = permutation_importance(
        rf_pipe, X_val, y_val,
        n_repeats=repeats,
        random_state=42,
        scoring=scoring
    )
    imp = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
    return imp


def find_best_k_features_cv(
    X_train, y_train, preprocess, n_estimators, out_dir: Path, label: str,
    k_grid=(5, 8, 10, 12, 15, 20),
    cv_splits=5
):
    out_dir.mkdir(exist_ok=True)
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # We can only do top-k over ORIGINAL columns (not one-hot expanded),
    # so we compute permutation importance on the pipeline using original feature names.
    feature_names = list(X_train.columns)

    results = []

    for k in k_grid:
        fold_scores = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
                bootstrap=True,
                max_features="sqrt",
            )
            pipe = Pipeline([
                ("preprocess", preprocess),
                ("rf", rf),
            ])
            pipe.fit(X_tr, y_tr)

            # rank ORIGINAL columns by permutation importance
            imp = top_features_from_fold(pipe, X_va, y_va, feature_names, repeats=3)

            top_k = list(imp.head(k).index)

            # retrain using only top-k ORIGINAL cols
            preprocess_k = build_preprocessor(X_tr[top_k])
            rf_k = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
                bootstrap=True,
                max_features="sqrt",
            )
            pipe_k = Pipeline([
                ("preprocess", preprocess_k),
                ("rf", rf_k),
            ])
            pipe_k.fit(X_tr[top_k], y_tr)

            pred = pipe_k.predict(X_va[top_k])
            score = balanced_accuracy_score(y_va, pred)
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        results.append({"k": k, "cv_balanced_accuracy": mean_score})
        print(f"[{label}] k={k:3d} | CV balanced accuracy={mean_score:.4f}")

    res_df = pd.DataFrame(results).sort_values("cv_balanced_accuracy", ascending=False)
    res_df.to_csv(out_dir / f"k_selection_{label}.csv", index=False)

    best_k = int(res_df.iloc[0]["k"])
    print(f"\n[{label}] Best k by CV balanced accuracy: {best_k}")
    return best_k, res_df


# -------------------------
# Step C: final train + eval + save
# -------------------------
def train_final_and_save(
    X_train, X_test, y_train, y_test,
    n_estimators, selected_features,
    out_dir: Path, label: str
):
    out_dir.mkdir(exist_ok=True)

    preprocess = build_preprocessor(X_train[selected_features])

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        bootstrap=True,
        oob_score=True,
        max_features="sqrt",
    )
    model = Pipeline([
        ("preprocess", preprocess),
        ("rf", rf),
    ])
    model.fit(X_train[selected_features], y_train)

    pred = model.predict(X_test[selected_features])

    bal_acc = balanced_accuracy_score(y_test, pred)
    report = classification_report(y_test, pred)
    labels_sorted = sorted(y_test.unique())
    cm = confusion_matrix(y_test, pred, labels=labels_sorted)

    # Save artifacts
    (out_dir / f"final_results_{label}.txt").write_text(
        f"Dataset: {label}\n"
        f"n_estimators: {n_estimators}\n"
        f"OOB score (accuracy): {model.named_steps['rf'].oob_score_}\n"
        f"Selected features ({len(selected_features)}): {selected_features}\n\n"
        f"Balanced accuracy (test): {bal_acc:.6f}\n\n"
        f"Classification report:\n{report}\n\n"
        f"Labels: {labels_sorted}\n"
        f"Confusion matrix:\n{cm}\n",
        encoding="utf-8"
    )

    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(
        out_dir / f"final_confusion_{label}.csv"
    )
    pd.DataFrame(classification_report(y_test, pred, output_dict=True)).T.to_csv(
        out_dir / f"final_report_{label}.csv"
    )

    # Save selected features JSON
    (out_dir / f"selected_features_{label}.json").write_text(
        json.dumps(selected_features, indent=2),
        encoding="utf-8"
    )

    print(f"\n[{label}] Final test balanced accuracy: {bal_acc:.4f}")
    print(f"Saved outputs to: {out_dir}")
    return model


# def run_pipeline(csv_path: str, label: str):
#     out_dir = Path("rf_oob_kselect_outputs")
#     out_dir.mkdir(exist_ok=True)

#     X, y = clean_df(csv_path)

#     # Hold-out test set
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     preprocess = build_preprocessor(X_train)

#     # A) best number of trees by OOB
#     best_n, _ = compute_optimized_tree_count_oob(
#         X_train, y_train, preprocess, out_dir=out_dir, label=label,
#         min_trees=50, max_trees=1000, step=50
#     )

#     # B) best k by CV (balanced accuracy)
#     best_k, _ = find_best_k_features_cv(
#         X_train, y_train, preprocess, n_estimators=best_n, out_dir=out_dir, label=label,
#         k_grid=(5, 8, 10, 12, 15, 20),
#         cv_splits=5
#     )

#     # C) get final ranking on full train set to pick top-k
#     rf_full = RandomForestClassifier(
#         n_estimators=best_n,
#         random_state=42,
#         n_jobs=-1,
#         class_weight="balanced",
#         bootstrap=True,
#         max_features="sqrt",
#     )
#     pipe_full = Pipeline([
#         ("preprocess", preprocess),
#         ("rf", rf_full),
#     ])
#     pipe_full.fit(X_train, y_train)

#     feature_names = list(X_train.columns)
#     imp_full = permutation_importance(
#         pipe_full, X_train, y_train, n_repeats=5, random_state=42, scoring="balanced_accuracy"
#     )
#     ranked = pd.Series(imp_full.importances_mean, index=feature_names).sort_values(ascending=False)
#     ranked.to_csv(out_dir / f"final_feature_ranking_{label}.csv")

#     selected_features = list(ranked.head(best_k).index)

    # # D) train final + save evaluation
    # train_final_and_save(
    #     X_train, X_test, y_train, y_test,
    #     n_estimators=best_n,
    #     selected_features=selected_features,
    #     out_dir=out_dir,
    #     label=label
    # )

def run_pipeline(csv_path: str, label: str, variant_name: str, drop_cols):
    out_dir = Path("rf_oob_kselect_outputs") / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = clean_df(csv_path, drop_cols=drop_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = build_preprocessor(X_train)

    best_n, _ = compute_optimized_tree_count_oob(
        X_train, y_train, preprocess, out_dir=out_dir, label=f"{label}_{variant_name}",
        min_trees=50, max_trees=1000, step=50
    )

    best_k, _ = find_best_k_features_cv(
        X_train, y_train, preprocess, n_estimators=best_n, out_dir=out_dir, label=f"{label}_{variant_name}",
        k_grid=(5, 8, 10, 12, 15, 20),
        cv_splits=5
    )

    rf_full = RandomForestClassifier(
        n_estimators=best_n,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        bootstrap=True,
        max_features="sqrt",
    )
    pipe_full = Pipeline([
        ("preprocess", preprocess),
        ("rf", rf_full),
    ])
    pipe_full.fit(X_train, y_train)

    feature_names = list(X_train.columns)
    imp_full = permutation_importance(
        pipe_full, X_train, y_train, n_repeats=5, random_state=42, scoring="balanced_accuracy"
    )
    ranked = pd.Series(imp_full.importances_mean, index=feature_names).sort_values(ascending=False)
    ranked.to_csv(out_dir / f"final_feature_ranking_{label}_{variant_name}.csv")

    selected_features = list(ranked.head(best_k).index)

    train_final_and_save(
        X_train, X_test, y_train, y_test,
        n_estimators=best_n,
        selected_features=selected_features,
        out_dir=out_dir,
        label=f"{label}_{variant_name}"
    )

# if __name__ == "__main__":
#     run_pipeline("KQ_model_ready.csv", "KQ")
#     run_pipeline("QQ_model_ready.csv", "QQ")\
if __name__ == "__main__":
    for variant_name, cfg in CONFIGS.items():
        run_pipeline("KQ_model_ready.csv", "KQ", variant_name, cfg["drop_cols"])
        run_pipeline("QQ_model_ready.csv", "QQ", variant_name, cfg["drop_cols"])