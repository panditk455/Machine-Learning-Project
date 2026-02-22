import os
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import scipy.sparse as sp


# -------------------------
# Keep this consistent with your training script
# -------------------------
def clean_df(csv_path: str, extra_drop_cols=None):
    extra_drop_cols = extra_drop_cols or []

    df = pd.read_csv(csv_path)

    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Drop columns that are entirely empty
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
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


def get_feature_names(preprocess, X: pd.DataFrame):
    """
    Try to recover the post-encoding feature names from the ColumnTransformer.
    Falls back to generic names if not available.
    """
    try:
        names = preprocess.get_feature_names_out()
        return list(names)
    except Exception:
        # Fallback: try older sklearn behavior
        try:
            # Sometimes transformers have get_feature_names_out but CT doesn't
            # We'll just create generic names
            n_features = preprocess.transform(X.iloc[:1]).shape[1]
            return [f"f{i}" for i in range(n_features)]
        except Exception:
            return [f"f{i}" for i in range(999999)]  # should never hit


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def _as_list_of_class_arrays(shap_values):
    """
    Normalize shap output across shap versions:

    Possible shapes:
    - list of arrays: [ (n, p), (n, p), (n, p) ]
    - ndarray: (n, p, k)

    Return: list of (n, p) arrays, one per class.
    """
    if isinstance(shap_values, list):
        return shap_values

    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        # (n, p, k) -> list of (n, p)
        return [sv[:, :, i] for i in range(sv.shape[2])]

    # binary sometimes returns (n, p); treat as single "class"
    return [sv]


def save_summary_plots(shap_values_list, X_transformed, feature_names, class_names, out_dir: Path, max_display=25):
    """
    1) Global summary plot with direction (beeswarm)
    2) Global bar plot (mean |SHAP|)
    3) Per-class summary plots
    """
    # ---- Global (aggregated across classes) ----
    # Aggregate by mean absolute across classes
    if len(shap_values_list) > 1:
        sv_stack = np.stack(shap_values_list, axis=2)  # (n,p,k)
        sv_global = np.mean(np.abs(sv_stack), axis=2)  # (n,p)
    else:
        sv_global = shap_values_list[0]

    # Beeswarm: direction + magnitude
    plt.figure()
    shap.summary_plot(
        sv_global,
        X_transformed,
        feature_names=feature_names,
        show=False,
        max_display=max_display
    )
    plt.tight_layout()
    plt.savefig(out_dir / "summary_global_beeswarm.png", dpi=200)
    plt.close()

    # Bar: global magnitude
    plt.figure()
    shap.summary_plot(
        sv_global,
        X_transformed,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=max_display
    )
    plt.tight_layout()
    plt.savefig(out_dir / "summary_global_bar.png", dpi=200)
    plt.close()

    # ---- Per-class plots ----
    for i, sv in enumerate(shap_values_list):
        cname = class_names[i] if i < len(class_names) else f"class_{i}"

        plt.figure()
        shap.summary_plot(
            sv,
            X_transformed,
            feature_names=feature_names,
            show=False,
            max_display=max_display
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"summary_{cname}_beeswarm.png", dpi=200)
        plt.close()

        plt.figure()
        shap.summary_plot(
            sv,
            X_transformed,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=max_display
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"summary_{cname}_bar.png", dpi=200)
        plt.close()


def save_single_waterfall(explainer, shap_values_list, X_transformed, feature_names, class_names, out_dir: Path, row_idx: int, predicted_class_idx: int):
    """
    Single-example waterfall plot (for the predicted class).
    """
    # For multiclass, pick predicted class explanation
    sv = shap_values_list[predicted_class_idx]
    base_values = explainer.expected_value

    # base_values can be scalar or list/array per class
    if isinstance(base_values, (list, np.ndarray)):
        base_value = base_values[predicted_class_idx]
    else:
        base_value = base_values

    x_row = X_transformed[row_idx]
    sv_row = sv[row_idx]

    # Waterfall uses Explanation object
    exp = shap.Explanation(
        values=sv_row,
        base_values=base_value,
        data=x_row,
        feature_names=feature_names
    )

    cname = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else f"class_{predicted_class_idx}"

    plt.figure()
    shap.plots.waterfall(exp, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"waterfall_row{row_idx}_{cname}.png", dpi=200)
    plt.close()

def to_dense_float(X_trans):
    """Convert sparse/object outputs from ColumnTransformer into dense float64 for SHAP."""
    if sp.issparse(X_trans):
        X_trans = X_trans.toarray()
    X_trans = np.asarray(X_trans, dtype=np.float64)
    return X_trans

def run_shap(
    model_path: str,
    csv_path: str,
    label: str,
    extra_drop_cols=None,
    random_state: int = 42,
    test_size: float = 0.2,
    max_test_rows: int = 600,
    waterfall_row_idx: int = 0
):
    """
    - Loads your saved Pipeline (.joblib)
    - Rebuilds the same train/test split
    - Computes SHAP on test set (optionally subsampled)
    - Saves plots
    """
    out_dir = ensure_dir(Path("rf_outputs") / "shap" / label)

    print(f"\nLoading model: {model_path}")
    model = joblib.load(model_path)

    # Your saved model is a Pipeline(preprocess -> rf)
    preprocess = model.named_steps["preprocess"]
    rf = model.named_steps["rf"]

    # Load data and reproduce split
    X, y = clean_df(csv_path, extra_drop_cols=extra_drop_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Optional: subsample test set for speed/clarity
    if len(X_test) > max_test_rows:
        X_test = X_test.sample(n=max_test_rows, random_state=random_state)
        y_test = y_test.loc[X_test.index]

    # # Transform to numeric feature space used by RF
    # X_test_trans = preprocess.transform(X_test)

    # # Feature names after preprocessing/one-hot
    # feature_names = get_feature_names(preprocess, X_train)

    # Transform to numeric feature space used by RF
    X_train_trans = preprocess.transform(X_train)
    X_test_trans  = preprocess.transform(X_test)

    # IMPORTANT: make them dense float64 for SHAP
    X_train_trans = to_dense_float(X_train_trans)
    X_test_trans  = to_dense_float(X_test_trans)

    # Feature names after preprocessing/one-hot
    feature_names = get_feature_names(preprocess, X_train)

    # Class names
    # class_names = list(getattr(model, "classes_", []))
    class_names = list(rf.classes_)
    if not class_names:
        # Pipeline doesn't have classes_, but rf does
        class_names = list(rf.classes_)

    # Build explainer (TreeExplainer is best for RF)
    # explainer = shap.TreeExplainer(rf)

    
    # # Compute SHAP values for multiclass
    # shap_values = explainer.shap_values(X_test_trans)

    # Build explainer (TreeExplainer for RF) using background data
    explainer = shap.TreeExplainer(rf, data=X_train_trans)

    # Compute SHAP values for multiclass
    shap_values = explainer.shap_values(X_test_trans)
    shap_values_list = _as_list_of_class_arrays(shap_values)

    # Predictions to pick predicted class for waterfall
    proba = rf.predict_proba(X_test_trans)
    pred_class_idx = int(np.argmax(proba[waterfall_row_idx]))

    # Save plots
    save_summary_plots(
        shap_values_list=shap_values_list,
        X_transformed=X_test_trans,
        feature_names=feature_names,
        class_names=class_names,
        out_dir=out_dir,
        max_display=25
    )

    save_single_waterfall(
        explainer=explainer,
        shap_values_list=shap_values_list,
        X_transformed=X_test_trans,
        feature_names=feature_names,
        class_names=class_names,
        out_dir=out_dir,
        row_idx=waterfall_row_idx,
        predicted_class_idx=pred_class_idx
    )

    # Save which test row was explained (helpful for write-up)
    meta = {
        "label": label,
        "model_path": model_path,
        "csv_path": csv_path,
        "extra_drop_cols": extra_drop_cols or [],
        "test_rows_used": int(X_test.shape[0]),
        "waterfall_row_original_index": int(X_test.index[waterfall_row_idx]),
        "waterfall_row_position_in_test_sample": int(waterfall_row_idx),
        "predicted_class": str(class_names[pred_class_idx]),
        "predicted_proba": {str(class_names[i]): float(proba[waterfall_row_idx, i]) for i in range(len(class_names))}
    }
    (out_dir / "shap_run_meta.json").write_text(pd.Series(meta).to_json(indent=2))

    print(f"Saved SHAP plots to: {out_dir}")

if __name__ == "__main__":

    # ---------- QQ ----------
    run_shap(
        model_path="rf_outputs/rf_randomsearch_bestmodel_QQ_FULL.joblib",
        csv_path="QQ_model_ready.csv",
        label="QQ_FULL",
        extra_drop_cols=[],  # FULL keeps everything
        waterfall_row_idx=0
    )

    run_shap(
        model_path="rf_outputs/rf_randomsearch_bestmodel_QQ_OBJECTIVE.joblib",
        csv_path="QQ_model_ready.csv",
        label="QQ_OBJECTIVE",
        extra_drop_cols=["Difficulty", "Comparability", "Phase"],
        waterfall_row_idx=0
    )

    # ---------- RQ (or KQ) ----------
    run_shap(
        model_path="rf_outputs/rf_randomsearch_bestmodel_KQ_FULL.joblib",
        csv_path="KQ_model_ready.csv",
        label="RQ_FULL",
        extra_drop_cols=[],
        waterfall_row_idx=0
    )

    run_shap(
        model_path="rf_outputs/rf_randomsearch_bestmodel_KQ_OBJECTIVE.joblib",
        csv_path="KQ_model_ready.csv",
        label="RQ_OBJECTIVE",
        extra_drop_cols=["Difficulty", "Comparability", "Phase"],
        waterfall_row_idx=0
    )