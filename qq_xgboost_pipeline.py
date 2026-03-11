"""
XGBoost Pipeline — KQ Bullet Black Box Dataset
Runs TWO versions:
  Version A → target: ConclusionClass  (Match / NonMatch / Inconclusive)
  Version B → target: ConclusionRaw    (ID / LeanID / Excl / LeanExcl / Insuff)

Features:
    Comparability, Intervening bullets (max), Difficulty,
    Subsession Q2-K3, Q1 Quality (prescreen), Q1 Quality (Voted),
    Caliber Q2-K3, Assigns (total), Intervening bullets (min),
    Phase, Caliber Q1, Assigns (Baseline+Repeat), Assigns (Baseline), Type
"""

import pandas as pd
import numpy as np
import json, os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, accuracy_score
)
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────────
DATA_PATH = "QQ_model_ready.csv"    
SEED      = 42

FEATURE_COLS = [
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
    "Assigns (Baseline)",
]

TARGETS = {
    "ConclusionClass": "qq_xgb_ConclusionClass",   # Match / NonMatch / Inconclusive
    "ConclusionRaw":   "qq_xgb_ConclusionRaw",      # ID / LeanID / Excl / LeanExcl / Insuff
}

# ──LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols\n")

# Verify all needed columns exist
all_needed = FEATURE_COLS + list(TARGETS.keys())
missing = [c for c in all_needed if c not in df.columns]
if missing:
    print(f"⚠️  Missing columns (check spelling): {missing}")
else:
    print("✅  All required columns found.\n")


# ──  RUNNER FUNCTION ────────────────────────────────────────────────────────
def run_pipeline(df, feature_cols, target_col, out_dir):

    os.makedirs(out_dir, exist_ok=True)
    shap_dir = os.path.join(out_dir, "shap")
    os.makedirs(shap_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TARGET: {target_col}  →  {out_dir}")
    print(f"{'='*60}")

    # ──  Clean ────────────────────────────────────────────────────────────
    subset = df.dropna(subset=feature_cols + [target_col]).copy()
    print(f"Rows after dropna: {subset.shape[0]}")
    print(f"Target distribution:\n{subset[target_col].value_counts()}\n")

    X = subset[feature_cols].copy()
    y_raw = subset[target_col].copy()

    # ── Encode target ────────────────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)
    print(f"Classes ({len(class_names)}): {class_names}")

    # ──  Encode features ──────────────────────────────────────────────────
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Categorical: {cat_cols}")
    print(f"Numeric    : {num_cols}")

    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = oe.fit_transform(X[cat_cols].astype(str))
    X = X.astype(float)

    # ──  Split ────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # ──  Model ────────────────────────────────────────────────────────────
    xgb = XGBClassifier(
        n_estimators      = 300,
        max_depth         = 5,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 3,
        gamma             = 0.1,
        use_label_encoder = False,
        eval_metric       = "mlogloss",
        random_state      = SEED,
        n_jobs            = -1,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    # ──  Evaluate ─────────────────────────────────────────────────────────
    y_pred  = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy : {acc:.4f}")

    # Only use labels actually present in the test set (handles rare classes)
    present_labels = sorted(set(y_test) | set(y_pred))
    present_names  = [class_names[i] for i in present_labels]
    print(classification_report(y_test, y_pred, labels=present_labels, target_names=present_names))

    try:
        auc = roc_auc_score(
            y_test,
            y_proba[:, present_labels],
            multi_class="ovr",
            average="macro",
            labels=present_labels,
        )
        print(f"ROC-AUC (macro OvR): {auc:.4f}")
    except Exception as e:
        auc = None
        print(f"ROC-AUC skipped: {e}")

    # Drop classes with < 5 members for CV (can't stratify otherwise)
    # Reset index to avoid pandas alignment errors after dropna

    # Drop classes with < 5 members for CV (can't stratify otherwise)
    # Reset index to avoid pandas alignment errors after dropna
    X_cv_reset   = X.reset_index(drop=True)
    y_cv_series  = pd.Series(y)
    class_counts = y_cv_series.value_counts()
    valid_mask   = y_cv_series.isin(class_counts[class_counts >= 5].index)
    X_cv         = X_cv_reset[valid_mask]
    y_cv_raw     = y_cv_series[valid_mask].values
    # Re-encode so labels are always 0-based (XGBoost requirement)
    y_cv         = LabelEncoder().fit_transform(y_cv_raw)
    n_splits     = min(5, int(class_counts[class_counts >= 5].min()))
    cv     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    cv_acc = cross_val_score(xgb, X_cv, y_cv, cv=cv, scoring="accuracy", n_jobs=-1)

    print(f"5-Fold CV: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

    metrics = {
        "target"        : target_col,
        "test_accuracy" : round(acc, 4),
        "roc_auc_macro" : round(auc, 4) if auc else None,
        "cv_mean"       : round(cv_acc.mean(), 4),
        "cv_std"        : round(cv_acc.std(),  4),
        "class_names"   : class_names,
        "features"      : feature_cols,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Confusion matrix ─────────────────────────────────────────────────

    cm   = confusion_matrix(y_test, y_pred, labels=present_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=present_names)
    fig, ax = plt.subplots(figsize=(max(5, len(present_names) * 1.5), max(4, len(present_names) * 1.2)))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {target_col}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ──  Feature importance ───────────────────────────────────────────────
    fi = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.4)))
    fi.plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"Feature Importance — {target_col}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close()

    # ──SHAP ─────────────────────────────────────────────────────────────
    print("\nComputing SHAP values…")
    explainer   = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

    # Global beeswarm
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False, plot_type="dot")
    plt.title(f"SHAP Global Beeswarm — {target_col}")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "summary_global_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Global bar
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False, plot_type="bar")
    plt.title(f"SHAP Global Bar — {target_col}")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "summary_global_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Per-class plots
    for i, cls in enumerate(class_names):
        sv = shap_values[i] if isinstance(shap_values, list) else shap_values[:, :, i]

        shap.summary_plot(sv, X_test, feature_names=feature_cols, show=False, plot_type="dot")
        plt.title(f"SHAP Beeswarm — {cls}")
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, f"summary_{cls}_beeswarm.png"), dpi=150, bbox_inches="tight")
        plt.close()

        shap.summary_plot(sv, X_test, feature_names=feature_cols, show=False, plot_type="bar")
        plt.title(f"SHAP Bar — {cls}")
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, f"summary_{cls}_bar.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Waterfall — first test row
    row_idx    = 0
    pred_class = int(y_pred[row_idx])
    pred_label = class_names[pred_class]
    sv_row     = shap_values[pred_class][row_idx] if isinstance(shap_values, list) else shap_values[row_idx, :, pred_class]

    shap_exp = shap.Explanation(
        values        = sv_row,
        base_values   = explainer.expected_value[pred_class],
        data          = X_test.iloc[row_idx].values,
        feature_names = feature_cols,
    )
    shap.plots.waterfall(shap_exp, show=False)
    plt.title(f"SHAP Waterfall — row 0, pred={pred_label}")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f"waterfall_row0_{pred_label}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    with open(os.path.join(shap_dir, "shap_run_meta.json"), "w") as f:
        json.dump({
            "waterfall_row"   : row_idx,
            "predicted_class" : pred_label,
            "predicted_proba" : {class_names[j]: round(float(y_proba[row_idx, j]), 4) for j in range(len(class_names))},
        }, f, indent=2)

    print(f"✅  Done — outputs saved to {out_dir}/\n")


# ── 3. RUN BOTH VERSIONS ──────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

for target_col, out_dir in TARGETS.items():
    run_pipeline(df, FEATURE_COLS, target_col, out_dir)