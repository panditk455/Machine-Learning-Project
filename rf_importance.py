import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.inspection import permutation_importance

CSV_PATH = "bullet_pairs_model_ready.csv"

df = pd.read_csv(CSV_PATH)

# ---------
# Target
# ---------
y = df["ConclusionClass"].copy()

# ---------
# Features: drop IDs + raw target columns to avoid leakage
# ---------
drop_cols = ["AnonID", "Cset", "ConclusionRaw", "ConclusionClass"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ---------
# Group split by examiner to avoid leakage
# ---------
groups = df["AnonID"]
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# ---------
# Preprocess: one-hot categorical, impute missing
# ---------
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
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

# ---------
# Model
# ---------
rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("rf", rf)
])

model.fit(X_train, y_train)

# ---------
# Evaluate
# ---------
pred = model.predict(X_test)
print("Balanced accuracy:", balanced_accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# ---------
# Permutation importance
# IMPORTANT: use the original (pre-one-hot) feature names
# ---------
perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring="balanced_accuracy"
)

importances = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
print("\nTop 20 features by permutation importance:")
print(importances.head(20))

importances.to_csv("rf_permutation_importance.csv")
print("\nSaved rf_permutation_importance.csv")