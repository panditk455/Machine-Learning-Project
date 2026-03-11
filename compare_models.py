import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULT_FILES = {
    "Random Forest QQ": "rf_results_QQ.json",
    "Random Forest KQ": "rf_results_KQ.json",
    "XGBoost QQ": "xgboost_results_QQ.json",
    "XGBoost KQ": "xgboost_results_KQ.json",
    "LogReg QQ": "logreg_results_QQ.json",
    "LogReg KQ": "logreg_results_KQ.json"
}

rows = []

for name, path in RESULT_FILES.items():
    if not Path(path).exists():
        print(f"Skipping missing file: {path}")
        continue

    with open(path) as f:
        data = json.load(f)

    model, dataset = name.split()

    rows.append({
        "Model": model,
        "Dataset": dataset,
        "Accuracy": data["test_accuracy"],
        "ROC_AUC": data["roc_auc_macro"],
        "CV_Mean": data["cv_mean"]
    })

df = pd.DataFrame(rows)

print("\nModel Comparison\n")
print(df.sort_values(["Dataset","Accuracy"], ascending=False))

df.to_csv("model_comparison.csv", index=False)

print("\nSaved: model_comparison.csv")

# ----------- Plot -----------

plt.figure(figsize=(8,5))

for dataset in df["Dataset"].unique():
    subset = df[df["Dataset"] == dataset]

    plt.bar(
        subset["Model"] + " (" + dataset + ")",
        subset["Accuracy"]
    )

plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("model_comparison_accuracy.png", dpi=300)

print("Saved: model_comparison_accuracy.png")