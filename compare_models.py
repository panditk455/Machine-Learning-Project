import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

rows = []

def load_json_result(path, model_name, dataset):
    with open(path) as f:
        data = json.load(f)
    rows.append({
        "Model": model_name,
        "Dataset": dataset,
        "Accuracy": data["test_accuracy"],
        "ROC_AUC": data["roc_auc_macro"],
        "CV_Mean": data["cv_mean"]
    })

def load_txt_result(path, model_name, dataset):
    text = Path(path).read_text()

    acc = re.search(r"Test balanced accuracy:\s*([0-9.]+)", text)
    cv  = re.search(r"Best CV balanced accuracy:\s*([0-9.]+)", text)

    rows.append({
        "Model": model_name,
        "Dataset": dataset,
        "Accuracy": float(acc.group(1)) if acc else None,
        "ROC_AUC": None,
        "CV_Mean": float(cv.group(1)) if cv else None
    })

# Logistic Regression
if Path("logreg_results_QQ.json").exists():
    load_json_result("logreg_results_QQ.json", "LogReg", "QQ")
if Path("logreg_results_KQ.json").exists():
    load_json_result("logreg_results_KQ.json", "LogReg", "KQ")

# Random Forest 
if Path("rf_outputs/rf_randomsearch_results_QQ.txt").exists():
    load_txt_result("rf_outputs/rf_randomsearch_results_QQ.txt", "RF", "QQ")
if Path("rf_outputs/rf_randomsearch_results_KQ.txt").exists():
    load_txt_result("rf_outputs/rf_randomsearch_results_KQ.txt", "RF", "KQ")

# XGBoost 
if Path("qq_xgb_ConclusionClass/metrics.json").exists():
    load_json_result("qq_xgb_ConclusionClass/metrics.json", "XGBoost", "QQ")
if Path("kq_xgb_ConclusionClass/metrics.json").exists():
    load_json_result("kq_xgb_ConclusionClass/metrics.json", "XGBoost", "KQ")

df = pd.DataFrame(rows)

print("\nModel Comparison\n")
print(df.sort_values(["Dataset", "Accuracy"], ascending=[True, False]))

df.to_csv("model_comparison.csv", index=False)
print("\nSaved: model_comparison.csv")

# Accuracy figure
plt.figure(figsize=(8, 5))
for dataset in df["Dataset"].unique():
    subset = df[df["Dataset"] == dataset]
    plt.bar(
        subset["Model"] + " (" + subset["Dataset"] + ")",
        subset["Accuracy"]
    )

plt.ylabel("Balanced Accuracy / Accuracy")
plt.title("Model Comparison by Dataset")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("model_comparison_accuracy.png", dpi=300)
print("Saved: model_comparison_accuracy.png")