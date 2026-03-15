# Interpretable Multiclass Modeling of Forensic Firearm Decisions

This repository contains our machine learning project on forensic firearm decision modeling.

## Authors
- Isha Patel
- Kritika Pandit
- Raaid Iqbal

## Project Overview

This project studies examiner decision outcomes from the **Bullet Black Box Study** using machine learning methods, with an emphasis on **interpretability** and **model comparison**.

The goal is to predict the final examiner decision in forensic bullet comparison tasks using structured features derived from the study data. We compare several classification approaches, including:

- Random Forest
- XGBoost
- Logistic Regression with K-best feature selection

We analyze two different comparison settings in the dataset:

- **KQ**
- **QQ**

These represent two different subsets or formulations of the forensic comparison problem, and we train separate models for each.

### Research Question

Can machine learning models help predict forensic firearm examiner decisions, and which methods are most effective and interpretable for this task?

---

## Decision Classes

The original forensic decision labels include outcomes such as:

- `ID` = Identification
- `LeanID` = Leaning Identification
- `Excl` = Exclusion
- `LeanExcl` = Leaning Exclusion
- `Insuff` = Insufficient
- `NoValue` = No Value

Depending on the script, these may be modeled either in their original form or grouped into broader conclusion classes.

Because the classes are imbalanced, we compare models using not only overall accuracy, but also class-aware metrics and interpretation tools.

---

## Main Files

### Datasets
- `KQ_model_ready.csv` — Final model-ready dataset for the KQ analysis
- `QQ_model_ready.csv` — Final model-ready dataset for the QQ analysis

### Data Preparation
- `make_rq_qq_csvs.py` — Prepares or generates the model-ready CSV files from the raw data

### Random Forest
- `train_rf_kq_qq.py` — Trains Random Forest models for both KQ and QQ datasets
- `train_rf_random_search.py` — Performs Random Forest hyperparameter tuning using random search
- `rf_outputs/` — Stores Random Forest outputs, metrics, plots, or saved artifacts

### XGBoost
- `kq_xgboost_pipeline.py` — Trains and evaluates the XGBoost pipeline for the KQ dataset
- `qq_xgboost_pipeline.py` — Trains and evaluates the XGBoost pipeline for the QQ dataset
- `kq_xgb_ConclusionClass/` — Stores XGBoost results and outputs for KQ
- `qq_xgb_ConclusionClass/` — Stores XGBoost results and outputs for QQ

### Logistic Regression
- `train_logreg_kbest.py` — Trains Logistic Regression models using K-best feature selection
- `logreg_results_KQ.json` — Saved Logistic Regression results for KQ
- `logreg_results_QQ.json` — Saved Logistic Regression results for QQ

### Model Comparison and Interpretation
- `compare_models.py` — Compares model outputs across approaches and generates summary comparison results
- `shap_explain.py` — Produces SHAP-based model interpretation for feature importance and explanation
- `model_comparison.csv` — Summary table comparing model results
- `final_comparison.txt` — Text summary of final model comparison
- `model_comparison_accuracy.png` — Plot comparing model accuracies

---

## Data

This repository includes the model-ready datasets needed to run the main experiments:

- `KQ_model_ready.csv`
- `QQ_model_ready.csv`

These files should allow the code to run without needing to separately download the final cleaned datasets.

There is also a `data/` folder, which may contain raw or intermediate files used during preprocessing.

---

## Methods Used

This project compares multiple machine learning methods for forensic decision prediction.

### Random Forest
Used as a nonlinear ensemble baseline and for feature importance analysis.

### XGBoost
Used to improve predictive performance through boosted tree ensembles.

### Logistic Regression with K-Best Features
Used as a simpler and more interpretable baseline model.

### SHAP Explanations
Used to interpret feature contributions in trained models, especially tree-based models.

---

## Requirements

This project uses **Python 3**.

Recommended Python packages include:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `xgboost`
- `shap`
- `scipy`

## Run Code

Run the main modeling workflow using the prepared datasets

Since KQ_model_ready.csv and QQ_model_ready.csv are already included, you can directly run the training scripts.

If needed, install dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib xgboost shap scipy


