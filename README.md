# Social Media's Impact on Teen Mental Health

Binary classification project predicting depression risk from social media usage patterns and health indicators in 1,200 teen records.

## Overview

This notebook explores whether depression can be predicted from behavioral and self-reported health features using classical ML models. The dataset is severely imbalanced (2.6% positive rate), making this a study in handling real-world class imbalance rather than a straightforward classification task.

## Dataset

**Source:** [Kaggle — Social Media Impact on Teen Mental Health](https://www.kaggle.com/datasets/algozee/teenager-menthal-healy/data)

| Property | Value |
|---|---|
| Records | 1,200 |
| Features | 11 |
| Target | `depression_label` (0/1) |
| Class ratio | 97.4% / 2.6% |

**Key features:** `daily_social_media_hours`, `sleep_hours`, `stress_level`, `anxiety_level`, `addiction_level`, `platform_usage`, `social_interaction_level`

## Pipeline

1. Load & inspect
2. Cleaning & encoding — ordinal map, binary encode, one-hot encode
3. EDA — correlation matrix, boxplots by class
4. Stratified 80/20 train/test split
5. Baseline models — Logistic Regression, Decision Tree, Random Forest
6. SMOTETomek resampling — applied on training data only
7. Retrain all models + XGBoost on balanced data
8. Hyperparameter tuning — Optuna for both DT and XGBoost (100 trials, 5-fold CV)
9. Final model evaluation
10. Robustness check — 10-fold cross-validation
11. Hard case analysis — 200-seed sweep to identify persistently misclassified samples

## Results

| Model | F1 (class 1) |
|---|---|
| Logistic Regression (baseline) | 0.44 |
| Logistic Regression (SMOTETomek) | 0.40 |
| Decision Tree (SMOTETomek) | 0.91 |
| Random Forest (SMOTETomek) | 0.80 |
| XGBoost (SMOTETomek) | 0.91 |
| Decision Tree (Optuna tuned) | 0.91 |
| XGBoost (Optuna tuned) | 0.91 |

Cross-validated mean F1: **0.930 ± 0.155** (10-fold, final DT)

The F1 ceiling of 0.91 is data-limited, not model-limited — a single statistically ambiguous sample (index 691) is responsible for all misclassifications across 200 random seeds.

## Tech Stack

- Python 3, [Marimo](https://marimo.io) (reactive notebook)
- pandas, numpy, scikit-learn, imbalanced-learn, XGBoost
- Optuna (hyperparameter tuning)
- Plotly, Matplotlib (visualization)

## Setup

```bash
pip install marimo pandas numpy scikit-learn imbalanced-learn xgboost optuna plotly matplotlib
marimo run socialmediaclean.py
```

Update the CSV path in the load cell to match your local directory.

## Why No Diagnosis Tool

The model achieves F1=0.91 on this dataset, but a user-facing diagnostic tool was deliberately not built. The dataset has no documented clinical instrument behind its labels — there is no way to verify that `depression_label = 1` maps to any recognized clinical threshold. The numeric scales (1–10) have no defined anchors, and the dataset's provenance is unverified.

A model trained on labels this loosely defined cannot produce a clinically meaningful prediction. Wrapping it in an input prompt and presenting a result to a real person risks being taken seriously when it shouldn't be. The ML work stands on its own without it.

## Limitations

- Depression, anxiety, and addiction labels are assumed self-reported with no documented clinical instrument (e.g. PHQ-9, GAD-7, AUDIT). Self-report validity for these constructs is low.
- Numeric scales (1–10) have no documented anchors — a score of 7 carries different meaning across respondents, time periods, and cultures.
- Dataset provenance is unverified. The original source and collection methodology are undocumented, making clinical generalization impossible.
- Model performance metrics reflect pattern-matching on potentially noisy labels, not genuine predictive validity for depression screening.
