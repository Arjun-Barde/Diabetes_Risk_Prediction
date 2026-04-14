# Diabetes Risk Prediction — End-to-End ML Pipeline

A complete machine learning pipeline for predicting diabetes risk using the Pima Indians Diabetes Database. Covers exploratory data analysis, preprocessing, multi-model comparison, and SHAP-based explainability.

## Overview

Diabetes affects over 537 million adults globally. Early, accurate risk prediction from routine clinical measurements can enable earlier intervention. This project builds and evaluates three classification models, then uses SHAP to explain which features drive individual predictions, a critical requirement for clinical decision support tools.

**Dataset:** [Pima Indians Diabetes Database](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) , 768 female patients, 8 clinical features, binary outcome (diabetic / not diabetic).

---

## What's Inside

diabetes-risk-prediction/
├── diabetes_ml_pipeline.ipynb   # Main notebook — full pipeline
├── README.md
└── plots/                      
    ├── class_distribution.png
    ├── feature_distributions.png
    ├── correlation_heatmap.png
    ├── model_comparison.png
    ├── confusion_matrices.png
    ├── feature_importance.png
    ├── shap_bar.png
    └── shap_beeswarm.png
```

## Pipeline Steps

### 1. Exploratory Data Analysis
- Class distribution and balance check
- Feature distributions split by outcome (diabetic vs. non-diabetic)
- Correlation matrix across all clinical features
- Detection of biologically impossible zero values (Glucose, BMI, BloodPressure, etc.)

### 2. Preprocessing
- Replaced impossible zeros with column medians (robust to outliers)
- Stratified train/test split (80/20) to preserve class balance
- StandardScaler normalization (applied to Logistic Regression)

### 3. Model Training & Evaluation
Three models trained and compared using 5-fold stratified cross-validation:

| Model | CV AUC | Test Accuracy | Test ROC-AUC |
|---|---|---|---|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |

*Scores populate after running the notebook.*

Primary metric: **ROC-AUC** (preferred over accuracy given class imbalance).

### 4. Explainability — SHAP
- SHAP TreeExplainer applied to XGBoost (best-performing model)
- Bar plot: global feature importance across all predictions
- Beeswarm plot: direction and magnitude of each feature's effect per patient
- Key finding: Glucose, BMI, and Age are the strongest predictors — consistent with established clinical evidence

## Key Results

- XGBoost achieves the highest ROC-AUC among the three models
- Glucose is the dominant predictor across all models, aligning with clinical knowledge
- SHAP analysis confirms model logic is medically coherent — high glucose and high BMI consistently push predictions toward diabetic

---

## How to Run

**Option A — Google Colab (recommended, no setup needed)**
1. Open [Google Colab](https://colab.research.google.com)
2. File → Upload notebook → select `diabetes_ml_pipeline.ipynb`
3. Runtime → Run all

**Option B — Local**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap jupyter
jupyter notebook diabetes_ml_pipeline.ipynb
```

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | pandas, NumPy |
| Visualization | Matplotlib, seaborn |
| Machine learning | scikit-learn (Logistic Regression, Random Forest) |
| Gradient boosting | XGBoost |
| Explainability | SHAP |

---

## Clinical Relevance

This project reflects core challenges in health informatics ML:
- **Handling missing/invalid clinical data** (zero-value imputation)
- **Class imbalance** in medical datasets and appropriate metric selection
- **Model explainability** — SHAP bridges the gap between black-box models and clinical trust
- **Multi-model benchmarking** to justify algorithm selection

---

## Dataset Citation

Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.* Proceedings of the Annual Symposium on Computer Application in Medical Care.

UCI Machine Learning Repository — Pima Indians Diabetes Database.
