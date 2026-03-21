# 🏥 Healthcare ML Portfolio

![Python](https://img.shields.io/badge/Python-3.10-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![Status](https://img.shields.io/badge/Status-Active-green)
![Domain](https://img.shields.io/badge/Domain-Healthcare-red)

> A hands-on ML learning journey built on top of real Data Engineering 
> patterns — medallion architecture, cloud pipelines, and production tooling.
> Updated daily. Built in public.

---


## 👤 Who is this for

Data engineers transitioning into ML and AI. Every project here connects
ML directly to data engineering concepts you already know — pipelines,
data quality, layered architecture. No toy examples. No iris dataset.

If you are a data engineer curious about ML, or an ML engineer curious
about production data pipelines — this repo is for you.


## 🚀 How to run any project

All notebooks run on Google Colab — no local setup needed.

1. Click the notebook link in the project section below
2. In Colab: `File → Save a copy in Drive`
3. Run cells from top to bottom
4. Each notebook is self-contained with install commands in Cell 1


## 📂 Projects

### Week 1 — ML Foundations on Healthcare Data

---

#### Random Forest Baseline
Colab link - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vkantimahanti/healthcare-ml-portfolio/blob/main/randomforest_regression_load_diabetes_dataset.ipynb)

git repo link - 📓 [randomforest_regression_load_diabetes_dataset.ipynb](./randomforest_regression_load_diabetes_dataset.ipynb)

| | |
|---|---|
| **Dataset** | Sklearn Diabetes (442 patients, 10 clinical features) |
| **Task** | Regression — predict disease progression score |
| **Algorithm** | Random Forest Regressor |
| **R² Score** | 0.441 |
| **MAE** | 43.2 |

**What I built:** Full ML loop — load data → train → evaluate → 
MLflow experiment tracking → push to GitHub.

**Key learning:** MLflow tracks every experiment run so you can 
compare models over time. Habit started from Day 1, not after.


---

#### Algorithm Comparison + Cross Validation
Colab link - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vkantimahanti/healthcare-ml-portfolio/blob/main/Algorithm_Comparison+Cross_Validation.ipynb)

git repo link - 📓[Algorithm_Comparison_Cross_Validation.ipynb](./Algorithm_Comparison_Cross_Validation.ipynb)

| Model | CV Mean R² | Overfit Gap | MAE |
|-------|-----------|-------------|-----|
| Linear Regression | **0.482** | **0.075** | 42.8 |
| Random Forest | 0.430 | 0.284 | 43.6 |
| Decision Tree | 0.206 | 0.334 | 45.9 |
| SVM | 0.147 | -0.015 | 56.0 |

**What I built:** 5-fold cross validation across 4 algorithms. 
Logged all runs to MLflow for side-by-side comparison.

**Key insight:** Linear Regression outperformed Random Forest 
(CV R² 0.482 vs 0.430) because the diabetes dataset has mostly 
linear feature relationships. Random Forest showed significant 
overfitting (gap=0.284) — will be fixed via GridSearchCV in Day 5. 
SVM collapsed with default parameters (C=1) but is recoverable 
with tuning — shows that algorithm choice and hyperparameter 
tuning are inseparable decisions.



#### sklearn Pipeline + ColumnTransformer
Colab link - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vkantimahanti/healthcare-ml-portfolio/blob/main/sklearn_pipeline.ipynb)

git repo link - 📓 [sklearn_pipeline.ipynb](./sklearn_pipeline.ipynb)

**What I built:** Production-grade sklearn Pipeline with ColumnTransformer
applying different preprocessing per feature type. Compared Random Forest,
Linear Regression and XGBoost inside the same pipeline.

**Key finding:** XGBoost default CV R² 0.283 → tuned 0.416 by reducing
max_depth 5→3, adding learning_rate=0.1 and subsample=0.8. Small datasets
favor simple models — XGBoost will win on Week 3 large CMS dataset.

---

#### random forest and xgboost with gridsearchcv for hyperparamter tuning.
Colab link - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vkantimahanti/healthcare-ml-portfolio/blob/main/Hyperparameters_gridsearchcv_breastcancer.ipynb)

git repo link - 📓 [sklearn_pipeline.ipynb](./Hyperparameters_gridsearchcv_breastcancer.ipynb)

**What I built:** GridSearchCV testing 36 parameter combinations across 5-fold cross validation, optimised for Recall (healthcare priority).

**Key finding:** Default pipeline Recall 0.931 → tuned 0.958 ↑ 0.028. Top 5 combinations scored within 0.003 of each other — rank 2 (50 trees, max_depth=7) is the smarter production choice: same clinical outcome, 4x faster inference. Always check std alongside mean — lower std means more stable predictions across different patient groups.


#### Capstone project on heart disease prediction
Colab link - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vkantimahanti/healthcare-ml-portfolio/blob/main/HeartDiseasePrediction.ipynb)

git repo link - 📓 [sklearn_pipeline.ipynb](./HeartDiseasePrediction.ipynb)

Project covers full clinical ML System:
   ├── EDA on real UCI Heart Disease data
         ├── ColumnTransformer pipeline
         ├── 3 models compared on Recall
         ├── GridSearchCV (confirmed defaults best)
         ├── SHAP explainability (thal, cp, ca top drivers)
         ├── FastAPI prediction endpoint
         └── 3 patient risk predictions validated clinically

## 👨‍💻 About
Goal - Deep diving into ML and AI.
Background in multi-cloud (Azure, GCP), Databricks, medallion architecture, and healthcare data (EHR, FHIR, pre-authorization, Hedis, Stars).

## How to run
All notebooks run on Google Colab — no local setup needed.
1. Click the notebook link
2. Click on colab link or
3. File → Save a copy in Drive 
4. Run cells top to bottom


## Stack
| Tool | Purpose |
|------|---------|
| scikit-learn | ML models and pipelines |
| MLflow | Experiment tracking |
| pandas / numpy | Data manipulation |
| matplotlib | Visualization |
