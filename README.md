# Healthcare ML Portfolio

## Stack
Python | scikit-learn | MLflow | pandas | matplotlib

## Project 1 — Diabetes Progression Predictor
- **Algorithm:** Random Forest Regressor
- **Dataset:** sklearn diabetes dataset (442 patients, 10 features)
- **Result:** R² = 0.44 | MAE = 43.2
- **Skills:** scikit-learn, MLflow, feature importance

**Files:** `randomforest_regression_load_diabetes_dataset.ipynb`


## Algorithm Comparison + Cross Validation
**Dataset:** Sklearn Diabetes dataset  
**Task:** Compare 4 algorithms using 5-fold cross validation  

| Model | CV Mean R² | Overfit Gap | MAE |
|-------|-----------|-------------|-----|
| Linear Regression | 0.482 | 0.075 | 42.8 |
| Random Forest | 0.430 | 0.284 | 43.6 |
| Decision Tree | 0.206 | 0.334 | 45.9 |
| SVM | 0.147 | -0.015 | 56.0 |


**Key finding:** Linear Regression wins on this dataset — relationships  
are mostly linear. Random Forest overfits (gap=0.284), needs tuning.  
SVM underperforms due to default hyperparameters (C=1) — tuning fixes this.  
**Files:** `Algorithm_Comparison+Cross_Validation.ipynb`
