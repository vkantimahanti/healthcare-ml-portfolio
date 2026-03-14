# 📚 ML Concepts — Learning Notes

## 🤖 Model selection — when to use what
```
Is relationship mostly linear?       → Linear Regression first
Need human-readable rules?           → Decision Tree
Best accuracy on tabular data?       → Random Forest → then XGBoost
Small data + high dimensions?        → SVM (but always tune C parameter)
```
---

## 🔧 Tools

| Tool | What it does |
|------|-------------|
| `mlflow` | Tracks experiments — parameters, metrics, models. Like a run history for every model you train. |
| `shap` | Explains individual predictions — why did THIS patient score high? Mandatory in healthcare AI. |
| `cross_val_score` | Runs cross validation in one line. Returns array of scores across folds. |
```python
# Install both from Day 1 — start the habit early
!pip install mlflow shap -q   # -q = quiet, hides progress noise
```
---

## 📊 Scores — what they mean

| Metric | What it measures | Good value |
|--------|-----------------|------------|
| R² | % of variance explained. 1.0 = perfect, 0 = model is useless | > 0.5 for clinical data |
| MAE | Average prediction error in original units (e.g. disease score points) | As low as possible |
| CV Mean R² | Average R² across 5 folds — more trustworthy than single split R² | Close to single split R² |
| Overfit Gap | Train R² minus Test R² — measures how much model memorizes vs generalizes | < 0.1 is healthy |

---

### Cross Validation
📅 🔗 [See it in code](./Algorithm_Comparison_Cross_Validation.ipynb)
"Cross validation gives a more reliable performance estimate by testing the model on multiple non-overlapping subsets of data, reducing the variance of a single train/test split."

## ⚖️ Overfitting vs Underfitting

**Parameter vs Hyperparameter — know this first:**

| | Who sets it | Example |
|---|---|---|
| Parameter | Model learns it during training | Tree split values |
| Hyperparameter | YOU set it before training | `max_depth`, `n_estimators` |

You don't learn hyperparameters — you tune them. `max_depth` directly
controls overfitting in Decision Trees and Random Forests.

**What it is:**
Splits data into 5 folds, trains on 4, tests on 1 — repeats 5 times 
and averages the scores.

**Why single train/test split is not enough:**
One split depends on luck — which patients landed in test set. 
CV removes that luck by testing on every part of the data.

**Data engineering analogy:**
Like testing your pipeline on 5 different months of data 
instead of just one month. Same idea — don't trust one sample.


### Overfitting vs Underfitting
**Overfitting:**
Model memorizes training data, fails on new data.
Symptom: train score 0.95, test score 0.44 — large gap.
Cause in Random Forest: `max_depth` too high = tree memorizes patients.
Fix: reduce `max_depth`, add more data, use cross validation.

```
Overfitting is when a model performs well on training data but poorly on unseen data — it has memorized noise instead of learning patterns. I detect it by comparing train vs test scores. A large gap means overfitting. 
how do you fix it - I fix it by tuning max_depth, using regularization, or getting more training data.
```

**Underfitting:**
Model too simple, misses real patterns.
Symptom: both train AND test scores are low.
Fix: increase model complexity, add more features.

**Sweet spot:**
Train and test scores are close to each other.
That's what you're always hunting for.

**Parameter:** learned by the model during training (e.g. tree split values)
**Hyperparameter:** set by YOU before training (e.g. `max_depth`, `n_estimators`)

### Pipeline
🔗 [See it in code](./sklearn_pipeline.ipynb)

**What it is:**
Chains preprocessing steps + model into one object.
Fits preprocessing on training data only — prevents data leakage.

**Data engineering analogy:**
Bronze → Silver → Gold — each layer transforms before passing forward.
Pipeline does the same: Imputer → Scaler → Model.

**Syntax:**
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # step 1
    ('scaler',  StandardScaler()),                  # step 2
    ('model',   RandomForestRegressor())            # step 3
])

pipeline.fit(X_train, y_train)   # fits all steps on train only
pipeline.predict(X_test)         # applies all steps to test
```
**Why not preprocess manually:**
Manual: scale full data → split → train. Scaler saw test data. Leakage.
Pipeline: split first → fit scaler on train only → apply to test. Clean.

**Imp Note:**
"Pipeline prevents data leakage by ensuring preprocessing is fitted
only on training data and applied to test data — never the reverse."

### Preprocessing — when to use what

| Tool | Purpose | Use when |
|------|---------|----------|
| `SimpleImputer(strategy='median')` | Fills null values with median | Numeric columns with outliers |
| `SimpleImputer(strategy='mean')` | Fills null values with mean | Numeric columns, no outliers |
| `SimpleImputer(strategy='most_frequent')` | Fills with most common value | Categorical columns |
| `StandardScaler` | Mean=0, Std=1. Scales using mean and std | Normal distribution, no outliers |
| `RobustScaler` | Scales using median and IQR | Clinical data with extreme lab values |

**When to use RobustScaler over StandardScaler:**
A patient with blood pressure of 300 (extreme outlier) pulls
StandardScaler badly — it shifts the mean for everyone.
RobustScaler uses median and IQR — outliers don't affect it.
Always use RobustScaler on clinical numeric features.

### ColumnTransformer
**What it is:**
Applies different preprocessing to different column groups.
Same thinking as your silver layer — different rules per column type.

**Syntax:**
```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    ('numeric', numeric_transformer, ['age', 'bmi', 'bp']),
    ('other',   other_transformer,   ['sex', 's4', 's5'])
])
```

**Rule:**
Numeric columns with outliers → RobustScaler
Numeric columns clean → StandardScaler
Categorical columns → OneHotEncoder (coming in Week 3)

---


**What it tracks:**

| Method | What it logs | Example |
|--------|-------------|---------|
| `mlflow.log_param()` | Settings you chose | `max_depth=5` |
| `mlflow.log_metric()` | Performance numbers | `r2=0.452` |
| `mlflow.sklearn.log_model()` | The trained model file | full pipeline |
| `mlflow.set_experiment()` | Groups related runs | `"diabetes-pipeline"` |

**Syntax:**
```python
mlflow.set_experiment("experiment-name")

with mlflow.start_run(run_name="run-name"):
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("r2", 0.452)
    mlflow.sklearn.log_model(pipeline, "model")
```

**Where it saves:**
Locally in `mlruns/` folder — resets when Colab session ends.
Permanent solution: DagsHub remote server — coming in Week 2.

---

### joblib — saving and loading pipelines

**What it does:**
Saves a trained pipeline to a `.pkl` file.
Load it later to make predictions without retraining.
In production this file goes to a model registry (MLflow, Vertex AI).
```python
import joblib

# Save
joblib.dump(pipeline, 'model.pkl')

# Load and predict
loaded = joblib.load('model.pkl')
loaded.predict(new_data)
```

**Why not pickle:**
joblib is faster than Python's built-in pickle for large numpy arrays.
Industry standard for sklearn models.


### XGBoost — why defaults fail on small data
Three parameters that fixed it:
- `max_depth 5→3`      : shallower trees, less memorization
- `learning_rate=0.1`  : slower correction, more stable  
- `subsample=0.8`      : each tree sees 80% of rows, forces generalization

Rule:
Small dataset (<1k rows)  → Linear Regression or Random Forest wins
Large dataset (10k+ rows) → XGBoost wins with proper tuning


Evaluation metrics that actually matter in healthcare ML — recall, precision, ROC-AUC. 
Accuracy is useless in healthcare. If 95% of patients are healthy and your model predicts "healthy" for everyone, you get 95% accuracy — but you missed every sick patient. That's catastrophic. Healthcare ML uses three metrics instead:

| Metric | What it means | Healthcare translation |
| :--- | :--- | :--- |
| **Recall** | Of all actual sick patients, how many did we catch? | Missing a sick patient = false negative = dangerous |
| **Precision** | Of all patients we flagged as sick, how many actually were? | False alarm = unnecessary treatment = costly |
| **ROC-AUC** | How well does the model separate sick from healthy at all thresholds? | 1.0 = perfect, 0.5 = random guess |



