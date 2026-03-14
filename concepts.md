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



