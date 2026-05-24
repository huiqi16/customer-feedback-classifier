# Customer Feedback Classification

> **Multiclass NLP classification** of 10,000 customer comments into 28 product categories using a stacked ensemble of Logistic Regression, Random Forest, and XGBoost.

**Best result: Weighted Log Loss = 0.0043** on the held-out test set.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Distribution Shift Handling](#distribution-shift-handling)
- [Future Improvements](#future-improvements)

---

## Problem Statement

In modern manufacturing, customer feedback must be efficiently routed to the correct department. This project builds a machine learning model to automatically classify customer comments into one of **28 product categories**, streamlining the feedback management process.

Each comment is pre-encoded as a 300-dimensional NLP feature vector. The core challenge is a **severe class imbalance** — class 5 accounts for over 4,000 samples while most other classes have fewer than 500.

---

## Dataset

| Split | Size | Labels |
|-------|------|--------|
| Training set | 10,000 samples | ✅ |
| Test set 1 | 1,000 samples | ❌ |
| Test set 2 | 202 labelled + 1,818 unlabelled | Partial |

- **Features:** 300 NLP-derived numerical features per sample (`X_train.csv`, `X_test_1.csv`, `X_test_2.csv`)
- **Labels:** 28 classes (`y_train.csv`)
- **Note:** Raw dataset files are not included in this repo. Place them in a `raw_datasets/` folder at the project root.

---

## Approach

### 1. Exploratory Data Analysis (`01EDA.py`)
- Checked for missing values, duplicates, and outliers (Z-score > 3)
- Found **< 0.4% of data** affected by outliers — no rows removed
- Most features have mean ≈ 0 and low variance, indicating many uninformative features
- Class 5 dominates (~4,000 samples); most classes have < 500 → severe class imbalance

### 2. Feature Selection (`02Features_.py`)
- Applied **Permutation Importance** using three separate models:
  - Logistic Regression
  - Random Forest
  - XGBoost (feature importance scores)
- Tested top-k features (k = 1 to 300, step 50) measuring Weighted Log Loss
- Found ~50 features is the optimal cutoff for Logistic Regression Permutation Importance
- Took the **union of top 50 features** from all three models → **105 final features**
- Union approach captures both linear (LR) and non-linear (RF, XGBoost) relationships

### 3. Model Development (`03_classifier.py`)
Models evaluated (3-fold stratified cross-validation throughout):

| Model | Train WLL | Test WLL |
|-------|-----------|----------|
| Logistic Regression (standalone) | 0.0052 | 0.0058 |
| Random Forest (standalone) | 0.0013 | 0.0202 ⚠️ overfit |
| XGBoost (standalone) | 6.5139 | 0.0092 |
| LR + RF (2-way stack) | 0.0032 | 0.0045 |
| RF + XGBoost (2-way stack) | 0.0034 | 0.0050 |
| **LR + RF + XGBoost (3-way stack)** | **0.0037** | **0.0043 ✅** |

**Final model:** 3-way stacking ensemble
- Base models: Logistic Regression, Random Forest, XGBoost
- Meta-learner: Logistic Regression (L2 regularisation)
- Class imbalance handled via `class_weight='balanced'` in all models
- Features standardised with `StandardScaler` before training

### Hyperparameter Tuning
Key hyperparameters for final model:
- `class_weight='balanced'` — inversely proportional to class frequency
- LR: `penalty='l2'`, `solver='lbfgs'`, `max_iter=1000`
- RF: `n_estimators=100`
- XGBoost: `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`

---

## Key Results

| Metric | Value |
|--------|-------|
| **Weighted Log Loss (Test Set 1)** | **0.0043** |
| Features selected (from 300) | 105 |
| Final model | 3-way Stacking Ensemble |
| Evaluation strategy | 3-fold stratified cross-validation |

The 3-way stacking method outperformed all individual models and 2-way stacks, achieving the best generalisation. Random Forest alone severely overfitted (train WLL 0.0013 vs test WLL 0.0202).

---

## Project Structure

```
.
├── raw_datasets/
│   ├── X_train.csv
│   ├── y_train.csv
│   ├── X_test_1.csv
│   └── X_test_2.csv
├── src/
│   ├── 01EDA.py             # Exploratory Data Analysis
│   ├── 02Features_.py       # Feature selection (Permutation Importance)
│   └── 03_classifier.py     # Model training & stacking ensemble
├── notebooks/
│   ├── EDA.ipynb            # EDA notebook (exploratory)
│   ├── feature_selection.ipynb
│   ├── model_comparison.ipynb
│   └── final_model.ipynb    # Final model training + prediction generation
├── requirements.txt
└── README.md
```

---

## Setup & Installation

**Requirements:** Python 3.8+

```bash
# Clone the repo
git clone https://github.com/huiqi16/customer-feedback-classifier.git
cd customer-feedback-classification

# Install dependencies
pip install -r requirements.txt
```

Add the dataset CSVs to a `raw_datasets/` folder at the project root before running any scripts.

---

## How to Run

Run the pipeline in order:

```bash
# Step 1: Exploratory Data Analysis
python src/01EDA.py

# Step 2: Feature Selection
python src/02Features_.py

# Step 3: Train stacking ensemble and evaluate
python src/03_classifier.py
```

For final predictions (generates `preds_1.npy` and `preds_2.npy`), open and run:
```
04_final_model.ipynb
```

---

## Distribution Shift Handling

Test Set 2 exhibits **covariate shift** — the input feature distribution differs from the training data. Evidence:
- Feature means differ significantly between the two datasets (even after standardisation)
- Distribution plots show the original dataset has tightly aligned peaks, while the new dataset has multiple shifted peaks

**Approach used:** Density ratio estimation to reweight training samples to better match the test distribution. Due to the complexity of integrating reweighting into the full stacking ensemble, Logistic Regression with density ratio estimation was used for Test Set 2 predictions as it produced the best result among base models.

---

## Future Improvements

- Integrate density ratio estimation directly into the stacking ensemble for Test Set 2
- Test SVM with reduced feature set (computational cost was the blocker)
- Explore LightGBM with better minority class tuning (showed unstable performance on minority classes)
- Investigate label-aware domain adaptation techniques for covariate shift

---

## Evaluation Metric

**Weighted Cross-Entropy Loss** was used over F1 score because it:
- Applies class-specific weights (inversely proportional to class frequency)
- Penalises misclassification of minority classes more heavily
- Produces better-calibrated probability outputs

$$L = -\sum_{i=1}^{N} w_{y_i} \log \hat{p}_i \quad \text{where} \quad w_{y_i} = \frac{1}{f_{y_i}}$$
