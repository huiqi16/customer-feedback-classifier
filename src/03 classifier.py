import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import log_loss

def clip_predictions(y_pred):
    """Safely clip predictions to avoid log(0)."""
    return np.clip(y_pred, 1e-15, 1 - 1e-15)


def load_and_prepare_data():
    """Load, split, and scale the dataset."""
    X = pd.read_csv("raw datasets/X_train.csv")
    y = pd.read_csv("raw datasets/y_train.csv")

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test

# =========================
# Base Models
# =========================

def train_base_models(x_train, y_train):
    """Train Logistic Regression, Random Forest, and XGBoost."""
    
    lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42)
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='mlogloss', random_state=42)

    lr.fit(x_train, y_train)
    rf.fit(x_train, y_train)
    xgb.fit(x_train, y_train)

    return lr, rf, xgb

# =========================
# Meta-Model Training
# =========================

def stack_models(lr, rf, xgb, x_train, y_train, x_test, y_test):
    """Stack base models and train a meta-learner."""

    # Get predictions (probabilities) from base models
    lr_pred_train = clip_predictions(lr.predict_proba(x_train))
    rf_pred_train = clip_predictions(rf.predict_proba(x_train))
    xgb_pred_train = clip_predictions(xgb.predict_proba(x_train))

    # Stack predictions horizontally
    stacked_train = np.hstack((lr_pred_train, rf_pred_train, xgb_pred_train))

    # Same for test set
    lr_pred_test = clip_predictions(lr.predict_proba(x_test))
    rf_pred_test = clip_predictions(rf.predict_proba(x_test))
    xgb_pred_test = clip_predictions(xgb.predict_proba(x_test))

    stacked_test = np.hstack((lr_pred_test, rf_pred_test, xgb_pred_test))

    # Train meta-model (Logistic Regression)
    meta_model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
    meta_model.fit(stacked_train, y_train)

    # Final predictions
    final_pred = clip_predictions(meta_model.predict_proba(stacked_test))

    # Evaluate
    y_test_ohe = label_binarize(y_test, classes=np.unique(y_train))
    loss = log_loss(y_test_ohe, final_pred)

    print(f"\n✅ Final Stacked Model Weighted Log Loss: {loss:.4f}")

    return meta_model

# =========================
# Main
# =========================

def main():
    print("\n📥 Loading and preparing data...")
    x_train, x_test, y_train, y_test = load_and_prepare_data()

    print("\n🏗️ Training base models...")
    lr, rf, xgb = train_base_models(x_train, y_train)

    print("\n🔗 Stacking base model predictions and training meta-model...")
    meta_model = stack_models(lr, rf, xgb, x_train, y_train, x_test, y_test)

    print("\n🎉 Done! Stacked model ready.")

if __name__ == "__main__":
    main()
