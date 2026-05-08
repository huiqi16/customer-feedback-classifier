import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def weighted_log_loss (y_true , y_pred) :
    """
    Compute the weighted cross - entropy (log loss ) given true labels and predicted
    probabilities.

    Parameters:
    - y_true : (N, C) One - hot encoded true labels
    - y_pred : (N, C) Predicted probabilities

    Returns:
    Weighted log loss (scalar).
    """

    # Compute class frequencies
    class_counts = np.sum(y_true , axis =0) # Sum over samples to get counts per class
    class_weights = 1.0 / class_counts
    class_weights /= np. sum (class_weights) # Normalize weights to sum to 1

    # Compute weighted loss
    sample_weights = np.sum (y_true * class_weights, axis =1) # Get weight for each sample
    loss = -np. mean (sample_weights * np. sum ( y_true * np.log( y_pred ) , axis =1) )
    
    return loss

def weighted_log_loss_adapter(estimator, X, y):
    y_pred = estimator.predict_proba(X)
    y_true_ohe = label_binarize(y, classes=np.unique(y))

    return -weighted_log_loss(y_true_ohe, y_pred)

def weighted_log_loss_adapter_rf(estimator, X, y):
    y_pred = estimator.predict_proba(X)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    y_true_ohe = label_binarize(y, classes=np.unique(y))

    return -weighted_log_loss(y_true_ohe, y_pred)

def plot_wll(k_values, wll_scores, label_name, color):
    plt.plot(k_values, wll_scores, marker='o', color=color, label=label_name)

# # ========= Main Models =========

def logistic_regression(x_train, x_test, y_train, y_test):
    print("Running Logistic Regression...")
    
    # train based on logistic regression
    model = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(x_train, y_train.ravel())

    # compute permutation importance
    perm_result_lr = permutation_importance(
        model,
        x_train,
        y_train.ravel(),
        scoring=weighted_log_loss_adapter,
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )
    importances_lr = perm_result_lr.importances_mean
    indices_lr = np.argsort(importances_lr)[::-1]

    # get top k values, test at each 50
    k_values = list(range(1, 300, 50))
    wll_scores_lr = []
    for k in k_values:
        top_k = indices_lr[:k]
        model_k = LogisticRegression(
            penalty='l2', solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42
        )
        model_k.fit(x_train.iloc[:, top_k], y_train)
        y_pred = model_k.predict_proba(x_test.iloc[:, top_k])
        y_true_ohe = label_binarize(y_test, classes=np.unique(y_train))
        wll = weighted_log_loss(y_true_ohe, y_pred)
        wll_scores_lr.append(wll)

    # plot weighted log loss vs top-k features
    plot_wll(k_values, wll_scores_lr, "Logistic Regression + Permutation", "green")
    return indices_lr


def random_forest(x_train, x_test, y_train, y_test):
    print("Running Random Forest...")

    # train based on random forest
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(x_train, y_train)

    # compute permutation importance
    perm_result = permutation_importance(
        model,
        x_train,
        y_train,
        scoring=weighted_log_loss_adapter_rf,
        n_repeats=5, # shuffle each feature 5 times and average the impact on the score
        random_state=42,
        n_jobs=-1 # how many CPU cores to use (use all available cores for maximum speed)
    )
    importances = perm_result.importances_mean
    indices = np.argsort(importances)[::-1]

    # get top k values, test at each 50
    k_values = list(range(1, 300, 50))
    wll_scores = []
    for k in k_values:
        top_k = indices[:k]
        model_k = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=0)
        model_k.fit(x_train.iloc[:, top_k], y_train)
        y_pred = model_k.predict_proba(x_test.iloc[:, top_k])
        y_true_ohe = label_binarize(y_test, classes=np.unique(y_train))
        wll = weighted_log_loss(y_true_ohe, y_pred)
        wll_scores.append(wll)

    # plot weighted log loss vs top-k features
    plot_wll(k_values, wll_scores, "Random Forest + Permutation", "blue")
    return indices


def xgboost(x_train_unscaled, x_test_unscaled, y_train, y_test):
    print("Running XGBoost...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(x_train_unscaled, y_train)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    k_values = list(range(1, 300, 50))
    wll_scores = []
    for k in k_values:
        top_k = indices[:k]
        model_k = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            eval_metric='mlogloss', random_state=42
        )
        model_k.fit(x_train_unscaled.iloc[:, top_k], y_train)
        y_pred = model_k.predict_proba(x_test_unscaled.iloc[:, top_k])
        y_true_ohe = label_binarize(y_test, classes=np.unique(y_train))
        wll = weighted_log_loss(y_true_ohe, y_pred)
        wll_scores.append(wll)

    plot_wll(k_values, wll_scores, "XGBoost + Feature Importance", "orange")
    return indices

# ========= Full Pipeline =========

def main():
    # load datasets
    X_train = pd.read_csv("../raw datasets/X_train.csv")
    y_train = pd.read_csv("../raw datasets/y_train.csv")

    # split training and test
    x_train, x_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    # flatten y to work with sklearn
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # scale training dataset
    scaler = StandardScaler()
    scaler.fit(x_train)

    # convert into dataset
    x_train = pd.DataFrame(scaler.transform(x_train), columns=X_train.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=X_train.columns)

    plt.figure(figsize=(8,6))

    # run models
    lr_ranking = logistic_regression(x_train, x_test, y_train, y_test)
    print('done LR')

    rf_ranking = random_forest(x_train, x_test, y_train, y_test)
    print('done RF')

    xgb_ranking = xgboost(x_train, x_test, y_train, y_test)
    print('done XGB')


    # plot final findings
    plt.xlabel('Top-k Features')
    plt.ylabel('Weighted Log Loss')
    plt.title('Weighted Log Loss vs Number of Features')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # find the union of all 3 models
    top_50_rf = set(rf_ranking[:50])
    top_50_lr = set(lr_ranking[:50])
    top_50_xgb = set(xgb_ranking[:50])

    top_features_union_3 = sorted(list(
        top_50_rf | top_50_lr | top_50_xgb
    ))

    print(f"Union of top 50 features from all 3 models: {len(top_features_union_3)} features")
    print(top_features_union_3)



if __name__ == "__main__":
    main()