import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
import numpy as np
import matplotlib.pyplot as plt

# load training files
X_train = pd.read_csv("../raw datasets/X_train.csv")
y_train = pd.read_csv("../raw datasets/y_train.csv")

def min_max_mean_std(X_train):
    # get a transposed table for min, max, mean and std of each feature
    feature_stats = X_train.describe().T[["min", "max", "mean", "std"]]

    # get feature indices
    x = np.arange(len(feature_stats))

    # extract each value for plotting
    mean = feature_stats["mean"].values
    std = feature_stats["std"].values
    min_vals = feature_stats["min"].values
    max_vals = feature_stats["max"].values

    # plot
    plt.figure(figsize=(15, 6))
    plt.fill_between(x, min_vals, max_vals, color='lightgray', label='Min to Max Range')
    plt.plot(x, mean, color='blue', label='Mean')
    plt.fill_between(x, mean - std, mean + std, color='blue', alpha=0.2, label='Mean ± Std')
    plt.title("Min, Max, Mean and Standard Deviation for Each Feature")
    plt.xlabel("Feature Index (0 to 299)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def outliers(X_train):
    # calculate z-score for each feature
    z_scores = np.abs(zscore(X_train))

    # flag any outliers with z-score > 3
    outlier_flags = z_scores > 3  

    # count the number of outliers
    outlier_counts = np.sum(outlier_flags, axis=0)

    # sort top 20 outliers
    top_features = np.argsort(outlier_counts)[:20]

    # plot
    plt.figure(figsize=(10,6))
    plt.barh([f'{i}' for i in top_features], outlier_counts.iloc[top_features])
    plt.title("Top 20 Features with Most Outliers (Z-score > 3)")
    plt.xlabel("Outlier Count")
    plt.ylabel("Feature")
    plt.grid(True)
    plt.show()

min_max_mean_std(X_train)
outliers(X_train)