"""
Baseline KNN Regression (s2, no embeddings, no Optuna, no PCA)

Method summary:
- Load s2 tabular data and one-hot encode categorical variables
- Use a simple KNN regressor in a pipeline with StandardScaler
- Evaluate 5-fold GroupKFold CV (geocluster) with MAE/RMSE
- Train final scaler+KNN pipeline on full data and report training metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error


train_df = pd.read_csv("../../data/train_s2.csv")

groups_train = train_df["geo_cluster"]
y_train = train_df["log_price"]

X_train = train_df.drop(columns=["log_price", "geo_cluster"])
X_train_enc = pd.get_dummies(X_train, drop_first=True)

X_train_enc = np.asarray(X_train_enc)
y_train = np.asarray(y_train)
groups_train = np.asarray(groups_train)

# Define baseline KNN model
baseline_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(
        n_neighbors=5,      # simple baseline
        weights="distance", # or "uniform"
        p=2,                # Euclidean distance
    )),
])

# 5-fold GroupKFold CV
gkf = GroupKFold(n_splits=5)
outer_mae_scores = []
outer_rmse_scores = []

for train_idx, val_idx in gkf.split(X_train_enc, y_train, groups=groups_train):
    X_tr, X_val = X_train_enc[train_idx], X_train_enc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = baseline_model
    model.fit(X_tr, y_tr)

    y_pred_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

    outer_mae_scores.append(mae)
    outer_rmse_scores.append(rmse)

print(f"Baseline KNN Geocluster CV MAE:  {np.mean(outer_mae_scores):.3f} +/- {np.std(outer_mae_scores):.3f}")
print(f"Baseline KNN Geocluster CV RMSE: {np.mean(outer_rmse_scores):.3f} +/- {np.std(outer_rmse_scores):.3f}")

# Train final model on full data
baseline_model.fit(X_train_enc, y_train)
y_pred_train = baseline_model.predict(X_train_enc)

train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

print(f"Final Training MAE (baseline KNN):  {train_mae:.3f}")
print(f"Final Training RMSE (baseline KNN): {train_rmse:.3f}")
print("Baseline KNN hyperparameters: n_neighbors=5, weights='distance', p=2 (no PCA, no Optuna)")
