"""
Random Forest Regression with Optuna Tuning (Geocluster CV, tabular features)

Method summary:
- Load s2 tabular data, one-hot encode categorical variables, align train/test
- Tune Random Forest hyperparameters with GroupKFold geocluster CV
- Fit final model on full training data and compute geocluster CV MAE/RMSE
- Report training performance; test-set features prepared for downstream use
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna

train_df = pd.read_csv("../../data/train_s2.csv")
test_df = pd.read_csv("../../data/test_s2.csv")
groups_train = train_df["geo_cluster"]

y_train = train_df["log_price"]
y_test = test_df["log_price"]
X_train = train_df.drop(columns=["log_price", "geo_cluster"])
X_test = train_df.drop(columns=["log_price", "geo_cluster"])

X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_test_enc = pd.get_dummies(X_test, drop_first=True)
X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1)
X_test_enc = X_test_enc.fillna(0)

# CV setup
gkf = GroupKFold(n_splits=5)

# Optuna objective
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        "random_state": 229,
        "n_jobs": -1
    }
    model = RandomForestRegressor(**params)
    mae_scores = -cross_val_score(
        model,
        X_train_enc,
        y_train,
        scoring="neg_mean_absolute_error",
        cv=gkf,
        groups=groups_train,
        n_jobs=-1
    )
    return mae_scores.mean()


# Run Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Best hyperparameters:", best_params)

# Train final model on full training set
final_model = RandomForestRegressor(**best_params, random_state=229, n_jobs=-1)
final_model.fit(X_train_enc, y_train)

# Outer CV for unbiased performance estimate
outer_mae = -cross_val_score(
    final_model,
    X_train_enc,
    y_train,
    scoring="neg_mean_absolute_error",
    cv=gkf,
    groups=groups_train,
    n_jobs=-1
)
outer_rmse = np.sqrt(-cross_val_score(
    final_model,
    X_train_enc,
    y_train,
    scoring="neg_mean_squared_error",
    cv=gkf,
    groups=groups_train,
    n_jobs=-1
))

print(f"Geocluster CV MAE: {outer_mae.mean():.3f} +/- {outer_mae.std():.3f}")
print(f"Geocluster CV RMSE: {outer_rmse.mean():.3f} +/- {outer_rmse.std():.3f}")

y_pred_train = final_model.predict(X_train_enc)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"Training MAE: {train_mae:.3f}")
print(f"Training RMSE: {train_rmse:.3f}")
