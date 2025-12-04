"""
Random Forest Regression with Optuna Tuning
(using tabular features + description embeddings, GroupKFold CV)

Method summary:
- Load tabular data + SBERT description embeddings (via data_s3_utils)
- Use GroupKFold CV to respect geographic clusters
- Tune Random Forest hyperparameters with Optuna
- Train final model on the full training set
- Evaluate performance with geocluster CV (MAE, RMSE)
- Report training and test-set performance
"""

import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna

from data_s3_utils import load_s2_with_embeddings

X_train_enc, X_test_enc, y_train, y_test, groups_train = load_s2_with_embeddings()

gkf = GroupKFold(n_splits=5)

# Optuna for Random Forest
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        "random_state": 229,
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**params)
    mae_scores = -cross_val_score(
        model,
        X_train_enc,
        y_train,
        scoring="neg_mean_absolute_error",
        cv=gkf,
        groups=groups_train,
        n_jobs=-1,
    )
    return mae_scores.mean()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Best hyperparameters:", best_params)

# Train final model on full training set
final_model = RandomForestRegressor(
    **best_params,
    random_state=229,
    n_jobs=-1,
)
final_model.fit(X_train_enc, y_train)

# Geocluster CV performance
outer_mae = -cross_val_score(
    final_model,
    X_train_enc,
    y_train,
    scoring="neg_mean_absolute_error",
    cv=gkf,
    groups=groups_train,
    n_jobs=-1,
)
outer_rmse = np.sqrt(
    -cross_val_score(
        final_model,
        X_train_enc,
        y_train,
        scoring="neg_mean_squared_error",
        cv=gkf,
        groups=groups_train,
        n_jobs=-1,
    )
)

print(f"\nGeocluster CV MAE: {outer_mae.mean():.3f} +/- {outer_mae.std():.3f}")
print(f"Geocluster CV RMSE: {outer_rmse.mean():.3f} +/- {outer_rmse.std():.3f}")

# Training performance
y_pred_train = final_model.predict(X_train_enc)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"\nTraining MAE: {train_mae:.3f}")
print(f"Training RMSE: {train_rmse:.3f}")

# Test-set performance (optional but nice for the report)
y_pred_test = final_model.predict(X_test_enc)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"\nTest MAE: {test_mae:.3f}")
print(f"Test RMSE: {test_rmse:.3f}")
