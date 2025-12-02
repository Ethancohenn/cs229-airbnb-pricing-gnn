import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor, plot_importance
import optuna
import matplotlib.pyplot as plt

# 1. Load train and test data
train_df = pd.read_csv("../../data/train_s2.csv")
groups_train = train_df["geo_cluster"]
y_train = train_df["log_price"]
X_train = train_df.drop(columns=["log_price", "geo_cluster"])

# 2. Encode categorical variables
X_train_enc = pd.get_dummies(X_train, drop_first=True)

# =========================
# 3. Double CV
# =========================
outer_gkf = GroupKFold(n_splits=5)
outer_mae_scores, outer_rmse_scores = [], []

for fold, (train_idx, test_idx) in enumerate(outer_gkf.split(X_train_enc, y_train, groups_train)):
    print(f"\n=== Outer fold {fold + 1} ===")
    X_tr, X_val = X_train_enc.iloc[train_idx], X_train_enc.iloc[test_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[test_idx]
    groups_tr = groups_train.iloc[train_idx]

    # 3a. Inner CV for hyperparameter tuning
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "objective": "reg:squarederror",
            "random_state": 229,
            "n_jobs": -1
        }
        model = XGBRegressor(**params)
        inner_gkf = GroupKFold(n_splits=3)
        mae_scores = -cross_val_score(
            model,
            X_tr,
            y_tr,
            scoring="neg_mean_absolute_error",
            cv=inner_gkf,
            groups=groups_tr,
            n_jobs=-1
        )
        return mae_scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print("Best params (inner CV):", study.best_params)

    # 3b. Train final model on outer train split
    best_params = study.best_params
    model = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=229,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    # 3c. Evaluate on outer validation split
    y_val_pred = model.predict(X_val)
    outer_mae = mean_absolute_error(y_val, y_val_pred)
    outer_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    outer_mae_scores.append(outer_mae)
    outer_rmse_scores.append(outer_rmse)

    print(f"Outer fold MAE: {outer_mae:.3f}, RMSE: {outer_rmse:.3f}")

# =========================
# 4. Overall CV results
# =========================
print(f"\nGeocluster CV MAE: {np.mean(outer_mae_scores):.3f} +/- {np.std(outer_mae_scores):.3f}")
print(f"Geocluster CV RMSE: {np.mean(outer_rmse_scores):.3f} +/- {np.std(outer_rmse_scores):.3f}")

# =========================
# 5. Train final model on full dataset with best params from last fold
# =========================
final_model = XGBRegressor(
    **study.best_params,
    objective="reg:squarederror",
    random_state=229,
    n_jobs=-1
)
final_model.fit(X_train_enc, y_train)

# Training MAE/RMSE
y_pred_train = final_model.predict(X_train_enc)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"\nTraining MAE: {train_mae:.3f}")
print(f"Training RMSE: {train_rmse:.3f}")

# Feature importance
plt.figure(figsize=(10, 6))
plot_importance(final_model, importance_type='gain', max_num_features=15)
plt.title('Top 15 Most Important Features (XGBoost)')
plt.show()
