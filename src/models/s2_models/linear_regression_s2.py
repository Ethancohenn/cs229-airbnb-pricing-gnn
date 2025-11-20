import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna

# 1. Load data
train_df = pd.read_csv("../../data/train_s2.csv")
test_df = pd.read_csv("../../data/test_s2.csv")
groups_train = train_df["geo_cluster"]

y_train = train_df["log_price"]
y_test = test_df["log_price"]
X_train = train_df.drop(columns=["log_price", "geo_cluster"])
X_test = test_df.drop(columns=["log_price", "geo_cluster"])

X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_test_enc = pd.get_dummies(X_test, drop_first=True)
X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1)
X_test_enc = X_test_enc.fillna(0)

# 2. CV setup
gkf = GroupKFold(n_splits=5)

# 3. Function to optimize Ridge/Lasso alpha
def objective(trial, model_class):
    alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    model = model_class(alpha=alpha)
    mae_scores = -cross_val_score(
        model,
        X_train_enc,
        y_train,
        scoring="neg_mean_absolute_error",
        cv=gkf,
        groups=groups_train
    )
    return mae_scores.mean()  # Optuna minimizes objective by default

# 4. Run Optuna study for Ridge
ridge_study = optuna.create_study(direction="minimize")
ridge_study.optimize(lambda trial: objective(trial, Ridge), n_trials=50)
best_ridge_alpha = ridge_study.best_params["alpha"]
print(f"Best Ridge alpha: {best_ridge_alpha:.5f}")

# 5. Run Optuna study for Lasso
lasso_study = optuna.create_study(direction="minimize")
lasso_study.optimize(lambda trial: objective(trial, Lasso), n_trials=50)
best_lasso_alpha = lasso_study.best_params["alpha"]
print(f"Best Lasso alpha: {best_lasso_alpha:.5f}")

# 6. Fit models with best alpha and evaluate
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=best_ridge_alpha),
    "Lasso": Lasso(alpha=best_lasso_alpha)
}

for name, model in models.items():
    # Geocluster CV
    mae_scores = -cross_val_score(
        model,
        X_train_enc,
        y_train,
        scoring="neg_mean_absolute_error",
        cv=gkf,
        groups=groups_train
    )
    rmse_scores = np.sqrt(-cross_val_score(
        model,
        X_train_enc,
        y_train,
        scoring="neg_mean_squared_error",
        cv=gkf,
        groups=groups_train
    ))
    print(f"\n=== {name} ===")
    print(f"Geocluster CV MAE: {mae_scores.mean():.3f} +/- {mae_scores.std():.3f}")
    print(f"Geocluster CV RMSE: {rmse_scores.mean():.3f} +/- {rmse_scores.std():.3f}")

    # Fit on full training set
    model.fit(X_train_enc, y_train)
    y_pred_train = model.predict(X_train_enc)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print(f"Training MAE: {train_mae:.3f}")
    print(f"Training RMSE: {train_rmse:.3f}")
