"""
Linear/Ridge/Lasso Regression with Optuna Alpha Tuning (Random CV, S1)

Method summary:
- Load S1 train/test tabular data and one-hot encode categorical variables
- Align feature matrices between train and test sets
- Use 5-fold KFold CV to tune Ridge and Lasso regularization strength with Optuna
- Evaluate LinearRegression, Ridge, and Lasso via random 5-fold MAE/RMSE
- Fit each model on full training data and report training performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna

# --------------------
# Load and preprocess data
# --------------------
train_df = pd.read_csv("../../data/train_s1.csv")
test_df = pd.read_csv("../../data/test_s1.csv")

y_train = train_df["log_price"]
y_test = test_df["log_price"]
X_train = train_df.drop(columns=["log_price"])
X_test = test_df.drop(columns=["log_price"])

# One-hot encode categorical variables
X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_test_enc = pd.get_dummies(X_test, drop_first=True)

# Align columns between train and test
X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1)
X_test_enc = X_test_enc.fillna(0)

# --------------------
# CV setup
# --------------------
kf = KFold(n_splits=5, shuffle=True, random_state=229)

# --------------------
# Optuna objective for Ridge/Lasso alpha
# --------------------
def objective(trial, model_class):
    alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    model = model_class(alpha=alpha)
    mae_scores = -cross_val_score(
        model,
        X_train_enc,
        y_train,
        scoring="neg_mean_absolute_error",
        cv=kf,
    )
    return mae_scores.mean()  # Optuna minimizes this

# --------------------
# Tune Ridge alpha
# --------------------
ridge_study = optuna.create_study(direction="minimize")
ridge_study.optimize(lambda trial: objective(trial, Ridge), n_trials=50)
best_ridge_alpha = ridge_study.best_params["alpha"]
print(f"Best Ridge alpha (S1): {best_ridge_alpha:.5f}")

# --------------------
# Tune Lasso alpha
# --------------------
lasso_study = optuna.create_study(direction="minimize")
lasso_study.optimize(lambda trial: objective(trial, Lasso), n_trials=50)
best_lasso_alpha = lasso_study.best_params["alpha"]
print(f"Best Lasso alpha (S1): {best_lasso_alpha:.5f}")

# --------------------
# Evaluate Linear, Ridge, Lasso
# --------------------
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=best_ridge_alpha),
    "Lasso": Lasso(alpha=best_lasso_alpha),
}

for name, model in models.items():
    # 5-fold CV
    mae_scores = -cross_val_score(
        model,
        X_train_enc,
        y_train,
        scoring="neg_mean_absolute_error",
        cv=kf,
    )
    rmse_scores = np.sqrt(
        -cross_val_score(
            model,
            X_train_enc,
            y_train,
            scoring="neg_mean_squared_error",
            cv=kf,
        )
    )

    print(f"\n=== {name} (S1) ===")
    print(f"5-Fold CV MAE: {mae_scores.mean():.3f} ± {mae_scores.std():.3f}")
    print(f"5-Fold CV RMSE: {rmse_scores.mean():.3f} ± {rmse_scores.std():.3f}")

    # Fit on full training set
    model.fit(X_train_enc, y_train)
    y_pred_train = model.predict(X_train_enc)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print(f"Training MAE: {train_mae:.3f}")
    print(f"Training RMSE: {train_rmse:.3f}")