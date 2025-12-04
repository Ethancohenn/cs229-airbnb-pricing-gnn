"""
Linear Models (Linear Regression, Ridge, Lasso) with Optuna Tuning
(using tabular features + description embeddings, GroupKFold CV)

Method summary:
- Load tabular data + SBERT description embeddings using data_s3_utils
- Use GroupKFold CV (to respect geoclusters)
- Tune Ridge and Lasso regularization strength (alpha) with Optuna
- Evaluate Linear Regression, Ridge, and Lasso with geocluster CV (MAE, RMSE)
- Train final models and report training performance
"""

import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import optuna

from data_s3_utils import load_s2_with_embeddings

X_train_enc, X_test_enc, y_train, y_test, groups_train = load_s2_with_embeddings()

gkf = GroupKFold(n_splits=5)


# Optimizing Ridge/Lasso alpha (with scaling)
def objective(trial, model_class):
    alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    model = make_pipeline(
        StandardScaler(),
        model_class(alpha=alpha, max_iter=10000),
    )
    mae_scores = -cross_val_score(
        model,
        X_train_enc,
        y_train,
        scoring="neg_mean_absolute_error",
        cv=gkf,
        groups=groups_train,
    )
    return mae_scores.mean()


# Optuna for Ridge
ridge_study = optuna.create_study(direction="minimize")
ridge_study.optimize(lambda trial: objective(trial, Ridge), n_trials=50)
best_ridge_alpha = ridge_study.best_params["alpha"]
print(f"Best Ridge alpha: {best_ridge_alpha:.5f}")

# Optuna for Lasso
lasso_study = optuna.create_study(direction="minimize")
lasso_study.optimize(lambda trial: objective(trial, Lasso), n_trials=50)
best_lasso_alpha = lasso_study.best_params["alpha"]
print(f"Best Lasso alpha: {best_lasso_alpha:.5f}")

# Final models
models = {
    "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=best_ridge_alpha)),
    "Lasso": make_pipeline(
        StandardScaler(), Lasso(alpha=best_lasso_alpha, max_iter=10000)
    ),
}

for name, model in models.items():
    mae_scores = -cross_val_score(
        model,
        X_train_enc,
        y_train,
        scoring="neg_mean_absolute_error",
        cv=gkf,
        groups=groups_train,
    )
    rmse_scores = np.sqrt(
        -cross_val_score(
            model,
            X_train_enc,
            y_train,
            scoring="neg_mean_squared_error",
            cv=gkf,
            groups=groups_train,
        )
    )
    print(f"\n=== {name} ===")
    print(f"Geocluster CV MAE: {mae_scores.mean():.3f} +/- {mae_scores.std():.3f}")
    print(f"Geocluster CV RMSE: {rmse_scores.mean():.3f} +/- {rmse_scores.std():.3f}")

    # Fit on full train to inspect coefficients
    model.fit(X_train_enc, y_train)
    y_pred_train = model.predict(X_train_enc)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print(f"Training MAE: {train_mae:.3f}")
    print(f"Training RMSE: {train_rmse:.3f}")

    # inspect Lasso sparsity
    if name == "Lasso":
        # Get the actual Lasso object inside the pipeline
        lasso = model.named_steps["lasso"]
        coefs = lasso.coef_

        n_total = coefs.size
        n_zero = np.sum(coefs == 0.0)
        n_nonzero = n_total - n_zero

        print(f"\nLasso coefficient sparsity:")
        print(f"  Total coefficients: {n_total}")
        print(f"  Zero coefficients: {n_zero}")
        print(f"  Non-zero coefficients: {n_nonzero}")
        print(f"  Sparsity: {100 * n_zero / n_total:.1f}% of coefficients are exactly 0")