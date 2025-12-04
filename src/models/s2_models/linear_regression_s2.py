import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import optuna

from data_s2_utils import load_s2_with_embeddings

# 1. Load preprocessed data (tabular + SBERT)
X_train_enc, X_test_enc, y_train, y_test, groups_train = load_s2_with_embeddings()

# 2. CV setup
gkf = GroupKFold(n_splits=5)


# 3. Function to optimize Ridge/Lasso alpha (with scaling)
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


# 4. Optuna for Ridge
ridge_study = optuna.create_study(direction="minimize")
ridge_study.optimize(lambda trial: objective(trial, Ridge), n_trials=50)
best_ridge_alpha = ridge_study.best_params["alpha"]
print(f"Best Ridge alpha: {best_ridge_alpha:.5f}")

# 5. Optuna for Lasso
lasso_study = optuna.create_study(direction="minimize")
lasso_study.optimize(lambda trial: objective(trial, Lasso), n_trials=50)
best_lasso_alpha = lasso_study.best_params["alpha"]
print(f"Best Lasso alpha: {best_lasso_alpha:.5f}")

# 6. Final models
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

    model.fit(X_train_enc, y_train)
    y_pred_train = model.predict(X_train_enc)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print(f"Training MAE: {train_mae:.3f}")
    print(f"Training RMSE: {train_rmse:.3f}")
