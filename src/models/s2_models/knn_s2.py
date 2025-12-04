import optuna
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_s2_utils import load_s2_with_embeddings

# ===========================
# Load data (tabular + SBERT embeddings)
# ===========================
X_train_enc, X_test_enc, y_train, y_test, groups_train = load_s2_with_embeddings()

# Make sure everything is numpy arrays
X_train_enc = np.asarray(X_train_enc)
y_train = np.asarray(y_train)
groups_train = np.asarray(groups_train)

# ===========================
# Outer CV for unbiased estimation (geocluster)
# ===========================
gkf_outer = GroupKFold(n_splits=5)
outer_mae_scores = []
outer_rmse_scores = []

for train_idx, val_idx in gkf_outer.split(X_train_enc, y_train, groups=groups_train):
    X_tr, X_val = X_train_enc[train_idx], X_train_enc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    groups_tr = groups_train[train_idx]

    # ===========================
    # Inner Optuna study (hyperparameter tuning)
    # ===========================
    def objective(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 2, 20)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        p = trial.suggest_int("p", 1, 2)
        n_components = trial.suggest_int(
            "n_components",
            5,
            min(X_tr.shape[1], 20)
        )

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("knn", KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights,
                p=p,
            )),
        ])

        mae = -cross_val_score(
            model,
            X_tr,
            y_tr,
            scoring="neg_mean_absolute_error",
            cv=GroupKFold(n_splits=3),
            groups=groups_tr,
            n_jobs=-1,
        ).mean()
        return mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    # ===========================
    # Fit best model on inner training fold
    # ===========================
    best_model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=study.best_params["n_components"])),
        ("knn", KNeighborsRegressor(
            n_neighbors=study.best_params["n_neighbors"],
            weights=study.best_params["weights"],
            p=study.best_params["p"],
        )),
    ])
    best_model.fit(X_tr, y_tr)

    # Predict on outer validation fold
    y_pred_val = best_model.predict(X_val)
    outer_mae_scores.append(mean_absolute_error(y_val, y_pred_val))
    outer_rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred_val)))

# ===========================
# Outer CV results
# ===========================
print(f"Nested Geocluster CV MAE: {np.mean(outer_mae_scores):.3f} +/- {np.std(outer_mae_scores):.3f}")
print(f"Nested Geocluster CV RMSE: {np.mean(outer_rmse_scores):.3f} +/- {np.std(outer_rmse_scores):.3f}")

# ===========================
# Train final model on full dataset (with CV-based tuning)
# ===========================
def final_objective(trial):
    n_neighbors = trial.suggest_int("n_neighbors", 2, 20)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    p = trial.suggest_int("p", 1, 2)
    n_components = trial.suggest_int(
        "n_components",
        5,
        min(X_train_enc.shape[1], 20)
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("knn", KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            p=p,
        )),
    ])

    # Use GroupKFold CV instead of pure training error for more sensible tuning
    mae = -cross_val_score(
        model,
        X_train_enc,
        y_train,
        scoring="neg_mean_absolute_error",
        cv=GroupKFold(n_splits=5),
        groups=groups_train,
        n_jobs=-1,
    ).mean()
    return mae

study_final = optuna.create_study(direction="minimize")
study_final.optimize(final_objective, n_trials=30, show_progress_bar=False)

best_final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=study_final.best_params["n_components"])),
    ("knn", KNeighborsRegressor(
        n_neighbors=study_final.best_params["n_neighbors"],
        weights=study_final.best_params["weights"],
        p=study_final.best_params["p"],
    )),
])
best_final_model.fit(X_train_enc, y_train)

y_pred_train = best_final_model.predict(X_train_enc)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

print(f"Final Training MAE: {train_mae:.3f}")
print(f"Final Training RMSE: {train_rmse:.3f}")
print(f"Best hyperparameters on full dataset: {study_final.best_params}")
