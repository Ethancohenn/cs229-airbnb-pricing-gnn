import optuna
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
train_df = pd.read_csv("../../data/train_s2.csv")
groups_train = train_df["geo_cluster"]

y_train = train_df["log_price"]
X_train = train_df.drop(columns=["log_price", "geo_cluster"])

X_train_enc = pd.get_dummies(X_train, drop_first=True)

# Geocluster CV
gkf = GroupKFold(n_splits=5)

# Objective function for Optuna
def objective(trial):
    n_neighbors = trial.suggest_int("n_neighbors", 2, 20)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    p = trial.suggest_int("p", 1, 2)  # 1=Manhattan, 2=Euclidean

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            p=p
        ))
    ])

    # Inner CV: MAE over geo clusters
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

# Run Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)

# Fit final model with best params
best_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(
        n_neighbors=study.best_params["n_neighbors"],
        weights=study.best_params["weights"],
        p=study.best_params["p"]
    ))
])
best_model.fit(X_train_enc, y_train)

# Outer CV for unbiased test error
mae_scores = -cross_val_score(
    best_model,
    X_train_enc,
    y_train,
    scoring="neg_mean_absolute_error",
    cv=gkf,
    groups=groups_train,
    n_jobs=-1
)
rmse_scores = np.sqrt(-cross_val_score(
    best_model,
    X_train_enc,
    y_train,
    scoring="neg_mean_squared_error",
    cv=gkf,
    groups=groups_train,
    n_jobs=-1
))

print(f"Geocluster CV MAE: {mae_scores.mean():.3f} +/- {mae_scores.std():.3f}")
print(f"Geocluster CV RMSE: {rmse_scores.mean():.3f} +/- {rmse_scores.std():.3f}")

# Training performance
y_pred_train = best_model.predict(X_train_enc)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"Training MAE: {train_mae:.3f}")
print(f"Training RMSE: {train_rmse:.3f}")
