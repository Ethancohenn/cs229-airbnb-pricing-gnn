import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor, plot_importance

# 1. Load train and test data
train_df = pd.read_csv("../../data/train_s2.csv")
test_df = pd.read_csv("../../data/test_s2.csv")
groups_train = train_df["geo_cluster"]

# 2. Split features & target
y_train = train_df["log_price"]
y_test = test_df["log_price"]
X_train = train_df.drop(columns=["log_price", "geo_cluster"])
X_test = test_df.drop(columns=["log_price", "geo_cluster"])

# 3. Encode categorical variables
X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_test_enc = pd.get_dummies(X_test, drop_first=True)

# Align feature columns
X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1)
X_test_enc = X_test_enc.fillna(0)

# 4. Split train / validation
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=229)
tr_idx, val_idx = next(gss.split(X_train_enc, y_train, groups=groups_train))
X_tr, X_val = X_train_enc.iloc[tr_idx], X_train_enc.iloc[val_idx]
y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

# 5. Tune regularization strength (L2 penalty)
lambdas = np.logspace(-3, 2, 15)  # from 0.001 to 100
train_scores, val_scores = [], []

for reg_lambda in lambdas:
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=reg_lambda,  # L2 regularization
        random_state=229,
        objective="reg:squarederror",
        n_jobs=-1
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    train_scores.append(model.score(X_tr, y_tr))
    val_scores.append(model.score(X_val, y_val))

# 6. Plot post-"pruning" (regularization) performance
plt.figure(figsize=(8, 5))
plt.semilogx(lambdas, train_scores, marker='o', label='Train R^2')
plt.semilogx(lambdas, val_scores, marker='o', label='Validation R^2')
plt.xlabel('reg_lambda (L2 regularization strength)')
plt.ylabel('R^2 score')
plt.legend()
plt.title('XGBoost - Regularization tuning (analogous to pruning)')
plt.grid(True)
plt.show()

# 7. Choose best regularization
best_lambda = lambdas[np.argmax(val_scores)]
print(f"Best reg_lambda based on validation: {best_lambda:.5f}")

# 8. Train final model with best regularization
final_xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=best_lambda,
    random_state=229,
    objective="reg:squarederror",
    n_jobs=-1
)
final_xgb.fit(X_train_enc, y_train)

# 9. Geocluster cross-validation
gkf = GroupKFold(n_splits=5)
mae_scores = -cross_val_score(
    final_xgb,
    X_train_enc,
    y_train,
    scoring="neg_mean_absolute_error",
    cv=gkf,
    groups=groups_train
)
rmse_scores = np.sqrt(
    -cross_val_score(
        final_xgb,
        X_train_enc,
        y_train,
        scoring="neg_mean_squared_error",
        cv=gkf,
        groups=groups_train
    )
)

print(f"Geocluster CV MAE: {mae_scores.mean():.3f} +/- {mae_scores.std():.3f}")
print(f"Geocluster CV RMSE: {rmse_scores.mean():.3f} +/- {rmse_scores.std():.3f}")

# 10. Evaluate on training set
y_pred_train = final_xgb.predict(X_train_enc)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"Training MAE: {train_mae:.3f}")
print(f"Training RMSE: {train_rmse:.3f}")

# 11. Feature importance plot
plt.figure(figsize=(10, 6))
plot_importance(final_xgb, importance_type='gain', max_num_features=15)
plt.title('Top 15 Most Important Features (XGBoost)')
plt.show()
