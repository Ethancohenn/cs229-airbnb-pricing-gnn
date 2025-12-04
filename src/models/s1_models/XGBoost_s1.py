"""
XGBoost Regression with Regularization Tuning (Tabular Features Only)

Method summary:
- Load train/test tabular data and one-hot encode categorical variables
- Split training data into train/validation sets
- Tune L2 regularization strength
- Select best regularization
- Train final XGBoost model on full training set
- Evaluate using 5-fold cross-validation (MAE, RMSE) and training performance
- Plot feature importances and regularization curve
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor, plot_importance

train_df = pd.read_csv("../../data/train_s1.csv")
test_df = pd.read_csv("../../data/test_s1.csv")

y_train = train_df["log_price"]
y_test = test_df["log_price"]
X_train = train_df.drop(columns=["log_price"])
X_test = test_df.drop(columns=["log_price"])

# Encode categorical variables
X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_test_enc = pd.get_dummies(X_test, drop_first=True)

# Align feature columns
X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1)
X_test_enc = X_test_enc.fillna(0)


X_tr, X_val, y_tr, y_val = train_test_split(X_train_enc, y_train, test_size=0.2, random_state=229)

# Tune regularization strength (L2 penalty)
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

#Plot post-"pruning" (regularization) performance
plt.figure(figsize=(8, 5))
plt.semilogx(lambdas, train_scores, marker='o', label='Train R²')
plt.semilogx(lambdas, val_scores, marker='o', label='Validation R²')
plt.xlabel('reg_lambda (L2 regularization strength)')
plt.ylabel('R² score')
plt.legend()
plt.title('XGBoost – Regularization tuning')
plt.grid(True)
plt.show()

# Choose best regularization
best_lambda = lambdas[np.argmax(val_scores)]
print(f"Best reg_lambda based on validation: {best_lambda:.5f}")

# Train final model with best regularization
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

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=229)
mae_scores = -cross_val_score(final_xgb, X_train_enc, y_train, scoring="neg_mean_absolute_error", cv=kf)
rmse_scores = np.sqrt(-cross_val_score(final_xgb, X_train_enc, y_train, scoring="neg_mean_squared_error", cv=kf))

print(f"5-Fold CV MAE: {mae_scores.mean():.3f} ± {mae_scores.std():.3f}")
print(f"5-Fold CV RMSE: {rmse_scores.mean():.3f} ± {rmse_scores.std():.3f}")

# Evaluate on training set
y_pred_train = final_xgb.predict(X_train_enc)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"Training MAE: {train_mae:.3f}")
print(f"Training RMSE: {train_rmse:.3f}")

plt.figure(figsize=(10, 6))
plot_importance(final_xgb, importance_type='gain', max_num_features=15)
plt.title('Top 15 Most Important Features (XGBoost)')
plt.show()
