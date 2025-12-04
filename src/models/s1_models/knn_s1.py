"""
KNN Regression (with Standardization)

Method summary:
- Load train/test data and one-hot encode categorical variables
- Align feature columns between train and test sets
- Build a pipeline: StandardScaler + KNeighborsRegressor
- Evaluate model using 5-fold cross-validation (MAE, RMSE)
- Fit final model on full training data and report training performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load train and test data
train_df = pd.read_csv("../../data/train_s1.csv")
test_df = pd.read_csv("../../data/test_s1.csv")

# Split features & target
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

# 5-fold cross-validation on train
kf = KFold(n_splits=5, shuffle=True, random_state=229)

# Build pipeline: standardize + KNN
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=5))
])

mae_scores = -cross_val_score(
    model, X_train_enc, y_train, scoring="neg_mean_absolute_error", cv=kf
)
rmse_scores = np.sqrt(
    -cross_val_score(model, X_train_enc, y_train, scoring="neg_mean_squared_error", cv=kf)
)

print(f"5-Fold CV MAE: {mae_scores.mean():.3f} ± {mae_scores.std():.3f}")
print(f"5-Fold CV RMSE: {rmse_scores.mean():.3f} ± {rmse_scores.std():.3f}")

# Fit on full training set
model.fit(X_train_enc, y_train)
y_pred_train = model.predict(X_train_enc)

train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

print(f"Training MAE: {train_mae:.3f}")
print(f"Training RMSE: {train_rmse:.3f}")
