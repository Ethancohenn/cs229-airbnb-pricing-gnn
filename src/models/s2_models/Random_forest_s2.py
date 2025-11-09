import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

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

# 4. Geocluster cross-validation on train
model = RandomForestRegressor(
    n_estimators=100,       # Number of trees in the forest
    max_depth=None,         # You can tune this later
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=229,
    n_jobs=-1               # Use all cores for faster training
)
gkf = GroupKFold(n_splits=5)

mae_scores = -cross_val_score(
    model,
    X_train_enc,
    y_train,
    scoring="neg_mean_absolute_error",
    cv=gkf,
    groups=groups_train,
    n_jobs=-1
)
rmse_scores = np.sqrt(
    -cross_val_score(
        model,
        X_train_enc,
        y_train,
        scoring="neg_mean_squared_error",
        cv=gkf,
        groups=groups_train,
        n_jobs=-1
    )
)

print(f"Geocluster CV MAE: {mae_scores.mean():.3f} +/- {mae_scores.std():.3f}")
print(f"Geocluster CV RMSE: {rmse_scores.mean():.3f} +/- {rmse_scores.std():.3f}")

# 5. Fit on full training set
model.fit(X_train_enc, y_train)
y_pred_train = model.predict(X_train_enc)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

print(f"Training MAE: {train_mae:.3f}")
print(f"Training RMSE: {train_rmse:.3f}")
