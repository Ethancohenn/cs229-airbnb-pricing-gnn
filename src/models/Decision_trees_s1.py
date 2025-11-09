import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load train and test data
train_df = pd.read_csv("../data/train_s1.csv")
test_df = pd.read_csv("../data/test_s1.csv")

# 2. Split features & target
y_train = train_df["log_price"]
y_test = test_df["log_price"]
X_train = train_df.drop(columns=["log_price"])
X_test = test_df.drop(columns=["log_price"])

# 3. Encode categorical variables
X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_test_enc = pd.get_dummies(X_test, drop_first=True)

# Align feature columns
X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1)
X_test_enc = X_test_enc.fillna(0)

# 4. Split train / validation for pruning plot
X_tr, X_val, y_tr, y_val = train_test_split(X_train_enc, y_train, test_size=0.2, random_state=229)

# Step 1: Grow the full tree
full_tree = DecisionTreeRegressor(random_state=229)
full_tree.fit(X_tr, y_tr)

# Step 2: Compute pruning path
path = full_tree.cost_complexity_pruning_path(X_tr, y_tr)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Step 3: Train trees for each alpha
trees = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeRegressor(random_state=229, ccp_alpha=ccp_alpha)
    clf.fit(X_tr, y_tr)
    trees.append(clf)

# Step 4: Evaluate on train & validation
train_scores = [clf.score(X_tr, y_tr) for clf in trees]
val_scores = [clf.score(X_val, y_val) for clf in trees]

# Step 5: Plot post-pruning performance
plt.figure(figsize=(8, 5))
plt.plot(ccp_alphas, train_scores, marker='o', label='train')
plt.plot(ccp_alphas, val_scores, marker='o', label='validation')
plt.xlabel('ccp_alpha (pruning strength)')
plt.ylabel('R² score')
plt.legend()
plt.title('Post-pruning performance')
plt.show()

# 6. Choose best alpha (e.g., max validation R²)
best_alpha = ccp_alphas[np.argmax(val_scores)]
print(f"Best ccp_alpha based on validation: {best_alpha:.5f}")

# 7. Train final model with best alpha
final_tree = DecisionTreeRegressor(random_state=229, ccp_alpha=best_alpha)
final_tree.fit(X_train_enc, y_train)

# 8. 10-fold cross-validation on full training set
kf = KFold(n_splits=10, shuffle=True, random_state=229)
mae_scores = -cross_val_score(final_tree, X_train_enc, y_train, scoring="neg_mean_absolute_error", cv=kf)
rmse_scores = np.sqrt(-cross_val_score(final_tree, X_train_enc, y_train, scoring="neg_mean_squared_error", cv=kf))

print(f"10-Fold CV MAE: {mae_scores.mean():.3f} ± {mae_scores.std():.3f}")
print(f"10-Fold CV RMSE: {rmse_scores.mean():.3f} ± {rmse_scores.std():.3f}")

# 9. Fit final model on full train & evaluate
y_pred_train = final_tree.predict(X_train_enc)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"Training MAE: {train_mae:.3f}")
print(f"Training RMSE: {train_rmse:.3f}")
