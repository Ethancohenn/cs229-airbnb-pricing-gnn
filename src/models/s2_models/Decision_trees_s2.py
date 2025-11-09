import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv("../../data/train_s2.csv")
test_df = pd.read_csv("../../data/test_s2.csv")

groups_train = train_df["geo_cluster"]

y_train = train_df["log_price"]
y_test = test_df["log_price"]
X_train = train_df.drop(columns=["log_price", "geo_cluster"])
X_test = test_df.drop(columns=["log_price", "geo_cluster"])

X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_test_enc = pd.get_dummies(X_test, drop_first=True)
X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1)
X_test_enc = X_test_enc.fillna(0)

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=229)
tr_idx, val_idx = next(gss.split(X_train_enc, y_train, groups=groups_train))
X_tr, X_val = X_train_enc.iloc[tr_idx], X_train_enc.iloc[val_idx]
y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

full_tree = DecisionTreeRegressor(random_state=229)
full_tree.fit(X_tr, y_tr)
path = full_tree.cost_complexity_pruning_path(X_tr, y_tr)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

trees = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeRegressor(random_state=229, ccp_alpha=ccp_alpha)
    clf.fit(X_tr, y_tr)
    trees.append(clf)

train_scores = [clf.score(X_tr, y_tr) for clf in trees]
val_scores = [clf.score(X_val, y_val) for clf in trees]

plt.figure(figsize=(8, 5))
plt.plot(ccp_alphas, train_scores, marker='o', label='train')
plt.plot(ccp_alphas, val_scores, marker='o', label='validation')
plt.xlabel('ccp_alpha (pruning strength)')
plt.ylabel('R^2 score')
plt.legend()
plt.title('Post-pruning performance')
plt.show()

best_alpha = ccp_alphas[np.argmax(val_scores)]
print(f"Best ccp_alpha based on validation: {best_alpha:.5f}")

final_tree = DecisionTreeRegressor(random_state=229, ccp_alpha=best_alpha)
final_tree.fit(X_train_enc, y_train)

gkf = GroupKFold(n_splits=5)
mae_scores = -cross_val_score(
    final_tree,
    X_train_enc,
    y_train,
    scoring="neg_mean_absolute_error",
    cv=gkf,
    groups=groups_train
)
rmse_scores = np.sqrt(-cross_val_score(
    final_tree,
    X_train_enc,
    y_train,
    scoring="neg_mean_squared_error",
    cv=gkf,
    groups=groups_train
))

print(f"Geocluster CV MAE: {mae_scores.mean():.3f} +/- {mae_scores.std():.3f}")
print(f"Geocluster CV RMSE: {rmse_scores.mean():.3f} +/- {rmse_scores.std():.3f}")

y_pred_train = final_tree.predict(X_train_enc)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"Training MAE: {train_mae:.3f}")
print(f"Training RMSE: {train_rmse:.3f}")
