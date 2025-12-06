"""
Decision Tree Regression with Pruning (SBERT + Tabular Features)

Method summary:
- Load features and targets using data_s3_utils (tabular + SBERT embeddings)
- Perform train/validation split with GroupShuffleSplit (geoclusters)
- Compute cost-complexity pruning path (ccp_alpha)
- Select best alpha based on validation R²
- Retrain pruned tree on the full training set
- Evaluate using 5-fold GroupKFold CV (MAE, RMSE)
- Report final training performance
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_s3_utils import load_s2_with_embeddings

X_train_enc, X_test_enc, y_train, y_test, groups_train = load_s2_with_embeddings()

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=229)
tr_idx, val_idx = next(gss.split(X_train_enc, y_train, groups=groups_train))

X_tr, X_val = X_train_enc[tr_idx], X_train_enc[val_idx]
y_tr, y_val = y_train[tr_idx], y_train[val_idx]

# Compute pruning path on a controlled tree
base_tree = DecisionTreeRegressor(
    random_state=229,
    max_depth=40,          # Here were avoid ultra-deep trees
    min_samples_leaf=10,
    min_samples_split=20
)
base_tree.fit(X_tr, y_tr)

path = base_tree.cost_complexity_pruning_path(X_tr, y_tr)
ccp_alphas = path.ccp_alphas

n_alphas = 40
if len(ccp_alphas) > n_alphas:
    idx = np.linspace(0, len(ccp_alphas)-1, n_alphas).astype(int)
    ccp_alphas_sub = ccp_alphas[idx]
else:
    ccp_alphas_sub = ccp_alphas


# Train pruned trees
trees = []
train_scores, val_scores = [], []

for alpha in ccp_alphas_sub:
    clf = DecisionTreeRegressor(
        random_state=229,
        ccp_alpha=alpha,
        max_depth=40,
        min_samples_leaf=10,
        min_samples_split=20
    )
    clf.fit(X_tr, y_tr)
    trees.append(clf)
    train_scores.append(clf.score(X_tr, y_tr))
    val_scores.append(clf.score(X_val, y_val))


# Plot R² vs ccp_alpha
plt.figure(figsize=(8, 5))
plt.plot(ccp_alphas_sub, train_scores, marker='o', label='train')
plt.plot(ccp_alphas_sub, val_scores, marker='o', label='validation')
plt.xlabel('ccp_alpha')
plt.ylabel('R^2 score')
plt.title('Post-pruning performance (optimized)')
plt.legend()
plt.tight_layout()
plt.show()

best_alpha = ccp_alphas_sub[np.argmax(val_scores)]
print(f"Best ccp_alpha: {best_alpha:.5e}")

# Train final tree on full training set
final_tree = DecisionTreeRegressor(
    random_state=229,
    ccp_alpha=best_alpha,
    max_depth=40,
    min_samples_leaf=10,
    min_samples_split=20
)
final_tree.fit(X_train_enc, y_train)

# Geocluster CV
gkf = GroupKFold(n_splits=5)

mae_scores = -cross_val_score(
    final_tree, X_train_enc, y_train,
    scoring="neg_mean_absolute_error",
    cv=gkf,
    groups=groups_train
)
rmse_scores = np.sqrt(
    -cross_val_score(
        final_tree, X_train_enc, y_train,
        scoring="neg_mean_squared_error",
        cv=gkf,
        groups=groups_train
    )
)

print(f"\nGeocluster CV MAE: {mae_scores.mean():.3f} +/- {mae_scores.std():.3f}")
print(f"Geocluster CV RMSE: {rmse_scores.mean():.3f} +/- {rmse_scores.std():.3f}")

# ============================
# 10. Training performance
# ============================
y_pred_train = final_tree.predict(X_train_enc)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

print(f"Training MAE: {train_mae:.3f}")
print(f"Training RMSE: {train_rmse:.3f}")