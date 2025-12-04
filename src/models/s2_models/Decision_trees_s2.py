import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_s2_utils import load_s2_with_embeddings

# 1. Load preprocessed data (tabular + SBERT embedding)
X_train_enc, X_test_enc, y_train, y_test, groups_train = load_s2_with_embeddings()

# 2. Train/validation split respecting geo clusters
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=229)
tr_idx, val_idx = next(gss.split(X_train_enc, y_train, groups=groups_train))

X_tr, X_val = X_train_enc[tr_idx], X_train_enc[val_idx]
y_tr, y_val = y_train[tr_idx], y_train[val_idx]

# 3. Train an unpruned tree and get CCP path
full_tree = DecisionTreeRegressor(random_state=229)
full_tree.fit(X_tr, y_tr)

path = full_tree.cost_complexity_pruning_path(X_tr, y_tr)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# 4. Train a sequence of pruned trees
trees = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeRegressor(random_state=229, ccp_alpha=ccp_alpha)
    clf.fit(X_tr, y_tr)
    trees.append(clf)

train_scores = [clf.score(X_tr, y_tr) for clf in trees]
val_scores = [clf.score(X_val, y_val) for clf in trees]

# 5. Plot R² vs ccp_alpha
plt.figure(figsize=(8, 5))
plt.plot(ccp_alphas, train_scores, marker='o', label='train')
plt.plot(ccp_alphas, val_scores, marker='o', label='validation')
plt.xlabel('ccp_alpha (pruning strength)')
plt.ylabel('R^2 score')
plt.legend()
plt.title('Post-pruning performance')
plt.tight_layout()
plt.show()

# 6. Choose best ccp_alpha based on validation R²
best_alpha = ccp_alphas[np.argmax(val_scores)]
print(f"Best ccp_alpha based on validation: {best_alpha:.5f}")

# 7. Train final tree on full training data
final_tree = DecisionTreeRegressor(random_state=229, ccp_alpha=best_alpha)
final_tree.fit(X_train_enc, y_train)

# 8. Geocluster CV evaluation
gkf = GroupKFold(n_splits=5)

mae_scores = -cross_val_score(
    final_tree,
    X_train_enc,
    y_train,
    scoring="neg_mean_absolute_error",
    cv=gkf,
    groups=groups_train,
)
rmse_scores = np.sqrt(
    -cross_val_score(
        final_tree,
        X_train_enc,
        y_train,
        scoring="neg_mean_squared_error",
        cv=gkf,
        groups=groups_train,
    )
)

print(f"\nGeocluster CV MAE: {mae_scores.mean():.3f} +/- {mae_scores.std():.3f}")
print(f"Geocluster CV RMSE: {rmse_scores.mean():.3f} +/- {rmse_scores.std():.3f}")

# 9. Training performance
y_pred_train = final_tree.predict(X_train_enc)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"Training MAE: {train_mae:.3f}")
print(f"Training RMSE: {train_rmse:.3f}")
