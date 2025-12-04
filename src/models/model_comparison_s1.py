"""
Baseline Model Comparison on S1 (Random Split, 10-Fold CV)

Method summary:
- Load S1 tabular training data and one-hot encode categorical features
- Define a wide range of regression baselines:
  linear models, trees, ensembles, KNN, SVR, MLP, XGBoost, CatBoost
- For the decision tree:
    * Use a train/validation split to select the best ccp_alpha (cost-complexity pruning)
- For XGBoost:
    * Tune L2 regularization (reg_lambda) via train/validation split
- For all models:
    * Compute training MAE/RMSE
    * Compute 10-fold CV MAE/RMSE (random KFold)
- Aggregate results into a comparison table and visualize train vs CV MAE
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt


train_df = pd.read_csv("../data/train_s1.csv")
y_train = train_df["log_price"]
X_train = train_df.drop(columns=["log_price"])
X_train_enc = pd.get_dummies(X_train, drop_first=True)


# Helper function for train evaluation + 10-fold CV
def eval_model(model, X, y, scale=False):
    if scale:
        pipe = make_pipeline(StandardScaler(), model)
    else:
        pipe = model
    pipe.fit(X, y)
    y_pred_train = pipe.predict(X)
    mae_train = mean_absolute_error(y, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y, y_pred_train))

    kf = KFold(n_splits=10, shuffle=True, random_state=229)
    mae_cv = -cross_val_score(pipe, X, y, scoring='neg_mean_absolute_error', cv=kf).mean()
    rmse_cv = np.sqrt(-cross_val_score(pipe, X, y, scoring='neg_mean_squared_error', cv=kf).mean())
    return mae_train, rmse_train, mae_cv, rmse_cv


# Evaluate Decision Tree with post-pruning
X_tr, X_val, y_tr, y_val = train_test_split(X_train_enc, y_train, test_size=0.2, random_state=229)
full_tree = DecisionTreeRegressor(random_state=229)
full_tree.fit(X_tr, y_tr)
path = full_tree.cost_complexity_pruning_path(X_tr, y_tr)
ccp_alphas = path.ccp_alphas

trees = [DecisionTreeRegressor(random_state=229, ccp_alpha=a).fit(X_tr, y_tr) for a in ccp_alphas]
val_scores = [t.score(X_val, y_val) for t in trees]
best_alpha = ccp_alphas[np.argmax(val_scores)]
final_tree = DecisionTreeRegressor(random_state=229, ccp_alpha=best_alpha)
tree_results = eval_model(final_tree, X_train_enc, y_train, scale=False)

# Evaluate XGBoost with regularization tuning
lambdas = np.logspace(-3, 2, 15)
train_scores, val_scores = [], []
for reg_lambda in lambdas:
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8,
                         reg_lambda=reg_lambda, random_state=229,
                         objective="reg:squarederror", n_jobs=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    val_scores.append(model.score(X_val, y_val))
best_lambda = lambdas[np.argmax(val_scores)]
final_xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8,
                         reg_lambda=best_lambda, random_state=229,
                         objective="reg:squarederror", n_jobs=-1)
xgb_results = eval_model(final_xgb, X_train_enc, y_train, scale=False)

# Evaluate other models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=229),
    "Lasso": Lasso(alpha=0.001, random_state=229, max_iter=10000),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=229, max_iter=10000),
    "Bayesian Ridge": BayesianRidge(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=229),
    "Extra Trees": ExtraTreesRegressor(n_estimators=300, random_state=229),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=229),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVR (RBF)": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "MLP": MLPRegressor(hidden_layer_sizes=(64,32), activation='relu', max_iter=2000, random_state=229),
    "CatBoost": CatBoostRegressor(n_estimators=300, learning_rate=0.05, depth=4, random_state=229, verbose=0)
}

results = [
    {"Model": "Decision Tree", "MAE_train": tree_results[0], "RMSE_train": tree_results[1],
     "MAE_CV": tree_results[2], "RMSE_CV": tree_results[3]},
    {"Model": "XGBoost", "MAE_train": xgb_results[0], "RMSE_train": xgb_results[1],
     "MAE_CV": xgb_results[2], "RMSE_CV": xgb_results[3]}
]

for name, model in models.items():
    scale = name in ["Linear Regression", "Ridge", "Lasso", "ElasticNet", "Bayesian Ridge", "SVR (RBF)", "KNN", "MLP"]
    mae_train, rmse_train, mae_cv, rmse_cv = eval_model(model, X_train_enc, y_train, scale=scale)
    results.append({"Model": name, "MAE_train": mae_train, "RMSE_train": rmse_train,
                    "MAE_CV": mae_cv, "RMSE_CV": rmse_cv})

df_results = pd.DataFrame(results)
df_results = df_results.sort_values("MAE_CV").reset_index(drop=True)
print(df_results.to_string(index=False))

# Plot MAE comparison
plt.figure(figsize=(14,7))
y_pos = np.arange(len(df_results))
bar_width = 0.35

plt.barh(y_pos - bar_width/2, df_results["MAE_train"], height=bar_width, color="lightgreen", label="Train MAE")
plt.barh(y_pos + bar_width/2, df_results["MAE_CV"], height=bar_width, color="skyblue", label="CV MAE")

for i in range(len(df_results)):
    plt.text(df_results["MAE_train"][i]+0.002, y_pos[i]-bar_width/2, f"{df_results['MAE_train'][i]:.3f}", va='center')
    plt.text(df_results["MAE_CV"][i]+0.002, y_pos[i]+bar_width/2, f"{df_results['MAE_CV'][i]:.3f}", va='center')

plt.yticks(y_pos, df_results["Model"])
plt.gca().invert_yaxis()
plt.xlabel("MAE")
plt.title("Model Comparison on Train Set vs 10-Fold CV (MAE)")
plt.legend()
plt.tight_layout()
plt.show()
