import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# ------------------------
# Models
# ------------------------
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# ------------------------
# 1. Load train data
# ------------------------
train_df = pd.read_csv("../data/train_s1.csv")
y_train = train_df["log_price"]
X_train = train_df.drop(columns=["log_price"])
X_train_enc = pd.get_dummies(X_train, drop_first=True)

# ------------------------
# 2. Define models
# ------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=229),
    "Lasso": Lasso(alpha=0.001, random_state=229, max_iter=10000),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=229, max_iter=10000),
    "Bayesian Ridge": BayesianRidge(),
    "Decision Tree": DecisionTreeRegressor(random_state=229),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=229),
    "Extra Trees": ExtraTreesRegressor(n_estimators=300, random_state=229),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=229, n_jobs=-1),
    "CatBoost": CatBoostRegressor(n_estimators=300, learning_rate=0.05, depth=4, random_state=229, verbose=0),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVR (RBF)": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=229),
    "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=2000, random_state=229)
}

# ------------------------
# 3. Evaluate models (train + 10-fold CV)
# ------------------------
kf = KFold(n_splits=10, shuffle=True, random_state=229)
results = []

for name, model in models.items():
    # Scaling pour modèles sensibles
    if name in ["Linear Regression", "Ridge", "Lasso", "ElasticNet", "Bayesian Ridge", "SVR (RBF)", "KNN", "MLP"]:
        pipe = make_pipeline(StandardScaler(), model)
    else:
        pipe = model

    # Fit on full train set
    pipe.fit(X_train_enc, y_train)
    y_pred_train = pipe.predict(X_train_enc)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

    # 10-fold CV
    mae_cv = -cross_val_score(pipe, X_train_enc, y_train, scoring='neg_mean_absolute_error', cv=kf).mean()
    rmse_cv = np.sqrt(-cross_val_score(pipe, X_train_enc, y_train, scoring='neg_mean_squared_error', cv=kf).mean())

    results.append({
        "Model": name,
        "MAE_train": mae_train,
        "RMSE_train": rmse_train,
        "MAE_CV": mae_cv,
        "RMSE_CV": rmse_cv
    })

df_results = pd.DataFrame(results).sort_values("MAE_train").reset_index(drop=True)
print("\n=== Train & CV Results ===")
print(df_results.to_string(index=False))

# ------------------------
# 4. Plot comparison (MAE)
# ------------------------
plt.figure(figsize=(14,7))
y_pos = np.arange(len(df_results))
bar_width = 0.35

plt.barh(y_pos - bar_width/2, df_results["MAE_train"], height=bar_width, color="lightgreen", label="Train MAE")
plt.barh(y_pos + bar_width/2, df_results["MAE_CV"], height=bar_width, color="skyblue", label="CV MAE")

# Annoter les valeurs
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

# ------------------------
# 5. Plot comparison (RMSE)
# ------------------------
plt.figure(figsize=(14,7))

plt.barh(y_pos - bar_width/2, df_results["RMSE_train"], height=bar_width, color="lightcoral", label="Train RMSE")
plt.barh(y_pos + bar_width/2, df_results["RMSE_CV"], height=bar_width, color="lightskyblue", label="CV RMSE")

# Annoter les valeurs
for i in range(len(df_results)):
    plt.text(df_results["RMSE_train"][i]+0.002, y_pos[i]-bar_width/2, f"{df_results['RMSE_train'][i]:.3f}", va='center')
    plt.text(df_results["RMSE_CV"][i]+0.002, y_pos[i]+bar_width/2, f"{df_results['RMSE_CV'][i]:.3f}", va='center')

plt.yticks(y_pos, df_results["Model"])
plt.gca().invert_yaxis()
plt.xlabel("RMSE")
plt.title("Model Comparison on Train Set vs 10-Fold CV (RMSE)")
plt.legend()
plt.tight_layout()
plt.show()
