import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

class BaselineModels:
    def __init__(self, data: pd.DataFrame, target_column: str):
        self.data = data
        self.target_column = target_column
        self.X = data.drop(columns=[target_column])
        self.y = data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        self.models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "KNN Regressor": KNeighborsRegressor(),
            "Random Forest Regressor": RandomForestRegressor()
        }
        self.results = {}

    def train_and_evaluate(self):
        for model_name, model in self.models.items():
            if "Forest" in model_name:  # No scaling for tree-based models
                X_train, X_test = self.X_train, self.X_test
            else:
                X_train, X_test = self.X_train_scaled, self.X_test_scaled

            model.fit(X_train, self.y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            self.results[model_name] = {"MAE": mae, "RMSE": np.sqrt(mse), "R2": r2}
            print(f"{model_name} - MAE: {mae:.3f}, RMSE: {np.sqrt(mse):.3f}, R²: {r2:.3f}")

    def cross_validate(self, cv=5):
        print("\nCross-validation scores:")
        for name, model in self.models.items():
            if "Forest" in name:
                X_data = self.X
            else:
                X_data = self.scaler.fit_transform(self.X)
            scores = cross_val_score(model, X_data, self.y, cv=cv, scoring="r2")
            print(f"{name}: mean R² = {scores.mean():.3f} ± {scores.std():.3f}")

    def plot_results(self):
        results_df = pd.DataFrame(self.results).T.sort_values("RMSE")
        results_df[["MAE", "RMSE"]].plot(kind='bar', figsize=(10,6))
        plt.title('Model Performance Comparison (Lower = Better)')
        plt.ylabel('Error')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def save_results(self, path="results_baseline.csv"):
        pd.DataFrame(self.results).T.to_csv(path)
        print(f"✅ Results saved to {path}")


df = pd.read_csv("clean_listings.csv")  
df.head()



