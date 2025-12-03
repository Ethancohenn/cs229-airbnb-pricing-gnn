import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===========================
# Load data
# ===========================
train_df = pd.read_csv("../../data/train_s2.csv")
groups_train = train_df["geo_cluster"]

y_train = train_df["log_price"].values
X_train = train_df.drop(columns=["log_price", "geo_cluster"])
X_train_enc = pd.get_dummies(X_train, drop_first=True).values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# MLP model with 3 hidden layers
# ===========================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, hidden3, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden3, 1)
        )

    def forward(self, x):
        return self.net(x)

# ===========================
# Training function
# ===========================
def train_model(model, loader, optimizer, criterion, epochs=50):
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

# ===========================
# Objective function for Optuna
# ===========================
def objective_factory(X_tr, y_tr, groups_tr):
    def objective(trial):
        hidden1 = trial.suggest_int("hidden1", 32, 128)
        hidden2 = trial.suggest_int("hidden2", 16, 64)
        hidden3 = trial.suggest_int("hidden3", 8, 32)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)

        X_tensor = torch.tensor(X_tr_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_tr.reshape(-1,1), dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor),
                            batch_size=batch_size, shuffle=True)

        model = MLP(X_tr_scaled.shape[1], hidden1, hidden2, hidden3, dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        train_model(model, loader, optimizer, criterion, epochs=30)

        gkf_inner = GroupKFold(n_splits=3)
        mae_scores = []
        for idx_tr, idx_val in gkf_inner.split(X_tr, y_tr, groups_tr):
            X_val = scaler.transform(X_tr[idx_val])
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

            model.eval()
            with torch.no_grad():
                preds = model(X_val_tensor).cpu().numpy().flatten()
            mae_scores.append(mean_absolute_error(y_tr[idx_val], preds))
        return np.mean(mae_scores)
    return objective

# ===========================
# Nested CV
# ===========================
gkf_outer = GroupKFold(n_splits=5)
outer_mae_scores, outer_rmse_scores = [], []

best_model_overall = None
best_val_mae_overall = np.inf
best_params_overall = None
best_scaler_overall = None

for train_idx, val_idx in gkf_outer.split(X_train_enc, y_train, groups_train):
    X_tr, X_val = X_train_enc[train_idx], X_train_enc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    groups_tr = groups_train.iloc[train_idx]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_factory(X_tr, y_tr, groups_tr),
                   n_trials=10, show_progress_bar=False)

    best_params = study.best_params

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    model = MLP(X_tr_scaled.shape[1],
                best_params["hidden1"],
                best_params["hidden2"],
                best_params["hidden3"],
                best_params["dropout"]).to(device)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_scaled, dtype=torch.float32),
                      torch.tensor(y_tr.reshape(-1,1), dtype=torch.float32)),
        batch_size=best_params["batch_size"], shuffle=True
    )
    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.MSELoss()
    train_model(model, loader, optimizer, criterion, epochs=40)

    model.eval()
    with torch.no_grad():
        y_pred_val = model(torch.tensor(X_val_scaled, dtype=torch.float32).to(device)).cpu().numpy().flatten()

    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

    outer_mae_scores.append(val_mae)
    outer_rmse_scores.append(val_rmse)

    print(f"Fold completed | CV MAE: {val_mae:.3f}, RMSE: {val_rmse:.3f} | Best params: {best_params}")

    if val_mae < best_val_mae_overall:
        best_val_mae_overall = val_mae
        best_model_overall = model
        best_params_overall = best_params
        best_scaler_overall = scaler

# ===========================
# Final training error using best model overall
# ===========================
X_train_scaled = best_scaler_overall.transform(X_train_enc)

best_model_overall.eval()
with torch.no_grad():
    y_pred_train = best_model_overall(torch.tensor(X_train_scaled, dtype=torch.float32).to(device)).cpu().numpy().flatten()

train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

print(f"\nFinal Training MAE (best fold model): {train_mae:.3f}")
print(f"Final Training RMSE (best fold model): {train_rmse:.3f}")
print(f"Best hyperparameters (best fold model): {best_params_overall}")

# Nested CV results
print(f"\nNested PyTorch MLP CV MAE: {np.mean(outer_mae_scores):.3f} +/- {np.std(outer_mae_scores):.3f}")
print(f"Nested PyTorch MLP CV RMSE: {np.mean(outer_rmse_scores):.3f} +/- {np.std(outer_rmse_scores):.3f}")