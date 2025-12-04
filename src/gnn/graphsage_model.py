# graphsage_model.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from graph_builder import (
    convert_embedding_string,
    build_edge_index_for_df,
)


# -------------------------------
# 1. GraphSAGE model definition
# -------------------------------
class GraphSAGE_Model(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=1, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.fc(x)
        return out.squeeze(-1)


# ---------------------------------------
# 2. Feature building (train vs. test)
# ---------------------------------------
NUMERIC_COLS = [
    "latitude",
    "longitude",
    "accommodates",
    "number_of_reviews",
    "review_scores_rating",
    "availability_30",
    "availability_365",
    "amenity_count",
    "bedrooms_per_guest",
    "bathrooms_per_guest",
    "beds_per_guest",
]

CATEGORICAL_COLS = [
    "neighbourhood",
    "room_type",
    "host_is_superhost",
]


def build_node_features_train(df_train):
    """
    Build node feature matrix X for training subset and record
    the dummy-column schema + normalization stats.
    df_train['embedding_sbert'] is assumed to already contain np.ndarray rows.
    """
    # ----- numeric -----
    X_num = df_train[NUMERIC_COLS].astype(np.float32).values
    num_mean = X_num.mean(axis=0, keepdims=True)
    num_std = X_num.std(axis=0, keepdims=True) + 1e-6
    X_num_norm = (X_num - num_mean) / num_std

    # ----- categorical → one-hot -----
    cat_dummies = pd.get_dummies(df_train[CATEGORICAL_COLS], drop_first=True)
    cat_dummy_cols = cat_dummies.columns.tolist()
    X_cat = cat_dummies.values.astype(np.float32)

    # ----- embeddings -----
    emb_list = df_train["embedding_sbert"].values  # already np arrays
    X_emb = np.stack(emb_list).astype(np.float32)
    emb_mean = X_emb.mean(axis=0, keepdims=True)
    emb_std = X_emb.std(axis=0, keepdims=True) + 1e-6
    X_emb_norm = (X_emb - emb_mean) / emb_std

    # ----- concat [num | cat | emb] -----
    X_train = np.concatenate([X_num_norm, X_cat, X_emb_norm], axis=1).astype(np.float32)

    feature_info = {
        "num_cols": NUMERIC_COLS,
        "cat_cols": CATEGORICAL_COLS,
        "cat_dummy_cols": cat_dummy_cols,
        "num_mean": num_mean,
        "num_std": num_std,
        "emb_mean": emb_mean,
        "emb_std": emb_std,
        "input_dim": X_train.shape[1],
    }
    return X_train, feature_info


def build_node_features_test(df_test, feature_info):
    """
    Build node feature matrix X for test subset, using the
    dummy-column schema + normalization stats from training.
    """
    # ----- numeric -----
    X_num = df_test[feature_info["num_cols"]].astype(np.float32).values
    X_num_norm = (X_num - feature_info["num_mean"]) / feature_info["num_std"]

    # ----- categorical → one-hot -----
    cat_dummies = pd.get_dummies(df_test[feature_info["cat_cols"]], drop_first=True)
    cat_dummies = cat_dummies.reindex(
        columns=feature_info["cat_dummy_cols"], fill_value=0
    )
    X_cat = cat_dummies.values.astype(np.float32)

    # ----- embeddings -----
    emb_list = df_test["embedding_sbert"].values
    X_emb = np.stack(emb_list).astype(np.float32)
    X_emb_norm = (X_emb - feature_info["emb_mean"]) / feature_info["emb_std"]

    # ----- concat -----
    X_test = np.concatenate([X_num_norm, X_cat, X_emb_norm], axis=1).astype(np.float32)
    return X_test


# -----------------------
# 3. Training helpers
# -----------------------
def make_train_val_masks(num_nodes, val_ratio=0.1, seed=229):
    rng = np.random.RandomState(seed)
    idx = np.arange(num_nodes)
    rng.shuffle(idx)
    split = int((1.0 - val_ratio) * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[idx[:split]] = True
    val_mask[idx[split:]] = True
    return train_mask, val_mask


def train_one_fold(
    df_train_fold,
    df_test_fold,
    device="cpu",
    k_spatial=10,
    k_text=10,
    hidden_channels=128,
    epochs=60,
    lr=5e-3,
):
    """
    Train GraphSAGE on one geocluster fold (inductive):
    - Build graph only on training rows.
    - Build a separate graph for test rows.
    - Evaluate on the unseen test graph.
    Also train a Linear Regression baseline on the same features.
    """

    # ---------- Build features ----------
    X_train_np, feature_info = build_node_features_train(df_train_fold)
    X_test_np = build_node_features_test(df_test_fold, feature_info)

    y_train_np = df_train_fold["log_price"].values.astype(np.float32)
    y_test_np = df_test_fold["log_price"].values.astype(np.float32)

    # ---------- Tabular baseline (Linear Regression) ----------
    linreg = LinearRegression()
    linreg.fit(X_train_np, y_train_np)
    y_pred_test_lin = linreg.predict(X_test_np)
    lin_mae = mean_absolute_error(y_test_np, y_pred_test_lin)
    lin_rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_test_lin))
    print(
        f"  [Baseline Linear] Test MAE: {lin_mae:.4f} | Test RMSE: {lin_rmse:.4f}"
    )

    # ---------- Build graphs ----------
    edge_index_train = build_edge_index_for_df(df_train_fold, k_spatial, k_text)
    edge_index_test = build_edge_index_for_df(df_test_fold, k_spatial, k_text)

    # ---------- Torch tensors ----------
    x_train = torch.from_numpy(X_train_np).to(device)
    y_train = torch.from_numpy(y_train_np).to(device)
    x_test = torch.from_numpy(X_test_np).to(device)
    y_test = torch.from_numpy(y_test_np).to(device)

    edge_index_train = edge_index_train.to(device)
    edge_index_test = edge_index_test.to(device)

    num_nodes_train = x_train.size(0)
    train_mask, val_mask = make_train_val_masks(num_nodes_train)

    data_train = Data(
        x=x_train,
        edge_index=edge_index_train,
        y=y_train,
        train_mask=train_mask,
        val_mask=val_mask,
    )

    data_test = Data(
        x=x_test,
        edge_index=edge_index_test,
        y=y_test,
    )

    # ---------- Model ----------
    in_channels = X_train_np.shape[1]
    model = GraphSAGE_Model(in_channels, hidden_channels, out_channels=1, dropout=0.2)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # ---------- Training loop ----------
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data_train.x, data_train.edge_index)
        loss = criterion(out[data_train.train_mask], data_train.y[data_train.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                out_val = model(data_train.x, data_train.edge_index)
                val_loss = criterion(
                    out_val[data_train.val_mask], data_train.y[data_train.val_mask]
                )
                val_mae = torch.mean(
                    torch.abs(
                        out_val[data_train.val_mask] - data_train.y[data_train.val_mask]
                    )
                )
            print(
                f"  Epoch {epoch:03d} | Train MSE: {loss.item():.4f} "
                f"| Val MSE: {val_loss.item():.4f} | Val MAE: {val_mae.item():.4f}"
            )

    # ---------- Test on unseen graph ----------
    model.eval()
    with torch.no_grad():
        preds_test = model(data_test.x, data_test.edge_index)
        mse_test = criterion(preds_test, data_test.y).item()
        mae_test = torch.mean(torch.abs(preds_test - data_test.y)).item()

    return mse_test, mae_test, lin_mae, lin_rmse


# ---------------------------
# 4. 5-fold geocluster CV + k tuning
# ---------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load full labeled data (CSV has SBERT as string)
    df = pd.read_csv("../data/train_s2.csv")

    # Parse embeddings ONCE into np.ndarray per row
    df["embedding_sbert"] = df["embedding_sbert"].apply(convert_embedding_string)

    groups = df["geo_cluster"].values
    gkf = GroupKFold(n_splits=5)

    # ----- Hyperparameter grid for k -----
    # Adjust if runtime is too long
    spatial_ks = [3, 5, 8, 10, 15, 20, 30]
    text_ks = [0, 3, 5, 8, 10]  # 0 = no text edges

    # store: (k_spatial, k_text) -> (mean_mse, mean_mae)
    results_gnn = {}
    results_lin = {}

    for k_spatial in spatial_ks:
        for k_text in text_ks:
            print(f"\n########## k_spatial={k_spatial}, k_text={k_text} ##########")

            fold_results_gnn = []
            fold_results_lin = []

            for fold, (train_idx, test_idx) in enumerate(
                gkf.split(df, groups=groups), start=1
            ):
                print(f"\n===== Fold {fold} / 5 =====")
                df_train_fold = df.iloc[train_idx].reset_index(drop=True)
                df_test_fold = df.iloc[test_idx].reset_index(drop=True)

                mse_test, mae_test, lin_mae, lin_rmse = train_one_fold(
                    df_train_fold,
                    df_test_fold,
                    device=device,
                    k_spatial=k_spatial,
                    k_text=k_text,
                    hidden_channels=128,
                    epochs=60,
                    lr=5e-3,
                )
                print(
                    f"  [GraphSAGE] Fold {fold} Test MSE: {mse_test:.4f} | "
                    f"Test RMSE: {np.sqrt(mse_test):.4f} | Test MAE: {mae_test:.4f}"
                )

                fold_results_gnn.append((mse_test, mae_test))
                fold_results_lin.append((lin_mae, lin_rmse))

            # aggregate for this (k_spatial, k_text)
            mse_arr = np.array([m for (m, a) in fold_results_gnn])
            mae_arr = np.array([a for (m, a) in fold_results_gnn])
            lin_mae_arr = np.array([m for (m, r) in fold_results_lin])
            lin_rmse_arr = np.array([r for (m, r) in fold_results_lin])

            print("\n===== Summary for this k setting – GraphSAGE =====")
            print(f"k_spatial={k_spatial}, k_text={k_text}")
            print(f"Test MSE:  {mse_arr.mean():.4f} ± {mse_arr.std():.4f}")
            print(
                f"Test RMSE: {np.sqrt(mse_arr).mean():.4f} "
                f"± {np.sqrt(mse_arr).std():.4f}"
            )
            print(f"Test MAE:  {mae_arr.mean():.4f} ± {mae_arr.std():.4f}")

            print("\n===== Summary for this k setting – Linear baseline =====")
            print(f"Test MAE:  {lin_mae_arr.mean():.4f} ± {lin_mae_arr.std():.4f}")
            print(f"Test RMSE: {lin_rmse_arr.mean():.4f} ± {lin_rmse_arr.std():.4f}")

            # store for final comparison
            results_gnn[(k_spatial, k_text)] = (mse_arr.mean(), mae_arr.mean())
            results_lin[(k_spatial, k_text)] = (
                lin_mae_arr.mean(),
                lin_rmse_arr.mean(),
            )

    # ----- Overall comparison across k -----
    print("\n########## Overall comparison across k ##########")
    for (k_spatial, k_text), (mse_mean, mae_mean) in results_gnn.items():
        lin_mae_mean, lin_rmse_mean = results_lin[(k_spatial, k_text)]
        print(
            f"k_spatial={k_spatial:2d}, k_text={k_text:2d} | "
            f"GNN MAE={mae_mean:.4f}, MSE={mse_mean:.4f} | "
            f"Linear MAE={lin_mae_mean:.4f}, RMSE={lin_rmse_mean:.4f}"
        )

    # pick best k according to GNN MAE
    best_k, (best_mse, best_mae) = min(results_gnn.items(), key=lambda x: x[1][1])
    print(
        f"\nBest k by GNN MAE: k_spatial={best_k[0]}, k_text={best_k[1]} "
        f"| MAE={best_mae:.4f}, MSE={best_mse:.4f}"
    )


if __name__ == "__main__":
    main()
