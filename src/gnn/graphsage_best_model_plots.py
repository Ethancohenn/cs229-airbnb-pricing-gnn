"""
GraphSAGE best model + qualitative plots (S2 with SBERT embeddings)

- Load train_s2.csv and test_s2.csv with SBERT embeddings
- Build train and test graphs with best k_spatial / k_text
- Train a single GraphSAGE model on the training graph
- Evaluate on the held-out test graph (inductive setting)
- Save:
    * scatter plot of true vs. predicted log-price
    * spatial error heatmap over San Francisco (Folium)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from graphsage_model import (
    GraphSAGE_Model,
    build_node_features_train,
    build_node_features_test,
    make_train_val_masks,
)
from graph_builder import convert_embedding_string, build_edge_index_for_df



THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

FIG_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TRAIN_CSV_PATH = os.path.join(PROJECT_ROOT, "src", "data", "train_s2.csv")
TEST_CSV_PATH = os.path.join(PROJECT_ROOT, "src", "data", "test_s2.csv")


# Best hyperparameters from 5-fold geo-CV
BEST_K_SPATIAL = 20
BEST_K_TEXT = 8
HIDDEN_CHANNELS = 128
EPOCHS = 60
LR = 5e-3
SEED = 229


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load train & test data (S2) and parse SBERT
    if not os.path.exists(TRAIN_CSV_PATH) or not os.path.exists(TEST_CSV_PATH):
        raise FileNotFoundError(
            f"train_s2.csv or test_s2.csv not found in src/data.\n"
            f"Expected:\n  {TRAIN_CSV_PATH}\n  {TEST_CSV_PATH}"
        )

    df_train = pd.read_csv(TRAIN_CSV_PATH)
    df_test = pd.read_csv(TEST_CSV_PATH)

    df_train["embedding_sbert"] = df_train["embedding_sbert"].apply(
        convert_embedding_string
    )
    df_test["embedding_sbert"] = df_test["embedding_sbert"].apply(
        convert_embedding_string
    )

    X_train_np, feature_info = build_node_features_train(df_train)
    X_test_np = build_node_features_test(df_test, feature_info)

    y_train_np = df_train["log_price"].values.astype(np.float32)
    y_test_np = df_test["log_price"].values.astype(np.float32)

    print(f"Train features: {X_train_np.shape}, Test features: {X_test_np.shape}")

    print(
        f"Building graphs with k_spatial={BEST_K_SPATIAL}, "
        f"k_text={BEST_K_TEXT}..."
    )
    edge_index_train = build_edge_index_for_df(
        df_train, k_spatial=BEST_K_SPATIAL, k_text=BEST_K_TEXT
    )
    edge_index_test = build_edge_index_for_df(
        df_test, k_spatial=BEST_K_SPATIAL, k_text=BEST_K_TEXT
    )


    x_train = torch.from_numpy(X_train_np).to(device)
    y_train = torch.from_numpy(y_train_np).to(device)
    x_test = torch.from_numpy(X_test_np).to(device)
    y_test = torch.from_numpy(y_test_np).to(device)

    edge_index_train = edge_index_train.to(device)
    edge_index_test = edge_index_test.to(device)

    num_nodes_train = x_train.size(0)
    train_mask, val_mask = make_train_val_masks(num_nodes_train, val_ratio=0.1, seed=SEED)

    class SimpleData:
        pass

    data_train = SimpleData()
    data_train.x = x_train
    data_train.y = y_train
    data_train.edge_index = edge_index_train
    data_train.train_mask = train_mask
    data_train.val_mask = val_mask

    data_test = SimpleData()
    data_test.x = x_test
    data_test.y = y_test
    data_test.edge_index = edge_index_test

    # Define and train GraphSAGE
    in_channels = X_train_np.shape[1]
    model = GraphSAGE_Model(
        in_channels=in_channels,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=1,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("Training GraphSAGE on train_s2 graph...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data_train.x, data_train.edge_index)
        loss = criterion(out[data_train.train_mask], data_train.y[data_train.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == EPOCHS:
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
                f"Epoch {epoch:03d} | Train MSE: {loss.item():.4f} "
                f"| Val MSE: {val_loss.item():.4f} | Val MAE: {val_mae.item():.4f}"
            )

    # Evaluate on train & test graphs
    model.eval()
    with torch.no_grad():
        preds_train = model(data_train.x, data_train.edge_index)
        preds_test = model(data_test.x, data_test.edge_index)

    y_pred_train_np = preds_train.cpu().numpy()
    y_pred_test_np = preds_test.cpu().numpy()

    train_mae = mean_absolute_error(y_train_np, y_pred_train_np)
    train_rmse = np.sqrt(mean_squared_error(y_train_np, y_pred_train_np))
    test_mae = mean_absolute_error(y_test_np, y_pred_test_np)
    test_rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_test_np))

    print(f"\n[GraphSAGE] Training MAE:  {train_mae:.3f}")
    print(f"[GraphSAGE] Training RMSE: {train_rmse:.3f}")
    print(f"[GraphSAGE] Test MAE:      {test_mae:.3f}")
    print(f"[GraphSAGE] Test RMSE:     {test_rmse:.3f}")

    # Scatter plot: true vs. predicted log-price on test
    plt.figure(figsize=(6, 6))
    min_val = min(y_test_np.min(), y_pred_test_np.min())
    max_val = max(y_test_np.max(), y_pred_test_np.max())

    plt.plot([min_val, max_val], [min_val, max_val], "k--", label="Ideal diagonal")
    plt.scatter(y_test_np, y_pred_test_np, s=12, alpha=0.4)

    plt.xlabel("True log-price")
    plt.ylabel("Predicted log-price")
    plt.title("True vs. Predicted log-price (GraphSAGE, geo-aware S2)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    scatter_path = os.path.join(FIG_DIR, "gnn_true_vs_pred_scatter.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"Saved scatter plot to {scatter_path}")

    # Spatial error heatmap over SF (Folium)
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        print("folium not installed, skipping folium heatmap.")
        return

    df_test["true_log_price"] = y_test_np
    df_test["pred_log_price_gnn"] = y_pred_test_np
    df_test["abs_error_gnn"] = np.abs(
        df_test["true_log_price"] - df_test["pred_log_price_gnn"]
    )

    # Normalize error for visualization
    q95 = df_test["abs_error_gnn"].quantile(0.95)
    df_test["err_norm"] = (df_test["abs_error_gnn"] / q95).clip(0, 1)

    center_lat, center_lon = 37.77, -122.42

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    heat_data = [
        [row.latitude, row.longitude, row.err_norm]
        for _, row in df_test.iterrows()
    ]

    gradient = {
        0.0: "rgba(0,0,0,0)",
        0.2: "#e0f2ff",
        0.4: "#60a5fa",
        0.6: "#4f46e5",
        0.8: "#ec4899",
        1.0: "#db2777",
    }

    HeatMap(
        heat_data,
        radius=22,
        blur=28,
        max_zoom=14,
        min_opacity=0.0,
        gradient=gradient,
    ).add_to(m)

    folium_path = os.path.join(FIG_DIR, "gnn_error_heatmap_sf.html")
    m.save(folium_path)
    print(f"Saved Folium error heatmap to {folium_path}")


if __name__ == "__main__":
    main()
