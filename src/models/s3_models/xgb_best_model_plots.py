"""
XGBoost best model + qualitative plots (S2 with SBERT embeddings)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from data_s3_utils import load_s2_with_embeddings


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))

FIG_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TEST_CSV_PATH = os.path.join(PROJECT_ROOT, "src", "data", "test_s2.csv")

X_train_enc, X_test_enc, y_train, y_test, groups_train = load_s2_with_embeddings()

X_train_enc = np.asarray(X_train_enc)
X_test_enc = np.asarray(X_test_enc)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

print(f"Train shape: {X_train_enc.shape}, Test shape: {X_test_enc.shape}")

# percentage of listings within ±10/20/30% in price
def print_within_thresholds(y_true_log, y_pred_log, label):
    """
    Print the share of listings whose predicted price is within
    ±10%, ±20% and ±30% of the true price.

    Assumes y_*_log are natural logs of price.
    """
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)

    rel_err = np.abs(y_true - y_pred) / y_true  # relative error

    print(f"\n[{label}] share of listings within a given % of true price:")
    for th in [0.10, 0.20, 0.30]:
        prop = np.mean(rel_err <= th)
        print(f"[{label}] {prop * 100:.1f}% of listings within ±{int(th * 100)}%")


# Best hyperparameters from nested CV / Optuna
BEST_PARAMS = {
    "n_estimators": 481,
    "max_depth": 4,
    "learning_rate": 0.062484137294537445,
    "subsample": 0.7632923996814528,
    "colsample_bytree": 0.9328241957015925,
    "reg_lambda": 1.8948364440954653,
    "reg_alpha": 0.005103456313519248,
}

model = XGBRegressor(
    **BEST_PARAMS,
    objective="reg:squarederror",
    random_state=229,
    n_jobs=-1,
)

# Train on full training data and evaluate on train & test
print("Fitting final XGBoost model with best hyperparameters...")
model.fit(X_train_enc, y_train)

y_pred_train = model.predict(X_train_enc)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"Training MAE:  {train_mae:.3f}")
print(f"Training RMSE: {train_rmse:.3f}")

y_pred_test = model.predict(X_test_enc)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"Test MAE:      {test_mae:.3f}")
print(f"Test RMSE:     {test_rmse:.3f}")

# Overall test-set accuracy in price space
print_within_thresholds(y_test, y_pred_test, label="XGBoost (test)")

# Cluster-level metrics on the test set
if os.path.exists(TEST_CSV_PATH):
    df_test_clusters = pd.read_csv(TEST_CSV_PATH)

    if len(df_test_clusters) != len(y_test):
        raise ValueError(
            f"Length mismatch: test_s2.csv has {len(df_test_clusters)} rows, "
            f"but y_test has {len(y_test)}."
        )

    if "geo_cluster" not in df_test_clusters.columns:
        raise KeyError(
            "Column 'geo_cluster' not found in test_s2.csv; "
            "cannot compute per-cluster metrics."
        )

    geo_test = df_test_clusters["geo_cluster"].values
    geo_test = df_test_clusters["geo_cluster"].values
    unique_clusters = np.unique(geo_test)

    # Map original cluster labels (e.g. 0, 8, 11, ...) to 1..
    cluster_id_map = {old: i + 1 for i, old in enumerate(sorted(unique_clusters))}

    print("\n--- Cluster-level price accuracy on test set (XGBoost) ---")
    for cl in sorted(unique_clusters):
        mask = (geo_test == cl)
        new_id = cluster_id_map[cl]
        label = f"XGBoost (test, cluster {new_id})"
        print_within_thresholds(y_test[mask], y_pred_test[mask], label=label)

    print("\nCluster ID mapping (original -> display):")
    for old, new in cluster_id_map.items():
        print(f"  {old} -> {new}")
else:
    print(f"WARNING: {TEST_CSV_PATH} not found, skipping per-cluster metrics.")


# Scatter plot: true vs. predicted log-price
plt.figure(figsize=(6, 6))

min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())

plt.plot([min_val, max_val], [min_val, max_val], "k--", label="Ideal diagonal")
plt.scatter(y_test, y_pred_test, s=12, alpha=0.4)

plt.xlabel("True log-price")
plt.ylabel("Predicted log-price")
plt.title("True vs. Predicted log-price (XGBoost, geo-aware with text embedding - S3)")
plt.legend()
plt.grid(True)
plt.tight_layout()

scatter_path = os.path.join(FIG_DIR, "xgb_true_vs_pred_scatter.png")
plt.savefig(scatter_path, dpi=300)
plt.close()
print(f"Saved scatter plot to {scatter_path}")

# Kernel density map of prediction error over SF
try:
    import folium
    from folium.plugins import HeatMap
except ImportError:
    print("folium not installed, skipping folium heatmap.")
else:
    if os.path.exists(TEST_CSV_PATH):
        df_test = pd.read_csv(TEST_CSV_PATH)

        if len(df_test) != len(y_test):
            raise ValueError(
                f"Length mismatch: test_s2.csv has {len(df_test)} rows, "
                f"but y_test has {len(y_test)}."
            )

        # Add errors
        df_test["true_log_price"] = y_test
        df_test["pred_log_price"] = y_pred_test
        df_test["abs_error"] = np.abs(
            df_test["true_log_price"] - df_test["pred_log_price"]
        )

        # Normalize error to [0, 1], clipping big outliers
        q95 = df_test["abs_error"].quantile(0.95)
        df_test["err_norm"] = (df_test["abs_error"] / q95).clip(0, 1)

        # Center of SF
        center_lat, center_lon = 37.77, -122.42

        # Map geo_cluster to display IDs 1..K (same logic as above)
        if "geo_cluster" in df_test.columns:
            unique_clusters = sorted(df_test["geo_cluster"].unique())
            cluster_id_map = {old: i + 1 for i, old in enumerate(unique_clusters)}
            df_test["cluster_display_id"] = df_test["geo_cluster"].map(cluster_id_map)
        else:
            df_test["cluster_display_id"] = np.nan
            cluster_id_map = {}

        # Base map (light background)
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles="CartoDB positron",
        )

        # HeatMap data: [lat, lon, normalized_error]
        heat_data = [
            [row.latitude, row.longitude, row.err_norm]
            for _, row in df_test.iterrows()
        ]

        # Gradient: low error quasi invisible, high error bien flashy
        gradient = {
            0.0: "rgba(0,0,0,0)",   # fully transparent
            0.2: "#e0f2ff",         # very light blue
            0.4: "#60a5fa",         # medium blue
            0.6: "#4f46e5",         # indigo
            0.8: "#ec4899",         # pink
            1.0: "#db2777",         # deep pink / violet
        }

        HeatMap(
            heat_data,
            radius=22,
            blur=28,
            max_zoom=14,
            min_opacity=0.0,
            gradient=gradient,
        ).add_to(m)

        # Add cluster number markers at cluster centroids
        if "cluster_display_id" in df_test.columns:
            centroids = (
                df_test.groupby("cluster_display_id")[["latitude", "longitude"]]
                .mean()
                .reset_index()
            )

            for _, row in centroids.iterrows():
                cluster_id = int(row["cluster_display_id"])
                lat = row["latitude"]
                lon = row["longitude"]

                # Circle marker as background
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=11,
                    color="black",
                    fill=True,
                    fill_opacity=0.7,
                ).add_to(m)

                # Text label with the cluster number
                folium.map.Marker(
                    [lat, lon],
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="font-size: 14px; '
                            f'font-weight: bold; color: white; '
                            f'text-align: center;">{cluster_id}</div>'
                        )
                    ),
                ).add_to(m)

        folium_path = os.path.join(FIG_DIR, "xgb_error_heatmap_sf.html")
        m.save(folium_path)
        print(f"Saved Folium error heatmap to {folium_path}")
    else:
        print(f"WARNING: {TEST_CSV_PATH} not found, skipping folium heatmap.")