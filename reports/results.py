import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# Data (Linear Reg + Random Forest removed)
# -----------------------------
models = ["KNN", "LASSO", "XGBoost", "Neural Net", "GNN"]

# MAE (S1, S2, S3)
mae_S1 = [0.345, 0.356, 0.271, 0.320, np.nan]
mae_S2 = [0.437, 0.412, 0.353, 0.426, np.nan]
mae_S3 = [0.514, 0.389, 0.341, 0.510, 0.491]

# -----------------------------
# Sorting by S3 MAE
# -----------------------------
order = np.argsort(mae_S3)

models_sorted = [models[i] for i in order]
mae_S1 = [mae_S1[i] for i in order]
mae_S2 = [mae_S2[i] for i in order]
mae_S3 = [mae_S3[i] for i in order]

# -----------------------------
# Plot layout (single MAE panel)
# -----------------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

x = np.arange(len(models_sorted))
h = 0.22

# grayscale
g1 = "0.75"   # S1
g2 = "0.45"   # S2
g3 = "0.15"   # S3

# -----------------------------
# MAE Panel
# -----------------------------
ax.barh(x - h, mae_S1, h, color=g1, label="S1")
ax.barh(x,     mae_S2, h, color=g2, label="S2")
ax.barh(x + h, mae_S3, h, color=g3, label="S3")

# Title and labels (English)
ax.set_title("MAE — Model Comparison (S1, S2, S3)", fontsize=15)
ax.set_xlabel("MAE", fontsize=13)
ax.set_yticks(x)
ax.set_yticklabels(models_sorted, fontsize=11)
ax.grid(axis="x", linestyle="--", alpha=0.3)

# Add MAE value labels next to each bar in the MAE panel
for i in range(len(models_sorted)):
    # S1 (left bar)
    val = mae_S1[i]
    if not np.isnan(val):
        ax.text(val + 0.005, x[i] - h, f"{val:.3f}", va='center', fontsize=10)
    # S2 (center bar)
    val = mae_S2[i]
    if not np.isnan(val):
        ax.text(val + 0.005, x[i], f"{val:.3f}", va='center', fontsize=10)
    # S3 (right bar)
    val = mae_S3[i]
    if not np.isnan(val):
        ax.text(val + 0.005, x[i] + h, f"{val:.3f}", va='center', fontsize=10)

# Legend: slightly reduce bottom margin and bring legend up closer to plot
# (closer to MAE axis but still below it)
fig.subplots_adjust(bottom=0.18)
fig.legend(["S1 (Random)", "S2 (Geo)", "S3 (SBERT + Geo)"],
           loc="lower center", ncol=3, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.06))

# Save figure to `reports/figures/results.png` (create folder if missing)
out_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "results.png")
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to: {out_path}")

# Use tight_layout but leave room at bottom for the external legend
plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.show()
