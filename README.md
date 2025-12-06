# Airbnb Price Prediction (CS229)

Team: Matthieu Hautsch, Ethan Cohen, Haotian (Owen) Cui

## TL;DR
- Predict Airbnb log_price in San Francisco.
- Compare tabular baselines across three settings: S1 (random split), S2 (geocluster-aware Cross Validation), S3 (tabular + (SBERT) description embeddings).
- Train a GraphSAGE model that links listings by spatial and semantic proximity.
- Best hold-out performer: XGBoost on S3 (tabular + SBERT), evaluated on the test set with saved plots/heatmaps.
- Evaluate with MAE/RMSE.

## Repository layout
```             
|- docs/, notebooks/           # proposal, EDA, geospatial HTML map
|- reports/
|  |- figures/                 # saved plots/heatmaps from best models
|- src/
|  |- data/                    # raw + processed CSVs (train_s1/s2, test_s1/s2, listings.csv, san_francisco_listings_details.csv)
|  |- features/                # feature builders (placeholders for now)
|  |- gnn/                     # graph construction + GraphSAGE training/plots
|  |- models/
|     |- catboost_info/        # CatBoost training logs from the S1 sweep
|     |- s1_models/            # tabular baselines on random split
|     |- s2_models/            # tabular baselines with geocluster-aware CV
|     |- s3_models/            # tabular + SBERT embedding baselines
|     |- model_comparison_s1.py  # wide baseline sweep on S1
```

## Data
- Source: InsideAirbnb listings for San Francisco (`listings.csv` stored in `data/`).
- Derived splits: `train_s1.csv`, `test_s1.csv`, `train_s2.csv`, `test_s2.csv` live in `src/data/`.
- Full raw scrape (rich text + amenities) for EDA/maps: `san_francisco_listings_details.csv` in `src/data/`.
- S1 uses a standard random train/test split.
- S2 groups by `geo_cluster` for GroupKFold evaluation and a geocluster-based train/test split.
- S3 augments S2 with SBERT text embeddings stored as strings in the `embedding_sbert` column.

## Environment setup
```bash
conda env create -f environment.yml
conda activate cs229-airbnb
```

## How to run experiments
All scripts print MAE/RMSE to stdout and some plot figures. Paths are relative to `src/models/` (run from repo root).

### S1 tabular baselines (random split)
- Quick baselines: `python src/models/s1_models/linear_regression_s1.py`, `.../Decision_trees_s1.py`, `.../Random_forest_s1.py`, `.../knn_s1.py`, `.../XGBoost_s1.py`.
- Wide sweep: `python src/models/model_comparison_s1.py` (10-fold CV across linear, tree, ensemble, KNN, SVR, MLP, CatBoost, XGBoost).

### S2 geocluster-aware tabular baselines
- Use `geo_cluster` with GroupKFold to avoid spatial leakage.
- Scripts: `python src/models/s2_models/linear_regression_s2.py`, `.../Decision_trees_s2.py`, `.../Random_forest_s2.py`, `.../knn_s2.py`, `.../XGBoost_s2.py`, `.../neural_network_s2.py`.
- Quick sanity check baseline: `python src/models/s2_models/baseline_knn_s2.py`.
- Optuna tunes key hyperparameters (alphas, depth, neighbors, learning rate) and outer CV reports unbiased MAE/RMSE.

### S3 tabular + SBERT embedding baselines
- `src/models/s3_models/data_s3_utils.py` parses SBERT strings to float arrays, one-hot encodes categoricals, and aligns train/test schemas.
- Scripts mirror S2 but operate on `[tabular | embedding]` features: `linear_regression_s3.py`, `Decision_trees_s3.py`, `Random_forest_s3.py`, `knn_s3.py`, `baseline_knn_s3.py`, `XGBoost_s3.py`, `neural_network_s3.py`.
- Best XGBoost (primary test-set model) + plots (scatter + Folium heatmap saved to `reports/figures/`): `python src/models/s3_models/xgb_best_model_plots.py`.
- CV remains GroupKFold on `geo_cluster`.

### Graph Neural Network
- Graph construction: `src/gnn/graph_builder.py` builds spatial (haversine) and text KNN edges and merges them.
- Model: `src/gnn/graphsage_model.py` implements a 2-layer GraphSAGE.
- Run 5-fold geocluster CV with different `k_spatial`/`k_text` values via `python src/gnn/graphsage_model.py`. Training happens on training subgraphs and evaluation is inductive on held-out geocluster subgraphs.
- Best GraphSAGE config with plots (scatter + Folium heatmap saved to `reports/figures/`): `python src/gnn/graphsage_best_model_plots.py`.
