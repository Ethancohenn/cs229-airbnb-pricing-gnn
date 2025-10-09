# Evaluating Graph-Based and Standard ML for Airbnb Pricing

**CS229 – Finance & Commerce**  
Team: Matthieu Hautsch · Ethan Cohen · Haotian (Owen) Cui

This project predicts Airbnb listing prices using tabular ML baselines (Linear/Ridge/Lasso, RF, CatBoost, kNN)
and a spatial **Graph Neural Network** that models listings as nodes with proximity edges.

## Repo layout
```
.
├── data/                  # (gitignored) raw & processed data
├── docs/                  # papers, proposal, notes
├── notebooks/             # EDA & experiments
├── reports/               # figures & final report assets
├── src/                   # package-style code
│   ├── data/              # data loading/cleaning
│   ├── features/          # feature engineering
│   ├── gnn/               # graph building & GNN models
│   └── models/            # baseline models & training loops
├── requirements.txt
├── environment.yml
├── .pre-commit-config.yaml
└── .github/workflows/ci.yml
```

## Getting started
1. Create env:
   ```bash
   conda env create -f environment.yml
   conda activate cs229-airbnb
   # or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   ```
2. Install pre-commit hooks (recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```
3. Put `listings.csv` from InsideAirbnb into `data/raw/` (not tracked by git).
4. Open `notebooks/01_data_preprocessing.ipynb` and start EDA.

## Notes on PyTorch Geometric
Installing `torch`/`torch_geometric` often depends on your CUDA version. We keep them **out** of
`requirements.txt`. Please follow the official instructions for your platform and then install
`torch-geometric` via:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # or cuda
pip install torch-geometric
```

## Project links
- Santorini paper: *Predicting prices of Airbnb listings via Graph Neural Networks and Document Embeddings.*
- Data: Inside Airbnb

---
> Use branches: `feature/*`, `data/*`, `model/*`. Open PRs to `main`. CI runs ruff/black/pytest.
