"""
In this file, we provide small utilities to load and preprocess the S2 Airbnb dataset.
We parse SBERT embedding strings into float32 arrays, one-hot encode tabular features,
and return sklearn-ready design matrices (with embeddings), targets, and geocluster groups.
"""

# data_s3_utils.py
import numpy as np
import pandas as pd


def convert_embedding_string(s: str) -> np.ndarray:
    """
    Convert a string like "[ 10.05e-02 -2.12e-02 ...]" into a 1D float32 array.
    """
    s = str(s).strip()
    # Remove optional surrounding quotes
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    # Remove brackets
    s = s.strip().lstrip('[').rstrip(']')
    # np.fromstring handles spaces & newlines
    return np.fromstring(s.replace("\n", " "), sep=" ", dtype=np.float32)


def load_s2_with_embeddings(
    train_path: str = "../../data/train_s2.csv",
    test_path: str = "../../data/test_s2.csv",
):
    """
    Load train/test s2, parse SBERT embeddings, one-hot categoricals,
    and return numpy arrays ready for sklearn models.

    Returns
    -------
    X_train : np.ndarray
    X_test  : np.ndarray
    y_train : np.ndarray
    y_test  : np.ndarray
    groups_train : np.ndarray
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    groups_train = train_df["geo_cluster"].to_numpy()
    y_train = train_df["log_price"].to_numpy()
    y_test = test_df["log_price"].to_numpy()

    # Drop target + group from features
    X_train_raw = train_df.drop(columns=["log_price", "geo_cluster"])
    X_test_raw = test_df.drop(columns=["log_price", "geo_cluster"])

    # Parse embeddings to dense numeric matrices
    emb_train = np.stack(
        X_train_raw["embedding_sbert"].apply(convert_embedding_string).values
    )
    emb_test = np.stack(
        X_test_raw["embedding_sbert"].apply(convert_embedding_string).values
    )

    # Remove embedding column from the tabular part
    X_train_tab = X_train_raw.drop(columns=["embedding_sbert"])
    X_test_tab = X_test_raw.drop(columns=["embedding_sbert"])

    # One-hot encode categoricals only
    X_train_tab_enc = pd.get_dummies(X_train_tab, drop_first=True)
    X_test_tab_enc = pd.get_dummies(X_test_tab, drop_first=True)

    X_train_tab_enc, X_test_tab_enc = X_train_tab_enc.align(
        X_test_tab_enc, join="left", axis=1
    )
    X_test_tab_enc = X_test_tab_enc.fillna(0)

    X_train_tab_arr = X_train_tab_enc.to_numpy()
    X_test_tab_arr = X_test_tab_enc.to_numpy()

    # Final design matrices: [tabular | embedding]
    X_train = np.hstack([X_train_tab_arr, emb_train])
    X_test = np.hstack([X_test_tab_arr, emb_test])

    return X_train, X_test, y_train, y_test, groups_train
