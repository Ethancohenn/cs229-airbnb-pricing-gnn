import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import re


def convert_embedding_string(s):
    """
    Convert a string like "[ 0.01 -0.02 ...]" into a numpy array (float32).
    If it's already an array, just return it.
    """
    if isinstance(s, str):
        # Remove brackets
        s = s.strip().replace("[", "").replace("]", "")
        # Split by whitespace
        nums = re.split(r"\s+", s.strip())
        # Convert to float32
        return np.array(nums, dtype=np.float32)
    return s


def build_spatial_knn_edges(latitudes, longitudes, k=10):
    """
    Build KNN edges based on geographic distance (lat/long).

    Args:
        latitudes: 1D numpy array of shape [num_nodes]
        longitudes: 1D numpy array of shape [num_nodes]
        k: number of nearest neighbors (excluding self)

    Returns:
        edge_index: torch.LongTensor of shape [2, num_edges]
    """
    num_nodes = len(latitudes)

    # [N, 2] coordinates
    coords_deg = np.vstack([latitudes, longitudes]).T
    coords_rad = np.radians(coords_deg)

    neighbors = NearestNeighbors(
        n_neighbors=k + 1,  # +1 to include self
        algorithm="ball_tree",
        metric="haversine",
    )
    neighbors.fit(coords_rad)

    distances, indices = neighbors.kneighbors(coords_rad)

    src_list, dst_list = [], []
    for i in range(num_nodes):
        for j in indices[i]:
            if i == j:
                continue
            src_list.append(i)
            dst_list.append(j)

    # make undirected
    src = np.array(src_list + dst_list, dtype=np.int64)
    dst = np.array(dst_list + src_list, dtype=np.int64)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


def build_text_knn_edges(embeddings, k=10):
    """
    Build KNN edges based on text embeddings (semantic similarity).

    Args:
        embeddings: 2D numpy array [num_nodes, emb_dim]
        k: number of nearest neighbors (excluding self)

    Returns:
        edge_index: torch.LongTensor of shape [2, num_edges]
    """
    num_nodes = embeddings.shape[0]

    neighbors = NearestNeighbors(
        n_neighbors=k + 1,  # +1 for self
        algorithm="auto",
        metric="cosine",
    )
    neighbors.fit(embeddings)

    distances, indices = neighbors.kneighbors(embeddings)

    src_list, dst_list = [], []
    for i in range(num_nodes):
        for j in indices[i]:
            if i == j:
                continue
            src_list.append(i)
            dst_list.append(j)

    src = np.array(src_list + dst_list, dtype=np.int64)
    dst = np.array(dst_list + src_list, dtype=np.int64)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


def merge_edge_indices(edge_index_list):
    """
    Merge multiple edge_index tensors and remove duplicates.

    Args:
        edge_index_list: list of edge_index tensors, each [2, E_i]

    Returns:
        merged_edge_index: torch.LongTensor [2, E_total_unique]
    """
    merged = torch.cat(edge_index_list, dim=1)  # [2, sum(E_i)]
    merged_t = merged.t()                       # [E_total, 2]
    merged_unique = torch.unique(merged_t, dim=0)
    merged_edge_index = merged_unique.t().contiguous()
    return merged_edge_index


def build_edge_index_for_df(df_subset, k_spatial=7, k_text=7):
    """
    Convenience function:
    Given a dataframe subset with latitude, longitude, embedding_sbert,
    build a merged edge_index using spatial + text KNN.

    Returns:
        edge_index: torch.LongTensor [2, num_edges]
    """
    latitudes = df_subset["latitude"].values
    longitudes = df_subset["longitude"].values

    # ensure embeddings are arrays
    emb_list = df_subset["embedding_sbert"].apply(convert_embedding_string).values
    embeddings = np.stack(emb_list)  # [N, emb_dim]

    edge_spatial = build_spatial_knn_edges(latitudes, longitudes, k=k_spatial)
    edge_text = build_text_knn_edges(embeddings, k=k_text)
    edge_index = merge_edge_indices([edge_spatial, edge_text])
    return edge_index
