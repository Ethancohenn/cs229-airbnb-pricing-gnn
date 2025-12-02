import numpy as np
import pandas as pd
import torch
import ast
import re
from sklearn.neighbors import NearestNeighbors

def build_spatial_knn_edges(latitudes, longitudes, k=10):
    """
    Build KNN edges based on geographic distance (latlatitudes/longitudes)

    Args:
        latitudes: 1D numpy array of shape [num_nodes]
        longitudes: 1D numpy array of shape [num_nodes]
        k: number of nearest neighbors (excluding self) !!! HYPERPATAMETER !!!

    Returns:
        edge_index: torch.LongTensor of shape [2, num_edges]
    """
    num_nodes = len(latitudes)

    # Stack lat/lon into a [num_nodes, 2] array
    coords_deg = np.vstack([latitudes, longitudes]).T  # shape: [N, 2]

    # Convert degrees to radians for haversine
    coords_rad = np.radians(coords_deg)

    # Use sklearn NearestNeighbors with haversine metric
    neighbors = NearestNeighbors(n_neighbors=k+1,  # +1 to include self, we'll drop it
                            algorithm='ball_tree',
                            metric='haversine')
    neighbors.fit(coords_rad)

    # distances: [N, k+1], indices: [N, k+1]
    distances, indices = neighbors.kneighbors(coords_rad)

    # Build edge list (i -> j for each neighbor j != i)
    src_list = []
    dst_list = []

    for i in range(num_nodes):
        for j in indices[i]:
            if i == j:
                continue  # skip self-loop
            src_list.append(i)
            dst_list.append(j)

    # Make it undirected by adding reverse edges
    src = np.array(src_list + dst_list)
    dst = np.array(dst_list + src_list)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index

def build_text_knn_edges(embeddings, k=10):
    """
    Build KNN edges based on text embeddings (semantic similarity)

    Args:
        embeddings: 2D numpy array [num_nodes, emb_dim]
        k: number of nearest neighbors (excluding self) !!! HYPERPATAMETER !!!

    Returns:
        edge_index: torch.LongTensor of shape [2, num_edges]
    """
    num_nodes = embeddings.shape[0]

    # We use cosine distance: distance = 1 - cosine_similarity
    neighbors = NearestNeighbors(
        n_neighbors=k+1,    # +1 for self
        algorithm='auto',
        metric='cosine'     # cosine distance
    )
    neighbors.fit(embeddings)

    distances, indices = neighbors.kneighbors(embeddings)

    src_list = []
    dst_list = []

    for i in range(num_nodes):
        for j in indices[i]:
            if i == j:
                continue  # skip self
            src_list.append(i)
            dst_list.append(j)

    # Make it undirected
    src = np.array(src_list + dst_list)
    dst = np.array(dst_list + src_list)

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
    # Concatenate along columns: [2, E_total]
    merged = torch.cat(edge_index_list, dim=1)  # shape [2, sum(E_i)]

    # Transpose to [E_total, 2] to dedupe rows
    merged_t = merged.t()  # [E_total, 2]

    # Remove duplicate edges
    merged_unique = torch.unique(merged_t, dim=0)

    # Transpose back to [2, E_total_unique]
    merged_edge_index = merged_unique.t().contiguous()

    return merged_edge_index

# Example: assume you have a DataFrame `df` with lat/lon and a numpy embedding matrix

# lat / lon as numpy arrays
df = pd.read_csv("../data/train_s2.csv")
latitudes = df["latitude"].values
longitudes = df["longitude"].values

# embeddings as numpy array [num_nodes, emb_dim] (if ready)
# If not ready yet, you can skip this part for now.
def convert_to_array(s):
    if isinstance(s, str):
        # Remove brackets
        s = s.strip().replace("[", "").replace("]", "")
        # Split by whitespace
        nums = re.split(r"\s+", s.strip())
        # Convert to float
        return np.array(nums, dtype=float)
    return s

df["embedding_sbert"] = df["embedding_sbert"].apply(convert_to_array)
embeddings = np.stack(df["embedding_sbert"].values)  # depends on how your friend stores it

# --- Build edges ---

# 1) Spatial KNN edges
edge_index_spatial = build_spatial_knn_edges(latitudes, longitudes, k=10)

# 2) Text KNN edges (optional at first)
edge_index_text = build_text_knn_edges(embeddings, k=10)

# 3) Merge the two
edge_index = merge_edge_indices([edge_index_spatial, edge_index_text])

print(edge_index.shape)  # [2, num_edges]