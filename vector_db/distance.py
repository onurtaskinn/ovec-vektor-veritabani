"""
Distance metrics for vector similarity computation.
All functions are implemented from scratch using only NumPy.
"""

import numpy as np
from typing import Literal


DistanceMetric = Literal["cosine", "euclidean", "dot_product"]


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    L2 normalize vectors for cosine similarity computation.
    
    Args:
        vectors: Array of shape (n_vectors, dimension)
        
    Returns:
        Normalized vectors
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


def cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and multiple vectors.
    
    Cosine similarity = (A · B) / (||A|| * ||B||)
    Range: [-1, 1], where 1 means identical direction
    
    Args:
        query: Query vector of shape (dimension,)
        vectors: Database vectors of shape (n_vectors, dimension)
        
    Returns:
        Similarities of shape (n_vectors,), sorted in descending order
    """
    # Normalize query
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(len(vectors), dtype=np.float32)
    query_normalized = query / query_norm
    
    # Normalize vectors
    vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    vector_norms = np.where(vector_norms == 0, 1, vector_norms)
    vectors_normalized = vectors / vector_norms
    
    # Dot product of normalized vectors
    similarities = np.dot(vectors_normalized, query_normalized)
    
    return similarities.astype(np.float32)


def euclidean_distance(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean (L2) distance between query and multiple vectors.
    
    L2 distance = sqrt(sum((A - B)^2))
    Range: [0, inf], where 0 means identical vectors
    
    Note: Returns negative distances so that higher values = more similar
    (consistent with other metrics for top-k retrieval)
    
    Args:
        query: Query vector of shape (dimension,)
        vectors: Database vectors of shape (n_vectors, dimension)
        
    Returns:
        Negative distances of shape (n_vectors,)
    """
    # Compute squared differences
    diff = vectors - query
    squared_diff = diff ** 2
    squared_distances = np.sum(squared_diff, axis=1)
    
    # Take square root to get actual Euclidean distance
    distances = np.sqrt(squared_distances)
    
    # Return negative so higher values = more similar
    return -distances.astype(np.float32)


def dot_product(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute dot product between query and multiple vectors.
    
    Dot product = sum(A * B)
    Range: [-inf, inf], higher values = more similar
    
    Note: Useful when vectors are already normalized or when 
    magnitude matters (e.g., for Maximum Inner Product Search - MIPS)
    
    Args:
        query: Query vector of shape (dimension,)
        vectors: Database vectors of shape (n_vectors, dimension)
        
    Returns:
        Dot products of shape (n_vectors,)
    """
    return np.dot(vectors, query).astype(np.float32)


def compute_distances(
    query: np.ndarray, 
    vectors: np.ndarray, 
    metric: DistanceMetric = "cosine"
) -> np.ndarray:
    """
    Compute distances/similarities using the specified metric.
    
    Args:
        query: Query vector of shape (dimension,)
        vectors: Database vectors of shape (n_vectors, dimension)
        metric: Distance metric to use
        
    Returns:
        Distances/similarities where higher values = more similar
        
    Raises:
        ValueError: If metric is not supported
    """
    if metric == "cosine":
        return cosine_similarity(query, vectors)
    elif metric == "euclidean":
        return euclidean_distance(query, vectors)
    elif metric == "dot_product":
        return dot_product(query, vectors)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'cosine', 'euclidean', or 'dot_product'")


def get_top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Get indices of top-k highest scores efficiently.
    
    Args:
        scores: Array of scores
        k: Number of top results to return
        
    Returns:
        Indices of top-k scores in descending order
    """
    if k >= len(scores):
        return np.argsort(scores)[::-1]
    
    # Use argpartition for efficiency when k << n
    # This is O(n) instead of O(n log n) for full sort
    indices = np.argpartition(scores, -k)[-k:]
    
    # Sort the top-k results
    indices = indices[np.argsort(scores[indices])[::-1]]
    
    return indices
