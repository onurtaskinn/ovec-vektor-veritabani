"""
Utility functions for the vector database.
"""

import numpy as np
from typing import Optional


def validate_vector(vector: np.ndarray, expected_dim: Optional[int] = None) -> np.ndarray:
    """
    Validate and normalize vector input.
    
    Args:
        vector: Input vector (can be list or numpy array)
        expected_dim: Expected dimension of the vector
        
    Returns:
        Normalized numpy array
        
    Raises:
        ValueError: If vector dimensions don't match or vector is invalid
    """
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector, dtype=np.float32)
    
    if vector.ndim != 1:
        raise ValueError(f"Vector must be 1-dimensional, got shape {vector.shape}")
    
    if expected_dim is not None and len(vector) != expected_dim:
        raise ValueError(f"Vector dimension {len(vector)} doesn't match expected {expected_dim}")
    
    return vector.astype(np.float32)


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


def generate_random_vectors(n_vectors: int, dimension: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random vectors for testing.
    
    Args:
        n_vectors: Number of vectors to generate
        dimension: Dimension of each vector
        seed: Random seed for reproducibility
        
    Returns:
        Random vectors of shape (n_vectors, dimension)
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(n_vectors, dimension).astype(np.float32)

