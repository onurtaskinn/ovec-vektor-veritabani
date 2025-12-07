"""
Core VectorDB implementation.
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
