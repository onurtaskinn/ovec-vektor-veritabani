"""
Unit tests for distance metrics.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_db.distance import (
    cosine_similarity, 
    euclidean_distance, 
    dot_product,
    compute_distances,
    get_top_k_indices
)


def test_cosine_similarity():
    """Test cosine similarity computation."""
    print("Testing cosine similarity...")
    
    # Test identical vectors (should be 1.0)
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([[1.0, 0.0, 0.0]])
    sim = cosine_similarity(v1, v2)
    assert abs(sim[0] - 1.0) < 1e-6, f"Expected 1.0, got {sim[0]}"
    
    # Test orthogonal vectors (should be 0.0)
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([[0.0, 1.0, 0.0]])
    sim = cosine_similarity(v1, v2)
    assert abs(sim[0]) < 1e-6, f"Expected 0.0, got {sim[0]}"
    
    # Test opposite vectors (should be -1.0)
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([[-1.0, 0.0, 0.0]])
    sim = cosine_similarity(v1, v2)
    assert abs(sim[0] + 1.0) < 1e-6, f"Expected -1.0, got {sim[0]}"
    
    # Test multiple vectors
    query = np.array([1.0, 1.0, 0.0])
    vectors = np.array([
        [1.0, 1.0, 0.0],  # Same
        [1.0, 0.0, 0.0],  # 45 degrees
        [0.0, 1.0, 0.0],  # 45 degrees
        [-1.0, -1.0, 0.0] # Opposite
    ])
    sims = cosine_similarity(query, vectors)
    assert sims[0] > sims[1], "Same vector should be most similar"
    assert abs(sims[1] - sims[2]) < 1e-6, "45 degree vectors should be equally similar"
    assert sims[3] < 0, "Opposite vector should have negative similarity"
    
    print("✓ Cosine similarity tests passed")


def test_euclidean_distance():
    """Test Euclidean distance computation."""
    print("Testing Euclidean distance...")
    
    # Test identical vectors (distance should be 0)
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([[1.0, 2.0, 3.0]])
    dist = euclidean_distance(v1, v2)
    assert abs(dist[0]) < 1e-6, f"Expected 0.0, got {dist[0]}"
    
    # Test known distance
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([[3.0, 4.0, 0.0]])
    dist = euclidean_distance(v1, v2)
    expected = -5.0  # sqrt(3^2 + 4^2) = 5, negative because we return negative distances
    assert abs(dist[0] - expected) < 1e-6, f"Expected {expected}, got {dist[0]}"
    
    # Test multiple vectors
    query = np.array([0.0, 0.0])
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [3.0, 4.0],
        [0.0, 0.0]
    ])
    dists = euclidean_distance(query, vectors)
    # Closer vectors (smaller distance) should have higher values (less negative)
    assert dists[3] > dists[0], "Identical vector should have highest score"
    assert dists[0] > dists[2], "Closer vector should have higher score"
    
    print("✓ Euclidean distance tests passed")


def test_dot_product():
    """Test dot product computation."""
    print("Testing dot product...")
    
    # Test known dot product
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([[1.0, 0.0, 0.0]])
    dp = dot_product(v1, v2)
    assert abs(dp[0] - 1.0) < 1e-6, f"Expected 1.0, got {dp[0]}"
    
    # Test with multiple vectors
    query = np.array([1.0, 2.0, 3.0])
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 2.0, 3.0]
    ])
    dps = dot_product(query, vectors)
    assert abs(dps[0] - 1.0) < 1e-6
    assert abs(dps[1] - 2.0) < 1e-6
    assert abs(dps[2] - 3.0) < 1e-6
    assert abs(dps[3] - 14.0) < 1e-6  # 1*1 + 2*2 + 3*3 = 14
    
    print("✓ Dot product tests passed")


def test_compute_distances():
    """Test the unified distance computation function."""
    print("Testing compute_distances...")
    
    query = np.array([1.0, 0.0, 0.0])
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    
    # Test cosine
    scores = compute_distances(query, vectors, "cosine")
    assert scores[0] > scores[1], "Cosine: identical vector should have higher score"
    
    # Test euclidean
    scores = compute_distances(query, vectors, "euclidean")
    assert scores[0] > scores[1], "Euclidean: closer vector should have higher score"
    
    # Test dot product
    scores = compute_distances(query, vectors, "dot_product")
    assert scores[0] > scores[1], "Dot product: aligned vector should have higher score"
    
    print("✓ compute_distances tests passed")


def test_get_top_k_indices():
    """Test top-k index retrieval."""
    print("Testing get_top_k_indices...")
    
    scores = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
    
    # Test k=1
    indices = get_top_k_indices(scores, 1)
    assert len(indices) == 1
    assert indices[0] == 3, f"Expected index 3, got {indices[0]}"
    
    # Test k=3
    indices = get_top_k_indices(scores, 3)
    assert len(indices) == 3
    assert list(indices) == [3, 1, 2], f"Expected [3, 1, 2], got {list(indices)}"
    
    # Test k > len(scores)
    indices = get_top_k_indices(scores, 10)
    assert len(indices) == len(scores)
    
    print("✓ get_top_k_indices tests passed")


if __name__ == "__main__":
    test_cosine_similarity()
    test_euclidean_distance()
    test_dot_product()
    test_compute_distances()
    test_get_top_k_indices()
    print("\n✅ All distance metric tests passed!")

