"""
Unit tests for IVF indexing.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_db.index import KMeans, IVFIndex


def test_kmeans_basic():
    """Test basic k-means clustering."""
    print("Testing K-Means clustering...")
    
    # Create simple clustered data
    np.random.seed(42)
    cluster1 = np.random.randn(50, 2) + np.array([0, 0])
    cluster2 = np.random.randn(50, 2) + np.array([5, 5])
    cluster3 = np.random.randn(50, 2) + np.array([10, 0])
    
    data = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)
    
    # Fit k-means
    kmeans = KMeans(n_clusters=3, seed=42)
    kmeans.fit(data)
    
    # Check that we got 3 clusters
    assert kmeans.n_clusters == 3
    assert kmeans.centroids.shape == (3, 2)
    assert len(kmeans.labels) == len(data)
    
    # Check that labels are in valid range
    assert np.all(kmeans.labels >= 0)
    assert np.all(kmeans.labels < 3)
    
    # Check that each cluster has some points
    for i in range(3):
        assert np.sum(kmeans.labels == i) > 0, f"Cluster {i} has no points"
    
    print("✓ K-Means basic tests passed")


def test_kmeans_predict():
    """Test k-means prediction on new data."""
    print("Testing K-Means prediction...")
    
    np.random.seed(42)
    data = np.random.randn(100, 5).astype(np.float32)
    
    kmeans = KMeans(n_clusters=10, seed=42)
    kmeans.fit(data)
    
    # Predict on new data
    new_data = np.random.randn(20, 5).astype(np.float32)
    labels = kmeans.predict(new_data)
    
    assert len(labels) == 20
    assert np.all(labels >= 0)
    assert np.all(labels < 10)
    
    print("✓ K-Means prediction tests passed")


def test_ivf_index_build():
    """Test IVF index building."""
    print("Testing IVF index build...")
    
    np.random.seed(42)
    n_vectors = 1000
    dimension = 128
    
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    ids = [f"vec_{i}" for i in range(n_vectors)]
    metadata = {id: {'index': i} for i, id in enumerate(ids)}
    
    # Build index
    index = IVFIndex(
        vectors=vectors,
        ids=ids,
        metadata=metadata,
        metric="cosine",
        n_clusters=50,
        nprobe=5
    )
    
    index.build()
    
    # Check that index is built
    assert index._is_built
    assert index.kmeans is not None
    assert len(index.inverted_lists) > 0
    
    # Check stats
    stats = index.get_stats()
    assert stats['built'] == True
    assert stats['n_vectors'] == n_vectors
    assert stats['n_clusters'] <= 50
    
    print("✓ IVF index build tests passed")


def test_ivf_index_search():
    """Test IVF index search."""
    print("Testing IVF index search...")
    
    np.random.seed(42)
    n_vectors = 500
    dimension = 64
    
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    ids = [f"vec_{i}" for i in range(n_vectors)]
    metadata = {}
    
    # Build index
    index = IVFIndex(
        vectors=vectors,
        ids=ids,
        metadata=metadata,
        metric="cosine",
        n_clusters=25,
        nprobe=5
    )
    
    index.build()
    
    # Search with a query
    query = vectors[0]  # Use first vector as query
    results = index.search(query, top_k=10)
    
    # Check results
    assert len(results) <= 10
    assert len(results) > 0
    
    # First result should be the query vector itself
    assert results[0]['id'] == 'vec_0'
    
    # Check that scores are in descending order
    scores = [r['score'] for r in results]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    
    print("✓ IVF index search tests passed")


def test_ivf_accuracy():
    """Test IVF index accuracy vs brute force."""
    print("Testing IVF index accuracy...")
    
    np.random.seed(42)
    n_vectors = 1000
    dimension = 32
    
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    ids = [f"vec_{i}" for i in range(n_vectors)]
    metadata = {}
    
    # Build IVF index
    index = IVFIndex(
        vectors=vectors,
        ids=ids,
        metadata=metadata,
        metric="cosine",
        n_clusters=50,
        nprobe=10  # Search more clusters for better accuracy
    )
    index.build()
    
    # Create a query
    query = np.random.randn(dimension).astype(np.float32)
    
    # Get IVF results
    ivf_results = index.search(query, top_k=20)
    ivf_ids = set(r['id'] for r in ivf_results)
    
    # Get brute force results
    from vector_db.distance import compute_distances, get_top_k_indices
    scores = compute_distances(query, vectors, "cosine")
    top_indices = get_top_k_indices(scores, 20)
    bf_ids = set(ids[i] for i in top_indices)
    
    # Calculate recall (how many of the true top-20 we found)
    recall = len(ivf_ids & bf_ids) / len(bf_ids)
    
    # With nprobe=10, we should get reasonable recall
    assert recall >= 0.5, f"Recall too low: {recall}"
    
    print(f"✓ IVF accuracy tests passed (recall@20: {recall:.2%})")


if __name__ == "__main__":
    test_kmeans_basic()
    test_kmeans_predict()
    test_ivf_index_build()
    test_ivf_index_search()
    test_ivf_accuracy()
    print("\n✅ All IVF index tests passed!")
