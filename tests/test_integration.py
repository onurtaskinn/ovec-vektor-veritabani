"""
Integration tests for the complete VectorDB system.
"""

import numpy as np
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_db import VectorDB


def test_basic_operations():
    """Test basic CRUD operations."""
    print("Testing basic CRUD operations...")
    
    db = VectorDB(dimension=128, metric="cosine")
    
    # Test insert
    vec1 = np.random.randn(128).astype(np.float32)
    db.insert(id="vec1", vector=vec1, metadata={"text": "hello"})
    
    assert len(db) == 1
    assert "vec1" in db
    
    # Test get
    result = db.get("vec1")
    assert result['id'] == "vec1"
    assert np.allclose(result['vector'], vec1)
    assert result['metadata']['text'] == "hello"
    
    # Test update
    vec1_new = np.random.randn(128).astype(np.float32)
    db.update("vec1", vector=vec1_new, metadata={"text": "world"})
    
    result = db.get("vec1")
    assert np.allclose(result['vector'], vec1_new)
    assert result['metadata']['text'] == "world"
    
    # Test delete
    db.delete("vec1")
    assert len(db) == 0
    assert "vec1" not in db
    
    print("✓ Basic CRUD operations passed")


def test_batch_insert():
    """Test batch insertion."""
    print("Testing batch insert...")
    
    db = VectorDB(dimension=64, metric="cosine")
    
    n_vectors = 100
    vectors = np.random.randn(n_vectors, 64).astype(np.float32)
    ids = [f"vec_{i}" for i in range(n_vectors)]
    metadata = [{"index": i} for i in range(n_vectors)]
    
    db.batch_insert(ids, vectors, metadata)
    
    assert len(db) == n_vectors
    
    # Check random samples
    for i in [0, 50, 99]:
        result = db.get(f"vec_{i}")
        assert np.allclose(result['vector'], vectors[i])
        assert result['metadata']['index'] == i
    
    print("✓ Batch insert tests passed")


def test_brute_force_search():
    """Test brute-force search with different metrics."""
    print("Testing brute-force search...")
    
    # Test with cosine similarity
    db = VectorDB(dimension=32, metric="cosine")
    
    n_vectors = 100
    vectors = np.random.randn(n_vectors, 32).astype(np.float32)
    ids = [f"vec_{i}" for i in range(n_vectors)]
    
    db.batch_insert(ids, vectors)
    
    # Search with first vector
    query = vectors[0]
    results = db.search(query, top_k=10)
    
    assert len(results) == 10
    assert results[0]['id'] == 'vec_0'  # Should find itself
    assert 'score' in results[0]
    assert 'metadata' in results[0]
    
    # Check scores are sorted
    scores = [r['score'] for r in results]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    
    # Test with euclidean distance
    db_l2 = VectorDB(dimension=32, metric="euclidean")
    db_l2.batch_insert(ids, vectors)
    results_l2 = db_l2.search(query, top_k=10)
    
    assert len(results_l2) == 10
    assert results_l2[0]['id'] == 'vec_0'
    
    # Test with dot product
    db_dot = VectorDB(dimension=32, metric="dot_product")
    db_dot.batch_insert(ids, vectors)
    results_dot = db_dot.search(query, top_k=10)
    
    assert len(results_dot) == 10
    
    print("✓ Brute-force search tests passed")


def test_ivf_search():
    """Test IVF indexed search."""
    print("Testing IVF indexed search...")
    
    db = VectorDB(dimension=64, metric="cosine", index_type="ivf", n_clusters=20, nprobe=5)
    
    n_vectors = 500
    vectors = np.random.randn(n_vectors, 64).astype(np.float32)
    ids = [f"vec_{i}" for i in range(n_vectors)]
    
    db.batch_insert(ids, vectors)
    
    # Build index
    db.build_index()
    
    # Search
    query = vectors[0]
    results = db.search(query, top_k=10)
    
    assert len(results) == 10
    assert 'score' in results[0]
    
    # The query vector should be in top results (may not be #1 due to approximation)
    result_ids = [r['id'] for r in results]
    assert 'vec_0' in result_ids
    
    print("✓ IVF indexed search tests passed")


def test_persistence():
    """Test save and load functionality."""
    print("Testing save/load...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create and populate database
        db = VectorDB(dimension=32, metric="cosine", index_type="ivf", n_clusters=10)
        
        n_vectors = 100
        vectors = np.random.randn(n_vectors, 32).astype(np.float32)
        ids = [f"vec_{i}" for i in range(n_vectors)]
        metadata = [{"index": i, "text": f"vector {i}"} for i in range(n_vectors)]
        
        db.batch_insert(ids, vectors, metadata)
        db.build_index()
        
        # Save
        save_path = os.path.join(temp_dir, "test_db")
        db.save(save_path)
        
        # Check files were created
        assert os.path.exists(os.path.join(save_path, "config.json"))
        assert os.path.exists(os.path.join(save_path, "vectors.npz"))
        assert os.path.exists(os.path.join(save_path, "metadata.json"))
        assert os.path.exists(os.path.join(save_path, "ivf_index.npz"))
        
        # Load
        db_loaded = VectorDB.load(save_path)
        
        # Check loaded database
        assert len(db_loaded) == n_vectors
        assert db_loaded.dimension == 32
        assert db_loaded.metric == "cosine"
        assert db_loaded.index_type == "ivf"
        
        # Check vectors and metadata
        for i in [0, 50, 99]:
            result = db_loaded.get(f"vec_{i}")
            assert np.allclose(result['vector'], vectors[i])
            assert result['metadata']['index'] == i
            assert result['metadata']['text'] == f"vector {i}"
        
        # Check search works on loaded database
        query = vectors[0]
        results = db_loaded.search(query, top_k=5)
        assert len(results) == 5
        
        print("✓ Save/load tests passed")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases...")
    
    db = VectorDB(dimension=10)
    
    # Test empty database search
    results = db.search(np.random.randn(10), top_k=5)
    assert len(results) == 0
    
    # Test duplicate ID insertion
    vec = np.random.randn(10).astype(np.float32)
    db.insert("dup", vec)
    
    try:
        db.insert("dup", vec)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test invalid dimension
    try:
        db.insert("bad", np.random.randn(20))  # Wrong dimension
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test get non-existent ID
    try:
        db.get("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test delete non-existent ID
    try:
        db.delete("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    print("✓ Edge case tests passed")


def benchmark_search_speed():
    """Benchmark search speed: brute-force vs IVF."""
    print("\nBenchmarking search speed...")
    
    import time
    
    n_vectors = 10000
    dimension = 128
    
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    ids = [f"vec_{i}" for i in range(n_vectors)]
    
    # Brute-force database
    db_bf = VectorDB(dimension=dimension, metric="cosine")
    db_bf.batch_insert(ids, vectors)
    
    # IVF database
    db_ivf = VectorDB(dimension=dimension, metric="cosine", index_type="ivf", n_clusters=100, nprobe=10)
    db_ivf.batch_insert(ids, vectors)
    
    # Time index building
    start = time.time()
    db_ivf.build_index()
    index_time = time.time() - start
    
    # Create test queries
    n_queries = 100
    queries = np.random.randn(n_queries, dimension).astype(np.float32)
    
    # Benchmark brute-force
    start = time.time()
    for query in queries:
        db_bf.search(query, top_k=10)
    bf_time = time.time() - start
    
    # Benchmark IVF
    start = time.time()
    for query in queries:
        db_ivf.search(query, top_k=10)
    ivf_time = time.time() - start
    
    print(f"\n📊 Benchmark Results ({n_vectors} vectors, {n_queries} queries):")
    print(f"  Index build time: {index_time:.3f}s")
    print(f"  Brute-force:      {bf_time:.3f}s ({bf_time/n_queries*1000:.2f}ms/query)")
    print(f"  IVF:              {ivf_time:.3f}s ({ivf_time/n_queries*1000:.2f}ms/query)")
    print(f"  Speedup:          {bf_time/ivf_time:.2f}x")


if __name__ == "__main__":
    test_basic_operations()
    test_batch_insert()
    test_brute_force_search()
    test_ivf_search()
    test_persistence()
    test_edge_cases()
    benchmark_search_speed()
    print("\n✅ All integration tests passed!")

