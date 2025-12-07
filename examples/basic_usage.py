"""
Basic usage examples for the vector database.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_db import VectorDB


def example_1_basic_usage():
    """Example 1: Basic insert and search."""
    print("=" * 60)
    print("Example 1: Basic Insert and Search")
    print("=" * 60)
    
    # Create a vector database
    db = VectorDB(dimension=128, metric="cosine")
    
    # Insert some vectors
    for i in range(10):
        vector = np.random.randn(128).astype(np.float32)
        db.insert(
            id=f"vector_{i}",
            vector=vector,
            metadata={"description": f"This is vector {i}"}
        )
    
    print(f"✓ Inserted {len(db)} vectors")
    
    # Search for similar vectors
    query = np.random.randn(128).astype(np.float32)
    results = db.search(query, top_k=5)
    
    print(f"\n🔍 Top 5 search results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['id']} (score: {result['score']:.4f})")
        print(f"     Metadata: {result['metadata']}")
    
    print()


def example_2_batch_operations():
    """Example 2: Batch insert for efficiency."""
    print("=" * 60)
    print("Example 2: Batch Insert")
    print("=" * 60)
    
    db = VectorDB(dimension=64, metric="cosine")
    
    # Generate many vectors at once
    n_vectors = 1000
    vectors = np.random.randn(n_vectors, 64).astype(np.float32)
    ids = [f"doc_{i}" for i in range(n_vectors)]
    metadata = [{"document": f"Document {i}", "category": i % 5} for i in range(n_vectors)]
    
    # Batch insert (much faster than individual inserts)
    db.batch_insert(ids, vectors, metadata)
    
    print(f"✓ Batch inserted {len(db)} vectors")
    
    # Search
    query = vectors[42]  # Use a known vector as query
    results = db.search(query, top_k=3)
    
    print(f"\n🔍 Searching for similar documents:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['id']} (score: {result['score']:.4f})")
        print(f"     Category: {result['metadata']['category']}")
    
    print()


def example_4_different_metrics():
    """Example 4: Different distance metrics."""
    print("=" * 60)
    print("Example 4: Different Distance Metrics")
    print("=" * 60)
    
    dimension = 32
    vectors = np.random.randn(100, dimension).astype(np.float32)
    ids = [f"vec_{i}" for i in range(100)]
    query = vectors[0]  # Use first vector as query
    
    # Cosine similarity
    db_cosine = VectorDB(dimension=dimension, metric="cosine")
    db_cosine.batch_insert(ids, vectors)
    results_cosine = db_cosine.search(query, top_k=5)
    
    # Euclidean distance
    db_euclidean = VectorDB(dimension=dimension, metric="euclidean")
    db_euclidean.batch_insert(ids, vectors)
    results_euclidean = db_euclidean.search(query, top_k=5)
    
    # Dot product
    db_dot = VectorDB(dimension=dimension, metric="dot_product")
    db_dot.batch_insert(ids, vectors)
    results_dot = db_dot.search(query, top_k=5)
    
    print("Top 5 results with different metrics:\n")
    
    print("Cosine Similarity:")
    for r in results_cosine[:3]:
        print(f"  {r['id']}: {r['score']:.4f}")
    
    print("\nEuclidean Distance:")
    for r in results_euclidean[:3]:
        print(f"  {r['id']}: {r['score']:.4f}")
    
    print("\nDot Product:")
    for r in results_dot[:3]:
        print(f"  {r['id']}: {r['score']:.4f}")
    
    print()


def example_5_persistence():
    """Example 5: Save and load database."""
    print("=" * 60)
    print("Example 5: Save and Load Database")
    print("=" * 60)
    
    import tempfile
    import shutil
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    save_path = os.path.join(temp_dir, "my_database")
    
    try:
        # Create and populate database
        db = VectorDB(dimension=64, metric="cosine")
        
        vectors = np.random.randn(100, 64).astype(np.float32)
        ids = [f"item_{i}" for i in range(100)]
        metadata = [{"name": f"Item {i}", "value": i} for i in range(100)]
        
        db.batch_insert(ids, vectors, metadata)
        
        print(f"✓ Created database with {len(db)} vectors")
        
        # Save to disk
        db.save(save_path)
        print(f"✓ Saved database to {save_path}")
        
        # Load from disk
        db_loaded = VectorDB.load(save_path)
        print(f"✓ Loaded database with {len(db_loaded)} vectors")
        
        # Verify data is intact
        original_vec = db.get("item_42")
        loaded_vec = db_loaded.get("item_42")
        
        assert np.allclose(original_vec['vector'], loaded_vec['vector'])
        assert original_vec['metadata'] == loaded_vec['metadata']
        
        print("✓ Data integrity verified")
        
        # Search works on loaded database
        query = vectors[0]
        results = db_loaded.search(query, top_k=3)
        print(f"✓ Search works on loaded database: found {len(results)} results")
        
        # Test clean up
        db_loaded.clear()
        
    finally:
        # Cleanup
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    print()


def example_6_crud_operations():
    """Example 6: CRUD operations."""
    print("=" * 60)
    print("Example 6: CRUD Operations")
    print("=" * 60)
    
    db = VectorDB(dimension=32, metric="cosine")
    
    # Create (Insert)
    vec1 = np.random.randn(32).astype(np.float32)
    db.insert("user_1", vec1, metadata={"name": "Alice", "age": 30})
    print("✓ Created vector for user_1")
    
    # Read (Get)
    result = db.get("user_1")
    print(f"✓ Read: {result['metadata']}")
    
    # Update
    db.update("user_1", metadata={"name": "Alice", "age": 31})
    result = db.get("user_1")
    print(f"✓ Updated: age is now {result['metadata']['age']}")
    
    # Update vector
    vec1_new = np.random.randn(32).astype(np.float32)
    db.update("user_1", vector=vec1_new)
    print("✓ Updated vector")
    
    # Delete
    db.delete("user_1")
    print(f"✓ Deleted user_1 (database now has {len(db)} vectors)")
    
    print()


if __name__ == "__main__":
    example_1_basic_usage()
    example_2_batch_operations()
    # example_3_ivf_index()
    example_4_different_metrics()
    example_5_persistence()
    example_6_crud_operations()
    # example_7_real_world_simulation()
    
    print("=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
