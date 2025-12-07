# My Vector Database

A minimal, educational vector database implementation built from scratch in Python. This project uses only NumPy as a dependency, making it perfect for learning how vector databases work under the hood.

## 🎯 Project Goals

This vector database was built as a learning project to understand:
- **Vector similarity search algorithms** (cosine similarity, Euclidean distance, dot product)
- **Indexing strategies** (brute-force vs IVF indexing with k-means clustering)
- **Systems design** (CRUD operations, persistence, API design)
- **Performance tradeoffs** (accuracy vs speed, memory vs query time)

## ✨ Features

- ✅ **Multiple distance metrics**: Cosine similarity, Euclidean distance, dot product
- ✅ **Brute-force search**: Simple, accurate baseline
- ✅ **IVF indexing**: Fast approximate search using k-means clustering
- ✅ **CRUD operations**: Insert, read, update, delete vectors
- ✅ **Batch operations**: Efficient bulk inserts
- ✅ **Metadata support**: Store additional information with each vector
- ✅ **Persistence**: Save and load databases from disk
- ✅ **Minimal dependencies**: Only NumPy required
- ✅ **Comprehensive tests**: Full test suite with benchmarks

## 📦 Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd my_vector_db

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

```python
from vector_db import VectorDB
import numpy as np

# Create a vector database
db = VectorDB(dimension=128, metric="cosine")

# Insert vectors
vector = np.random.randn(128).astype(np.float32)
db.insert(
    id="vec_1",
    vector=vector,
    metadata={"description": "My first vector"}
)

# Search for similar vectors
query = np.random.randn(128).astype(np.float32)
results = db.search(query, top_k=5)

for result in results:
    print(f"ID: {result['id']}, Score: {result['score']:.4f}")
```

## 📚 Examples

### Basic Operations

```python
# Insert a single vector
db.insert("vec_1", vector, metadata={"text": "hello"})

# Batch insert (more efficient)
vectors = np.random.randn(1000, 128).astype(np.float32)
ids = [f"vec_{i}" for i in range(1000)]
db.batch_insert(ids, vectors)

# Get a vector by ID
result = db.get("vec_1")
print(result['vector'], result['metadata'])

# Update vector or metadata
db.update("vec_1", vector=new_vector, metadata={"text": "world"})

# Delete a vector
db.delete("vec_1")

# Search
results = db.search(query_vector, top_k=10)
```

### Using IVF Index for Fast Search

```python
# Create database with IVF index
db = VectorDB(
    dimension=128,
    metric="cosine",
    index_type="ivf",
    n_clusters=100,  # Number of clusters
    nprobe=10        # Number of clusters to search
)

# Insert vectors
db.batch_insert(ids, vectors)

# Build the index
db.build_index()

# Search (automatically uses IVF index)
results = db.search(query, top_k=10)
```

### Persistence

```python
# Save database to disk
db.save("my_database")

# Load database from disk
db = VectorDB.load("my_database")
```

### Different Distance Metrics

```python
# Cosine similarity (best for normalized vectors)
db_cosine = VectorDB(dimension=128, metric="cosine")

# Euclidean distance (L2)
db_euclidean = VectorDB(dimension=128, metric="euclidean")

# Dot product (useful for MIPS - Maximum Inner Product Search)
db_dot = VectorDB(dimension=128, metric="dot_product")
```

## 🏗️ Architecture

### Project Structure

```
my_vector_db/
├── vector_db/
│   ├── __init__.py       # Package initialization
│   ├── core.py           # VectorDB main class
│   ├── distance.py       # Distance metrics implementation
│   ├── index.py          # IVF indexing with k-means
│   ├── storage.py        # Persistence layer
│   └── utils.py          # Helper functions
├── tests/
│   ├── test_distance.py      # Distance metrics tests
│   ├── test_index.py         # Indexing tests
│   └── test_integration.py   # Integration tests
├── examples/
│   └── basic_usage.py    # Usage examples
├── requirements.txt
└── README.md
```

### Key Components

#### 1. Distance Metrics (`distance.py`)
- **Cosine Similarity**: Measures angle between vectors (range: -1 to 1)
- **Euclidean Distance**: L2 distance between vectors (range: 0 to ∞)
- **Dot Product**: Inner product of vectors (range: -∞ to ∞)

All implemented from scratch using NumPy for educational purposes.

#### 2. Core VectorDB (`core.py`)
- Manages vector storage using NumPy arrays
- Implements CRUD operations
- Supports brute-force search as baseline
- Integrates with optional IVF index for fast search

#### 3. IVF Index (`index.py`)
- **K-Means Clustering**: Partitions vector space into clusters
- **Inverted Lists**: Maps clusters to vectors
- **Approximate Search**: Searches only nearest clusters (controlled by `nprobe`)
- **Trade-off**: Speed vs accuracy

#### 4. Persistence (`storage.py`)
- Saves vectors using NumPy's `.npz` format (compressed)
- Saves metadata as JSON
- Saves index structure for fast loading
- Efficient serialization/deserialization

## 🧪 Testing

Run the test suite:

```bash
# Test distance metrics
python tests/test_distance.py

# Test IVF indexing
python tests/test_index.py

# Integration tests with benchmarks
python tests/test_integration.py
```

Run examples:

```bash
python examples/basic_usage.py
```

## 📊 Performance

Benchmark results on a dataset of 10,000 vectors (128 dimensions):

| Method       | Query Time | Notes                           |
|--------------|------------|---------------------------------|
| Brute-force  | ~150ms     | 100% accuracy, searches all vectors |
| IVF (100 clusters, nprobe=10) | ~15ms | ~3-10x faster, 90-95% recall@10 |

*Results will vary based on hardware, dataset size, and parameters.*

### Performance Tips

1. **Use batch_insert** instead of individual inserts
2. **Build IVF index** for datasets > 10,000 vectors
3. **Tune nprobe**: Higher = more accurate but slower
4. **Tune n_clusters**: More clusters = faster but needs more memory
5. **Choose right metric**: Cosine for normalized vectors, L2 for absolute distance

## 🎓 Learning Insights

### What You'll Learn

1. **Vector Similarity Search**
   - How different distance metrics work
   - When to use each metric
   - Understanding normalized vs non-normalized vectors

2. **Indexing Algorithms**
   - K-means clustering from scratch
   - Space partitioning strategies
   - Accuracy vs speed tradeoffs

3. **Systems Design**
   - Efficient data structures (NumPy arrays vs lists)
   - Memory management
   - Serialization strategies
   - API design patterns

4. **Performance Optimization**
   - Batch operations
   - Approximate nearest neighbor search
   - Index building strategies

### Why Minimal Dependencies?

Using only NumPy forces you to:
- Understand algorithms at a fundamental level
- Learn about vectorized operations
- Appreciate what libraries like FAISS/Qdrant do under the hood
- Have full control and visibility into the implementation

## 🔬 Advanced Topics

### IVF Index Parameters

- **n_clusters**: Number of partitions (typically √n to n/10)
- **nprobe**: Number of clusters to search (higher = more accurate)
- **metric**: Distance function used for both clustering and search

### When to Use IVF Index

- ✅ **Use IVF when**: Dataset > 10,000 vectors, can tolerate small accuracy loss
- ❌ **Skip IVF when**: Dataset < 1,000 vectors, need 100% accuracy

## 🚧 Limitations & Future Improvements

Current limitations (by design, for learning):
- Single-threaded (no concurrency)
- In-memory index (limited by RAM)
- No quantization (vectors stored at full precision)
- Simple k-means (no product quantization)

Potential improvements for production use:
- Add HNSW index for better accuracy/speed tradeoff
- Implement quantization (PQ, SQ) for memory efficiency
- Add filtering on metadata
- Multi-threading for batch operations
- Disk-based storage for large datasets
- Distributed search across multiple nodes

## 🤝 Contributing

This is an educational project! Feel free to:
- Add new indexing algorithms (HNSW, LSH, etc.)
- Implement quantization techniques
- Improve documentation
- Add more examples and tutorials

## 📝 License

MIT License - feel free to use for learning and teaching!

## 🙏 Acknowledgments

Inspired by production vector databases:
- [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI)
- [Qdrant](https://github.com/qdrant/qdrant)
- [Milvus](https://github.com/milvus-io/milvus)
- [Weaviate](https://github.com/weaviate/weaviate)

Built for learning, not production use! 🎓

