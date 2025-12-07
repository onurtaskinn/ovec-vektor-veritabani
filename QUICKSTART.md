# Quick Start Guide

## Installation & Setup

```bash
# Navigate to project directory
cd my_vector_db

# Activate virtual environment
source venv/bin/activate

# Dependencies already installed (numpy)
```

## Run Tests

```bash
# Test distance metrics
python tests/test_distance.py

# Test IVF indexing
python tests/test_index.py

# Test full integration (includes benchmarks)
python tests/test_integration.py
```

## Run Examples

```bash
# Run all 7 examples
python examples/basic_usage.py
```

## Basic Usage in Your Code

```python
from vector_db import VectorDB
import numpy as np

# 1. Create a vector database
db = VectorDB(dimension=128, metric="cosine")

# 2. Add vectors
vector = np.random.randn(128).astype(np.float32)
db.insert(id="vec_1", vector=vector, metadata={"name": "example"})

# 3. Batch insert (more efficient)
vectors = np.random.randn(1000, 128).astype(np.float32)
ids = [f"vec_{i}" for i in range(1000)]
db.batch_insert(ids, vectors)

# 4. Search for similar vectors
query = np.random.randn(128).astype(np.float32)
results = db.search(query, top_k=5)

for result in results:
    print(f"{result['id']}: {result['score']:.4f}")

# 5. Save and load
db.save("my_database")
db_loaded = VectorDB.load("my_database")
```

## Use IVF Index for Speed

```python
# Create database with IVF index
db = VectorDB(
    dimension=128,
    metric="cosine",
    index_type="ivf",
    n_clusters=100,  # Adjust based on dataset size
    nprobe=10        # Higher = more accurate but slower
)

# Add vectors
db.batch_insert(ids, vectors)

# Build index (one-time cost)
db.build_index()

# Now searches are much faster!
results = db.search(query, top_k=10)
```

## Available Distance Metrics

```python
# Cosine similarity (best for normalized embeddings)
db = VectorDB(dimension=128, metric="cosine")

# Euclidean distance (L2 distance)
db = VectorDB(dimension=128, metric="euclidean")

# Dot product (for Maximum Inner Product Search)
db = VectorDB(dimension=128, metric="dot_product")
```

## CRUD Operations

```python
# Create (Insert)
db.insert("id_1", vector, metadata={"key": "value"})

# Read (Get)
result = db.get("id_1")
print(result['vector'], result['metadata'])

# Update
db.update("id_1", vector=new_vector, metadata={"key": "new_value"})

# Delete
db.delete("id_1")

# Check existence
if "id_1" in db:
    print("Vector exists")

# Get database size
print(f"Database has {len(db)} vectors")
```

## Performance Tips

1. **Use batch_insert** for multiple vectors (much faster)
2. **Build IVF index** for datasets > 10,000 vectors
3. **Tune nprobe**: Higher values = more accurate but slower
4. **Choose right metric**: Cosine for normalized vectors
5. **Save/load** to avoid rebuilding index every time

## Troubleshooting

### NumPy not found
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Import errors
Make sure you're in the project root directory when running scripts.

### Slow searches
Build an IVF index if you have > 10,000 vectors.

## Learn More

- Read [README.md](README.md) for detailed documentation
- Check [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for implementation details
- Explore [examples/basic_usage.py](examples/basic_usage.py) for more examples
- Run tests to understand the algorithms better

## Next Steps

Try implementing your own features:
- Add new distance metrics
- Implement HNSW index
- Add quantization for memory efficiency
- Create a REST API wrapper
- Build a simple web UI

