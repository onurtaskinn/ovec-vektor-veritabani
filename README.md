# My Vector Database

A minimal, educational implementation of a vector database from scratch in Python.

## Features

- **Vector Storage**: Efficient storage using NumPy.
- **Distance Metrics**: Cosine Similarity, Euclidean Distance, Dot Product.
- **Indexing**: IVF (Inverted File Index) with K-Means clustering for fast approximate search.
- **Persistence**: Save and load database state to disk.
- **Metadata**: Store arbitrary metadata with vectors.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd my_vector_db

# Install dependencies
pip install numpy
# For the document search example:
pip install -r examples/document_search/requirements.txt
```

## Usage

See `examples/basic_usage.py` for comprehensive examples.

```python
from vector_db import VectorDB
import numpy as np

# Create database
db = VectorDB(dimension=128, metric="cosine")

# Insert
db.insert("vec1", np.random.randn(128))

# Search
results = db.search(np.random.randn(128), top_k=5)
```

## Document Search Demo

Check `examples/document_search/` for a full application using Google's Gemini API for embeddings.
