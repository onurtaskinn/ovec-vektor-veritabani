# OVec — a vector database written from scratch

A working vector database in pure Python and NumPy. No FAISS, no Annoy, no
scikit-learn — the clustering, the distance metrics and the index are all
implemented here.

The goal was never to produce something to use in production. It was to find out
what is actually inside Pinecone, Weaviate or Qdrant by building one, and the
answer turns out to be about eight steps long.

## Built here, not imported

| Piece | What it is |
|---|---|
| `distance.py` | L2 normalisation, cosine similarity, Euclidean distance and dot product, written against NumPy directly, plus `compute_distances` to select between them and `get_top_k_indices` to take the best k |
| `index.py` — `KMeans` | k-means from scratch, including **k-means++ initialisation**, iteration to a tolerance, and `predict` |
| `index.py` — `IVFIndex` | an inverted-file index: partition the space into `nlist` clusters, search only the `nprobe` nearest ones at query time |
| `core.py` — `VectorDB` | storage, CRUD, search, persistence |

## The design decision at the centre

`search()` dispatches between two private methods:

```
search()
 ├── _search_brute_force()   exact, every vector compared
 └── _search_ivf()           approximate, only nprobe clusters visited
```

Which one runs depends on whether `build_index()` has been called. A fresh
database is exact and slow; building the index trades recall for speed, and that
trade is an explicit action rather than a hidden default. This is the whole
premise of approximate nearest-neighbour search expressed in one dispatch.

## The rest of the surface

`VectorDB` implements a full CRUD surface — `insert`, `batch_insert`, `get`,
`update`, `delete`, `search`, `clear`, `ids`, `vectors` — plus `__len__` and
`__contains__`, so `len(db)` and `"vec1" in db` work the way Python users expect.

Input validation is centralised: lists become NumPy arrays, dimension mismatches
raise a message that says what was expected, and everything is pinned to
`float32`. Metadata storage is optional.

**Persistence** writes a directory, not a blob: `vectors.npz` for the vectors,
`metadata.json` for the metadata, `config.json` for the database settings, and
`ivf_index.npz` for the index when one has been built. `load()` is a static
method that reconstructs the database, index included.

## Usage

```python
from vector_db import VectorDB
import numpy as np

db = VectorDB(dimension=128, metric="cosine")
db.insert("vec1", np.random.randn(128))

results = db.search(np.random.randn(128), top_k=5)   # exact

db.build_index()                                      # switch to IVF
results = db.search(np.random.randn(128), top_k=5)   # approximate

db.save("./mydb")
db2 = VectorDB.load("./mydb")
```

`examples/basic_usage.py` covers the API in full.

## Document search — the end-to-end demo

`examples/document_search/` is the database doing real work: a chunker splits
documents, Gemini's embedding model turns the chunks into vectors, OVec stores
and searches them, and an `npz_viewer` lets you inspect what ended up on disk.
Chunking strategy, chunk size and overlap are configurable.

The Gemini key is read from `GEMINI_API_KEY`; nothing is hardcoded.

```bash
pip install numpy
pip install -r examples/document_search/requirements.txt
```

## Tests

16 tests across three files — `test_distance.py`, `test_index.py`,
`test_integration.py` — covering the metrics, the IVF index and the database end
to end.

## How it was built

The commit log is a curriculum, and all of it landed on 7 December 2025:

```
Day 1: Foundations & Math
Day 2: Core Database Engine
Day 3: Persistence Layer
Day 4: Indexing Algorithms
Day 5: Optimization & Integration
Day 6: Testing & Stability
Day 7: Document Search Tools
Day 8: Full Application & Polish
```

Eight numbered steps, planned before they were written, each one adding a layer
the previous one made possible: you cannot index what you cannot store, and you
cannot optimise what you have not tested.

## Status

**Archive.** December 2025, not under development.

Consolidated in September 2026. The work previously lived in two repositories —
this one, and a second holding a re-commit of the first two days under a cleaner
history. The second repository's tree was byte-identical to this one's Day 2
commit, so it was dropped rather than merged. The working tree had also been left
on a detached checkout at Day 2, which made the project look like it stopped
there; it did not.
