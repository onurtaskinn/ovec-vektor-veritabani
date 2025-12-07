"""
Core VectorDB implementation with basic CRUD operations and search.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from .distance import compute_distances, get_top_k_indices, DistanceMetric
from .utils import validate_vector


class VectorDB:
    """
    A simple vector database with support for:
    - Basic CRUD operations (insert, delete, update)
    - Similarity search with multiple distance metrics
    - Optional metadata storage
    - Brute-force and indexed search
    """
    
    def __init__(
        self, 
        dimension: int,
        metric: DistanceMetric = "cosine",
        index_type: Optional[str] = None,
        **index_params
    ):
        """
        Initialize the vector database.
        
        Args:
            dimension: Dimensionality of vectors
            metric: Distance metric ('cosine', 'euclidean', 'dot_product')
            index_type: Type of index ('ivf' or None for brute-force)
            **index_params: Additional parameters for the index
        """
        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type
        self.index_params = index_params
        
        # Storage
        self._vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self._ids: List[str] = []
        self._metadata: Dict[str, Any] = {}
        self._id_to_idx: Dict[str, int] = {}
        
        # Index (will be initialized when needed)
        self._index = None
        self._index_built = False
        
    def __len__(self) -> int:
        """Return the number of vectors in the database."""
        return len(self._ids)
    
    def __contains__(self, vector_id: str) -> bool:
        """Check if a vector ID exists in the database."""
        return vector_id in self._id_to_idx
    
    def insert(
        self, 
        id: str, 
        vector: Union[np.ndarray, List[float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Insert a new vector into the database.
        
        Args:
            id: Unique identifier for the vector
            vector: Vector to insert
            metadata: Optional metadata to associate with the vector
            
        Raises:
            ValueError: If ID already exists or vector is invalid
        """
        if id in self._id_to_idx:
            raise ValueError(f"Vector with ID '{id}' already exists. Use update() to modify it.")
        
        # Validate vector
        vector = validate_vector(vector, self.dimension)
        
        # Add to storage
        self._vectors = np.vstack([self._vectors, vector.reshape(1, -1)])
        self._ids.append(id)
        self._id_to_idx[id] = len(self._ids) - 1
        
        if metadata is not None:
            self._metadata[id] = metadata
        
        # Mark index as stale
        self._index_built = False
    
    def batch_insert(
        self,
        ids: List[str],
        vectors: Union[np.ndarray, List[List[float]]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Insert multiple vectors at once (more efficient than individual inserts).
        
        Args:
            ids: List of unique identifiers
            vectors: Array of vectors, shape (n_vectors, dimension)
            metadata: Optional list of metadata dicts
            
        Raises:
            ValueError: If any ID already exists or vectors are invalid
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match expected {self.dimension}")
        
        if len(ids) != len(vectors):
            raise ValueError(f"Number of IDs ({len(ids)}) doesn't match number of vectors ({len(vectors)})")
        
        # Check for duplicates
        for id in ids:
            if id in self._id_to_idx:
                raise ValueError(f"Vector with ID '{id}' already exists.")
        
        # Add to storage
        start_idx = len(self._ids)
        self._vectors = np.vstack([self._vectors, vectors])
        self._ids.extend(ids)
        
        for i, id in enumerate(ids):
            self._id_to_idx[id] = start_idx + i
            if metadata is not None and i < len(metadata):
                self._metadata[id] = metadata[i]
        
        self._index_built = False
    
    def delete(self, id: str) -> None:
        """
        Delete a vector from the database.
        
        Args:
            id: Identifier of the vector to delete
            
        Raises:
            KeyError: If ID doesn't exist
        """
        if id not in self._id_to_idx:
            raise KeyError(f"Vector with ID '{id}' not found.")
        
        idx = self._id_to_idx[id]
        
        # Remove from arrays
        self._vectors = np.delete(self._vectors, idx, axis=0)
        self._ids.pop(idx)
        
        # Remove metadata
        if id in self._metadata:
            del self._metadata[id]
        
        # Rebuild ID mapping
        self._id_to_idx = {id: i for i, id in enumerate(self._ids)}
        
        self._index_built = False
    
    def update(
        self,
        id: str,
        vector: Optional[Union[np.ndarray, List[float]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update a vector and/or its metadata.
        
        Args:
            id: Identifier of the vector to update
            vector: New vector (optional)
            metadata: New metadata (optional, replaces existing)
            
        Raises:
            KeyError: If ID doesn't exist
        """
        if id not in self._id_to_idx:
            raise KeyError(f"Vector with ID '{id}' not found.")
        
        idx = self._id_to_idx[id]
        
        if vector is not None:
            vector = validate_vector(vector, self.dimension)
            self._vectors[idx] = vector
            self._index_built = False
        
        if metadata is not None:
            self._metadata[id] = metadata
    
    def get(self, id: str) -> Dict[str, Any]:
        """
        Retrieve a vector and its metadata by ID.
        
        Args:
            id: Identifier of the vector
            
        Returns:
            Dictionary with 'id', 'vector', and 'metadata' keys
            
        Raises:
            KeyError: If ID doesn't exist
        """
        if id not in self._id_to_idx:
            raise KeyError(f"Vector with ID '{id}' not found.")
        
        idx = self._id_to_idx[id]
        return {
            'id': id,
            'vector': self._vectors[idx].copy(),
            'metadata': self._metadata.get(id, {})
        }
    
    def search(
        self,
        query: Union[np.ndarray, List[float]],
        top_k: int = 10,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for the most similar vectors.
        
        Args:
            query: Query vector
            top_k: Number of results to return
            return_scores: Whether to include similarity scores
            
        Returns:
            List of dictionaries with 'id', 'score', 'metadata' keys
        """
        if len(self._vectors) == 0:
            return []
        
        query = validate_vector(query, self.dimension)
        
        # Use index if available, otherwise brute-force
        if self.index_type == 'ivf' and self._index_built:
            return self._search_ivf(query, top_k, return_scores)
        else:
            return self._search_brute_force(query, top_k, return_scores)
    
    def _search_brute_force(
        self,
        query: np.ndarray,
        top_k: int,
        return_scores: bool
    ) -> List[Dict[str, Any]]:
        """
        Brute-force search over all vectors.
        """
        # Compute distances to all vectors
        scores = compute_distances(query, self._vectors, self.metric)
        
        # Get top-k indices
        top_k = min(top_k, len(scores))
        top_indices = get_top_k_indices(scores, top_k)
        
        # Build results
        results = []
        for idx in top_indices:
            result = {
                'id': self._ids[idx],
                'metadata': self._metadata.get(self._ids[idx], {})
            }
            if return_scores:
                result['score'] = float(scores[idx])
            results.append(result)
        
        return results
    
    def _search_ivf(
        self,
        query: np.ndarray,
        top_k: int,
        return_scores: bool
    ) -> List[Dict[str, Any]]:
        """
        Search using IVF index.
        """
        if self._index is None:
            return self._search_brute_force(query, top_k, return_scores)
        
        return self._index.search(query, top_k, return_scores)
    
    def build_index(self) -> None:
        """
        Build the index for faster search.
        Only applicable when index_type is set.
        """
        if self.index_type is None:
            return
        
        if self.index_type == 'ivf':
            from .index import IVFIndex
            self._index = IVFIndex(
                vectors=self._vectors,
                ids=self._ids,
                metadata=self._metadata,
                metric=self.metric,
                **self.index_params
            )
            self._index.build()
            self._index_built = True
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def clear(self) -> None:
        """Remove all vectors from the database."""
        self._vectors = np.empty((0, self.dimension), dtype=np.float32)
        self._ids = []
        self._metadata = {}
        self._id_to_idx = {}
        self._index = None
        self._index_built = False
    
    @property
    def ids(self) -> List[str]:
        """Get all vector IDs."""
        return self._ids.copy()
    
    @property
    def vectors(self) -> np.ndarray:
        """Get all vectors as a numpy array."""
        return self._vectors.copy()

