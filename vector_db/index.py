"""
IVF (Inverted File) Index implementation with k-means clustering.
This index partitions the vector space into clusters for faster similarity search.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from .distance import compute_distances, get_top_k_indices, DistanceMetric


class KMeans:
    """
    Simple K-Means clustering implementation from scratch.
    Used for partitioning the vector space in IVF index.
    """
    
    def __init__(self, n_clusters: int, max_iter: int = 100, tol: float = 1e-4, seed: Optional[int] = None):
        """
        Initialize K-Means clustering.
        
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            seed: Random seed for initialization
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
    
    def fit(self, vectors: np.ndarray) -> 'KMeans':
        """
        Fit k-means on the given vectors.
        
        Args:
            vectors: Array of shape (n_vectors, dimension)
            
        Returns:
            Self for chaining
        """
        n_vectors, dimension = vectors.shape
        
        # Adjust n_clusters if we have fewer vectors
        n_clusters = min(self.n_clusters, n_vectors)
        
        # Initialize centroids using k-means++
        if self.seed is not None:
            np.random.seed(self.seed)
        
        centroids = self._kmeans_plus_plus_init(vectors, n_clusters)
        
        # Iterative refinement
        for iteration in range(self.max_iter):
            # Assign each vector to nearest centroid
            labels = self._assign_clusters(vectors, centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(vectors, labels, n_clusters)
            
            # Check for convergence
            centroid_shift = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids
            
            if centroid_shift < self.tol:
                break
        
        self.centroids = centroids
        self.labels = labels
        self.n_clusters = n_clusters
        
        return self
    
    def _kmeans_plus_plus_init(self, vectors: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Initialize centroids using k-means++ algorithm.
        This gives better initial positions than random selection.
        """
        n_vectors = len(vectors)
        centroids = np.zeros((n_clusters, vectors.shape[1]), dtype=np.float32)
        
        # Choose first centroid randomly
        centroids[0] = vectors[np.random.randint(n_vectors)]
        
        # Choose remaining centroids with probability proportional to distance
        for i in range(1, n_clusters):
            # Compute distances to nearest existing centroid
            distances = np.min(
                np.sum((vectors[:, np.newaxis] - centroids[:i]) ** 2, axis=2),
                axis=1
            )
            
            # Choose next centroid with probability proportional to distance^2
            probabilities = distances / np.sum(distances)
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.rand()
            
            for j, prob in enumerate(cumulative_probs):
                if r < prob:
                    centroids[i] = vectors[j]
                    break
        
        return centroids
    
    def _assign_clusters(self, vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each vector to the nearest centroid.
        """
        # Compute distances to all centroids
        # Shape: (n_vectors, n_clusters)
        distances = np.sum((vectors[:, np.newaxis] - centroids) ** 2, axis=2)
        
        # Assign to nearest centroid
        labels = np.argmin(distances, axis=1)
        
        return labels
    
    def _update_centroids(self, vectors: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Update centroids as the mean of assigned vectors.
        """
        centroids = np.zeros((n_clusters, vectors.shape[1]), dtype=np.float32)
        
        for i in range(n_clusters):
            mask = labels == i
            if np.any(mask):
                centroids[i] = np.mean(vectors[mask], axis=0)
            else:
                # If no vectors assigned, keep previous centroid or assign random vector
                centroids[i] = vectors[np.random.randint(len(vectors))]
        
        return centroids
    
    def predict(self, vectors: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new vectors.
        
        Args:
            vectors: Array of shape (n_vectors, dimension)
            
        Returns:
            Cluster labels of shape (n_vectors,)
        """
        if self.centroids is None:
            raise ValueError("KMeans must be fitted before prediction")
        
        return self._assign_clusters(vectors, self.centroids)


class IVFIndex:
    """
    Inverted File Index for efficient similarity search.
    
    The IVF index works by:
    1. Clustering vectors into groups using k-means
    2. Building inverted lists mapping clusters to vectors
    3. At query time, searching only the nearest clusters
    
    This trades some accuracy for significant speed improvements.
    """
    
    def __init__(
        self,
        vectors: np.ndarray,
        ids: List[str],
        metadata: Dict[str, Any],
        metric: DistanceMetric = "cosine",
        n_clusters: int = 100,
        nprobe: int = 10,
        **kwargs
    ):
        """
        Initialize IVF index.
        
        Args:
            vectors: Array of vectors, shape (n_vectors, dimension)
            ids: List of vector IDs
            metadata: Dictionary mapping IDs to metadata
            metric: Distance metric to use
            n_clusters: Number of clusters for partitioning
            nprobe: Number of clusters to search at query time
            **kwargs: Additional parameters (for future extensions)
        """
        self.vectors = vectors
        self.ids = ids
        self.metadata = metadata
        self.metric = metric
        self.n_clusters = n_clusters
        self.nprobe = nprobe
        
        # Will be initialized during build()
        self.kmeans: Optional[KMeans] = None
        self.inverted_lists: Dict[int, List[int]] = {}
        self._is_built = False
    
    def build(self) -> None:
        """
        Build the IVF index by clustering vectors.
        """
        if len(self.vectors) == 0:
            return
        
        # Adjust n_clusters based on data size
        # Rule of thumb: at least sqrt(n) clusters, but not more than n/10
        n_vectors = len(self.vectors)
        adjusted_n_clusters = min(
            self.n_clusters,
            max(int(np.sqrt(n_vectors)), n_vectors // 10)
        )
        
        if adjusted_n_clusters < 1:
            adjusted_n_clusters = 1
        
        # Cluster the vectors
        self.kmeans = KMeans(n_clusters=adjusted_n_clusters, max_iter=50)
        self.kmeans.fit(self.vectors)
        
        # Build inverted lists
        self.inverted_lists = {i: [] for i in range(self.kmeans.n_clusters)}
        
        for idx, label in enumerate(self.kmeans.labels):
            self.inverted_lists[label].append(idx)
        
        self._is_built = True
    
    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using the IVF index.
        
        Args:
            query: Query vector
            top_k: Number of results to return
            return_scores: Whether to include similarity scores
            
        Returns:
            List of dictionaries with 'id', 'score', 'metadata' keys
        """
        if not self._is_built or self.kmeans is None:
            raise ValueError("Index must be built before searching")
        
        if len(self.vectors) == 0:
            return []
        
        # Find nearest clusters to query
        cluster_label = self.kmeans.predict(query.reshape(1, -1))[0]
        
        # Get distances to all cluster centroids
        centroid_distances = compute_distances(
            query, 
            self.kmeans.centroids, 
            "euclidean"  # Use euclidean for cluster selection
        )
        
        # Get nprobe nearest clusters
        nprobe = min(self.nprobe, self.kmeans.n_clusters)
        nearest_clusters = get_top_k_indices(centroid_distances, nprobe)
        
        # Collect candidate vectors from nearest clusters
        candidate_indices = []
        for cluster_id in nearest_clusters:
            candidate_indices.extend(self.inverted_lists[int(cluster_id)])
        
        if len(candidate_indices) == 0:
            return []
        
        # Search within candidates
        candidate_vectors = self.vectors[candidate_indices]
        scores = compute_distances(query, candidate_vectors, self.metric)
        
        # Get top-k
        top_k = min(top_k, len(scores))
        top_indices = get_top_k_indices(scores, top_k)
        
        # Build results
        results = []
        for idx in top_indices:
            original_idx = candidate_indices[idx]
            result = {
                'id': self.ids[original_idx],
                'metadata': self.metadata.get(self.ids[original_idx], {})
            }
            if return_scores:
                result['score'] = float(scores[idx])
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        if not self._is_built:
            return {'built': False}
        
        cluster_sizes = [len(lst) for lst in self.inverted_lists.values()]
        
        return {
            'built': True,
            'n_vectors': len(self.vectors),
            'n_clusters': self.kmeans.n_clusters,
            'nprobe': self.nprobe,
            'avg_cluster_size': np.mean(cluster_sizes),
            'min_cluster_size': np.min(cluster_sizes),
            'max_cluster_size': np.max(cluster_sizes),
            'metric': self.metric
        }

