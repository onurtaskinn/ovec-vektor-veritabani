"""
Persistence layer for saving and loading vector databases.
Uses NumPy's .npz format for efficient storage.
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class StorageManager:
    """
    Manages saving and loading of vector database state.
    """
    
    @staticmethod
    def save(db: 'VectorDB', path: str) -> None:
        """
        Save a VectorDB instance to disk.
        
        Args:
            db: VectorDB instance to save
            path: Directory path to save to (will be created if doesn't exist)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save vectors and IDs
        vectors_file = path / "vectors.npz"
        np.savez_compressed(
            vectors_file,
            vectors=db._vectors,
            ids=np.array(db._ids, dtype=object)
        )
        
        # Save metadata as JSON
        metadata_file = path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(db._metadata, f, indent=2)
        
        # Save configuration
        config = {
            'dimension': db.dimension,
            'metric': db.metric,
            'index_type': db.index_type,
            'index_params': db.index_params,
            'n_vectors': len(db)
        }
        config_file = path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save index if built
        if db.index_type == 'ivf' and db._index_built and db._index is not None:
            StorageManager._save_ivf_index(db._index, path)
    
    @staticmethod
    def _save_ivf_index(index: 'IVFIndex', path: Path) -> None:
        """
        Save IVF index state.
        
        Args:
            index: IVFIndex instance
            path: Directory path to save to
        """
        index_file = path / "ivf_index.npz"
        
        # Save centroids and inverted lists
        # Convert inverted lists to a format that can be saved
        inverted_lists_array = np.array(
            [np.array(lst, dtype=np.int32) for lst in index.inverted_lists.values()],
            dtype=object
        )
        
        np.savez_compressed(
            index_file,
            centroids=index.kmeans.centroids,
            labels=index.kmeans.labels,
            n_clusters=np.array([index.kmeans.n_clusters]),
            inverted_lists=inverted_lists_array,
            nprobe=np.array([index.nprobe])
        )
    
    @staticmethod
    def load(path: str) -> 'VectorDB':
        """
        Load a VectorDB instance from disk.
        
        Args:
            path: Directory path to load from
            
        Returns:
            Loaded VectorDB instance
            
        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If data is corrupted or incompatible
        """
        from .core import VectorDB
        
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Database path '{path}' not found")
        
        # Load configuration
        config_file = path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found in '{path}'")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Create VectorDB instance
        db = VectorDB(
            dimension=config['dimension'],
            metric=config['metric'],
            index_type=config.get('index_type'),
            **config.get('index_params', {})
        )
        
        # Load vectors and IDs
        vectors_file = path / "vectors.npz"
        if vectors_file.exists():
            data = np.load(vectors_file, allow_pickle=True)
            db._vectors = data['vectors']
            db._ids = data['ids'].tolist()
            
            # Rebuild ID to index mapping
            db._id_to_idx = {id: i for i, id in enumerate(db._ids)}
        
        # Load metadata
        metadata_file = path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                db._metadata = json.load(f)
        
        # Load index if it exists
        if db.index_type == 'ivf':
            index_file = path / "ivf_index.npz"
            if index_file.exists():
                StorageManager._load_ivf_index(db, index_file)
        
        return db
    
    @staticmethod
    def _load_ivf_index(db: 'VectorDB', index_file: Path) -> None:
        """
        Load IVF index state.
        
        Args:
            db: VectorDB instance to load index into
            index_file: Path to index file
        """
        from .index import IVFIndex, KMeans
        
        data = np.load(index_file, allow_pickle=True)
        
        # Create IVF index
        # Use saved values, not the ones from index_params to avoid conflicts
        db._index = IVFIndex(
            vectors=db._vectors,
            ids=db._ids,
            metadata=db._metadata,
            metric=db.metric,
            n_clusters=int(data['n_clusters'][0]),
            nprobe=int(data['nprobe'][0])
        )
        
        # Restore k-means state
        db._index.kmeans = KMeans(n_clusters=int(data['n_clusters'][0]))
        db._index.kmeans.centroids = data['centroids']
        db._index.kmeans.labels = data['labels']
        db._index.kmeans.n_clusters = int(data['n_clusters'][0])
        
        # Restore inverted lists
        inverted_lists_array = data['inverted_lists']
        db._index.inverted_lists = {
            i: lst.tolist() for i, lst in enumerate(inverted_lists_array)
        }
        
        db._index._is_built = True
        db._index_built = True


# Add save/load methods to VectorDB
def save_db(self, path: str) -> None:
    """
    Save the database to disk.
    
    Args:
        path: Directory path to save to
    """
    StorageManager.save(self, path)


def load_db(path: str) -> 'VectorDB':
    """
    Load a database from disk.
    
    Args:
        path: Directory path to load from
        
    Returns:
        Loaded VectorDB instance
    """
    return StorageManager.load(path)


# These will be monkey-patched onto VectorDB in __init__.py
__all__ = ['StorageManager', 'save_db', 'load_db']

