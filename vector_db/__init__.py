"""
My Vector Database - A minimal vector database implementation from scratch.
Built for learning and understanding vector similarity search.
"""

from .core import VectorDB
from .storage import save_db, load_db

# Add save/load methods to VectorDB
VectorDB.save = save_db
VectorDB.load = staticmethod(load_db)

__version__ = "0.1.0"
__all__ = ["VectorDB"]

