"""
Configuration management for real embeddings demo.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Configuration for Gemini embeddings and vector database."""
    
    def __init__(self):
        # Gemini API settings
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        self.embedding_model: str = "models/embedding-001"
        
        # Chunking settings
        self.chunk_size: int = 500  # characters
        self.chunk_overlap: int = 50  # characters
        self.chunking_strategy: str = "simple"  # simple, sentence, or paragraph
        
        # Vector DB settings
        self.vector_dimension: int = 768  # Gemini embedding-001 dimension
        self.distance_metric: str = "cosine"
        self.use_index: bool = True
        self.index_type: str = "ivf"
        self.n_clusters: int = 50
        self.nprobe: int = 10
        
        # Search settings
        self.top_k: int = 5
        
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables. "
                "Please set it using: export GEMINI_API_KEY='your-key-here' "
                "or create a .env file with GEMINI_API_KEY=your-key-here"
            )
        
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        return True
    
    def __str__(self) -> str:
        """String representation (hides API key)."""
        return f"""Config(
    embedding_model={self.embedding_model},
    chunk_size={self.chunk_size},
    chunk_overlap={self.chunk_overlap},
    vector_dimension={self.vector_dimension},
    distance_metric={self.distance_metric},
    use_index={self.use_index},
    index_type={self.index_type if self.use_index else 'None'},
    api_key={'***' + self.gemini_api_key[-4:] if self.gemini_api_key else 'NOT SET'}
)"""


# Global config instance
config = Config()

