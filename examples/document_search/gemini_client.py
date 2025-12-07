"""
Gemini API client for generating embeddings.
"""

import time
from typing import List, Union
import google.generativeai as genai

# Handle both direct execution and package import
try:
    from .config import config
except ImportError:
    from config import config


class GeminiEmbedder:
    """
    Wrapper for Google's Gemini embedding API.
    Handles API calls, batching, and error handling.
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Gemini embedder.
        
        Args:
            api_key: Gemini API key (uses config if not provided)
            model: Embedding model name (uses config if not provided)
        """
        self.api_key = api_key or config.gemini_api_key
        self.model = model or config.embedding_model
        
        if not self.api_key:
            raise ValueError("API key required. Set GEMINI_API_KEY environment variable.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        print(f"✓ Initialized Gemini embedder with model: {self.model}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error embedding text: {e}")
            raise
    
    def embed_batch(
        self, 
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        total = len(texts)
        
        if show_progress:
            print(f"Generating embeddings for {total} texts...")
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            batch_end = min(i + batch_size, total)
            
            try:
                # Process batch
                for text in batch:
                    embedding = self.embed_text(text)
                    embeddings.append(embedding)
                    
                    # Rate limiting: small delay between requests
                    time.sleep(0.1)
                
                if show_progress:
                    print(f"  Progress: {batch_end}/{total} texts embedded")
                    
            except Exception as e:
                print(f"Error in batch {i}-{batch_end}: {e}")
                print("Retrying after delay...")
                time.sleep(2)
                
                # Retry failed batch
                for text in batch:
                    try:
                        embedding = self.embed_text(text)
                        embeddings.append(embedding)
                        time.sleep(0.2)
                    except Exception as retry_error:
                        print(f"Failed to embed text: {retry_error}")
                        # Use zero vector as fallback
                        embeddings.append([0.0] * 768)
        
        if show_progress:
            print(f"✓ Generated {len(embeddings)} embeddings")
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            result = genai.embed_content(
                model=self.model,
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error embedding query: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from this model.
        
        Returns:
            Embedding dimension
        """
        # Gemini embedding-001 produces 768-dimensional embeddings
        return 768


def test_connection():
    """Test Gemini API connection."""
    try:
        embedder = GeminiEmbedder()
        test_text = "This is a test."
        embedding = embedder.embed_text(test_text)
        print(f"✓ Successfully generated embedding of dimension {len(embedding)}")
        return True
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the embedder
    print("Testing Gemini embedder...")
    test_connection()

