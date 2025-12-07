"""
Document search demo using Gemini embeddings and the vector database.
Demonstrates real semantic search on user-provided documents.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vector_db import VectorDB

# Handle both direct execution and package import
try:
    from .config import config
    from .gemini_client import GeminiEmbedder
    from .chunker import DocumentChunker, TextChunk
except ImportError:
    from config import config
    from gemini_client import GeminiEmbedder
    from chunker import DocumentChunker, TextChunk


class DocumentSearchSystem:
    """
    Complete document search system with chunking, embedding, and retrieval.
    """
    
    def __init__(
        self,
        docs_dir: str = None,
        use_index: bool = True,
        index_type: str = "ivf"
    ):
        """
        Initialize document search system.
        
        Args:
            docs_dir: Directory containing documents to index
            use_index: Whether to use IVF index
            index_type: Type of index to use
        """
        self.docs_dir = docs_dir
        self.use_index = use_index
        self.index_type = index_type if use_index else None
        
        # Validate configuration
        config.validate()
        
        # Initialize components
        print("\n" + "=" * 60)
        print("Document Search System with Gemini Embeddings")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Embedding model: {config.embedding_model}")
        print(f"  Chunk size: {config.chunk_size} chars")
        print(f"  Chunk overlap: {config.chunk_overlap} chars")
        print(f"  Chunking strategy: {config.chunking_strategy}")
        print(f"  Distance metric: {config.distance_metric}")
        print(f"  Use index: {self.use_index}")
        if self.use_index:
            print(f"  Index type: {self.index_type}")
            print(f"  n_clusters: {config.n_clusters}")
            print(f"  nprobe: {config.nprobe}")
        
        self.embedder = GeminiEmbedder()
        self.chunker = DocumentChunker()
        self.db: VectorDB = None
        self.chunks: List[TextChunk] = []
    
    def load_documents(self, docs_dir: str = None) -> Dict[str, str]:
        """
        Load all text documents from a directory.
        
        Args:
            docs_dir: Directory path (uses instance default if not provided)
            
        Returns:
            Dictionary mapping filenames to document text
        """
        docs_dir = docs_dir or self.docs_dir
        
        if not docs_dir:
            raise ValueError("docs_dir must be provided")
        
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            raise FileNotFoundError(f"Directory not found: {docs_dir}")
        
        documents = {}
        
        print(f"\n📂 Loading documents from: {docs_dir}")
        
        # Support common text formats
        extensions = ['.txt', '.md', '.text', '.doc']
        
        for file_path in docs_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        documents[file_path.name] = text
                        print(f"  ✓ {file_path.name} ({len(text)} chars)")
                except Exception as e:
                    print(f"  ✗ {file_path.name}: {e}")
        
        print(f"\n✓ Loaded {len(documents)} documents")
        return documents
    
    def index_documents(self, documents: Dict[str, str] = None):
        """
        Index documents: chunk, embed, and store in vector database.
        
        Args:
            documents: Dictionary of documents (loads from docs_dir if not provided)
        """
        if documents is None:
            documents = self.load_documents()
        
        if not documents:
            raise ValueError("No documents to index")
        
        # Step 1: Chunk documents
        print(f"\n📝 Step 1: Chunking documents...")
        self.chunks = self.chunker.chunk_documents(documents, show_progress=True)
        
        if not self.chunks:
            raise ValueError("No chunks created")
        
        # Step 2: Generate embeddings
        print(f"\n🔮 Step 2: Generating embeddings...")
        start_time = time.time()
        
        chunk_texts = [chunk.text for chunk in self.chunks]
        embeddings = self.embedder.embed_batch(chunk_texts, show_progress=True)
        
        embed_time = time.time() - start_time
        print(f"  Embedding time: {embed_time:.2f}s ({embed_time/len(self.chunks):.3f}s per chunk)")
        
        # Step 3: Store in vector database
        print(f"\n💾 Step 3: Storing in vector database...")
        
        # Create database
        if self.use_index:
            self.db = VectorDB(
                dimension=config.vector_dimension,
                metric=config.distance_metric,
                index_type=self.index_type,
                n_clusters=config.n_clusters,
                nprobe=config.nprobe
            )
        else:
            self.db = VectorDB(
                dimension=config.vector_dimension,
                metric=config.distance_metric
            )
        
        # Insert chunks
        ids = [f"chunk_{i}" for i in range(len(self.chunks))]
        # Include text in metadata for reconstruction when loading
        metadata = []
        for chunk in self.chunks:
            meta = chunk.metadata.copy()
            meta['text'] = chunk.text  # Store the actual text
            metadata.append(meta)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        self.db.batch_insert(ids, embeddings_array, metadata)
        print(f"  ✓ Inserted {len(self.chunks)} chunks")
        
        # Build index if using
        if self.use_index:
            print(f"\n🔨 Step 4: Building IVF index...")
            start_time = time.time()
            self.db.build_index()
            index_time = time.time() - start_time
            print(f"  ✓ Index built in {index_time:.2f}s")
        
        print(f"\n✅ Indexing complete!")
        print(f"  Total chunks: {len(self.chunks)}")
        print(f"  Database size: {len(self.db)} vectors")
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search for relevant document chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        if self.db is None:
            raise ValueError("Database not initialized. Call index_documents() first.")
        
        top_k = top_k or config.top_k
        
        # Generate query embedding
        print(f"\n🔍 Searching for: '{query}'")
        start_time = time.time()
        
        query_embedding = self.embedder.embed_query(query)
        embed_time = time.time() - start_time
        
        # Search in database
        search_start = time.time()
        results = self.db.search(
            np.array(query_embedding, dtype=np.float32),
            top_k=top_k
        )
        search_time = time.time() - search_start
        
        # Add chunk text to results
        for result in results:
            chunk_id = int(result['id'].split('_')[1])
            result['text'] = self.chunks[chunk_id].text
        
        print(f"  Query embedding: {embed_time:.3f}s")
        print(f"  Search time: {search_time:.3f}s")
        print(f"  Total time: {embed_time + search_time:.3f}s")
        
        return results
    
    def display_results(self, results: List[Dict], max_text_length: int = 500):
        """
        Display search results in a formatted way.
        
        Args:
            results: Search results
            max_text_length: Maximum characters to show per result
        """
        print(f"\n{'=' * 60}")
        print(f"📋 Top {len(results)} Results:")
        print(f"{'=' * 60}\n")
        
        for i, result in enumerate(results, 1):
            score = result['score']
            source = result['metadata'].get('source', 'Unknown')
            chunk_id = result['metadata'].get('chunk_id', '?')
            text = result['text']
            
            # Truncate text if too long
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            
            # Color code by score
            if score > 0.7:
                score_indicator = "🟢"
            elif score > 0.5:
                score_indicator = "🟡"
            else:
                score_indicator = "🔴"
            
            print(f"{i}. {score_indicator} Score: {score:.4f}")
            print(f"   Source: {source} (chunk #{chunk_id})")
            print(f"   Text: {text}")
            print()
    
    def interactive_search(self):
        """
        Interactive search interface.
        """
        if self.db is None:
            raise ValueError("Database not initialized. Call index_documents() first.")
        
        print(f"\n{'=' * 60}")
        print("🔎 Interactive Search")
        print("=" * 60)
        print("Enter your search queries (or 'quit' to exit)\n")
        
        while True:
            try:
                query = input("Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Search
                results = self.search(query)
                
                # Display
                self.display_results(results)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def save_database(self, path: str):
        """Save the database to disk."""
        if self.db is None:
            raise ValueError("Database not initialized")
        
        print(f"\n💾 Saving database to: {path}")
        self.db.save(path)
        print(f"✓ Database saved")
    
    def load_database(self, path: str):
        """Load a database from disk."""
        print(f"\n📂 Loading database from: {path}")
        self.db = VectorDB.load(path)
        
        # Reconstruct chunks from metadata
        self.chunks = []
        # Get all chunk IDs sorted numerically
        chunk_ids = sorted(
            [chunk_id for chunk_id in self.db._metadata.keys() if chunk_id.startswith('chunk_')],
            key=lambda x: int(x.split('_')[1])
        )
        
        for chunk_id in chunk_ids:
            metadata = self.db._metadata[chunk_id]
            # Create a simple chunk object (namedtuple-like)
            chunk = type('Chunk', (), {
                'text': metadata.get('text', ''),
                'source': metadata['source'],
                'chunk_id': metadata['chunk_id'],
                'start': metadata.get('start', 0),
                'end': metadata.get('end', 0)
            })()
            self.chunks.append(chunk)
        
        print(f"✓ Database loaded ({len(self.db)} vectors)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Document search using Gemini embeddings and vector database"
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        required=True,
        help="Directory containing documents to index"
    )
    parser.add_argument(
        "--index-type",
        type=str,
        choices=["ivf", "none"],
        default="ivf",
        help="Index type to use (default: ivf)"
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Use brute-force search without index"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save database to specified path after indexing"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Load database from specified path (skips indexing)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to run (non-interactive mode)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Create system
    use_index = not args.no_index and args.index_type != "none"
    system = DocumentSearchSystem(
        docs_dir=args.docs_dir,
        use_index=use_index,
        index_type=args.index_type if use_index else None
    )
    
    # Load or index
    if args.load:
        system.load_database(args.load)
    else:
        system.index_documents()
        
        if args.save:
            system.save_database(args.save)
    
    # Search
    if args.query:
        # Single query mode
        config.top_k = args.top_k
        results = system.search(args.query)
        system.display_results(results)
    else:
        # Interactive mode
        system.interactive_search()


if __name__ == "__main__":
    main()

