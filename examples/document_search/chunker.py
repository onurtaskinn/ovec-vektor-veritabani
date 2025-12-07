"""
Document chunking utilities for splitting text into manageable pieces.
"""

import re
from typing import List, Dict, Any

# Handle both direct execution and package import
try:
    from .config import config
except ImportError:
    from config import config


class TextChunk:
    """Represents a chunk of text with metadata."""
    
    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        """
        Initialize a text chunk.
        
        Args:
            text: The chunk text
            metadata: Optional metadata (source file, position, etc.)
        """
        self.text = text
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"TextChunk(text='{self.text[:50]}...', metadata={self.metadata})"
    
    def __len__(self):
        return len(self.text)


class DocumentChunker:
    """
    Splits documents into chunks for embedding.
    Supports multiple chunking strategies.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        strategy: str = None
    ):
        """
        Initialize document chunker.
        
        Args:
            chunk_size: Target size for each chunk (characters)
            chunk_overlap: Overlap between consecutive chunks
            strategy: Chunking strategy ('simple', 'sentence', 'paragraph')
        """
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
        self.strategy = strategy or config.chunking_strategy
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
    
    def chunk_text(self, text: str, source: str = None) -> List[TextChunk]:
        """
        Chunk text according to the selected strategy.
        
        Args:
            text: Text to chunk
            source: Source identifier (filename, URL, etc.)
            
        Returns:
            List of TextChunk objects
        """
        if self.strategy == "simple":
            return self._chunk_simple(text, source)
        elif self.strategy == "sentence":
            return self._chunk_sentence(text, source)
        elif self.strategy == "paragraph":
            return self._chunk_paragraph(text, source)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
    
    def _chunk_simple(self, text: str, source: str = None) -> List[TextChunk]:
        """
        Simple chunking: fixed size with overlap.
        Fast but may break sentences.
        """
        chunks = []
        text = text.strip()
        
        if len(text) == 0:
            return chunks
        
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # Get chunk text
            chunk_text = text[start:end]
            
            # Create chunk with metadata
            chunk = TextChunk(
                text=chunk_text,
                metadata={
                    'source': source,
                    'chunk_id': chunk_id,
                    'start': start,
                    'end': min(end, len(text)),
                    'strategy': 'simple'
                }
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = start + self.chunk_size - self.chunk_overlap
            chunk_id += 1
        
        return chunks
    
    def _chunk_sentence(self, text: str, source: str = None) -> List[TextChunk]:
        """
        Sentence-based chunking: split on sentences, combine to target size.
        Better semantic boundaries than simple chunking.
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size and we have content, save chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk = TextChunk(
                    text=chunk_text,
                    metadata={
                        'source': source,
                        'chunk_id': chunk_id,
                        'num_sentences': len(current_chunk),
                        'strategy': 'sentence'
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
                
                # Start new chunk with overlap (keep last few sentences)
                if self.chunk_overlap > 0:
                    # Calculate how many sentences to keep for overlap
                    overlap_text = ''
                    overlap_sentences = []
                    for s in reversed(current_chunk):
                        if len(overlap_text) + len(s) < self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_text = ' '.join(overlap_sentences)
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_length = len(overlap_text)
                else:
                    current_chunk = []
                    current_length = 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = TextChunk(
                text=chunk_text,
                metadata={
                    'source': source,
                    'chunk_id': chunk_id,
                    'num_sentences': len(current_chunk),
                    'strategy': 'sentence'
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_paragraph(self, text: str, source: str = None) -> List[TextChunk]:
        """
        Paragraph-based chunking: split on paragraphs, combine to target size.
        Respects natural document structure.
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            # If adding this paragraph exceeds chunk size and we have content, save chunk
            if current_length + paragraph_length > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunk = TextChunk(
                    text=chunk_text,
                    metadata={
                        'source': source,
                        'chunk_id': chunk_id,
                        'num_paragraphs': len(current_chunk),
                        'strategy': 'paragraph'
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
                
                # No overlap for paragraph-based chunking (would break structure)
                current_chunk = []
                current_length = 0
            
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_length += paragraph_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk = TextChunk(
                text=chunk_text,
                metadata={
                    'source': source,
                    'chunk_id': chunk_id,
                    'num_paragraphs': len(current_chunk),
                    'strategy': 'paragraph'
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        Simple implementation using regex.
        """
        # Handle common sentence endings
        text = text.replace('\n', ' ')
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def chunk_documents(
        self,
        documents: Dict[str, str],
        show_progress: bool = True
    ) -> List[TextChunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: Dictionary mapping source names to document text
            show_progress: Whether to show progress
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        if show_progress:
            print(f"Chunking {len(documents)} documents...")
        
        for source, text in documents.items():
            chunks = self.chunk_text(text, source)
            all_chunks.extend(chunks)
            
            if show_progress:
                print(f"  {source}: {len(chunks)} chunks")
        
        if show_progress:
            print(f"✓ Total: {len(all_chunks)} chunks")
        
        return all_chunks


def test_chunker():
    """Test document chunking."""
    print("Testing document chunker...\n")
    
    sample_text = """
    This is the first paragraph. It contains multiple sentences. Each sentence adds meaning.
    
    This is the second paragraph. It discusses a different topic. Chunking helps with retrieval.
    
    The third paragraph continues the discussion. It provides additional context. Document structure matters.
    """
    
    print("1. Simple chunking (chunk_size=100, overlap=20):")
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20, strategy="simple")
    chunks = chunker.chunk_text(sample_text, "test.txt")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: '{chunk.text[:50]}...' (length: {len(chunk)})")
    
    print(f"\n2. Sentence-based chunking (chunk_size=150, overlap=30):")
    chunker = DocumentChunker(chunk_size=150, chunk_overlap=30, strategy="sentence")
    chunks = chunker.chunk_text(sample_text, "test.txt")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk.metadata['num_sentences']} sentences, {len(chunk)} chars")
        print(f"    '{chunk.text[:60]}...'")
    
    print(f"\n3. Paragraph-based chunking (chunk_size=200):")
    chunker = DocumentChunker(chunk_size=200, strategy="paragraph")
    chunks = chunker.chunk_text(sample_text, "test.txt")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk.metadata['num_paragraphs']} paragraphs, {len(chunk)} chars")


if __name__ == "__main__":
    test_chunker()

