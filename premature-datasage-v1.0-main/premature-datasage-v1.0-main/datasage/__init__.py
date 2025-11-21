# datasage/__init__.py

# Only import what's commonly used
from datasage.indexing.index_engine import IndexingEngine
from datasage.indexing.embedder import Embedder
from datasage.indexing.vector_store import VectorStore
from datasage.ingestion.chunker import TextChunker

# Don't import generator/retriever at package level to avoid dependency issues
# Users can import them directly when needed:
# from datasage.query.generator import LLMGenerator

__all__ = [
    "IndexingEngine",
    "Embedder", 
    "VectorStore",
    "TextChunker",
]