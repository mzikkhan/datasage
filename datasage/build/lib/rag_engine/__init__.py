"""DataSage: A local RAG engine for document Q&A"""

__version__ = "0.1.0"

# Import core components using relative imports
from .indexing.index_engine import IndexingEngine
from .indexing.embedder import Embedder
from .indexing.vector_store import VectorStore
from .ingestion.chunker import TextChunker
from .ingestion.loaders import DocumentLoader, PDFLoader, CSVLoader

# Only import retrieval components if they exist
try:
    from .retrieval.retriever import Retriever
    from .retrieval.generator import LLMGenerator
    _has_retrieval = True
except ImportError as e:
    print(f"Warning: Could not import retrieval components: {e}")
    _has_retrieval = False
    Retriever = None
    LLMGenerator = None

__all__ = [
    "IndexingEngine",
    "Embedder",
    "VectorStore",
    "TextChunker",
    "DocumentLoader",
    "PDFLoader",
    "CSVLoader",
]

if _has_retrieval:
    __all__.extend(["Retriever", "LLMGenerator"])