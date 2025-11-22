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
    __all__.extend(["Retriever", "LLMGenerator", "RagEngine"])

# High-level RAG Engine class
class RagEngine:
    """High-level API for RAG - handles everything automatically."""
    
    def __init__(self, data_files, model_name: str = "llama3.1", persist_dir: str = "./vector_db"):
        """
        Initialize RAG engine with automatic document processing.
        
        Args:
            data_files: Path to file(s) to index (string or list)
            model_name: Ollama model name
            persist_dir: Directory for vector database
        """
        if not _has_retrieval:
            raise ImportError("Retrieval components not available. Check installation.")
        
        # Ensure list
        if isinstance(data_files, str):
            data_files = [data_files]
        
        # 1. Initialize indexing engine
        indexer = IndexingEngine(persist_dir=persist_dir)
        
        # 2. Index all files
        for file_path in data_files:
            indexer.index(file_path)
        
        # 3. Set up retriever
        embedder = Embedder()
        self.retriever = Retriever(
            vector_store=indexer.vector_store,
            embedder=embedder
        )
        
        # 4. Set up generator
        self.generator = LLMGenerator(model=model_name)
    
    def query(self, question: str, top_k: int = 5) -> str:
        """
        Ask a question and get an answer based on indexed documents.
        
        Args:
            question: The question to ask
            top_k: Number of relevant documents to retrieve
            
        Returns:
            Generated answer string
        """
        # Retrieve relevant documents
        docs = self.retriever.retrieve(question, k=top_k)
        
        # Generate answer
        answer = self.generator.generate_answer(question, docs)
        
        return answer