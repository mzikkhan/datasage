from .indexing.index_engine import IndexingEngine
from .indexing.embedder import Embedder
from .indexing.vector_store import VectorStore
from .ingestion.chunker import TextChunker 
from .ingestion.loaders import DocumentLoader
from .ingestion.loaders import PDFLoader
from .ingestion.loaders import CSVLoader
from .retrieval.retriever import Retriever
from .retrieval.generator import LLMGenerator 
from typing import List

__all__ = [
    "IndexingEngine",
    "Embedder", 
    "VectorStore",
    "TextChunker",
    "DocumentLoader",
    "PDFLoader",
    "CSVLoader",
    "Retriever",
    "LLMGenerator"
]

class RagEngine:
    def __init__(self, data_files: List[str], metadata: dict = None):

        # 1) Load documents
        loader = self._choose_loader(data_files)
        docs = loader.load(data_files)

        # 2) Split into chunks
        chunker = TextChunker()
        chunks = chunker.chunk_documents(docs)

        # 3) Initialize embedding and vector store
        embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
        vs = VectorStore(embedding_model=embedder, persist_dir="./vector_db")

        # stores chunks
        vs.add_documents(chunks)  

        # 4) Prepare retriever and generator
        self.retriever = Retriever(vector_store=vs, embedder=embedder)
        self.generator = LLMGenerator(model="llama3.1:latest")

    def query(self, question: str, top_k: int = 5) -> str:
        # Retrieve relevant chunks
        docs = self.retriever.retrieve(question, k=top_k)
        # Generate answer with context
        answer = self.generator.generate_answer(question, docs)
        return answer