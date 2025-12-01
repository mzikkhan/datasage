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
    def __init__(self, data_files: List[str], model_name: str = "llama3.1", metadata: dict = None):

        # 1) Load documents
        loader = self._choose_loader(data_files)
        docs = loader.load(data_files)

        # 2) Split into chunks
        chunker = TextChunker()
        chunks = chunker.chunk_documents(docs)

        # 3) Initialize embedding and vector store
        embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
        vs = VectorStore(
            embedding_model=embedder.model,
            persist_dir="./vector_db"
        )

        # stores chunks
        vs.add_documents(chunks)  

        # 4) Prepare retriever and generator
        self.retriever = Retriever(vector_store=vs, embedder=embedder)
        self.generator = LLMGenerator(model=model_name)

    def _choose_loader(self, data_files):
        # Ensure list
        if isinstance(data_files, str):
            data_files = [data_files]

        filename = data_files[0].lower()

        if filename.endswith(".pdf"):
            return PDFLoader()
        elif filename.endswith(".csv"):
            return CSVLoader()
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    # Function 1
    def query(self, question: str, top_k: int = 5) -> str:
        docs = self.retriever.retrieve(question, k=top_k)
        answer = self.generator.generate_answer(question, docs)
        return answer

    # Function 2
    def summary(self, topic: str = "", top_k: int = 5) -> str:
        """Generate a summary of the data.
        
        Args:
            topic: Optional topic/query to retrieve relevant documents. If empty, retrieves documents broadly.
            top_k: Number of documents to retrieve for summarization.
            
        Returns:
            A bulleted summary of key ideas from the retrieved documents.
        """
        # Use topic to retrieve relevant documents, or a generic query if no topic
        query_text = topic if topic else "main topics and key information"
        docs = self.retriever.retrieve(query_text, k=top_k)
        
        # Fixed prompt for summarization
        summary_prompt = "Summarize in concise bullet points the key ideas in the data"
        summary = self.generator.generate_answer(summary_prompt, docs)
        return summary