from datasage.ingestion.loaders import PDFLoader, CSVLoader
from datasage.ingestion.chunker import TextChunker
from datasage.indexing.embedder import Embedder
from datasage.indexing.vector_store import VectorStore
from datasage.query.retriever import Retriever
from datasage.query.generator import LLMGenerator
from typing import List

class RagEngine:
    def __init__(self, data_files: List[str], metadata: dict = None):
        # 1) Load documents
        loader = self._choose_loader(data_files)  # e.g. PDFLoader or CSVLoader
        docs = loader.load(data_files)
        # 2) Split into chunks
        chunker = TextChunker()
        chunks = chunker.chunk_documents(docs)
        # 3) Initialize embedding and vector store
        embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
        vs = VectorStore(embedding_model=embedder, persist_dir="./db")
        vs.add_documents(chunks)  # stores embeddings+chunks
        # 4) Prepare retriever and generator
        self.retriever = Retriever(vector_store=vs, embedder=embedder)
        self.generator = LLMGenerator(model="llama3.1:latest")

    def query(self, question: str, top_k: int = 5) -> str:
        # Retrieve relevant chunks
        docs = self.retriever.retrieve(question, k=top_k)
        # Generate answer with context
        answer = self.generator.generate_answer(question, docs)
        return answer
