from typing import List
from langchain_core.documents import Document
from indexing.vector_store import VectorStore
from indexing.embedder import Embedder

class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vs = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5, filter: dict = None) -> List[Document]:
        return self.vs.search(query, k=k, filter=filter)
