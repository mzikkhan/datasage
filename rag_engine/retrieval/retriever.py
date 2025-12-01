from typing import List
from .data_models import Document
from ..indexing.vector_store import VectorStore
from ..indexing.embedder import Embedder

class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vs = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5, filter: dict = None) -> List[Document]:
        results = self.vs.search(query, k=k, filter=filter)
        return [Document(page_content=d.page_content, metadata=d.metadata) for d in results]
