from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma

class VectorStore:
    def __init__(self, embedding_model, persist_dir: str = None):
        from langchain_chroma import Chroma
        self.store = Chroma(embedding_function=embedding_model, persist_directory=persist_dir, collection_name="datasage")
    
    def add_documents(self, docs: List[Document]):
        self.store.add_documents(docs)

    def search(self, query: str, k: int = 5, filter: dict = None) -> List[Document]:
        return self.store.similarity_search(query, k=k, filter=filter)
