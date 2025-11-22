from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from langchain_huggingface import HuggingFaceEmbeddings
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)
