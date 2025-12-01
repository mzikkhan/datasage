from typing import List
import json
import urllib.request
import urllib.error
from .data_models import Document
from ..indexing.vector_store import VectorStore

class OllamaEmbedder:
    """
    A simple Ollama embedder to embed documents and queriesß.
    """
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def _get_embedding(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text
        }
        data = json.dumps(payload).encode("utf-8")
        
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                if response.status != 200:
                    raise Exception(f"Ollama API returned status {response.status}")
                result = json.loads(response.read().decode("utf-8"))
                return result.get("embedding", [])
        except urllib.error.URLError as e:
            raise Exception(f"Failed to connect to Ollama at {self.base_url}: {e}")

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(text) for text in texts]

class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: OllamaEmbedder):
        self.vs = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5, filter: dict = None) -> List[Document]:
        embedding = self.embedder.embed_query(query)
        results = self.vs.store.similarity_search_by_vector(embedding, k=k, filter=filter)
        return [Document(page_content=d.page_content, metadata=d.metadata) for d in results]
