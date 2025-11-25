import os
from typing import List, Optional
from langchain_core.documents import Document
from ..ingestion.chunker import TextChunker
from ..ingestion.loaders import (
    DocumentLoader,
    PDFLoader,
    CSVLoader,
)
from .embedder import Embedder
from .vector_store import VectorStore


class IndexingEngine:
    """
    High-level pipeline that:
      1. chooses correct loader based on file type
      2. loads raw documents
      3. optionally applies metadata
      4. chunks documents
      5. stores chunks inside the VectorStore
    """

    def __init__(
        self,
        persist_dir: Optional[str] = "./datasage_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        overlap: int = 200,
    ):
        # Components already written in your package
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedder = Embedder(model_name=embedding_model)

        # Vector store requires the low-level embedding function
        self.vector_store = VectorStore(
            embedding_model=self.embedder.model,
            persist_dir=persist_dir,
        )

        self.persist_dir = persist_dir

    # ----------------------------------------------------------------------
    # Loader selector based on extension
    # ----------------------------------------------------------------------
    def _get_loader(self, file_path: str) -> DocumentLoader:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return PDFLoader()
        elif ext == ".csv":
            return CSVLoader()
        elif ext == ".txt":
            # lightweight inline loader since your code doesn't include TextLoader
            from langchain_core.documents import Document
            class _TxtLoader(DocumentLoader):
                def load(self, fp: str) -> List[Document]:
                    with open(fp, "r", encoding="utf-8") as f:
                        return [Document(page_content=f.read(), metadata={"source": fp})]

            return _TxtLoader()

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ----------------------------------------------------------------------
    # MAIN API METHOD
    # ----------------------------------------------------------------------
    def index(self, file_path: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Runs the full indexing pipeline — returns chunked documents.
        """

        # 1. Load using correct loader
        loader = self._get_loader(file_path)
        docs = loader.load(file_path)

        # 2. Apply metadata to each raw document
        if metadata:
            for d in docs:
                d.metadata.update(metadata)

        # 3. Split into chunks
        chunks = self.chunker.chunk_documents(docs)

        # 4. Store chunks in vector store
        self.vector_store.add_documents(chunks)

        # done
        return chunks

    # ----------------------------------------------------------------------
    def search(self, query: str, k: int = 5, filter: Optional[dict] = None):
        """
        Convenience method: search the index directly.
        """
        return self.vector_store.search(query, k=k, filter=filter)
s