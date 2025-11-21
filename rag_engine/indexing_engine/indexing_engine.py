# indexing_engine.py
import os
from pathlib import Path
from typing import List, Union, Dict, Any

import pandas as pd
from langchain.schema import Document
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS

def load_file(path: Union[str, Path]) -> List[Document]:
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    elif ext in [".txt", ".md"]:
        return TextLoader(str(path)).load()
    elif ext == ".csv":
        return CSVLoader(str(path)).load()
    elif ext in [".docx", ".doc"]:
        return UnstructuredWordDocumentLoader(str(path)).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


class IndexingEngine:
    """
    Responsible for:
    - Loading documents
    - Chunking
    - Embeddings
    - Creating/storing the vector index
    """

    def __init__(
        self,
        embed_method: str = "hf",   # "hf" = sentence-transformers, "openai" = OpenAI embeddings
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        vector_store: str = "chroma",  # "chroma" or "faiss"
        persist_dir: str = "./datasage_index",
    ):
        self.embed_method = embed_method
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = vector_store
        self.persist_dir = persist_dir

    # --------------------------
    # Step 1: Ingestion
    # --------------------------
    def ingest(self, data, metadata: Dict[str, Any] = None) -> List[Document]:
        docs = []

        if isinstance(data, (str, Path)) and Path(data).exists():
            docs.extend(load_file(data))

        elif isinstance(data, list) and all(Path(p).exists() for p in data):
            for p in data:
                docs.extend(load_file(p))

        elif isinstance(data, pd.DataFrame):
            for _, row in data.iterrows():
                text = "\n".join(f"{col}: {row[col]}" for col in data.columns)
                docs.append(Document(page_content=text, metadata=metadata or {}))

        elif isinstance(data, str):
            docs.append(Document(page_content=data, metadata=metadata or {}))

        elif isinstance(data, list) and all(isinstance(x, str) for x in data):
            for t in data:
                docs.append(Document(page_content=t, metadata=metadata or {}))

        else:
            raise ValueError("Unsupported data type for ingestion")

        return docs

    # --------------------------
    # Step 2: Chunking
    # --------------------------
    def chunk(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_documents(docs)

    def _create_embeddings(self):
        if self.embed_method == "hf":
            return HuggingFaceEmbeddings(model_name=self.embed_model)
        elif self.embed_method == "openai":
            return OpenAIEmbeddings()
        else:
            raise ValueError("embed_method must be 'hf' or 'openai'")

    def create_index(self, chunks: List[Document]):
        embeddings = self._create_embeddings()

        if self.vector_store == "chroma":
            db = Chroma.from_documents(
                chunks,
                embedding=embeddings,
                persist_directory=self.persist_dir,
            )
            db.persist()
            return db

        elif self.vector_store == "faiss":
            return FAISS.from_documents(chunks, embedding=embeddings)

        else:
            raise ValueError("Unsupported vector_store")

    # --------------------------
    # Main entry point
    # --------------------------
    def build(self, data, metadata=None):
        docs = self.ingest(data, metadata)
        chunks = self.chunk(docs)
        db = self.create_index(chunks)
        return db
