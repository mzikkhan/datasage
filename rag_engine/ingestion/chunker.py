from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        # Split each Document into smaller Documents
        return self.splitter.split_documents(documents)

    def chunk_text(self, text: str, metadata: dict = None) -> List[Document]:
        # Split a raw text string into chunks (wrapping in Document)
        docs = [Document(page_content=text, metadata=metadata)]
        return self.splitter.split_documents(docs)
