from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

class TextChunker:
    #break up documents into smaller chunks with some overlap so context is not lost
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        #takes a bunch of docs and splits them up
        return self.splitter.split_documents(documents)

    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        #converts raw text into chunked documents
        doc = Document(page_content=text, metadata=metadata or {})
        return self.splitter.split_documents([doc])
